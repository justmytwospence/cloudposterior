"""Derive human-readable and machine-correct identifiers from PyMC models.

Used by cache (directory names + keys), notify (topic names), and volumes.

Two layers:
- Human-readable: model_slug (for directory browsability)
- Machine-correct: payload_hash, cache_key (for identity checks)
"""

from __future__ import annotations

import hashlib
import re


def get_model_name(model) -> str:
    """Get the best human-readable name for a PyMC model.

    Tries two strategies:
    1. model.name (explicit PyMC model name)
    2. Free RV names (e.g. "mu_tau_theta")
    """
    # 1. Explicit model name
    if model is not None and hasattr(model, "name") and model.name:
        return model.name

    # 2. Derive from free RV names
    if model is not None and hasattr(model, "free_RVs") and model.free_RVs:
        names = [rv.name.split("::")[-1] for rv in model.free_RVs[:4]]
        result = "_".join(names)
        if len(model.free_RVs) > 4:
            result += f"_plus{len(model.free_RVs) - 4}"
        return result

    return "unnamed"


def slugify(name: str, separator: str = "_") -> str:
    """Convert a name to a filesystem/URL-safe slug."""
    return re.sub(r"[^a-zA-Z0-9]+", separator, name).strip(separator).lower()


def model_slug(model) -> str:
    """Filesystem-safe slug from a PyMC model name, e.g. 'radon'."""
    return slugify(get_model_name(model))


def payload_hash(model_bytes: bytes) -> str:
    """SHA-256 hex prefix of serialized model bytes (16 chars).

    Used for Volume payload filenames. Captures model + data identity
    since PyMC bundles observed data into the model pickle.
    """
    return hashlib.sha256(model_bytes).hexdigest()[:16]


def data_digest(model) -> str:
    """Content hash of a model's data arrays: pm.Data and observed values.

    Covers what the logp graph text does not. Constant observed arrays are
    inlined into the graph, but ``pm.Data`` values live in shared variables the
    graph only references by name -- so both are hashed here, by content rather
    than by any summary statistic (a permutation or a sign flip must register).
    """
    import numpy as np

    h = hashlib.sha256()

    def _feed(label: str, arr) -> None:
        arr = np.ascontiguousarray(np.asarray(arr))
        h.update(f"{label}|{arr.shape}|{arr.dtype}\x00".encode())
        h.update(arr.tobytes())

    try:
        from pytensor.compile.sharedvalue import SharedVariable

        for name in sorted(model.named_vars):
            var = model.named_vars[name]
            if isinstance(var, SharedVariable):
                try:
                    _feed(f"shared:{name}", var.get_value(borrow=True))
                except Exception:
                    continue

        for rv in sorted(model.observed_RVs, key=lambda v: v.name):
            value = model.rvs_to_values.get(rv)
            data = getattr(value, "data", None)
            if data is None:
                data = getattr(value, "value", None)
            if data is None:
                continue
            try:
                _feed(f"observed:{rv.name}", data)
            except Exception:
                continue
    except Exception:
        return ""
    return h.hexdigest()


def model_digest(model) -> str:
    """Structural identity of a PyMC model: its graph plus its data.

    Deliberately *not* a hash of the cloudpickle bytes. Rebuilding the same
    model in a new interpreter produces different pickle bytes (each RV's
    shared RNG is re-seeded from system entropy at build time), and forward
    sampling advances that state in place -- so a bytes hash makes the
    persistent disk cache miss every time across sessions. The logp graph text
    is stable across processes and sensitive to priors, structure, and inlined
    constants; data_digest covers the mutable arrays it only references.
    """
    h = hashlib.sha256()

    graph_text = ""
    try:
        import pytensor

        graph_text = pytensor.printing.debugprint(
            model.logp(), file="str", print_type=True
        )
    except Exception:
        try:
            graph_text = model.str_repr()
        except Exception:
            graph_text = ""
    h.update(b"graph\x00")
    h.update(graph_text.encode())

    # Variable names and static types: cheap, and keeps two models with the
    # same logp text but different naming/shape metadata distinct.
    h.update(b"vars\x00")
    try:
        for name in sorted(model.named_vars):
            h.update(f"{name}|{model.named_vars[name].type}\x00".encode())
    except Exception:
        pass

    h.update(b"coords\x00")
    try:
        for cname in sorted(model.coords or {}):
            values = model.coords[cname]
            rendered = None if values is None else tuple(str(v) for v in values)
            h.update(f"{cname}|{rendered}\x00".encode())
    except Exception:
        pass

    h.update(b"data\x00")
    h.update(data_digest(model).encode())
    return h.hexdigest()


def _kwarg_token(v) -> str:
    """Deterministic string for a sample kwarg value.

    ``str(v)`` is unstable for numpy arrays (whitespace/truncation), random
    Generators, and callables (their reprs embed a memory address), which would
    scramble the cache key. Arrays and generator states hash by content;
    callables reduce to their qualified name.
    """
    try:
        import numpy as np

        if isinstance(v, np.ndarray):
            digest = hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest()[:16]
            return f"ndarray:{v.shape}:{v.dtype}:{digest}"
        if isinstance(v, np.random.Generator):
            # Hash the actual bit-generator state. Collapsing every Generator
            # to one constant made two runs with genuinely different streams
            # share a cache entry.
            state = hashlib.sha256(repr(v.bit_generator.state).encode()).hexdigest()
            return f"rng:{state[:16]}"
        if isinstance(v, np.random.RandomState):
            state = hashlib.sha256(repr(v.get_state()).encode()).hexdigest()
            return f"rng:{state[:16]}"
        if isinstance(v, np.generic):
            return repr(v.item())
    except Exception:
        pass
    if isinstance(v, (list, tuple)):
        return "[" + ",".join(_kwarg_token(x) for x in v) + "]"
    if isinstance(v, dict):
        return "{" + ",".join(f"{k}:{_kwarg_token(val)}" for k, val in sorted(v.items())) + "}"
    if callable(v):
        # repr() would embed an address, so a step= or callback= kwarg could
        # never hit its own cache entry twice.
        module = getattr(v, "__module__", "?")
        qualname = getattr(v, "__qualname__", None) or getattr(v, "__name__", None)
        if qualname is None:
            qualname = type(v).__qualname__
        return f"callable:{module}.{qualname}"
    return repr(v)


def cache_key(model_identity: str | bytes, sample_kwargs: dict) -> str:
    """Full SHA-256 of model identity + sampling config.

    ``model_identity`` is a ``model_digest`` string (bytes are accepted and
    hashed directly, for callers that only hold a serialized payload).
    """
    h = hashlib.sha256()
    if isinstance(model_identity, bytes):
        h.update(model_identity)
    else:
        h.update(model_identity.encode())
    # Length-delimit so ("ab", "c") and ("a", "bc") can't collide.
    for k, v in sorted(sample_kwargs.items()):
        token = _kwarg_token(v)
        h.update(f"{len(k)}:{k}={len(token)}:{token}\x00".encode())
    return h.hexdigest()
