"""Derive human-readable names from PyMC models.

Used by cache (directory names) and notify (topic names). Tries three
strategies in order:

1. model.name (explicit PyMC model name)
2. Caller's variable name (stack frame introspection)
3. Free RV names (e.g. "mu_tau_theta")
"""

from __future__ import annotations

import inspect
import re


def get_model_name(model, stack_offset: int = 2) -> str:
    """Get the best human-readable name for a PyMC model.

    Args:
        model: A PyMC model object.
        stack_offset: How many frames to go up to find the caller's variable.
                      Default 2 works when called from a function that was
                      called by user code.
    """
    # 1. Explicit model name
    if model is not None and hasattr(model, "name") and model.name:
        return model.name

    # 2. Introspect the variable name from the caller's frame
    var_name = _introspect_var_name(model, stack_offset)
    if var_name and var_name not in ("model", "m", "self", "_"):
        return var_name

    # 3. Derive from free RV names
    if model is not None and hasattr(model, "free_RVs") and model.free_RVs:
        names = [rv.name.split("::")[-1] for rv in model.free_RVs[:4]]
        result = "_".join(names)
        if len(model.free_RVs) > 4:
            result += f"_plus{len(model.free_RVs) - 4}"
        return result

    return "unnamed"


def _introspect_var_name(obj, stack_offset: int) -> str | None:
    """Try to find the variable name bound to obj in a caller's frame."""
    try:
        frame = inspect.currentframe()
        for _ in range(stack_offset + 1):  # +1 for this function's own frame
            if frame is None:
                return None
            frame = frame.f_back
        if frame is None:
            return None
        for name, val in frame.f_locals.items():
            if val is obj:
                return name
    except Exception:
        pass
    finally:
        del frame  # avoid reference cycles
    return None


def slugify(name: str, separator: str = "_") -> str:
    """Convert a name to a filesystem/URL-safe slug."""
    return re.sub(r"[^a-zA-Z0-9]+", separator, name).strip(separator).lower()
