"""arviz 0.x / 1.x compatibility shims.

PyMC 6 hard-requires ``arviz>=1.1``, a DataTree-based rewrite that:

- makes ``InferenceData.groups()`` a ``DataTree.groups`` *property* returning
  slash-prefixed paths (``/posterior``) plus a ``/`` root,
- removes ``arviz.convert_to_inference_data``,
- changes ``arviz.ess(..., method="tail")`` to require a ``prob`` argument,
- returns ``xarray.DataTree`` from samplers / ``from_netcdf`` instead of
  ``arviz.InferenceData``.

PyMC 5 ships arviz 0.x with the old API. These helpers work on both majors so
cloudposterior (and the remote worker) stay version-agnostic.
"""

from __future__ import annotations


def group_names(idata) -> list[str]:
    """Clean group names (no leading slash, no DataTree root) for both majors."""
    groups = idata.groups
    raw = list(groups() if callable(groups) else groups)
    out = []
    for name in raw:
        name = name.strip("/")
        if name:  # drop the DataTree root group ("/")
            out.append(name)
    return out


def get_group(idata, name):
    """Return a group dataset/node by name for both majors (attr or item access)."""
    try:
        group = getattr(idata, name)
        if group is not None:
            return group
    except Exception:
        pass
    try:
        return idata[name]
    except Exception:
        return None


def group_attrs(idata, name=None):
    """attrs dict of a group, or the top-level attrs when ``name`` is None."""
    if name is None:
        return getattr(idata, "attrs", None)
    return getattr(get_group(idata, name), "attrs", None)


def load_all(idata) -> None:
    """Best-effort eager load of every group so a temp NetCDF file can be deleted."""
    for name in group_names(idata):
        loader = getattr(get_group(idata, name), "load", None)
        if callable(loader):
            try:
                loader()
            except Exception:
                pass


def to_inference_data(trace):
    """Normalize a sampler result to an arviz object across majors.

    arviz 0.x: convert non-InferenceData via ``convert_to_inference_data``.
    arviz 1.x: nutpie / pm.sample already return a DataTree -- use it as-is.
    """
    import arviz as az

    inference_data = getattr(az, "InferenceData", None)
    if inference_data is not None and isinstance(trace, inference_data):
        return trace
    conv = getattr(az, "convert_to_inference_data", None)
    if conv is not None:
        try:
            return conv(trace)
        except Exception:
            pass
    return trace


def ess_tail(arr) -> float:
    """Tail-ESS across majors (arviz 1.x changed ``ess(method="tail")``)."""
    import arviz as az

    try:
        return float(az.ess(arr, method="tail"))
    except TypeError:
        for kwargs in ({"method": "tail", "prob": (0.025, 0.975)}, {"method": "tail", "prob": 0.05}):
            try:
                return float(az.ess(arr, **kwargs))
            except Exception:
                continue
        return float(az.ess(arr))  # last resort: bulk ESS


def sanitize_inference_data(idata):
    """Make all attrs (top-level + each group) NetCDF-serializable, in place.

    nutpie stores a dict-valued ``sample_stats`` attr that xarray's NetCDF writer
    rejects (it only accepts str/Number/ndarray/list/tuple/bytes). Any other value
    is JSON-encoded. Idempotent for already-clean objects.
    """
    import json

    try:
        import numpy as np

        ok = (str, bytes, int, float, list, tuple, np.ndarray, np.number)
    except Exception:
        ok = (str, bytes, int, float, list, tuple)

    def _fix(attrs):
        if not isinstance(attrs, dict):
            return
        for key, value in list(attrs.items()):
            if not isinstance(value, ok):
                try:
                    attrs[key] = json.dumps(value, default=str)
                except Exception:
                    attrs[key] = str(value)

    _fix(group_attrs(idata, None))
    for name in group_names(idata):
        _fix(group_attrs(idata, name))
    return idata
