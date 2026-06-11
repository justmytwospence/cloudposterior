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


def add_group(idata, name: str, group) -> None:
    """Add ``group`` to ``idata`` under ``name`` in place, across arviz majors.

    Used to merge a remotely-computed group (e.g. ``log_likelihood``) into the
    caller's local idata so cloudposterior matches PyMC's
    ``extend_inferencedata=True`` in-place semantics. ``group`` may be an
    xarray Dataset or a DataTree node; it is normalized to a Dataset.
    """
    import xarray as xr

    ds = group
    if not isinstance(group, xr.Dataset):
        to_ds = getattr(group, "to_dataset", None)
        if callable(to_ds):
            try:
                ds = to_ds()
            except Exception:
                ds = group

    # arviz 0.x InferenceData exposes add_groups({name: ds}).
    adder = getattr(idata, "add_groups", None)
    if callable(adder):
        try:
            adder({name: ds})
            return
        except Exception:
            pass

    # arviz 1.x DataTree (and general fallback): item assignment creates a child.
    try:
        idata[name] = ds
        return
    except Exception:
        pass

    setattr(idata, name, ds)


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


_ess_tail_fallback_warned = False


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
        # Last resort: bulk ESS. Warn once -- silently relabeling bulk as tail
        # would misreport convergence across future arviz API changes.
        global _ess_tail_fallback_warned
        if not _ess_tail_fallback_warned:
            _ess_tail_fallback_warned = True
            import warnings

            warnings.warn(
                "arviz tail-ESS API unavailable; reporting bulk ESS in place "
                "of tail ESS.", stacklevel=2,
            )
        return float(az.ess(arr))


def sanitize_inference_data(idata):
    """Make all attrs (top-level + each group) NetCDF-serializable, in place.

    nutpie stores a dict-valued ``sample_stats`` attr that xarray's NetCDF writer
    rejects (it only accepts str/Number/ndarray/list/tuple/bytes). Any other value
    is JSON-encoded. Idempotent for already-clean objects.
    """
    import json

    try:
        import numpy as np

        # np.bool_ is not an np.number; without it boolean attrs (which the
        # NetCDF writer accepts) would be needlessly JSON-encoded.
        ok = (str, bytes, int, float, list, tuple, np.ndarray, np.number, np.bool_)
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

    def _coerce_object_datavars(group):
        """Coerce object-dtype numeric data variables to float64 in place.

        PyMC's SMC sample_stats (beta, accept_rate, log_marginal_likelihood)
        come back as object arrays of mixed Python float/int that NetCDF can't
        write (even native ``idata.to_netcdf()`` raises). The values are regular
        (chain x stage) numbers, so float64 is lossless.
        """
        data_vars = getattr(group, "data_vars", None)
        if data_vars is None:
            return
        for name in list(data_vars):
            try:
                if group[name].dtype == object:
                    group[name] = group[name].astype("float64")
            except (ValueError, TypeError, KeyError):
                pass

    _fix(group_attrs(idata, None))
    for name in group_names(idata):
        _fix(group_attrs(idata, name))
        _coerce_object_datavars(get_group(idata, name))
    return idata
