"""PyMC 5 / PyMC 6 API differences the interceptors sit on top of.

PyMC 6 changed several things this library wraps directly: nuts_sampler
auto-selects nutpie, tune resolves per sampler, sample_posterior_predictive's
var_names became output-only, and a backend= kwarg appeared.
"""

import pymc as pm
import pytest

from cloudposterior._idata import pymc_major
from cloudposterior.api import _validate_sample_kwargs, resolve_tune

PYMC6 = pymc_major() >= 6


def test_pymc_major_matches_the_installed_version():
    assert pymc_major() == int(pm.__version__.split(".")[0])


def test_explicit_tune_is_used_verbatim():
    assert resolve_tune({"tune": 250}, "nutpie") == 250
    assert resolve_tune({"tune": 250}, "pymc") == 250


def test_unset_tune_resolves_per_sampler():
    """PyMC 6 made tune=None the default and resolves it per sampler, so a
    progress bar assuming 1000 mis-totals a nutpie run."""
    assert resolve_tune({}, "pymc") == 1000
    assert resolve_tune({}, "nutpie") == (400 if PYMC6 else 1000)


@pytest.mark.skipif(PYMC6, reason="backend= exists on PyMC 6")
def test_backend_kwarg_is_rejected_on_pymc5():
    with pytest.raises(TypeError, match="backend= requires PyMC 6"):
        _validate_sample_kwargs({"backend": "numba"})


@pytest.mark.skipif(not PYMC6, reason="backend= is PyMC 6 only")
def test_backend_kwarg_is_accepted_on_pymc6():
    _validate_sample_kwargs({"backend": "numba"})  # must not raise


@pytest.mark.skipif(not PYMC6, reason="the semantic change is PyMC 6 only")
def test_posterior_predictive_var_names_warns_on_pymc6():
    """var_names no longer decides what is resampled, so 5.x code that used it
    to force deterministics to recompute silently gets different results."""
    from cloudposterior.api import _warn_predictive_kwarg_drift

    with pytest.warns(UserWarning, match="only\\s+selects which variables are stored"):
        _warn_predictive_kwarg_drift("posterior", {"var_names": ["y"]})


@pytest.mark.skipif(not PYMC6, reason="the semantic change is PyMC 6 only")
def test_no_warning_when_sample_vars_is_explicit():
    import warnings

    from cloudposterior.api import _warn_predictive_kwarg_drift

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _warn_predictive_kwarg_drift(
            "posterior", {"var_names": ["y"], "sample_vars": ["y"]}
        )


def test_manifest_pins_the_arviz_metapackage_when_present():
    """Under arviz 1.x the effective API comes from arviz-base/-stats/-plots,
    so an unpinned container can resolve a different API than the client."""
    import importlib.util

    from cloudposterior.serialize import get_version_manifest

    manifest = get_version_manifest()
    if importlib.util.find_spec("arviz_base") is not None:
        assert "arviz-base" in manifest
    assert "pymc" in manifest and "cloudpickle" in manifest
