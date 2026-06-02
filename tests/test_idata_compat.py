"""arviz 0.x / 1.x compat shims and cache-key stability."""

import numpy as np
import pymc as pm

from cloudposterior._idata import (
    ess_tail,
    get_group,
    group_names,
    sanitize_inference_data,
)
from cloudposterior.naming import _kwarg_token, cache_key


def _idata():
    with pm.Model():
        pm.Normal("mu", 0, 1)
        pm.Normal("y", 0, 1, observed=np.zeros(5))
        return pm.sample(draws=6, tune=6, chains=2, nuts_sampler="pymc", progressbar=False, random_seed=0)


# -- compat shims -----------------------------------------------------------

def test_group_names_clean_and_version_agnostic():
    names = group_names(_idata())
    assert "posterior" in names
    assert all("/" not in n and n for n in names)  # no slashes, no empty root


def test_ess_tail_returns_positive_number():
    val = ess_tail(np.random.default_rng(0).standard_normal((2, 200)))
    assert val > 0


def test_sanitize_makes_dict_attr_netcdf_safe(tmp_path):
    idata = _idata()
    grp = get_group(idata, "posterior")
    grp.attrs["nutpie_like"] = {"settings": {"a": 1}, "b": [1, 2, 3]}  # dict-valued attr
    sanitize_inference_data(idata)
    # The dict attr must now be a serializable scalar (JSON string).
    assert isinstance(get_group(idata, "posterior").attrs.get("nutpie_like"), str)
    path = tmp_path / "out.nc"
    idata.to_netcdf(str(path))  # must not raise
    assert path.exists()


# -- cache-key stability (D6) ----------------------------------------------

def test_cache_key_order_independent():
    mb = b"model-bytes"
    assert cache_key(mb, {"draws": 1000, "chains": 4}) == cache_key(mb, {"chains": 4, "draws": 1000})


def test_cache_key_array_kwarg_is_deterministic():
    mb = b"m"
    a = np.array([1.0, 2.0, 3.0])
    assert cache_key(mb, {"initval": a}) == cache_key(mb, {"initval": a.copy()})
    assert cache_key(mb, {"initval": a}) != cache_key(mb, {"initval": a + 1})


def test_cache_key_rng_kwarg_does_not_crash():
    # numpy Generators have an unstable repr; the key must still compute.
    key = cache_key(b"m", {"random_seed": np.random.default_rng(0)})
    assert isinstance(key, str) and len(key) == 64


def test_kwarg_token_array_stable():
    a = np.arange(5)
    assert _kwarg_token(a) == _kwarg_token(a.copy())
    assert _kwarg_token(a) != _kwarg_token(a + 1)
