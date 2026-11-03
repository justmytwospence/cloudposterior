"""Test serialization round-trip without Modal."""

import numpy as np
import pymc as pm

from cloudposterior.serialize import (
    create_payload,
    deserialize_model,
    get_version_manifest,
    payload_size_mb,
    serialize_model,
)


def _make_eight_schools():
    y = np.array([28, 8, -3, 7, -1, 1, 18, 12], dtype=np.float64)
    sigma = np.array([15, 10, 16, 11, 9, 11, 10, 18], dtype=np.float64)
    with pm.Model() as model:
        mu = pm.Normal("mu", 0, 5)
        tau = pm.HalfCauchy("tau", 5)
        theta = pm.Normal("theta", mu=mu, sigma=tau, shape=8)
        pm.Normal("obs", mu=theta, sigma=sigma, observed=y)
    return model


def test_version_manifest():
    manifest = get_version_manifest()
    assert "python" in manifest
    assert "pymc" in manifest
    assert "numpy" in manifest
    assert "pytensor" in manifest


def test_model_serialize_roundtrip():
    model = _make_eight_schools()
    serialized = serialize_model(model)
    assert isinstance(serialized, bytes)
    assert len(serialized) > 0

    deserialized = deserialize_model(serialized)
    orig_names = sorted(rv.name for rv in model.free_RVs)
    deser_names = sorted(rv.name for rv in deserialized.free_RVs)
    assert orig_names == deser_names


def test_observed_data_survives_roundtrip():
    """Observed data is bundled inside the cloudpickled model -- verify it
    survives the serialize/deserialize round-trip without a separate payload."""
    model = _make_eight_schools()
    deserialized = deserialize_model(serialize_model(model))

    orig_obs = {rv.name: rv for rv in model.observed_RVs}
    deser_obs = {rv.name: rv for rv in deserialized.observed_RVs}
    assert set(orig_obs) == set(deser_obs)
    for name in orig_obs:
        orig_data = np.asarray(orig_obs[name].tag.observations.data)
        deser_data = np.asarray(deser_obs[name].tag.observations.data)
        np.testing.assert_array_equal(orig_data, deser_data)


def test_create_payload():
    model = _make_eight_schools()
    payload = create_payload(model, {"draws": 100, "tune": 50})
    assert payload.model_bytes
    assert "pymc" in payload.version_manifest
    assert payload.sample_kwargs == {"draws": 100, "tune": 50}

    size = payload_size_mb(payload)
    assert size > 0


def test_deserialized_model_can_sample():
    """Verify that a round-tripped model can actually run pm.sample()."""
    model = _make_eight_schools()
    serialized = serialize_model(model)
    deserialized = deserialize_model(serialized)

    with deserialized:
        idata = pm.sample(draws=10, tune=10, chains=1, progressbar=False)

    from cloudposterior._idata import group_names

    assert "posterior" in group_names(idata)
    assert "mu" in idata.posterior.data_vars
    assert "tau" in idata.posterior.data_vars
    assert "theta" in idata.posterior.data_vars


def test_deserialize_inference_data_is_eagerly_loaded(tmp_path, monkeypatch):
    """The temp NetCDF file is deleted on return, so every group must be
    eagerly loaded -- a lazy idata would fail when xarray's file manager
    reopens the deleted path."""
    import arviz as az
    import numpy as np

    from cloudposterior._idata import add_group, group_names
    from cloudposterior.serialize import (
        deserialize_inference_data,
        serialize_inference_data,
    )

    idata = az.InferenceData()
    add_group(idata, "posterior", az.dict_to_dataset({"mu": np.random.randn(2, 50)}))

    out = deserialize_inference_data(serialize_inference_data(idata))
    assert "posterior" in group_names(out)

    # Force-close any cached file handles; values must still be readable
    # because the data was loaded into memory before the temp file vanished.
    import xarray as xr

    xr.backends.file_manager.FILE_CACHE.clear()
    assert out.posterior["mu"].values.shape == (2, 50)


def test_wire_roundtrip_preserves_object_nan_and_warning_none():
    """Nutpie can emit NaN missing messages; PyMC emits all-None warnings."""
    import math

    import arviz as az
    import xarray as xr

    from cloudposterior._idata import add_group
    from cloudposterior.serialize import deserialize_inference_data, serialize_inference_data

    messages = np.array([[float("nan"), None, "diverged"]], dtype=object)
    warnings = np.array([[None, None, None]], dtype=object)
    idata = az.InferenceData()
    add_group(idata, "posterior", az.dict_to_dataset({"mu": np.zeros((1, 3))}))
    # az.dict_to_dataset normalizes None to NaN when a float NaN is also
    # present; build the diagnostic group directly to exercise the wire codec.
    add_group(idata, "sample_stats", xr.Dataset({
        "divergence_message": (("chain", "draw"), messages),
        "warning": (("chain", "draw"), warnings),
    }))
    # xarray also treats None as missing during Variable construction; restore
    # the actual mixed object values before the wire boundary being tested.
    diagnostic = idata.sample_stats["divergence_message"]
    diagnostic.values[0, 0] = float("nan")
    diagnostic.values[0, 1] = None
    diagnostic.values[0, 2] = "diverged"
    idata.sample_stats["warning"].values[:] = None
    assert diagnostic.dtype == object
    assert type(diagnostic.values[0, 0]) is float and math.isnan(diagnostic.values[0, 0])
    assert diagnostic.values[0, 1] is None
    assert diagnostic.values[0, 2] == "diverged"
    assert idata.sample_stats["warning"].dtype == object
    assert all(value is None for value in idata.sample_stats["warning"].values.flat)
    loaded = deserialize_inference_data(serialize_inference_data(idata))
    actual = loaded.sample_stats["divergence_message"].values
    assert type(actual[0, 0]) is float and math.isnan(actual[0, 0])
    assert actual[0, 1:].tolist() == [None, "diverged"]
    assert loaded.sample_stats["warning"].values.tolist() == warnings.tolist()
    assert loaded.sample_stats["warning"].values.dtype == object


def test_wire_roundtrip_preserves_nutpie_divergence_messages():
    """The cache codec preserves tagged None/text/bytes across LZ4+NetCDF."""
    import arviz as az

    from cloudposterior._idata import add_group
    from cloudposterior.serialize import deserialize_inference_data, serialize_inference_data

    messages = np.array([[None, "bad energy", b"raw"],
                         ["second", None, b"\x00binary"]], dtype=object)
    idata = az.InferenceData()
    add_group(idata, "posterior", az.dict_to_dataset({"mu": np.zeros((2, 3))}))
    add_group(idata, "sample_stats", az.dict_to_dataset({"divergence_message": messages}))
    roundtripped = deserialize_inference_data(serialize_inference_data(idata))
    actual = roundtripped.sample_stats["divergence_message"].values
    assert actual.tolist() == messages.tolist()
    assert [type(value) for value in actual.flat] == [type(value) for value in messages.flat]
    assert idata.sample_stats["divergence_message"].values.dtype == object
