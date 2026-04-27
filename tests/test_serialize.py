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
    print(f"Payload size: {size:.2f} MB")


def test_deserialized_model_can_sample():
    """Verify that a round-tripped model can actually run pm.sample()."""
    model = _make_eight_schools()
    serialized = serialize_model(model)
    deserialized = deserialize_model(serialized)

    with deserialized:
        idata = pm.sample(draws=10, tune=10, chains=1, progressbar=False)

    assert "posterior" in idata.groups()
    assert "mu" in idata.posterior
    assert "tau" in idata.posterior
    assert "theta" in idata.posterior
