"""Test naming utilities."""

import numpy as np
import pymc as pm

from cloudposterior.naming import (
    cache_key,
    data_slug,
    get_model_name,
    model_slug,
    payload_hash,
    slugify,
)


def _make_model(name=""):
    y = np.array([1.0, 2.0, 3.0])
    with pm.Model(name=name) as model:
        mu = pm.Normal("mu", 0, 1)
        pm.Normal("obs", mu, 1, observed=y)
    return model


def test_slugify():
    assert slugify("Eight Schools") == "eight_schools"
    assert slugify("my-model v2") == "my_model_v2"
    assert slugify("UPPER") == "upper"
    assert slugify("a/b/c", separator="-") == "a-b-c"


def test_get_model_name_explicit():
    model = _make_model("radon")
    assert get_model_name(model) == "radon"


def test_get_model_name_from_rvs():
    model = _make_model()
    name = get_model_name(model)
    assert "mu" in name


def test_model_slug_named():
    model = _make_model("Eight Schools")
    assert model_slug(model) == "eight_schools"


def test_model_slug_unnamed():
    model = _make_model()
    slug = model_slug(model)
    assert "mu" in slug


def test_data_slug_deterministic():
    data = b"some observed data bytes"
    slug1 = data_slug(data)
    slug2 = data_slug(data)
    assert slug1 == slug2
    assert slug1.startswith("data-")


def test_data_slug_changes_with_data():
    slug1 = data_slug(b"data version 1")
    slug2 = data_slug(b"data version 2")
    assert slug1 != slug2


def test_payload_hash_deterministic():
    model_bytes = b"serialized model bytes"
    h1 = payload_hash(model_bytes)
    h2 = payload_hash(model_bytes)
    assert h1 == h2
    assert len(h1) == 16  # hex prefix


def test_payload_hash_changes_with_model():
    h1 = payload_hash(b"model version 1")
    h2 = payload_hash(b"model version 2")
    assert h1 != h2


def test_cache_key_deterministic():
    model_bytes = b"serialized model"
    kwargs = {"draws": 1000, "tune": 500}
    k1 = cache_key(model_bytes, kwargs)
    k2 = cache_key(model_bytes, kwargs)
    assert k1 == k2
    assert len(k1) == 64  # full SHA-256 hex


def test_cache_key_changes_with_kwargs():
    model_bytes = b"serialized model"
    k1 = cache_key(model_bytes, {"draws": 1000})
    k2 = cache_key(model_bytes, {"draws": 2000})
    assert k1 != k2


def test_cache_key_changes_with_model():
    k1 = cache_key(b"model v1", {"draws": 1000})
    k2 = cache_key(b"model v2", {"draws": 1000})
    assert k1 != k2


def test_cache_key_kwargs_order_independent():
    model_bytes = b"model"
    k1 = cache_key(model_bytes, {"draws": 1000, "tune": 500})
    k2 = cache_key(model_bytes, {"tune": 500, "draws": 1000})
    assert k1 == k2
