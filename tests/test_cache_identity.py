"""Cache identity is structural, not a hash of the pickle bytes.

Rebuilding the same model in a new interpreter produces different cloudpickle
bytes (every RV's shared RNG is re-seeded from system entropy at build time),
so a bytes-keyed cache missed on every cross-session lookup -- which is the one
thing the persistent disk cache exists to do.
"""

import subprocess
import sys
import textwrap

import numpy as np
import pymc as pm

from cloudposterior.naming import cache_key, model_digest

_BUILD = """
import numpy as np, pymc as pm
with pm.Model(coords={"g": ["a", "b"]}) as m:
    mu = pm.Normal("mu", 0, %(sigma)s)
    s = pm.HalfNormal("s", 1)
    pm.Normal("obs", mu, s, observed=np.array(%(data)s))
"""


def _digest_in_subprocess(sigma="5.0", data="[1.0, 2.0, 3.0, 4.0]") -> str:
    script = textwrap.dedent(
        _BUILD % {"sigma": sigma, "data": data}
        + "\nfrom cloudposterior.naming import model_digest\nprint(model_digest(m))\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=True
    )
    return out.stdout.strip().splitlines()[-1]


def _model(sigma=5.0, data=(1.0, 2.0, 3.0, 4.0)):
    with pm.Model(coords={"g": ["a", "b"]}) as m:
        mu = pm.Normal("mu", 0, sigma)
        s = pm.HalfNormal("s", 1)
        pm.Normal("obs", mu, s, observed=np.array(data))
    return m


# -- the acceptance test for the phase --------------------------------------

def test_digest_is_stable_across_interpreter_sessions():
    """The scenario a bytes hash could never satisfy."""
    assert _digest_in_subprocess() == _digest_in_subprocess()


def test_digest_stable_across_rebuilds_in_one_session():
    assert model_digest(_model()) == model_digest(_model())


# -- sensitivity -------------------------------------------------------------

def test_digest_changes_with_a_prior():
    assert model_digest(_model(sigma=5.0)) != model_digest(_model(sigma=2.0))


def test_digest_changes_with_observed_values():
    assert model_digest(_model()) != model_digest(_model(data=(1.0, 2.0, 3.0, 9.0)))


def test_digest_changes_when_observed_data_is_permuted():
    """A summed fingerprint was blind to this."""
    assert model_digest(_model()) != model_digest(_model(data=(2.0, 1.0, 3.0, 4.0)))


def test_digest_tracks_pm_data_mutation():
    with pm.Model() as m:
        x = pm.Data("x", np.array([1.0, 2.0, 3.0]))
        mu = pm.Normal("mu", 0, 1)
        pm.Normal("obs", mu * x, 1, observed=np.array([1.0, 2.0, 3.0]))

    before = model_digest(m)
    with m:
        pm.set_data({"x": np.array([10.0, 20.0, 30.0])})
    assert model_digest(m) != before


# -- kwarg tokens ------------------------------------------------------------

def test_different_rng_streams_do_not_share_a_cache_entry():
    """Collapsing every Generator to one constant made two runs with different
    streams return each other's posterior."""
    ident = model_digest(_model())
    a = cache_key(ident, {"random_seed": np.random.default_rng(1)})
    b = cache_key(ident, {"random_seed": np.random.default_rng(2)})
    assert a != b


def test_same_rng_state_shares_a_cache_entry():
    ident = model_digest(_model())
    a = cache_key(ident, {"random_seed": np.random.default_rng(7)})
    b = cache_key(ident, {"random_seed": np.random.default_rng(7)})
    assert a == b


def test_callable_kwargs_are_stable_across_instances():
    """repr() embeds a memory address, so a callback= kwarg could never hit
    its own cache entry twice."""

    def make():
        def cb(*a, **k):
            return None

        return cb

    ident = model_digest(_model())
    assert cache_key(ident, {"callback": make()}) == cache_key(
        ident, {"callback": make()}
    )


def test_key_material_is_length_delimited():
    ident = model_digest(_model())
    assert cache_key(ident, {"ab": "c"}) != cache_key(ident, {"a": "bc"})
