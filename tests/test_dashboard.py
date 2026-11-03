"""Tests for the shared state reducer and Modal-hosted dashboard adapter."""

import base64
import hashlib
import html as html_module
import json
import re

import pytest

from cloudposterior.backends.modal_backend import _dashboard_cors_headers
from cloudposterior.dashboard import (
    DashboardSink,
    _modal_endpoint_url,
    dashboard_content_security_policy,
    render_dashboard_html,
)
from cloudposterior.progress import (
    ChainProgress,
    ConvergenceUpdate,
    JobPhase,
    ParamConvergence,
    PhaseUpdate,
    SamplingProgress,
)

ORIGIN = "https://workspace--proj-abc-dev.modal.run/"


def _render(**kwargs):
    return render_dashboard_html(
        progress_label="proj-abc-progress", stop_label="proj-abc-stop",
        dashboard_label="proj-abc", stop_token="tok", origin=ORIGIN, **kwargs,
    )


def test_dashboard_endpoint_cors_requires_well_formed_origin():
    assert _dashboard_cors_headers("https://dash.modal.run") == {
        "Vary": "Origin", "Access-Control-Allow-Origin": "https://dash.modal.run",
    }
    assert _dashboard_cors_headers("http://localhost:8000") == {
        "Vary": "Origin", "Access-Control-Allow-Origin": "http://localhost:8000",
    }
    for origin in ("null", "http://remote.example", "https://bad.example/path",
                   "https://[", "https://user@bad.example", "https://@",
                   "https://bad.example:bogus", "https://bad.example/a b", ""):
        assert _dashboard_cors_headers(origin) == {"Vary": "Origin"}


def test_dashboard_sink_writes_phase_and_sampling():
    store = {}
    sink = DashboardSink(store)
    sink.show_phase(PhaseUpdate(JobPhase.SAMPLING, "in_progress", "sampling", 0.0))
    sink.show_sampling(SamplingProgress(
        chains={0: ChainProgress(10, 100, "sampling", divergences=1,
                                  step_size=0.1, tree_size=7)},
        total_divergences=1, elapsed=1.0, total_draws=10,
    ))
    data = store["progress"]
    assert data["sampling"]["chains"]["0"]["draw"] == 10
    assert any(p["label"] == "sampling" for p in data["phases"])


def test_dashboard_sink_writes_convergence_and_traces():
    store = {}
    sink = DashboardSink(store)
    sink.show_convergence(ConvergenceUpdate(
        params={"mu": ParamConvergence(rhat=1.005, ess_bulk=500, ess_tail=450)},
        draws=200, traces={"mu": [[0.1, 0.2], [0.3, 0.4]]},
    ))
    data = store["progress"]
    assert data["convergence"]["params"]["mu"]["rhat"] == 1.005
    assert data["convergence"]["draws"] == 200
    assert data["traces"]["mu"] == [[0.1, 0.2], [0.3, 0.4]]


def test_dashboard_completion_waits_for_download():
    store = {}
    sink = DashboardSink(store)
    sink.show_phase(PhaseUpdate(JobPhase.SAMPLING, "done", "sampled", 0.1))
    assert store["progress"]["complete"] is False
    sink.show_phase(PhaseUpdate(JobPhase.DOWNLOADING, "done", "trace loaded", 0.5))
    assert store["progress"]["complete"] is True


def test_dashboard_retains_stop_status_until_trace_is_downloaded():
    store = {}
    sink = DashboardSink(store)
    sink.finish("stopped")
    assert store["progress"]["status"] == "stopped"
    assert store["progress"]["complete"] is False
    sink.show_phase(PhaseUpdate(JobPhase.DOWNLOADING, "done", "trace loaded", 0.5))
    assert store["progress"]["complete"] is True
    assert store["progress"]["status"] == "stopped"


def test_dashboard_sink_is_best_effort_on_write_failure():
    class BadDict:
        def __setitem__(self, key, value):
            raise RuntimeError("modal dict unavailable")

    DashboardSink(BadDict()).show_phase(PhaseUpdate(JobPhase.SAMPLING, "done", "x", 0.2))


def test_modal_endpoint_url_replaces_exact_hostname_segment():
    assert _modal_endpoint_url(ORIGIN, "proj-abc", "proj-abc-progress") == (
        "https://workspace--proj-abc-progress-dev.modal.run"
    )
    prod = "https://workspace--proj-abc.modal.run/"
    assert _modal_endpoint_url(prod, "proj-abc", "proj-abc-stop") == (
        "https://workspace--proj-abc-stop.modal.run"
    )
    for bad_origin in ("https://unrelated.example/", "https://user@workspace--proj-abc-dev.modal.run/",
                       "https://workspace--proj-abc-dev.modal.run/path", "https://[",
                       "https://workspace--proj-abc-dev.modal.run/?x=1"):
        with pytest.raises(ValueError, match="unexpected Modal endpoint origin"):
            _modal_endpoint_url(bad_origin, "proj-abc", "proj-abc-stop")
    with pytest.raises(ValueError, match="invalid Modal endpoint label"):
        _modal_endpoint_url(ORIGIN, "proj-abc", "stop?token=stolen")


def _render_config(page):
    encoded = re.search(r'data-config="([^"]+)"', page)
    assert encoded is not None
    return json.loads(html_module.unescape(encoded.group(1)))


def test_render_dashboard_html_uses_shared_inline_map_page():
    page = _render()
    assert _render_config(page) == {
        "progress_url": "https://workspace--proj-abc-progress-dev.modal.run",
        "stop_url": "https://workspace--proj-abc-stop-dev.modal.run",
        "token": "tok",
    }
    assert "window.location.origin" not in page
    assert "cdn.jsdelivr.net" not in page
    assert "uPlot" in page
    assert "function normalize" in page  # flat and {models, runs} shapes
    assert "function requestStop" in page
    assert 'id="overview"' in page and 'id="detail"' in page
    assert '<script type="module"' not in page
    assert "assets/" not in page


def test_dashboard_csp_hashes_inline_assets_and_allows_only_api_origins():
    page = _render()
    csp = dashboard_content_security_policy(
        "proj-abc-progress", "proj-abc-stop", "proj-abc", "tok", origin=ORIGIN)
    for tag, directive in (("style", "style-src"), ("script", "script-src")):
        body = re.search(rf"<{tag}>(.*?)</{tag}>", page, re.S)
        assert body is not None
        digest = base64.b64encode(hashlib.sha256(body.group(1).encode()).digest()).decode()
        assert f"{directive} 'sha256-{digest}'" in csp
    assert "connect-src 'self' https://workspace--proj-abc-progress-dev.modal.run " in csp
    assert "https://workspace--proj-abc-stop-dev.modal.run" in csp
    assert "default-src 'none'" in csp


def test_dashboard_requires_token_and_escapes_json_in_html_attribute():
    with pytest.raises(ValueError, match="stop_token"):
        render_dashboard_html("p", "s", "proj-abc", "", origin=ORIGIN)
    hostile = 'tok"; alert(1); // <img src=x>'
    page = render_dashboard_html("p", "s", "proj-abc", hostile, origin=ORIGIN)
    assert hostile not in page
    assert _render_config(page)["token"] == hostile
    assert "<img src=x>" not in page


def test_dashboard_sink_writes_under_custom_map_key():
    store = {}
    sink = DashboardSink(store, key="pooled-0")
    sink.show_phase(PhaseUpdate(JobPhase.SAMPLING, "in_progress", "go", 0.0))
    assert "pooled-0" in store and "progress" not in store
    sink.show_sampling(SamplingProgress(
        chains={0: ChainProgress(10, 20, "sampling")}, elapsed=1, total_draws=10,
    ))
    assert store["pooled-0"]["sampling"]["chains"]["0"]["draw"] == 10
