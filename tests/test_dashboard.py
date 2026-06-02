"""Tests for the live dashboard sink and HTML rendering."""

from cloudposterior.dashboard import DashboardSink, render_dashboard_html
from cloudposterior.progress import (
    ChainProgress,
    ConvergenceUpdate,
    JobPhase,
    ParamConvergence,
    PhaseUpdate,
    SamplingProgress,
)


def test_dashboard_sink_writes_phase_and_sampling():
    store = {}
    sink = DashboardSink(store)
    sink.show_phase(PhaseUpdate(phase=JobPhase.SAMPLING, status="in_progress", message="sampling", elapsed=0.0))
    sink.show_sampling(SamplingProgress(
        chains={0: ChainProgress(draw=10, total=100, phase="sampling", divergences=1, step_size=0.1, tree_size=7)},
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


def test_dashboard_sink_sets_complete_on_download_done():
    store = {}
    sink = DashboardSink(store)
    sink.show_phase(PhaseUpdate(phase=JobPhase.DOWNLOADING, status="done", message="trace loaded", elapsed=0.5))
    assert store["progress"]["complete"] is True


def test_dashboard_sink_is_best_effort_on_write_failure():
    class BadDict:
        def __setitem__(self, key, value):
            raise RuntimeError("modal dict unavailable")

    sink = DashboardSink(BadDict())
    # Must not raise -- progress writes are best-effort.
    sink.show_phase(PhaseUpdate(phase=JobPhase.SAMPLING, status="done", message="x", elapsed=0.2))


def test_render_dashboard_html_bakes_labels_and_vehtari_thresholds():
    html = render_dashboard_html(
        progress_label="proj-abc-progress", stop_label="proj-abc-stop", dashboard_label="proj-abc",
    )
    assert "proj-abc-progress" in html and "proj-abc-stop" in html
    assert "__PROGRESS_LABEL__" not in html  # placeholders substituted
    assert "1.01" in html and "400" in html  # rank-normalized R-hat / ESS thresholds


def test_render_dashboard_html_bakes_stop_token():
    html = render_dashboard_html(
        progress_label="p", stop_label="s", dashboard_label="d", stop_token="secret-tok-123",
    )
    assert "secret-tok-123" in html
    assert "__STOP_TOKEN__" not in html
