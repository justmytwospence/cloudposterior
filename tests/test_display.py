"""Rendering tests for the progress displays (Rich terminal + ipywidgets notebook)."""

import pytest

from cloudposterior import display
from cloudposterior.progress import ChainProgress, JobPhase, PhaseUpdate, SamplingProgress


def _sampling(divergences=0):
    return SamplingProgress(
        chains={
            0: ChainProgress(draw=50, total=100, phase="sampling", draws_per_sec=20.0,
                             eta_seconds=2.5, divergences=divergences, step_size=0.12, tree_size=7),
            1: ChainProgress(draw=30, total=100, phase="tuning"),
        },
        total_divergences=divergences, elapsed=12.3, total_draws=80,
    )


def _phase(status="done"):
    return PhaseUpdate(phase=JobPhase.SAMPLING, status=status, message="MCMC sampling", elapsed=1.5)


# -- pure formatters --------------------------------------------------------

def test_format_time_seconds_and_minutes():
    assert display._format_time(5.0).endswith("s")
    assert "m" in display._format_time(125.0)


def test_bar_html_clamps_out_of_range_fractions():
    for frac in (0.5, 2.0, -1.0):
        assert "div" in display._bar_html(frac)


def test_sampling_table_html_renders_rows():
    html = display._sampling_table_html(_sampling())
    assert "<table" in html and "Chain 0" in html and "50/100" in html


def test_sampling_table_html_colors_divergences_red():
    assert "#d9534f" in display._sampling_table_html(_sampling(divergences=3))


def test_phase_html_status_icons():
    html = display._phase_html([("done", "a", "done"), ("error", "b", "boom"), ("in_progress", "c", "wip")])
    assert "&#10003;" in html and "&#10007;" in html


def test_is_notebook_returns_bool():
    assert isinstance(display._is_notebook(), bool)


def test_is_marimo_returns_bool():
    assert isinstance(display._is_marimo(), bool)


# -- display backends -------------------------------------------------------

def test_terminal_display_updates_without_error():
    d = display.TerminalDisplay("Modal (auto-sized: 4 cores, 8GB)")
    d.show_phase(_phase("in_progress"))
    d.show_phase(_phase("done"))
    d.show_sampling(_sampling(divergences=2))  # exercises table build + red coloring


def test_notebook_display_updates_widget_html(monkeypatch):
    pytest.importorskip("anywidget")
    # No notebook kernel in tests -- skip the mount, just drive the sink.
    monkeypatch.setattr(display.NotebookDisplay, "_mount", lambda self: None)
    d = display.NotebookDisplay("test")
    d.show_phase(_phase("in_progress"))
    d.show_phase(_phase("done"))
    d.show_sampling(_sampling())
    html = d._compose_html()
    assert "<table" in html and "Chain 0" in html and "50/100" in html
    assert "&#10003;" in html  # done-phase check icon
    assert "<table" in d._widget.html  # trait was set


def test_notebook_display_stop_button(monkeypatch):
    pytest.importorskip("anywidget")
    monkeypatch.setattr(display.NotebookDisplay, "_mount", lambda self: None)
    d = display.NotebookDisplay("test", stop_url="https://x.modal.run", stop_token="tok")
    assert d._widget.stop_url == "https://x.modal.run"
    assert d._widget.stop_token == "tok"
    assert "Stop sampling" in display._PROGRESS_ESM  # button rendered by the ESM
    # No stop URL -> traits empty (button stays hidden in the frontend).
    d2 = display.NotebookDisplay("test")
    assert d2._widget.stop_url == ""


def test_notebook_display_render_degrades_when_trait_set_raises(monkeypatch):
    pytest.importorskip("anywidget")
    monkeypatch.setattr(display.NotebookDisplay, "_mount", lambda self: None)
    d = display.NotebookDisplay("test")

    class _Boom:
        @property
        def html(self):
            return ""

        @html.setter
        def html(self, value):
            raise RuntimeError("no runtime context")

    d._widget = _Boom()
    d.show_sampling(_sampling())  # must not raise
