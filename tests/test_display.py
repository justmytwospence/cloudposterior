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


def _filled_px(html: str) -> int:
    """Width of the inner (filled) bar."""
    import re

    return int(re.findall(r"width:(\d+)px", html)[-1])


def test_bar_html_clamps_out_of_range_fractions():
    """Named for clamping but only asserted that a <div> came back, so it
    passed for any implementation."""
    assert _filled_px(display._bar_html(0.5, width_px=200)) == 100
    assert _filled_px(display._bar_html(2.0, width_px=200)) == 200   # clamped high
    assert _filled_px(display._bar_html(-1.0, width_px=200)) == 0    # clamped low


def test_sampling_table_html_renders_rows():
    html = display._sampling_table_html(_sampling())
    assert "<table" in html and "Chain 0" in html and "50/100" in html


def test_sampling_table_html_colors_divergences_red():
    assert "#d9534f" in display._sampling_table_html(_sampling(divergences=3))


def test_phase_html_status_icons():
    html = display._phase_html([("done", "a", "done"), ("error", "b", "boom"), ("in_progress", "c", "wip")])
    assert "&#10003;" in html and "&#10007;" in html


def test_frontend_detection_is_false_under_plain_pytest():
    """Tautological isinstance(..., bool) checks asserted nothing; under a
    bare pytest run neither frontend is present."""
    assert display._is_notebook() is False
    assert display._is_marimo() is False


# -- display backends -------------------------------------------------------

def test_terminal_display_accumulates_phase_and_sampling_state():
    d = display.TerminalDisplay("Modal (auto-sized: 4 cores, 8GB)")
    d.show_phase(_phase("in_progress"))
    d.show_phase(_phase("done"))
    d.show_sampling(_sampling(divergences=2))

    assert d._sampling is not None
    assert d._sampling.total_divergences == 2
    # The phase is recorded once, updated in place rather than duplicated.
    assert len([p for p in d._phases if p[1] == "sampling"]) == 1


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


def test_notebook_display_stop_button_hides_when_sampling_done(monkeypatch):
    pytest.importorskip("anywidget")
    monkeypatch.setattr(display.NotebookDisplay, "_mount", lambda self: None)
    d = display.NotebookDisplay("test", stop_url="https://x.modal.run", stop_token="tok")
    d.show_phase(_phase("in_progress"))
    assert d._widget.stop_url == "https://x.modal.run"  # still stoppable
    d.show_phase(_phase("done"))
    assert d._widget.stop_url == ""  # cleared -> ESM hides the button


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


def test_phase_html_escapes_worker_messages():
    """Phase details carry worker messages (including exception text) -- a
    stray '<' must render as text, not inject markup into the widget."""
    from cloudposterior.display import _phase_html

    html = _phase_html([("error", "sampling", "<script>alert(1)</script>")])
    assert "<script>" not in html
    assert "&lt;script&gt;" in html


def test_notebook_display_escapes_active_phase():
    from cloudposterior.display import NotebookDisplay

    display = NotebookDisplay.__new__(NotebookDisplay)
    display._instance_desc = ""
    display._phases = []
    display._active_phase = None
    display._sampling = None

    display._active_phase = "compiling <model>"
    html = display._compose_html()
    assert "<model>" not in html
    assert "&lt;model&gt;" in html
