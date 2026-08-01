"""Progress display backends for terminal (Rich TUI) and notebooks (anywidget).

The notebook backend uses anywidget, which renders live in both Jupyter
(ipywidgets under the hood) and marimo via the shared Jupyter Comm protocol.
"""

from __future__ import annotations

from typing import Iterator

from cloudposterior.progress import (
    PhaseUpdate,
    ProgressEvent,
    SamplingProgress,
)


def _is_notebook() -> bool:
    try:
        from IPython import get_ipython
        shell = get_ipython().__class__.__name__
        return shell == "ZMQInteractiveShell"
    except Exception:
        return False


def _is_marimo() -> bool:
    try:
        import marimo
        return marimo.running_in_notebook()
    except Exception:
        return False


def _emit_oneshot_html(parts: list[str], *, terminal_fallback) -> None:
    """Emit one-shot HTML to whichever notebook frontend is active.

    ``parts`` are HTML fragments for browser frontends (Jupyter and marimo);
    ``terminal_fallback`` is a zero-arg callable that renders the terminal
    equivalent (e.g. a Rich print, or an ASCII QR code). Used for non-streaming
    output like cache-hit indicators and notification links/QR codes.
    """
    html = "".join(parts)
    try:
        if _is_marimo():
            import marimo as mo
            mo.output.append(mo.Html(html))
            return
        if _is_notebook():
            from IPython.display import HTML, display
            display(HTML(html))
            return
    except Exception:
        pass
    terminal_fallback()


# ---------------------------------------------------------------------------
# Notebook display (anywidget -- works in Jupyter and marimo)
# ---------------------------------------------------------------------------

def _format_time(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    secs = seconds % 60
    return f"{minutes}m {secs:.0f}s"


def _bar_html(fraction: float, width_px: int = 200, color: str = "#1764f4") -> str:
    filled = max(0, min(width_px, int(fraction * width_px)))
    return (
        f'<div style="display:inline-block;width:{width_px}px;height:14px;'
        f'background:#333;border-radius:3px;overflow:hidden;vertical-align:middle;">'
        f'<div style="width:{filled}px;height:100%;background:{color};"></div>'
        f'</div>'
    )


def _sampling_table_html(progress: SamplingProgress) -> str:
    """Build an HTML table matching PyMC's progress layout."""
    rows = []
    for chain_id in sorted(progress.chains.keys()):
        cp = progress.chains[chain_id]
        pct = cp.draw / cp.total if cp.total > 0 else 0
        bar_color = "#d9534f" if cp.divergences > 0 else "#1764f4"
        bar = _bar_html(pct, width_px=180, color=bar_color)

        phase_label = f'<span style="color:#888;">[{cp.phase}]</span>'
        speed_str = f"{cp.draws_per_sec:.0f} draws/s" if cp.draws_per_sec > 0 else "--"
        elapsed = _format_time(cp.draw / cp.draws_per_sec if cp.draws_per_sec > 0 else 0)
        remaining = _format_time(cp.eta_seconds) if cp.eta_seconds > 0 else "--"

        rows.append(
            f"<tr>"
            f"<td style='padding:2px 8px;white-space:nowrap;'>Chain {chain_id} {phase_label}</td>"
            f"<td style='padding:2px 4px;'>{bar}</td>"
            f"<td style='padding:2px 8px;text-align:right;'>{cp.draw}/{cp.total}</td>"
            f"<td style='padding:2px 8px;text-align:right;'>{cp.divergences}</td>"
            f"<td style='padding:2px 8px;text-align:right;'>{cp.step_size:.3f}</td>"
            f"<td style='padding:2px 8px;text-align:right;'>{cp.tree_size}</td>"
            f"<td style='padding:2px 8px;text-align:right;'>{speed_str}</td>"
            f"<td style='padding:2px 8px;text-align:right;'>{elapsed}</td>"
            f"<td style='padding:2px 8px;text-align:right;'>{remaining}</td>"
            f"</tr>"
        )

    header = (
        "<tr style='border-bottom:1px solid #555;'>"
        "<th style='padding:2px 8px;text-align:left;'>Chain</th>"
        "<th style='padding:2px 4px;text-align:left;'>Progress</th>"
        "<th style='padding:2px 8px;text-align:right;'>Draws</th>"
        "<th style='padding:2px 8px;text-align:right;'>Divergences</th>"
        "<th style='padding:2px 8px;text-align:right;'>Step size</th>"
        "<th style='padding:2px 8px;text-align:right;'>Grad evals</th>"
        "<th style='padding:2px 8px;text-align:right;'>Speed</th>"
        "<th style='padding:2px 8px;text-align:right;'>Elapsed</th>"
        "<th style='padding:2px 8px;text-align:right;'>Remaining</th>"
        "</tr>"
    )

    footer_parts = [f"Total divergences: {progress.total_divergences}"]
    footer_parts.append(f"Elapsed: {_format_time(progress.elapsed)}")
    footer = " | ".join(footer_parts)

    return (
        f"<table style='font-family:monospace;font-size:13px;border-collapse:collapse;'>"
        f"{header}{''.join(rows)}"
        f"<tr><td colspan='9' style='padding:6px 8px;color:#888;'>{footer}</td></tr>"
        f"</table>"
    )


_CSS_SPINNER = (
    '<style>@keyframes cp-spin{to{transform:rotate(360deg)}}</style>'
    '<span style="display:inline-block;width:10px;height:10px;'
    'border:2px solid #555;border-top-color:#f0ad4e;border-radius:50%;'
    'animation:cp-spin 0.8s linear infinite;vertical-align:middle;'
    'margin-right:6px;"></span>'
)


def _phase_html(phases: list[tuple[str, str, str]]) -> str:
    """Render phase checklist as HTML. Each tuple: (status, label, detail)."""
    import html

    lines = []
    for status, label, detail in phases:
        if status == "done":
            icon = '<span style="color:#5cb85c;">&#10003;</span>'
        elif status == "in_progress":
            icon = '<span style="color:#f0ad4e;">&#9679;</span>'
        else:
            icon = '<span style="color:#d9534f;">&#10007;</span>'
        lines.append(
            f'<div style="font-family:monospace;font-size:13px;padding:1px 0;">'
            f'  {icon} '
            # detail carries worker messages, including exception text --
            # escape so a stray "<" can't break (or script) the widget HTML.
            f'<span style="color:#888;">{html.escape(detail)}</span>'
            f'</div>'
        )
    return "".join(lines)


# anywidget ES module: mirror the Python-composed HTML into the cell. State
# flows over the Jupyter Comm protocol, which both Jupyter and marimo flush to
# the frontend immediately (mid-cell), so the table animates live during a
# blocking pm.sample() call. A persistent Stop button (shown only when a
# stop_url is set) POSTs to the worker's /stop endpoint from the browser -- no
# Python round-trip, so it works even though the kernel is blocked streaming.
_PROGRESS_ESM = """
function render({ model, el }) {
  const content = document.createElement("div");
  const btn = document.createElement("button");
  btn.textContent = "Stop sampling";
  btn.style.cssText = "margin:4px 0;padding:4px 12px;font:13px monospace;cursor:pointer;" +
    "background:#d9534f;color:#fff;border:none;border-radius:3px;";
  let stopped = false;
  btn.addEventListener("click", async () => {
    if (stopped) return;
    stopped = true;
    btn.textContent = "Stopping...";
    btn.disabled = true;
    btn.style.opacity = "0.6";
    const url = model.get("stop_url");
    const token = model.get("stop_token");
    try {
      await fetch(url + (token ? ("?token=" + encodeURIComponent(token)) : ""), {method: "POST"});
    } catch (e) {}
  });
  const syncHtml = () => { content.innerHTML = model.get("html"); };
  const syncBtn = () => { btn.style.display = model.get("stop_url") ? "inline-block" : "none"; };
  syncHtml(); syncBtn();
  model.on("change:html", syncHtml);
  model.on("change:stop_url", syncBtn);
  el.appendChild(content);
  el.appendChild(btn);
}
export default { render };
"""

_progress_widget_cls = None


def _progress_widget_class():
    """Lazily define and cache the anywidget subclass (keeps the import off the
    terminal/CLI path)."""
    global _progress_widget_cls
    if _progress_widget_cls is None:
        import anywidget
        import traitlets

        class _CPProgressWidget(anywidget.AnyWidget):
            _esm = _PROGRESS_ESM
            html = traitlets.Unicode("").tag(sync=True)
            stop_url = traitlets.Unicode("").tag(sync=True)
            stop_token = traitlets.Unicode("").tag(sync=True)

        _progress_widget_cls = _CPProgressWidget
    return _progress_widget_cls


class NotebookDisplay:
    """anywidget-based progress display for Jupyter and marimo notebooks.

    Composes the same HTML the terminal/Jupyter views always used and pushes it
    through a single synced ``html`` trait; the frontend re-renders on each
    change. Works identically in Jupyter and marimo.
    """

    def __init__(self, instance_desc: str = "", *,
                 stop_url: str | None = None, stop_token: str | None = None):
        self._instance_desc = instance_desc
        self._phases: list[tuple[str, str, str]] = []
        self._active_phase: str | None = None
        self._sampling: SamplingProgress | None = None

        # anywidget is a hard dependency, but a stripped environment or a
        # traitlets/ipywidgets version skew must not take down the whole
        # sampling call -- every other frontend interaction here is already
        # defensive. Callers fall back to TerminalDisplay (see _build_sinks).
        self._widget = _progress_widget_class()()
        # Setting stop_url before mount makes the Stop button appear in the
        # initial render (remote runs only; empty string => button hidden).
        self._widget.stop_url = stop_url or ""
        self._widget.stop_token = stop_token or ""
        self._mount()
        self._render()

    def stop(self) -> None:
        """Clear the Stop button.

        stop_url is otherwise cleared only on a terminal sampling phase, so a
        failure during provisioning/compiling left a live-looking Stop button
        in the notebook forever.
        """
        try:
            self._widget.stop_url = ""
        except Exception:
            pass

    def _mount(self) -> None:
        """Display the widget once. marimo mounts via mo.output.append; Jupyter
        (and other IPython frontends) via IPython.display."""
        try:
            if _is_marimo():
                import marimo as mo
                mo.output.append(self._widget)
            else:
                from IPython.display import display
                display(self._widget)
        except Exception:
            pass

    def show_phase(self, update: PhaseUpdate):
        detail = update.message
        if update.status == "done" and update.elapsed > 0.1:
            detail += f" ({_format_time(update.elapsed)})"

        if update.status == "in_progress":
            # Show only the spinner, not a checklist entry
            self._active_phase = update.message
        else:
            # Clear spinner and add completed/error entry to checklist
            self._active_phase = None
            found = False
            for i, (s, label, d) in enumerate(self._phases):
                if label == update.phase.value:
                    self._phases[i] = (update.status, update.phase.value, detail)
                    found = True
                    break
            if not found:
                self._phases.append((update.status, update.phase.value, detail))

        # Hide the Stop button once sampling reaches a terminal state -- there's
        # nothing left to stop. Clearing stop_url drives the ESM to hide it.
        if update.phase.value == "sampling" and update.status != "in_progress":
            try:
                self._widget.stop_url = ""
            except Exception:
                pass

        self._render()

    def show_sampling(self, progress: SamplingProgress):
        self._sampling = progress
        self._render()

    def _compose_html(self) -> str:
        """Build the full view as one HTML string from the pure builders."""
        parts = [
            f'<div style="font-family:monospace;font-size:14px;font-weight:bold;'
            f'padding:8px 0 4px 0;">cloudposterior'
            f'{" -- " + self._instance_desc if self._instance_desc else ""}</div>'
        ]
        if self._phases:
            parts.append(_phase_html(self._phases))
        if self._active_phase:
            import html

            parts.append(
                f'<div style="font-family:monospace;font-size:13px;padding:1px 0;">'
                f'  {_CSS_SPINNER}'
                f'<span style="color:#888;">{html.escape(self._active_phase)}...</span>'
                f'</div>'
            )
        if self._sampling and self._sampling.chains:
            parts.append(_sampling_table_html(self._sampling))
        return f'<div>{"".join(parts)}</div>'

    def _render(self) -> None:
        # Defensive: setting the trait off the main thread / outside a marimo
        # runtime context (the local-sampling background thread) no-ops rather
        # than crashing the sample call.
        try:
            self._widget.html = self._compose_html()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Terminal display (Rich TUI)
# ---------------------------------------------------------------------------

class TerminalDisplay:
    """Rich-based TUI display for terminal."""

    def __init__(self, instance_desc: str = ""):
        from rich.console import Console
        from rich.live import Live

        self._console = Console()
        self._instance_desc = instance_desc
        self._phases: list[tuple[str, str, str]] = []  # completed/error phases only
        self._active_phase: str | None = None  # in-progress message for spinner
        self._sampling: SamplingProgress | None = None
        self._live = Live(console=self._console, refresh_per_second=4)

    def start(self):
        self._live.start()
        self._update_live()

    def stop(self):
        self._update_live()
        self._live.stop()

    def show_phase(self, update: PhaseUpdate):
        detail = update.message
        if update.status == "done" and update.elapsed > 0.1:
            detail += f" ({_format_time(update.elapsed)})"

        if update.status == "in_progress":
            self._active_phase = update.message
        else:
            self._active_phase = None
            # Add to completed checklist
            found = False
            for i, (s, label, d) in enumerate(self._phases):
                if label == update.phase.value:
                    self._phases[i] = (update.status, update.phase.value, detail)
                    found = True
                    break
            if not found:
                self._phases.append((update.status, update.phase.value, detail))

        self._update_live()

    def show_sampling(self, progress: SamplingProgress):
        self._sampling = progress
        self._update_live()

    def _update_live(self):
        from rich.columns import Columns
        from rich.console import Group
        from rich.spinner import Spinner
        from rich.table import Table
        from rich.text import Text

        parts = []

        # Header
        header = f"cloudposterior{' -- ' + self._instance_desc if self._instance_desc else ''}"
        parts.append(Text(header, style="bold"))
        parts.append(Text(""))

        # Completed phases
        for status, label, detail in self._phases:
            if status == "done":
                parts.append(Text.from_markup(
                    f"  [green]\u2713[/green] [dim]{detail}[/dim]"
                ))
            else:
                parts.append(Text.from_markup(
                    f"  [red]\u2717[/red] [dim]{detail}[/dim]"
                ))

        # Active spinner for in-progress phase
        if self._active_phase:
            spinner = Spinner("dots", f"[dim]{self._active_phase}...[/dim]", style="yellow")
            parts.append(Columns(["  ", spinner], padding=0))

        # Sampling table
        if self._sampling and self._sampling.chains:
            parts.append(Text(""))
            table = Table(
                show_header=True,
                header_style="bold",
                box=None,
                padding=(0, 1),
                show_edge=False,
            )
            table.add_column("Chain", style="cyan", no_wrap=True)
            table.add_column("Progress", min_width=25)
            table.add_column("Draws", justify="right")
            table.add_column("Divergences", justify="right")
            table.add_column("Step size", justify="right")
            table.add_column("Grad evals", justify="right")
            table.add_column("Speed", justify="right")
            table.add_column("Elapsed", justify="right")
            table.add_column("Remaining", justify="right")

            for chain_id in sorted(self._sampling.chains.keys()):
                cp = self._sampling.chains[chain_id]
                pct = cp.draw / cp.total if cp.total > 0 else 0
                bar_width = 20
                filled = int(pct * bar_width)

                bar_color = "red" if cp.divergences > 0 else "blue"
                bar = f"[{bar_color}]" + "\u2501" * filled + f"[/{bar_color}]" + "[dim]\u2501[/dim]" * (bar_width - filled)

                phase_tag = f"[dim][{cp.phase}][/dim]"
                speed_str = f"{cp.draws_per_sec:.0f} dr/s" if cp.draws_per_sec > 0 else "--"
                elapsed = _format_time(cp.draw / cp.draws_per_sec if cp.draws_per_sec > 0 else 0)
                remaining = _format_time(cp.eta_seconds) if cp.eta_seconds > 0 else "--"

                div_style = "red" if cp.divergences > 0 else ""

                table.add_row(
                    f"Chain {chain_id} {phase_tag}",
                    bar,
                    f"{cp.draw}/{cp.total}",
                    Text(str(cp.divergences), style=div_style),
                    f"{cp.step_size:.3f}",
                    str(cp.tree_size),
                    speed_str,
                    elapsed,
                    remaining,
                )

            # Footer
            footer_parts = [f"Divergences: {self._sampling.total_divergences}"]
            footer_parts.append(f"Elapsed: {_format_time(self._sampling.elapsed)}")
            table.add_row("", "", "", "", "", "", "", "", "")
            table.add_row(
                Text(" | ".join(footer_parts), style="dim"),
                "", "", "", "", "", "", "", "",
            )

            parts.append(table)

        self._live.update(Group(*parts))


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def display_progress_stream(
    events: Iterator[ProgressEvent],
    instance_desc: str = "",
):
    """Consume a stream of progress events and display them.

    Automatically selects notebook or terminal backend.
    """
    if _is_marimo() or _is_notebook():
        display = NotebookDisplay(instance_desc)
        for event in events:
            if isinstance(event, PhaseUpdate):
                display.show_phase(event)
            elif isinstance(event, SamplingProgress):
                display.show_sampling(event)
    else:
        display = TerminalDisplay(instance_desc)
        display.start()
        try:
            for event in events:
                if isinstance(event, PhaseUpdate):
                    display.show_phase(event)
                elif isinstance(event, SamplingProgress):
                    display.show_sampling(event)
        finally:
            display.stop()
