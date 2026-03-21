"""Progress display backends for terminal (Rich TUI) and Jupyter (ipywidgets)."""

from __future__ import annotations

import time
from typing import Iterator

from cloudposterior.progress import (
    ChainProgress,
    JobPhase,
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


# ---------------------------------------------------------------------------
# Notebook display (ipywidgets)
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


def _phase_html(phases: list[tuple[str, str, str]]) -> str:
    """Render phase checklist as HTML. Each tuple: (status, label, detail)."""
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
            f'  {icon} {label} '
            f'<span style="color:#888;">{detail}</span>'
            f'</div>'
        )
    return "".join(lines)


class NotebookDisplay:
    """ipywidgets-based display for Jupyter notebooks."""

    def __init__(self, instance_desc: str = ""):
        from IPython.display import display, HTML
        import ipywidgets as widgets

        self._display = display
        self._HTML = HTML

        self._instance_desc = instance_desc
        self._phases: list[tuple[str, str, str]] = []

        # Widgets
        self._header = widgets.HTML(
            value=f'<div style="font-family:monospace;font-size:14px;font-weight:bold;'
            f'padding:8px 0 4px 0;">cloudposterior'
            f'{" -- " + instance_desc if instance_desc else ""}</div>'
        )
        self._phase_widget = widgets.HTML(value="")
        self._sampling_widget = widgets.HTML(value="")
        self._modal_output = widgets.Output(layout=widgets.Layout(
            max_height="200px",
            overflow_y="auto",
            border="1px solid #444",
            padding="4px",
            margin="2px 0 4px 0",
        ))
        self._modal_accordion = widgets.Accordion(children=[self._modal_output])
        self._modal_accordion.set_title(0, "Modal build logs")
        self._modal_accordion.selected_index = None  # collapsed

        self._container = widgets.VBox([
            self._header,
            self._phase_widget,
            self._sampling_widget,
        ])
        self._handle = self._display(self._container, display_id=True)
        self._modal_shown = False

    @property
    def modal_output_widget(self):
        """The Output widget that captures Modal's stdout."""
        return self._modal_output

    def show_phase(self, update: PhaseUpdate):
        detail = update.message
        if update.status == "done":
            detail += f" ({_format_time(update.elapsed)})"

        # Update or add phase
        found = False
        for i, (s, label, d) in enumerate(self._phases):
            if label == update.phase.value:
                self._phases[i] = (update.status, update.phase.value, detail)
                found = True
                break
        if not found:
            self._phases.append((update.status, update.phase.value, detail))

        self._phase_widget.value = _phase_html(self._phases)

        # Show modal accordion during provisioning
        if update.phase == JobPhase.PROVISIONING and update.status == "in_progress":
            if not self._modal_shown:
                self._container.children = [
                    self._header,
                    self._phase_widget,
                    self._modal_accordion,
                    self._sampling_widget,
                ]
                self._modal_accordion.selected_index = 0  # expanded
                self._modal_shown = True
        elif update.phase == JobPhase.PROVISIONING and update.status == "done":
            self._modal_accordion.selected_index = None  # collapse

    def show_sampling(self, progress: SamplingProgress):
        self._sampling_widget.value = _sampling_table_html(progress)


# ---------------------------------------------------------------------------
# Terminal display (Rich TUI)
# ---------------------------------------------------------------------------

class TerminalDisplay:
    """Rich-based TUI display for terminal."""

    def __init__(self, instance_desc: str = ""):
        from rich.console import Console, Group
        from rich.live import Live
        from rich.panel import Panel
        from rich.style import Style
        from rich.table import Table
        from rich.text import Text

        self._console = Console()
        self._instance_desc = instance_desc
        self._phases: list[tuple[str, str, str]] = []
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
        if update.status == "done":
            detail += f" ({_format_time(update.elapsed)})"

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
        from rich.console import Group
        from rich.table import Table
        from rich.text import Text
        from rich.panel import Panel
        from rich.style import Style

        parts = []

        # Header
        header = f"cloudposterior{' -- ' + self._instance_desc if self._instance_desc else ''}"
        parts.append(Text(header, style="bold"))
        parts.append(Text(""))

        # Phase checklist
        for status, label, detail in self._phases:
            if status == "done":
                icon = "[green]\u2713[/green]"
            elif status == "in_progress":
                icon = "[yellow]\u25cf[/yellow]"
            else:
                icon = "[red]\u2717[/red]"
            parts.append(Text.from_markup(f"  {icon} {label}  [dim]{detail}[/dim]"))

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
    if _is_notebook():
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
