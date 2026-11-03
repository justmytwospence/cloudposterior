"""Modal Dict adapter and hosted URLs for pymc-monitor's shared dashboard."""

from __future__ import annotations

from cloudposterior.backends.modal_backend import _run_blocking
from cloudposterior.progress import PhaseUpdate, SamplingProgress


class DashboardSink:
    """Write the monitor's transport-neutral state snapshots into a Modal Dict."""

    def __init__(self, progress_dict, key: str = "progress"):
        from pymc_monitor.dashboard import DashboardState

        self._dict = progress_dict
        self._key = key
        # Sampling completion is not delivery: keep polling until NetCDF is
        # compressed, transferred and loaded by the client.
        self._state = DashboardState(completion_phase="downloading")

    def show_phase(self, update: PhaseUpdate):
        self._state.update(update)
        self._write()

    def show_sampling(self, progress: SamplingProgress):
        self._state.update(progress)
        self._write()

    def show_convergence(self, update):
        self._state.update(update)
        self._write()

    def finish(self, status: str = "complete"):
        from pymc_monitor.events import RunFinished

        self._state.update(RunFinished(status))
        self._write()

    def _write(self):
        try:
            _run_blocking(self._dict.__setitem__, self._key, self._state.snapshot())
        except Exception:
            pass  # best-effort: dashboard failures cannot kill sampling


def _modal_endpoint_url(origin: str, dashboard_label: str, target_label: str) -> str:
    """Replace exactly the Modal endpoint label in a validated request origin."""
    import re
    from urllib.parse import urlsplit, urlunsplit

    if any(re.fullmatch(r"[a-z0-9-]+", label) is None
           for label in (dashboard_label, target_label)):
        raise ValueError("invalid Modal endpoint label")
    try:
        parsed = urlsplit(origin)
        host = parsed.hostname or ""
        port = parsed.port
    except (ValueError, TypeError) as exc:
        raise ValueError(f"unexpected Modal endpoint origin: {origin!r}") from exc
    markers = [f"--{dashboard_label}-", f"--{dashboard_label}."]
    matches = [marker for marker in markers if host.count(marker) == 1]
    if (parsed.scheme != "https" or not host.endswith(".modal.run") or
            len(matches) != 1 or parsed.username is not None or
            parsed.password is not None or parsed.path not in ("", "/") or
            parsed.query or parsed.fragment):
        raise ValueError(f"unexpected Modal endpoint origin: {origin!r}")
    marker = matches[0]
    host = host.replace(marker, f"--{target_label}{marker[-1]}", 1)
    netloc = host + (f":{port}" if port is not None else "")
    return urlunsplit((parsed.scheme, netloc, "", "", ""))


def _dashboard_urls(progress_label: str, stop_label: str,
                    dashboard_label: str, stop_token: str, origin: str) -> tuple[str, str]:
    if not all((progress_label, stop_label, dashboard_label, stop_token)):
        raise ValueError("render_dashboard_html requires all endpoint labels and stop_token")
    return (_modal_endpoint_url(origin, dashboard_label, progress_label),
            _modal_endpoint_url(origin, dashboard_label, stop_label))


def render_dashboard_html(progress_label: str, stop_label: str,
                          dashboard_label: str, stop_token: str, *, origin: str) -> str:
    """Use the monitor's self-contained flat/map page with explicit Modal URLs."""
    from pymc_monitor.dashboard import render_dashboard_html as monitor_html

    progress_url, stop_url = _dashboard_urls(
        progress_label, stop_label, dashboard_label, stop_token, origin)
    return monitor_html(progress_url=progress_url, stop_url=stop_url,
                        token=stop_token, title="cloudposterior")


def dashboard_content_security_policy(progress_label: str, stop_label: str,
                                      dashboard_label: str, stop_token: str, *, origin: str) -> str:
    """Hash exactly the inline monitor assets and permit only API origins."""
    from pymc_monitor.dashboard import dashboard_csp

    progress_url, stop_url = _dashboard_urls(
        progress_label, stop_label, dashboard_label, stop_token, origin)
    return dashboard_csp(progress_url=progress_url, stop_url=stop_url)
