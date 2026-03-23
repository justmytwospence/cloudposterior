"""Live progress dashboard served via Modal web endpoint."""

from __future__ import annotations

from cloudposterior.progress import (
    JobPhase,
    PhaseUpdate,
    SamplingProgress,
)


class DashboardSink:
    """Sink that writes progress state to a Modal Dict for the dashboard endpoint."""

    def __init__(self, progress_dict):
        self._dict = progress_dict
        self._phases: list[dict] = []
        self._sampling: dict | None = None
        self._complete = False

    def show_phase(self, update: PhaseUpdate):
        detail = update.message
        if update.status == "done" and update.elapsed > 0.1:
            detail += f" ({update.elapsed:.1f}s)"

        found = False
        for i, phase in enumerate(self._phases):
            if phase["label"] == update.phase.value:
                self._phases[i] = {"status": update.status, "label": update.phase.value, "detail": detail}
                found = True
                break
        if not found:
            self._phases.append({"status": update.status, "label": update.phase.value, "detail": detail})

        if update.phase == JobPhase.DOWNLOADING and update.status == "done":
            self._complete = True

        self._write()

    def show_sampling(self, progress: SamplingProgress):
        chains = {}
        for chain_id, cp in progress.chains.items():
            chains[str(chain_id)] = {
                "draw": cp.draw,
                "total": cp.total,
                "phase": cp.phase,
                "draws_per_sec": cp.draws_per_sec,
                "eta_seconds": cp.eta_seconds,
                "divergences": cp.divergences,
                "step_size": cp.step_size,
                "tree_size": cp.tree_size,
            }
        self._sampling = {
            "chains": chains,
            "total_divergences": progress.total_divergences,
            "elapsed": progress.elapsed,
        }
        self._write()

    def show_convergence(self, update):
        import sys
        self._convergence = {
            name: {"rhat": p.rhat, "ess_bulk": p.ess_bulk, "ess_tail": p.ess_tail}
            for name, p in update.params.items()
        }
        self._convergence_draws = update.draws
        self._traces = update.traces if update.traces else {}
        print(f"[DashboardSink] convergence: {len(self._convergence)} params, {self._convergence_draws} draws, {len(self._traces)} traces")
        self._write()

    def _write(self):
        try:
            data = {
                "phases": self._phases,
                "sampling": self._sampling,
                "complete": self._complete,
            }
            if hasattr(self, "_convergence") and self._convergence:
                data["convergence"] = {
                    "params": self._convergence,
                    "draws": self._convergence_draws,
                }
            if hasattr(self, "_traces") and self._traces:
                data["traces"] = self._traces
            self._dict["progress"] = data
        except Exception:
            pass  # best-effort


def render_dashboard_html(progress_label: str = "", stop_label: str = "",
                          dashboard_label: str = "") -> str:
    """Render dashboard HTML with endpoint labels baked in.

    The JS constructs full URLs from the labels by deriving the Modal
    workspace URL pattern from window.location.
    """
    return (DASHBOARD_HTML
        .replace("__PROGRESS_LABEL__", progress_label)
        .replace("__STOP_LABEL__", stop_label)
        .replace("__DASHBOARD_LABEL__", dashboard_label)
    )


DASHBOARD_HTML = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>cloudposterior</title>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/uplot@1.6.31/dist/uPlot.min.css">
<script src="https://cdn.jsdelivr.net/npm/uplot@1.6.31/dist/uPlot.iife.min.js"></script>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { font-family: 'SF Mono', 'Menlo', 'Monaco', monospace; font-size: 14px;
         background: #1a1a2e; color: #e0e0e0; padding: 20px; }
  h1 { font-size: 18px; color: #fff; margin-bottom: 16px; }
  .phase { padding: 3px 0; }
  .phase .icon { display: inline-block; width: 16px; }
  .done { color: #5cb85c; }
  .in_progress { color: #f0ad4e; }
  .error { color: #d9534f; }
  .detail { color: #888; }
  table { width: 100%; border-collapse: collapse; margin-top: 12px; }
  th { text-align: left; padding: 6px 8px; border-bottom: 1px solid #333; color: #aaa; font-weight: normal; }
  td { padding: 4px 8px; }
  .bar-bg { background: #333; border-radius: 3px; height: 14px; overflow: hidden; }
  .bar-fill { height: 100%; border-radius: 3px; transition: width 0.3s; }
  .bar-ok { background: #1764f4; }
  .bar-div { background: #d9534f; }
  .footer { color: #666; margin-top: 8px; font-size: 12px; }
  .complete-banner { background: #2d4a2d; color: #5cb85c; padding: 12px; border-radius: 6px;
                     margin-top: 16px; text-align: center; font-size: 16px; }
  .stop-btn { width: 100%; padding: 14px; margin: 12px 0; border: none; border-radius: 6px;
              font-family: inherit; font-size: 15px; font-weight: bold; cursor: pointer;
              background: #d9534f; color: #fff; transition: opacity 0.2s; }
  .stop-btn:hover { opacity: 0.85; }
  .stop-btn:disabled { background: #555; cursor: not-allowed; opacity: 0.6; }
  .conv-good { color: #5cb85c; }
  .conv-warn { color: #f0ad4e; }
  .conv-bad { color: #d9534f; }
  .verdict { padding: 8px 12px; border-radius: 4px; margin: 8px 0; font-size: 13px; text-align: center; }
  .verdict-good { background: #2d4a2d; color: #5cb85c; }
  .verdict-bad { background: #4a2d2d; color: #d9534f; }
  .verdict-warn { background: #4a3d2d; color: #f0ad4e; }
  .spinner { display: inline-block; width: 12px; height: 12px; border: 2px solid #555;
             border-top-color: #f0ad4e; border-radius: 50%; animation: spin 0.8s linear infinite; }
  @keyframes spin { to { transform: rotate(360deg); } }
  @media (max-width: 600px) { body { padding: 12px; font-size: 13px; } }
</style>
</head>
<body>
<h1>cloudposterior</h1>
<div id="phases"><div class="phase"><span class="spinner"></span> <span class="detail">waiting for sampling to start...</span></div></div>
<button id="stopBtn" class="stop-btn" disabled>Waiting for sampling...</button>
<div id="convergence"></div>
<div id="sampling"></div>
<div id="traces"></div>
<div id="banner"></div>
<script>
let polling = true;
let stopRequested = false;
const stopBtn = document.getElementById('stopBtn');

// Construct sibling endpoint URLs from our own URL
// Dashboard: https://workspace--{dash-label}-env.modal.run
// Progress: https://workspace--{prog-label}-env.modal.run
const dashLabel = '__DASHBOARD_LABEL__';
const progLabel = '__PROGRESS_LABEL__';
const stopLabel = '__STOP_LABEL__';
const origin = window.location.origin; // https://workspace--dash-label-env.modal.run
const progressUrl = origin.replace(dashLabel, progLabel);
const stopUrl = origin.replace(dashLabel, stopLabel);

stopBtn.addEventListener('click', async () => {
  if (stopRequested) return;
  stopRequested = true;
  stopBtn.textContent = 'Stopping...';
  stopBtn.disabled = true;
  try {
    await fetch(stopUrl, {method: 'POST'});
  } catch (e) {}
});

async function poll() {
  if (!polling) return;
  try {
    const r = await fetch(progressUrl);
    const data = await r.json();
    renderPhases(data.phases || []);
    renderSampling(data.sampling);
    if (data.convergence) renderConvergence(data.convergence);
    if (data.traces) renderTraces(data.traces);

    // Enable stop button during sampling (not tuning)
    const isSampling = (data.sampling && data.sampling.chains &&
      Object.values(data.sampling.chains).some(c => c.phase === 'sampling'));
    if (isSampling && !stopRequested && !data.complete) {
      stopBtn.textContent = 'Stop sampling';
      stopBtn.disabled = false;
    } else if (data.complete) {
      stopBtn.textContent = 'Sampling complete';
      stopBtn.disabled = true;
      document.getElementById('banner').innerHTML =
        '<div class="complete-banner">Sampling complete</div>';
      polling = false;
    }
  } catch (e) {
    document.getElementById('banner').innerHTML =
      '<div style="color:#d9534f;padding:8px;font-size:12px;">fetch error: ' + e.message + '</div>';
  }
  if (polling) setTimeout(poll, 1000);
}
function renderPhases(phases) {
  let html = '';
  for (const p of phases) {
    let icon;
    if (p.status === 'done') icon = '<span class="done">&#10003;</span>';
    else if (p.status === 'in_progress') icon = '<span class="spinner"></span>';
    else icon = '<span class="error">&#10007;</span>';
    html += '<div class="phase">' + icon + ' <span class="detail">' + p.detail + '</span></div>';
  }
  document.getElementById('phases').innerHTML = html;
}
function renderSampling(s) {
  if (!s || !s.chains) { document.getElementById('sampling').innerHTML = ''; return; }
  let html = '<table><tr><th>Chain</th><th>Progress</th><th>Draws</th><th>Div</th><th>Step</th><th>Speed</th><th>ETA</th></tr>';
  const ids = Object.keys(s.chains).sort((a,b) => +a - +b);
  for (const id of ids) {
    const c = s.chains[id];
    const pct = c.total > 0 ? (c.draw / c.total * 100) : 0;
    const barClass = c.divergences > 0 ? 'bar-div' : 'bar-ok';
    const speed = c.draws_per_sec > 0 ? Math.round(c.draws_per_sec) + '/s' : '--';
    const eta = c.eta_seconds > 0 ? c.eta_seconds.toFixed(0) + 's' : '--';
    html += '<tr>'
      + '<td>' + id + ' <span class="detail">[' + c.phase.slice(0,4) + ']</span></td>'
      + '<td><div class="bar-bg"><div class="bar-fill ' + barClass + '" style="width:' + pct + '%"></div></div></td>'
      + '<td>' + c.draw + '/' + c.total + '</td>'
      + '<td' + (c.divergences > 0 ? ' class="error"' : '') + '>' + c.divergences + '</td>'
      + '<td>' + c.step_size.toFixed(3) + '</td>'
      + '<td>' + speed + '</td>'
      + '<td>' + eta + '</td>'
      + '</tr>';
  }
  html += '</table>';
  html += '<div class="footer">Divergences: ' + s.total_divergences + ' | Elapsed: ' + s.elapsed.toFixed(1) + 's</div>';
  document.getElementById('sampling').innerHTML = html;
}
const traceCharts = {};
const chainColors = ['#1764f4', '#d9534f', '#5cb85c', '#f0ad4e', '#9b59b6', '#1abc9c', '#e67e22', '#3498db'];

function renderTraces(traces) {
  const container = document.getElementById('traces');
  const paramNames = Object.keys(traces).sort();

  for (const param of paramNames) {
    const chainData = traces[param];
    if (!chainData || chainData.length === 0) continue;

    const maxLen = Math.max(...chainData.map(c => c.length));
    const xValues = Array.from({length: maxLen}, (_, i) => i);
    const data = [xValues];
    const series = [{label: 'Draw'}];

    for (let c = 0; c < chainData.length; c++) {
      data.push(chainData[c]);
      series.push({
        label: 'Chain ' + c,
        stroke: chainColors[c % chainColors.length],
        width: 1,
      });
    }

    const divId = 'trace-' + param;
    if (!traceCharts[param]) {
      // Create new chart
      let div = document.getElementById(divId);
      if (!div) {
        const wrapper = document.createElement('div');
        wrapper.style.marginTop = '12px';
        wrapper.innerHTML = '<div style="color:#888;font-size:12px;margin-bottom:4px;">' + param + '</div>';
        div = document.createElement('div');
        div.id = divId;
        wrapper.appendChild(div);
        container.appendChild(wrapper);
      }
      const width = Math.min(container.clientWidth || 400, 600);
      const opts = {
        width: width,
        height: 120,
        series: series,
        axes: [{show: false}, {}],
        legend: {show: false},
        cursor: {show: false},
      };
      traceCharts[param] = new uPlot(opts, data, div);
    } else {
      // Update existing chart
      traceCharts[param].setData(data);
    }
  }
}

function renderConvergence(conv) {
  if (!conv || !conv.params) { document.getElementById('convergence').innerHTML = ''; return; }
  const params = conv.params;
  const names = Object.keys(params).sort();
  if (names.length === 0) return;

  function rhatClass(v) { return v < 1.01 ? 'conv-good' : v < 1.05 ? 'conv-warn' : 'conv-bad'; }
  function essClass(v) { return v >= 400 ? 'conv-good' : v >= 100 ? 'conv-warn' : 'conv-bad'; }

  let allGood = true;
  let html = '<table><tr><th>Parameter</th><th>R-hat</th><th>Bulk ESS</th><th>Tail ESS</th></tr>';
  for (const name of names) {
    const p = params[name];
    if (p.rhat >= 1.01 || p.ess_bulk < 400 || p.ess_tail < 400) allGood = false;
    html += '<tr>'
      + '<td>' + name + '</td>'
      + '<td class="' + rhatClass(p.rhat) + '">' + p.rhat.toFixed(3) + '</td>'
      + '<td class="' + essClass(p.ess_bulk) + '">' + p.ess_bulk + '</td>'
      + '<td class="' + essClass(p.ess_tail) + '">' + p.ess_tail + '</td>'
      + '</tr>';
  }
  html += '</table>';

  const verdictClass = allGood ? 'verdict-good' : 'verdict-warn';
  const verdictText = allGood ? 'Converged (' + conv.draws + ' draws)' : 'Not yet converged (' + conv.draws + ' draws)';
  html = '<div class="verdict ' + verdictClass + '">' + verdictText + '</div>' + html;

  document.getElementById('convergence').innerHTML = html;
}
poll();
</script>
</body>
</html>"""
