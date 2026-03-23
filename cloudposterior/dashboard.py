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
            "total_draws": progress.total_draws,
        }
        self._write()

    def show_convergence(self, update):
        self._convergence = {
            name: {"rhat": p.rhat, "ess_bulk": p.ess_bulk, "ess_tail": p.ess_tail}
            for name, p in update.params.items()
        }
        self._convergence_draws = update.draws
        self._traces = update.traces if update.traces else {}
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
  :root {
    --bg: #0f1117; --bg-card: #1a1d27; --bg-hover: #22262f;
    --text: #e4e4e7; --text-muted: #71717a; --text-dim: #52525b;
    --border: #27272a; --accent: #3b82f6;
    --green: #22c55e; --yellow: #eab308; --red: #ef4444;
    --green-bg: #052e16; --yellow-bg: #422006; --red-bg: #450a0a;
    --sp-1: 4px; --sp-2: 8px; --sp-3: 12px; --sp-4: 16px; --sp-5: 24px; --sp-6: 32px;
    --radius: 8px;
  }
  @media (prefers-color-scheme: light) {
    :root {
      --bg: #fafafa; --bg-card: #ffffff; --bg-hover: #f4f4f5;
      --text: #18181b; --text-muted: #71717a; --text-dim: #a1a1aa;
      --border: #e4e4e7;
      --green-bg: #dcfce7; --yellow-bg: #fef9c3; --red-bg: #fee2e2;
    }
  }
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
         font-size: 14px; background: var(--bg); color: var(--text);
         padding: var(--sp-4); max-width: 1200px; margin: 0 auto; line-height: 1.5; }
  .header { display: flex; justify-content: space-between; align-items: center; margin-bottom: var(--sp-5); }
  .header h1 { font-size: 18px; font-weight: 600; letter-spacing: -0.02em; }
  .stop-btn { padding: var(--sp-2) var(--sp-4); border: 2px solid var(--red); border-radius: var(--radius);
              font-family: inherit; font-size: 13px; font-weight: 600; cursor: pointer;
              background: transparent; color: var(--red); transition: all 0.15s; }
  .stop-btn:hover { background: var(--red); color: #fff; }
  .stop-btn:disabled { border-color: var(--text-dim); color: var(--text-dim); cursor: not-allowed; }
  .stop-btn:disabled:hover { background: transparent; }
  .section { background: var(--bg-card); border: 1px solid var(--border); border-radius: var(--radius);
             padding: var(--sp-4); margin-bottom: var(--sp-3); }
  .section-title { font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em;
                   color: var(--text-muted); margin-bottom: var(--sp-3); }
  .phase { padding: 2px 0; font-size: 13px; font-family: 'SF Mono', 'Menlo', monospace; }
  .done { color: var(--green); }
  .error { color: var(--red); }
  .detail { color: var(--text-muted); }
  table { width: 100%; border-collapse: collapse; font-size: 13px; }
  th { text-align: left; padding: var(--sp-2); border-bottom: 1px solid var(--border);
       color: var(--text-muted); font-weight: 500; font-size: 12px; }
  td { padding: var(--sp-2); }
  tr:hover td { background: var(--bg-hover); }
  .bar-bg { background: var(--border); border-radius: 4px; height: 8px; overflow: hidden; }
  .bar-fill { height: 100%; border-radius: 4px; transition: width 0.3s ease; }
  .bar-ok { background: var(--accent); }
  .bar-div { background: var(--red); }
  .footer { color: var(--text-dim); margin-top: var(--sp-2); font-size: 12px; }
  .complete-banner { background: var(--green-bg); color: var(--green); padding: var(--sp-3);
                     border-radius: var(--radius); margin-top: var(--sp-4); text-align: center;
                     font-weight: 600; border: 1px solid var(--green); }
  .conv-good { color: var(--green); }
  .conv-warn { color: var(--yellow); }
  .conv-bad { color: var(--red); font-weight: 600; }
  .verdict { padding: var(--sp-2) var(--sp-3); border-radius: var(--radius); margin-bottom: var(--sp-3);
             font-size: 13px; text-align: center; font-weight: 600; }
  .verdict-good { background: var(--green-bg); color: var(--green); border: 1px solid var(--green); }
  .verdict-warn { background: var(--yellow-bg); color: var(--yellow); border: 1px solid var(--yellow); }
  .spinner { display: inline-block; width: 12px; height: 12px; border: 2px solid var(--border);
             border-top-color: var(--yellow); border-radius: 50%; animation: spin 0.8s linear infinite;
             vertical-align: middle; }
  @keyframes spin { to { transform: rotate(360deg); } }
  .confirm-overlay { position: fixed; inset: 0; background: rgba(0,0,0,0.5); display: flex;
                     align-items: center; justify-content: center; z-index: 100; }
  .confirm-dialog { background: var(--bg-card); border: 1px solid var(--border); border-radius: var(--radius);
                    padding: var(--sp-5); max-width: 360px; text-align: center; }
  .confirm-dialog p { margin-bottom: var(--sp-4); color: var(--text-muted); font-size: 13px; }
  .confirm-dialog button { padding: var(--sp-2) var(--sp-5); border-radius: var(--radius); font-family: inherit;
                           font-size: 13px; font-weight: 600; cursor: pointer; border: none; margin: 0 var(--sp-2); }
  .confirm-yes { background: var(--red); color: #fff; }
  .confirm-no { background: var(--bg-hover); color: var(--text); border: 1px solid var(--border) !important; }
  @media (max-width: 600px) { body { padding: var(--sp-3); } .section { padding: var(--sp-3); } }
</style>
</head>
<body>
<div class="header">
  <h1>cloudposterior</h1>
  <button id="stopBtn" class="stop-btn" disabled>Waiting...</button>
</div>
<div class="section">
  <div class="section-title">Status</div>
  <div id="phases"><div class="phase"><span class="spinner"></span> <span class="detail">waiting for sampling to start...</span></div></div>
</div>
<div class="section">
  <div class="section-title">Chains</div>
  <div id="sampling"></div>
</div>
<div id="convergence"></div>
<div id="traces"></div>
<div id="banner"></div>
<div id="confirmOverlay" class="confirm-overlay" style="display:none">
  <div class="confirm-dialog">
    <p>Stop sampling early? You'll keep all draws collected so far.</p>
    <button class="confirm-yes" id="confirmYes">Stop</button>
    <button class="confirm-no" id="confirmNo">Cancel</button>
  </div>
</div>
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

const overlay = document.getElementById('confirmOverlay');
document.getElementById('confirmNo').addEventListener('click', () => { overlay.style.display = 'none'; });
document.getElementById('confirmYes').addEventListener('click', async () => {
  overlay.style.display = 'none';
  stopRequested = true;
  stopBtn.textContent = 'Stopping...';
  stopBtn.disabled = true;
  try { await fetch(stopUrl, {method: 'POST'}); } catch (e) {}
});
stopBtn.addEventListener('click', (e) => {
  e.preventDefault();
  if (stopRequested || stopBtn.disabled) return;
  overlay.style.display = 'flex';
});

// Clean param names: strip "modelname::" prefix
function cleanName(name) {
  const idx = name.indexOf('::');
  return idx >= 0 ? name.substring(idx + 2) : name;
}

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
      stopBtn.textContent = 'Stop';
      stopBtn.disabled = false;
    } else if (data.complete) {
      stopBtn.textContent = 'Done';
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
const kdeCharts = {};
const chainColors = ['#1764f4', '#d9534f', '#5cb85c', '#f0ad4e', '#9b59b6', '#1abc9c', '#e67e22', '#3498db'];

// Gaussian KDE in JS
function kde(values, nPoints) {
  nPoints = nPoints || 100;
  if (values.length < 2) return {x: [0], y: [0]};
  const n = values.length;
  const sorted = values.slice().sort((a, b) => a - b);
  const q1 = sorted[Math.floor(n * 0.25)];
  const q3 = sorted[Math.floor(n * 0.75)];
  const iqr = q3 - q1;
  const std = Math.sqrt(values.reduce((s, v) => { const d = v - values.reduce((a, b) => a + b, 0) / n; return s + d * d; }, 0) / n);
  const bw = 0.9 * Math.min(std, iqr / 1.34) * Math.pow(n, -0.2); // Silverman's rule
  if (bw === 0 || isNaN(bw)) return {x: [0], y: [0]};
  // Use 0.5th-99.5th percentile to exclude outliers (like ArviZ)
  const lo = sorted[Math.max(0, Math.floor(n * 0.005))] - 3 * bw;
  const hi = sorted[Math.min(n - 1, Math.floor(n * 0.995))] + 3 * bw;
  const step = (hi - lo) / (nPoints - 1);
  const x = Array.from({length: nPoints}, (_, i) => lo + i * step);
  const y = x.map(xi => {
    let sum = 0;
    for (let j = 0; j < n; j++) {
      const z = (xi - values[j]) / bw;
      sum += Math.exp(-0.5 * z * z);
    }
    return sum / (n * bw * Math.sqrt(2 * Math.PI));
  });
  return {x, y};
}

function renderTraces(traces) {
  const container = document.getElementById('traces');
  if (!container.classList.contains('section')) {
    container.classList.add('section');
    container.innerHTML = '<div class="section-title">Traces</div>';
  }
  const paramNames = Object.keys(traces).sort();
  const cw = container.clientWidth || 700;
  const narrow = cw < 600;
  const chartW = narrow ? cw - 40 : Math.floor((cw - 40) / 2);

  for (const param of paramNames) {
    const chainData = traces[param];
    if (!chainData || chainData.length === 0) continue;
    const nChains = chainData.length;

    // Compute robust y-range from all chains (0.5th-99.5th percentile)
    const allVals = chainData.flat().slice().sort((a, b) => a - b);
    const yLo = allVals[Math.max(0, Math.floor(allVals.length * 0.005))];
    const yHi = allVals[Math.min(allVals.length - 1, Math.floor(allVals.length * 0.995))];
    const yPad = (yHi - yLo) * 0.05;
    const yMin = yLo - yPad;
    const yMax = yHi + yPad;

    // -- Build trace data (right panel) --
    const maxLen = Math.max(...chainData.map(c => c.length));
    const traceX = Array.from({length: maxLen}, (_, i) => i);
    const traceData = [traceX];
    const traceSeries = [{label: 'Draw'}];
    for (let c = 0; c < nChains; c++) {
      traceData.push(chainData[c]);
      traceSeries.push({label: 'Chain ' + c, stroke: chainColors[c % chainColors.length], width: 1});
    }

    // -- Build KDE data (left panel) --
    const kdeResults = chainData.map(vals => kde(vals, 80));
    // Shared x-axis: union of all KDE x ranges
    const allX = kdeResults.flatMap(k => k.x);
    const kdeXmin = Math.min(...allX);
    const kdeXmax = Math.max(...allX);
    const nPts = 80;
    const kdeStep = (kdeXmax - kdeXmin) / (nPts - 1);
    const kdeX = Array.from({length: nPts}, (_, i) => kdeXmin + i * kdeStep);
    const kdeData = [kdeX];
    const kdeSeries = [{label: 'Value'}];
    for (let c = 0; c < nChains; c++) {
      // Interpolate each chain's KDE onto the shared x-axis
      const k = kdeResults[c];
      const interp = kdeX.map(xi => {
        if (xi <= k.x[0]) return k.y[0];
        if (xi >= k.x[k.x.length - 1]) return k.y[k.y.length - 1];
        let idx = 0;
        while (idx < k.x.length - 1 && k.x[idx + 1] < xi) idx++;
        const t = (xi - k.x[idx]) / (k.x[idx + 1] - k.x[idx]);
        return k.y[idx] + t * (k.y[idx + 1] - k.y[idx]);
      });
      kdeData.push(interp);
      kdeSeries.push({label: 'Chain ' + c, stroke: chainColors[c % chainColors.length], width: 2, fill: chainColors[c % chainColors.length] + '20', points: {show: false}});
    }

    const traceId = 'trace-' + param;
    const kdeId = 'kde-' + param;

    // Recreate charts if number of chains changed
    if (traceCharts[param] && traceCharts[param].series.length !== nChains + 1) {
      traceCharts[param].destroy();
      delete traceCharts[param];
      kdeCharts[param].destroy();
      delete kdeCharts[param];
      const old = document.getElementById(traceId);
      if (old) old.parentElement.parentElement.remove();
    }

    if (!traceCharts[param]) {
      // Create wrapper with label and two chart divs side by side
      const wrapper = document.createElement('div');
      wrapper.style.marginTop = '16px';
      wrapper.innerHTML = '<div style="color:var(--text-muted);font-size:12px;margin-bottom:4px;font-weight:600;font-family:monospace;">' + cleanName(param) + '</div>';
      const row = document.createElement('div');
      row.style.display = 'flex';
      row.style.gap = '8px';
      row.style.flexWrap = 'wrap';
      const kdeDiv = document.createElement('div');
      kdeDiv.id = kdeId;
      const traceDiv = document.createElement('div');
      traceDiv.id = traceId;
      row.appendChild(kdeDiv);
      row.appendChild(traceDiv);
      wrapper.appendChild(row);
      container.appendChild(wrapper);

      const chartH = 140;
      kdeCharts[param] = new uPlot({
        width: chartW, height: chartH, series: kdeSeries,
        scales: {x: {range: (u, dMin, dMax) => [yMin, yMax]}},
        axes: [{size: 30, stroke: '#555', ticks: {stroke: '#333'}}, {size: 40, stroke: '#555', ticks: {stroke: '#333'}}],
        legend: {show: false}, cursor: {show: false},
      }, kdeData, kdeDiv);
      traceCharts[param] = new uPlot({
        width: chartW, height: chartH, series: traceSeries,
        scales: {y: {range: (u, dMin, dMax) => [yMin, yMax]}},
        axes: [{size: 30, stroke: '#555', ticks: {stroke: '#333'}}, {size: 40, stroke: '#555', ticks: {stroke: '#333'}}],
        legend: {show: false}, cursor: {show: false},
      }, traceData, traceDiv);
    } else {
      // Resize charts and update scale ranges
      const newCw = container.clientWidth || 700;
      const newNarrow = newCw < 600;
      const newChartW = newNarrow ? newCw - 40 : Math.floor((newCw - 40) / 2);
      kdeCharts[param].setSize({width: newChartW, height: 140});
      traceCharts[param].setSize({width: newChartW, height: 140});
      // Update scale ranges for new data
      kdeCharts[param].scales.x.range = (u, dMin, dMax) => [yMin, yMax];
      traceCharts[param].scales.y.range = (u, dMin, dMax) => [yMin, yMax];
      kdeCharts[param].setData(kdeData);
      traceCharts[param].setData(traceData);
    }
  }
}

let lastConvTotalDraws = 0;
function renderConvergence(conv) {
  if (!conv || !conv.params) { document.getElementById('convergence').innerHTML = ''; return; }
  const params = conv.params;
  const names = Object.keys(params).sort();
  if (names.length === 0) return;
  lastConvTotalDraws = conv.total_draws || 0;

  function rhatClass(v) { return v < 1.01 ? 'conv-good' : v < 1.05 ? 'conv-warn' : 'conv-bad'; }
  function essClass(v) { return v >= 400 ? 'conv-good' : v >= 100 ? 'conv-warn' : 'conv-bad'; }

  let allGood = true;
  let html = '<div class="section"><div class="section-title">Convergence</div>';

  const verdictClass = allGood ? 'verdict-good' : 'verdict-warn';
  let tableHtml = '<table><tr><th>Parameter</th><th>R-hat</th><th>Bulk ESS</th><th>Tail ESS</th></tr>';
  for (const name of names) {
    const p = params[name];
    if (p.rhat >= 1.01 || p.ess_bulk < 400 || p.ess_tail < 400) allGood = false;
    tableHtml += '<tr>'
      + '<td style="font-family:monospace;font-size:12px;">' + cleanName(name) + '</td>'
      + '<td class="' + rhatClass(p.rhat) + '">' + p.rhat.toFixed(3) + '</td>'
      + '<td class="' + essClass(p.ess_bulk) + '">' + p.ess_bulk + '</td>'
      + '<td class="' + essClass(p.ess_tail) + '">' + p.ess_tail + '</td>'
      + '</tr>';
  }
  tableHtml += '</table>';

  const vClass = allGood ? 'verdict-good' : 'verdict-warn';
  const vText = allGood ? 'Converged (' + conv.draws + ' draws)' : 'Not yet converged (' + conv.draws + ' draws)';
  html += '<div class="verdict ' + vClass + '">' + vText + '</div>' + tableHtml;
  html += '</div>';

  document.getElementById('convergence').innerHTML = html;
}
poll();
</script>
</body>
</html>"""
