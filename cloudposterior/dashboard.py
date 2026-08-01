"""Live progress dashboard served via Modal web endpoint."""

from __future__ import annotations

from functools import lru_cache

from cloudposterior.progress import (
    JobPhase,
    PhaseUpdate,
    SamplingProgress,
)


class DashboardSink:
    """Sink that writes progress state to a Modal Dict for the dashboard endpoint.

    ``key`` is the Dict key the run-state is written under. Single-model runs use
    the default ``"progress"``; ``cp.map`` gives each model its own key (its label)
    so N concurrent workers write independent keys without clobbering each other.
    """

    def __init__(self, progress_dict, key: str = "progress"):
        self._dict = progress_dict
        self._key = key
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
            # Off the event loop: this per-event blocking Dict write would warn
            # and stall the loop in async hosts (marimo). _run_blocking runs it
            # in a worker thread when a loop is active.
            from cloudposterior.backends.modal_backend import _run_blocking

            _run_blocking(self._dict.__setitem__, self._key, data)
        except Exception:
            pass  # best-effort


def render_dashboard_html(progress_label: str, stop_label: str,
                          dashboard_label: str, stop_token: str) -> str:
    """Render dashboard HTML with endpoint labels baked in.

    The JS constructs full URLs from the labels by deriving the Modal workspace
    URL pattern from window.location. ``stop_token`` is the secret sent with
    every progress and stop request.

    Every value is substituted as a JSON literal: a bare replace into a quoted
    JS string breaks out of the literal on a single quote or backslash. All
    four are required -- they previously defaulted to "", which made
    ``origin.replace('', '')`` a silent no-op that pointed the progress URL at
    the dashboard itself.
    """
    import json

    values = {
        "__PROGRESS_LABEL__": progress_label,
        "__STOP_LABEL__": stop_label,
        "__DASHBOARD_LABEL__": dashboard_label,
        "__STOP_TOKEN__": stop_token,
    }
    missing = [k for k, v in values.items() if not v]
    if missing:
        raise ValueError(f"render_dashboard_html requires: {', '.join(missing)}")

    html = DASHBOARD_HTML
    for placeholder, value in values.items():
        html = html.replace(placeholder, json.dumps(value))
    return (html
        .replace("__UPLOT_CSS__", _static("uPlot.min.css"))
        .replace("__UPLOT_JS__", _static("uPlot.iife.min.js"))
    )


@lru_cache(maxsize=None)
def _static(name: str) -> str:
    """Read a vendored asset from cloudposterior/static.

    uPlot is inlined rather than pulled from a CDN: this page holds the stop
    token and polls posterior draws, so third-party script execution on its
    origin would be enough to exfiltrate both.
    """
    from importlib.resources import files

    return files("cloudposterior").joinpath("static", name).read_text(encoding="utf-8")


DASHBOARD_HTML = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>cloudposterior</title>
<style>__UPLOT_CSS__</style>
<script>__UPLOT_JS__</script>
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
  /* -- multi-model (cp.map) overview + detail -- */
  .header-actions { display: flex; align-items: center; gap: var(--sp-3); }
  .back-link { color: var(--accent); text-decoration: none; font-size: 13px; font-weight: 600; cursor: pointer; }
  .back-link:hover { text-decoration: underline; }
  .stop-sm { padding: var(--sp-1) var(--sp-3); font-size: 12px; border-width: 1px; }
  .mv-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: var(--sp-3); gap: var(--sp-3); }
  .mv-name { font-size: 15px; font-weight: 600; letter-spacing: -0.01em; }
  /* compact overview card: one clickable summary per model */
  .model-card { background: var(--bg-card); border: 1px solid var(--border); border-radius: var(--radius);
                padding: var(--sp-4); margin-bottom: var(--sp-3); cursor: pointer; transition: border-color 0.15s; }
  .model-card:hover { border-color: var(--accent); }
  .model-card .mv-name::after { content: ' \\2192'; color: var(--text-dim); }
  /* compact mode flattens the shared template into a lean summary card */
  .model-card.compact .section { border: none; background: transparent; padding: 0; margin-bottom: var(--sp-2); }
  .model-card.compact .section-title { display: none; }
  .model-card.compact .traces, .model-card.compact .banner { display: none; }
  .model-card.compact .sampling { margin-top: var(--sp-2); }
  @media (max-width: 600px) { body { padding: var(--sp-3); } .section { padding: var(--sp-3); } }
</style>
</head>
<body>
<div class="header">
  <h1>cloudposterior</h1>
  <div class="header-actions">
    <a id="backLink" class="back-link" style="display:none">&#8592; All models</a>
    <button id="stopBtn" class="stop-btn" disabled>Waiting...</button>
  </div>
</div>
<div id="overview"></div>
<div id="detail"></div>
<div id="confirmOverlay" class="confirm-overlay" style="display:none">
  <div class="confirm-dialog">
    <p id="confirmText">Stop sampling early? You'll keep all draws collected so far.</p>
    <button class="confirm-yes" id="confirmYes">Stop</button>
    <button class="confirm-no" id="confirmNo">Cancel</button>
  </div>
</div>
<!-- One per-model view, cloned for the full detail page and (compact) for each
     overview card. A single-model cp.cloud run is just N=1 of this. -->
<template id="modelTpl">
  <div class="mv-header"><span class="mv-name"></span><button class="mv-stop stop-btn stop-sm" disabled>Waiting...</button></div>
  <div class="section"><div class="section-title">Status</div><div class="phases"><div class="phase"><span class="spinner"></span> <span class="detail">waiting for sampling to start...</span></div></div></div>
  <div class="section"><div class="section-title">Chains</div><div class="sampling"></div></div>
  <div class="convergence"></div>
  <div class="traces"></div>
  <div class="banner"></div>
</template>
<script>
let polling = true;

// Construct sibling endpoint URLs from our own URL
// Dashboard: https://workspace--{dash-label}-env.modal.run
// Progress: https://workspace--{prog-label}-env.modal.run
const dashLabel = __DASHBOARD_LABEL__;
const progLabel = __PROGRESS_LABEL__;
const stopLabel = __STOP_LABEL__;
const stopToken = __STOP_TOKEN__;
const origin = window.location.origin; // https://workspace--dash-label-env.modal.run
const progressUrl = origin.replace(dashLabel, progLabel);
const stopUrl = origin.replace(dashLabel, stopLabel);
// /progress serves parameter names and posterior draws, so it is token-gated
// like /stop.
const progressFetchUrl = progressUrl
  + (stopToken ? ('?token=' + encodeURIComponent(stopToken)) : '');

const headerStop = document.getElementById('stopBtn');
const backLink = document.getElementById('backLink');
const overviewEl = document.getElementById('overview');
const detailEl = document.getElementById('detail');
const overlay = document.getElementById('confirmOverlay');
const confirmText = document.getElementById('confirmText');
const tpl = document.getElementById('modelTpl');

// One stop path for both the global "Stop all" and every per-model Stop.
// null label => global stop ("stop" key); a label => per-model ("stop:<label>").
const stopReq = {};      // label -> true once a per-model stop is requested
let stopAllReq = false;
let pendingStop;          // remembered while the confirm dialog is open
function requestStop(label) {
  pendingStop = label;
  confirmText.textContent = label
    ? "Stop this model early? You'll keep all draws collected so far."
    : "Stop all models early? You'll keep all draws collected so far.";
  overlay.style.display = 'flex';
}
document.getElementById('confirmNo').addEventListener('click', () => { overlay.style.display = 'none'; });
document.getElementById('confirmYes').addEventListener('click', async () => {
  overlay.style.display = 'none';
  const label = pendingStop;
  if (label) stopReq[label] = true; else stopAllReq = true;
  const params = [];
  if (label) params.push('model=' + encodeURIComponent(label));
  if (stopToken) params.push('token=' + encodeURIComponent(stopToken));
  const url = stopUrl + (params.length ? ('?' + params.join('&')) : '');
  try { await fetch(url, {method: 'POST'}); } catch (e) {}
});
headerStop.addEventListener('click', (e) => {
  e.preventDefault();
  if (!headerStop.disabled) requestStop(null);  // header button always stops all
});
backLink.addEventListener('click', (e) => { e.preventDefault(); location.hash = '#/'; });

// Clean param names: strip "modelname::" prefix
function cleanName(name) {
  const idx = name.indexOf('::');
  return idx >= 0 ? name.substring(idx + 2) : name;
}

// Escape dynamic strings (param names, worker messages) before they land in
// innerHTML -- a parameter named "beta<sub>" must render, not inject markup.
function esc(s) {
  return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;')
    .replace(/>/g, '&gt;').replace(/"/g, '&quot;').replace(/'/g, '&#39;');
}

// A single-model cp.cloud run is just N=1 of the map shape -- normalize so the
// rest of the UI has exactly one code path.
function normalize(data) {
  if (data && data.models) return data;
  return {models: [{label: 'progress', name: 'cloudposterior'}], runs: {progress: data || {}}};
}

// One per-model view, cloned from the shared template; cached per (mode,label)
// so polls update content in place rather than rebuilding DOM (and charts).
const views = {};
function makeView(label, name, compact) {
  const root = document.createElement('div');
  root.className = compact ? 'model-card compact' : 'model-view';
  root.appendChild(tpl.content.cloneNode(true));
  const refs = {
    root,
    name: root.querySelector('.mv-name'),
    stop: root.querySelector('.mv-stop'),
    phases: root.querySelector('.phases'),
    sampling: root.querySelector('.sampling'),
    convergence: root.querySelector('.convergence'),
    traces: root.querySelector('.traces'),
    banner: root.querySelector('.banner'),
  };
  refs.name.textContent = cleanName(name);
  refs.stop.addEventListener('click', (e) => {
    e.stopPropagation();
    if (!refs.stop.disabled) requestStop(label);
  });
  if (compact) root.addEventListener('click', () => { location.hash = '#/' + label; });
  return refs;
}
function getView(label, name, compact) {
  const key = (compact ? 'c:' : 'd:') + label;
  if (!views[key]) views[key] = makeView(label, name, compact);
  return views[key];
}

function setStopButton(btn, state, requested) {
  if (!btn) return;
  const isSampling = state.sampling && state.sampling.chains &&
    Object.values(state.sampling.chains).some(c => c.phase === 'sampling');
  if (state.complete) { btn.textContent = 'Done'; btn.disabled = true; }
  else if (requested) { btn.textContent = 'Stopping...'; btn.disabled = true; }
  else if (isSampling) { btn.textContent = 'Stop'; btn.disabled = false; }
  else { btn.textContent = 'Waiting...'; btn.disabled = true; }
}

// Update one cached view from its run-state, reusing the shared renderers.
function updateView(refs, state, compact, label, showName) {
  state = state || {};
  refs.name.style.display = showName ? '' : 'none';
  renderPhases(refs.phases, state.phases || [], compact);
  renderSampling(refs.sampling, state.sampling, compact);
  renderConvergence(refs.convergence, state.convergence, compact);
  if (!compact && state.traces && Object.keys(state.traces).length) {
    renderTraces(refs.traces, state.traces, label);
  }
  refs.banner.innerHTML = state.complete ? '<div class="complete-banner">Sampling complete</div>' : '';
  setStopButton(refs.stop, state, stopReq[label] || stopAllReq);
}

function selectedLabel(view) {
  const r = location.hash.replace(/^#\\/?/, '');
  return (r && view.runs && view.runs[r] !== undefined) ? r : null;
}

// One router: overview (N>1, no selection) vs a single model's detail page.
let lastView = null;
function render(view) {
  lastView = view;
  const models = view.models || [];
  const multi = models.length > 1;
  const sel = selectedLabel(view);
  backLink.style.display = (multi && sel) ? '' : 'none';

  if (sel || !multi) {
    const label = sel || (models[0] || {}).label;
    overviewEl.style.display = 'none';
    detailEl.style.display = '';
    headerStop.style.display = multi ? 'none' : '';  // per-model Stop lives in the view when multi
    const name = (models.find(m => m.label === label) || {}).name || label;
    const refs = getView(label, name, false);
    if (detailEl.firstChild !== refs.root) { detailEl.innerHTML = ''; detailEl.appendChild(refs.root); }
    updateView(refs, view.runs[label], false, label, multi);
    refs.stop.style.display = multi ? '' : 'none';   // N=1: header Stop covers it
    if (!multi) setStopButton(headerStop, view.runs[label] || {}, stopAllReq);
    return;
  }

  // Overview: one compact card per model, in manifest order.
  detailEl.style.display = 'none';
  overviewEl.style.display = '';
  headerStop.style.display = '';
  for (const m of models) {
    const refs = getView(m.label, m.name, true);
    if (refs.root.parentElement !== overviewEl) overviewEl.appendChild(refs.root);
    updateView(refs, view.runs[m.label], true, m.label, true);
  }
  const anySampling = models.some(m => {
    const st = view.runs[m.label] || {};
    return st.sampling && st.sampling.chains &&
      Object.values(st.sampling.chains).some(c => c.phase === 'sampling');
  });
  const allComplete = models.every(m => (view.runs[m.label] || {}).complete);
  if (allComplete) { headerStop.textContent = 'Done'; headerStop.disabled = true; }
  else if (stopAllReq) { headerStop.textContent = 'Stopping...'; headerStop.disabled = true; }
  else if (anySampling) { headerStop.textContent = 'Stop all'; headerStop.disabled = false; }
  else { headerStop.textContent = 'Waiting...'; headerStop.disabled = true; }
}
window.addEventListener('hashchange', () => { if (lastView) render(lastView); });

let failCount = 0;
function showOffline() {
  const note = '<div style="background:#eee;color:#555;padding:8px 12px;border-radius:6px;' +
    'font-size:13px;margin-bottom:12px;">Dashboard offline &mdash; the run has ended. ' +
    'Check your notebook for results.</div>';
  (overviewEl.style.display !== 'none' ? overviewEl : detailEl).insertAdjacentHTML('afterbegin', note);
  headerStop.disabled = true;
}
async function poll() {
  if (!polling) return;
  let data;
  try {
    const r = await fetch(progressFetchUrl);
    data = await r.json();
    failCount = 0;
  } catch (e) {
    // The /progress endpoint becomes unreachable when the run ends and the
    // Modal app shuts down with the notebook cell. Tolerate a couple of
    // transient misses, then stop with a calm message (not a red error loop).
    failCount++;
    if (failCount >= 3) { showOffline(); polling = false; }
    if (polling) setTimeout(poll, 1000);
    return;
  }
  // Rendering is deliberately outside the fetch try: a render bug (say a
  // missing field hitting .toFixed) is not the run going offline, and
  // counting it as one told the user their live run had ended.
  try {
    const view = normalize(data);
    render(view);
    if (view.models.length && view.models.every(m => (view.runs[m.label] || {}).complete)) {
      polling = false;
    }
  } catch (e) {
    console.error('cloudposterior: dashboard render failed', e);
  }
  if (polling) setTimeout(poll, 1000);
}
function phaseIcon(status) {
  if (status === 'done') return '<span class="done">&#10003;</span>';
  if (status === 'in_progress') return '<span class="spinner"></span>';
  return '<span class="error">&#10007;</span>';
}
function renderPhases(el, phases, compact) {
  phases = phases || [];
  if (compact) {
    // Overview: collapse to the single most-relevant phase line.
    const active = [...phases].reverse().find(p => p.status === 'in_progress') || phases[phases.length - 1];
    el.innerHTML = active
      ? '<div class="phase">' + phaseIcon(active.status) + ' <span class="detail">' + esc(active.detail) + '</span></div>'
      : '<div class="phase"><span class="spinner"></span> <span class="detail">waiting...</span></div>';
    return;
  }
  let html = '';
  for (const p of phases) {
    html += '<div class="phase">' + phaseIcon(p.status) + ' <span class="detail">' + esc(p.detail) + '</span></div>';
  }
  el.innerHTML = html;
}
function renderSampling(el, s, compact) {
  if (!s || !s.chains) { el.innerHTML = ''; return; }
  const ids = Object.keys(s.chains).sort((a,b) => +a - +b);
  if (compact) {
    // Overview: one overall bar driven by the slowest chain + a summary line.
    let minDraw = Infinity, total = 0;
    for (const id of ids) { const c = s.chains[id]; minDraw = Math.min(minDraw, c.draw); total = Math.max(total, c.total); }
    if (minDraw === Infinity) minDraw = 0;
    const pct = total > 0 ? (minDraw / total * 100) : 0;
    const barClass = s.total_divergences > 0 ? 'bar-div' : 'bar-ok';
    el.innerHTML = '<div class="bar-bg"><div class="bar-fill ' + barClass + '" style="width:' + pct + '%"></div></div>'
      + '<div class="footer">' + ids.length + ' chains | ' + minDraw + '/' + total + ' draws | Div: '
      + s.total_divergences + ' | ' + s.elapsed.toFixed(1) + 's</div>';
    return;
  }
  let html = '<table><tr><th>Chain</th><th>Progress</th><th>Draws</th><th>Div</th><th>Step</th><th>Speed</th><th>ETA</th></tr>';
  for (const id of ids) {
    const c = s.chains[id];
    const pct = c.total > 0 ? (c.draw / c.total * 100) : 0;
    const barClass = c.divergences > 0 ? 'bar-div' : 'bar-ok';
    const speed = c.draws_per_sec > 0 ? Math.round(c.draws_per_sec) + '/s' : '--';
    const eta = c.eta_seconds > 0 ? c.eta_seconds.toFixed(0) + 's' : '--';
    html += '<tr>'
      + '<td>' + esc(id) + ' <span class="detail">[' + esc(c.phase.slice(0,4)) + ']</span></td>'
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
  el.innerHTML = html;
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

function renderTraces(container, traces, prefix) {
  if (!container.classList.contains('section')) {
    container.classList.add('section');
    container.innerHTML = '<div class="section-title">Traces</div>';
  }
  const paramNames = Object.keys(traces).sort();
  const cw = container.clientWidth || 700;
  const narrow = cw < 600;
  const chartW = narrow ? cw - 40 : Math.floor((cw - 40) / 2);

  for (const param of paramNames) {
    // Namespace chart state + element ids by model so per-model charts (which
    // may share param names) never collide.
    const ckey = prefix + '::' + param;
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

    const traceId = 'trace-' + ckey;
    const kdeId = 'kde-' + ckey;

    // Recreate charts if number of chains changed
    if (traceCharts[ckey] && traceCharts[ckey].series.length !== nChains + 1) {
      traceCharts[ckey].destroy();
      delete traceCharts[ckey];
      kdeCharts[ckey].destroy();
      delete kdeCharts[ckey];
      const old = document.getElementById(traceId);
      if (old) old.parentElement.parentElement.remove();
    }

    if (!traceCharts[ckey]) {
      // Create wrapper with label and two chart divs side by side
      const wrapper = document.createElement('div');
      wrapper.style.marginTop = '16px';
      wrapper.innerHTML = '<div style="color:var(--text-muted);font-size:12px;margin-bottom:4px;font-weight:600;font-family:monospace;">' + esc(cleanName(param)) + '</div>';
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
      kdeCharts[ckey] = new uPlot({
        width: chartW, height: chartH, series: kdeSeries,
        scales: {x: {range: (u, dMin, dMax) => [yMin, yMax]}},
        axes: [{size: 30, stroke: '#555', ticks: {stroke: '#333'}}, {size: 40, stroke: '#555', ticks: {stroke: '#333'}}],
        legend: {show: false}, cursor: {show: false},
      }, kdeData, kdeDiv);
      traceCharts[ckey] = new uPlot({
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
      kdeCharts[ckey].setSize({width: newChartW, height: 140});
      traceCharts[ckey].setSize({width: newChartW, height: 140});
      // Update scale ranges for new data
      kdeCharts[ckey].scales.x.range = (u, dMin, dMax) => [yMin, yMax];
      traceCharts[ckey].scales.y.range = (u, dMin, dMax) => [yMin, yMax];
      kdeCharts[ckey].setData(kdeData);
      traceCharts[ckey].setData(traceData);
    }
  }
}

function renderConvergence(el, conv, compact) {
  if (!conv || !conv.params) { el.innerHTML = ''; return; }
  const params = conv.params;
  const names = Object.keys(params).sort();
  if (names.length === 0) { el.innerHTML = ''; return; }

  let allGood = true;
  for (const name of names) {
    const p = params[name];
    if (p.rhat >= 1.01 || p.ess_bulk < 400 || p.ess_tail < 400) allGood = false;
  }
  const vClass = allGood ? 'verdict-good' : 'verdict-warn';
  const vText = allGood ? 'Converged (' + conv.draws + ' draws)' : 'Not yet converged (' + conv.draws + ' draws)';
  if (compact) {
    // Overview: verdict badge only.
    el.innerHTML = '<div class="verdict ' + vClass + '">' + vText + '</div>';
    return;
  }

  function rhatClass(v) { return v < 1.01 ? 'conv-good' : v < 1.05 ? 'conv-warn' : 'conv-bad'; }
  function essClass(v) { return v >= 400 ? 'conv-good' : v >= 100 ? 'conv-warn' : 'conv-bad'; }
  let tableHtml = '<table><tr><th>Parameter</th><th>R-hat</th><th>Bulk ESS</th><th>Tail ESS</th></tr>';
  for (const name of names) {
    const p = params[name];
    tableHtml += '<tr>'
      + '<td style="font-family:monospace;font-size:12px;">' + esc(cleanName(name)) + '</td>'
      + '<td class="' + rhatClass(p.rhat) + '">' + p.rhat.toFixed(3) + '</td>'
      + '<td class="' + essClass(p.ess_bulk) + '">' + p.ess_bulk + '</td>'
      + '<td class="' + essClass(p.ess_tail) + '">' + p.ess_tail + '</td>'
      + '</tr>';
  }
  tableHtml += '</table>';
  el.innerHTML = '<div class="section"><div class="section-title">Convergence</div>'
    + '<div class="verdict ' + vClass + '">' + vText + '</div>' + tableHtml + '</div>';
}
poll();
</script>
</body>
</html>"""
