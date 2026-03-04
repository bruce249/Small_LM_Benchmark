/* ═══════════════════════════════════════════════════════════════════════
   Arena – Frontend Application Logic
   ═══════════════════════════════════════════════════════════════════════ */

const API = window.location.origin;

// ── Capability colours ───────────────────────────────────────────────────
const CAP_COLOURS = {
  text_generation:      { bg: 'rgba(59,130,246,0.12)', fg: '#60a5fa', border: 'rgba(59,130,246,0.25)' },
  summarization:        { bg: 'rgba(168,85,247,0.12)', fg: '#c084fc', border: 'rgba(168,85,247,0.25)' },
  question_answering:   { bg: 'rgba(34,211,238,0.12)', fg: '#22d3ee', border: 'rgba(34,211,238,0.25)' },
  chat:                 { bg: 'rgba(52,211,153,0.12)', fg: '#34d399', border: 'rgba(52,211,153,0.25)' },
  code_generation:      { bg: 'rgba(251,191,36,0.12)', fg: '#fbbf24', border: 'rgba(251,191,36,0.25)' },
  math_reasoning:       { bg: 'rgba(244,114,182,0.12)', fg: '#f472b6', border: 'rgba(244,114,182,0.25)' },
  translation:          { bg: 'rgba(129,140,248,0.12)', fg: '#818cf8', border: 'rgba(129,140,248,0.25)' },
  speech_to_text:       { bg: 'rgba(251,146,60,0.12)',  fg: '#fb923c', border: 'rgba(251,146,60,0.25)' },
  text_to_speech:       { bg: 'rgba(232,121,249,0.12)', fg: '#e879f9', border: 'rgba(232,121,249,0.25)' },
  image_generation:     { bg: 'rgba(248,113,113,0.12)', fg: '#f87171', border: 'rgba(248,113,113,0.25)' },
  image_classification: { bg: 'rgba(74,222,128,0.12)',  fg: '#4ade80', border: 'rgba(74,222,128,0.25)' },
  image_to_text:        { bg: 'rgba(45,212,191,0.12)',  fg: '#2dd4bf', border: 'rgba(45,212,191,0.25)' },
  object_detection:     { bg: 'rgba(253,186,116,0.12)', fg: '#fdba74', border: 'rgba(253,186,116,0.25)' },
  embedding:            { bg: 'rgba(148,163,184,0.12)', fg: '#94a3b8', border: 'rgba(148,163,184,0.25)' },
  sentiment_analysis:   { bg: 'rgba(196,181,253,0.12)', fg: '#c4b5fd', border: 'rgba(196,181,253,0.25)' },
  data_extraction:      { bg: 'rgba(134,239,172,0.12)', fg: '#86efac', border: 'rgba(134,239,172,0.25)' },
};

const CAP_ICONS = {
  text_generation: '✏️', summarization: '📝', question_answering: '❓', chat: '💬',
  code_generation: '💻', math_reasoning: '🧮', translation: '🌐', speech_to_text: '🎙️',
  text_to_speech: '🔊', image_generation: '🎨', image_classification: '🏷️',
  image_to_text: '📸', object_detection: '🔍', embedding: '📐',
  sentiment_analysis: '😊', data_extraction: '📊',
};

const MEDAL = { 1: '🥇', 2: '🥈', 3: '🥉' };

const EXAMPLE_REQUESTS = [
  'I want to build a podcast summariser that takes audio, transcribes it, summarises the text, and generates a short audio summary.',
  'Build me a chatbot that can translate between English and Spanish, answer questions about documents, and generate code snippets.',
  'I need an app that takes product images, classifies them, generates a description, and detects objects in the image.',
  'Create a math tutoring assistant that solves equations step by step, generates practice problems, and explains concepts clearly.',
];

const BENCHMARK_MODELS = [
  'Qwen/Qwen2.5-7B-Instruct',
  'meta-llama/Llama-3.1-8B-Instruct',
  'meta-llama/Llama-3.2-3B-Instruct',
  'Qwen/Qwen2.5-Coder-32B-Instruct',
];

// ═══════════════════════════════════════════════════════════════════════
//  Router
// ═══════════════════════════════════════════════════════════════════════

function navigateTo(page) {
  document.querySelectorAll('.page').forEach(p => p.classList.add('hidden'));
  const el = document.getElementById('page-' + page);
  if (el) { el.classList.remove('hidden'); el.style.animation = 'fadeIn 0.3s ease-out'; }
  document.querySelectorAll('.nav-link').forEach(n => n.classList.toggle('active', n.dataset.page === page));
  window.location.hash = page;
}

window.addEventListener('hashchange', () => {
  const page = location.hash.replace('#', '') || 'home';
  navigateTo(page);
});

// ═══════════════════════════════════════════════════════════════════════
//  Initialisation
// ═══════════════════════════════════════════════════════════════════════

document.addEventListener('DOMContentLoaded', () => {
  // Route based on hash
  const page = location.hash.replace('#', '') || 'home';
  navigateTo(page);

  // Populate example chips
  const chips = document.getElementById('example-chips');
  EXAMPLE_REQUESTS.forEach((ex, i) => {
    const chip = document.createElement('button');
    chip.className = 'chip';
    chip.textContent = ex.length > 70 ? ex.slice(0, 67) + '…' : ex;
    chip.title = ex;
    chip.onclick = () => {
      document.getElementById('wf-input').value = ex;
      updateBuildBtn();
    };
    chips.appendChild(chip);
  });

  // Populate capability legend
  const legend = document.getElementById('capability-legend');
  for (const [cap, col] of Object.entries(CAP_COLOURS)) {
    const row = document.createElement('div');
    row.className = 'flex items-center gap-2';
    row.innerHTML = `
      <span style="color:${col.fg}">${CAP_ICONS[cap] || '⚡'}</span>
      <span class="text-gray-400">${cap.replace(/_/g, ' ')}</span>
    `;
    legend.appendChild(row);
  }

  // Populate benchmark model checkboxes
  const checks = document.getElementById('bm-model-checks');
  BENCHMARK_MODELS.forEach((m, i) => {
    const lbl = document.createElement('label');
    lbl.className = 'model-check';
    lbl.innerHTML = `<input type="checkbox" value="${m}" ${i < 3 ? 'checked' : ''} /> ${m.split('/')[1]}`;
    checks.appendChild(lbl);
  });

  // Enable build button when text present
  document.getElementById('wf-input').addEventListener('input', updateBuildBtn);
  updateBuildBtn();
});

function updateBuildBtn() {
  const btn = document.getElementById('btn-build');
  const val = document.getElementById('wf-input').value.trim();
  btn.disabled = val.length < 5;
}

// ═══════════════════════════════════════════════════════════════════════
//  Toast notifications
// ═══════════════════════════════════════════════════════════════════════

function showToast(msg, type = 'info') {
  const container = document.getElementById('toast-container');
  const t = document.createElement('div');
  t.className = `toast toast-${type}`;
  const icons = { success: '✓', error: '✕', info: 'ℹ' };
  t.innerHTML = `<span>${icons[type] || 'ℹ'}</span> ${msg}`;
  container.appendChild(t);
  setTimeout(() => { t.style.opacity = '0'; t.style.transition = 'opacity 0.3s'; setTimeout(() => t.remove(), 300); }, 4000);
}

// ═══════════════════════════════════════════════════════════════════════
//  API helpers
// ═══════════════════════════════════════════════════════════════════════

async function apiPost(path, body) {
  const resp = await fetch(API + path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!resp.ok) {
    const text = await resp.text();
    throw new Error(`HTTP ${resp.status}: ${text.slice(0, 300)}`);
  }
  return resp.json();
}

// ═══════════════════════════════════════════════════════════════════════
//  WORKFLOW BUILDER
// ═══════════════════════════════════════════════════════════════════════

let currentWorkflow = null;

async function buildWorkflow() {
  const input = document.getElementById('wf-input').value.trim();
  if (input.length < 5) return;

  // Show loading
  hide('wf-results'); hide('wf-error');
  show('wf-loading');
  document.getElementById('btn-build').disabled = true;
  animatePhases();

  const body = {
    user_request: input,
    decomposer_model: document.getElementById('cfg-decomposer').value,
    quality_weight: parseFloat(document.getElementById('cfg-quality').value),
    latency_weight: parseFloat(document.getElementById('cfg-latency').value),
    cost_weight: parseFloat(document.getElementById('cfg-cost').value),
  };

  try {
    const wf = await apiPost('/workflow', body);
    currentWorkflow = wf;
    renderWorkflow(wf);
    hide('wf-loading');
    show('wf-results');
    showToast('Workflow built successfully!', 'success');
  } catch (err) {
    hide('wf-loading');
    document.getElementById('wf-error-msg').textContent = err.message;
    show('wf-error');
    showToast('Workflow build failed', 'error');
  } finally {
    document.getElementById('btn-build').disabled = false;
    updateBuildBtn();
  }
}

function animatePhases() {
  const phases = [
    { el: 'phase-1', delay: 0 },
    { el: 'phase-2', delay: 3000 },
    { el: 'phase-3', delay: 8000 },
  ];
  phases.forEach(p => {
    const el = document.getElementById(p.el);
    el.className = 'flex items-center gap-2';
  });
  document.getElementById('phase-1').classList.add('phase-active');

  setTimeout(() => {
    document.getElementById('phase-1').classList.replace('phase-active', 'phase-done');
    document.getElementById('phase-2').classList.add('phase-active');
    document.getElementById('wf-loading-msg').textContent = 'Benchmarking candidate models for each step';
  }, 3000);

  setTimeout(() => {
    document.getElementById('phase-2').classList.replace('phase-active', 'phase-done');
    document.getElementById('phase-3').classList.add('phase-active');
    document.getElementById('wf-loading-msg').textContent = 'Ranking models and assembling recommendation';
  }, 8000);
}

// ── Render workflow ──────────────────────────────────────────────────

function renderWorkflow(wf) {
  renderWfSummary(wf);
  renderWfPipeline(wf);
  renderWfBenchmarks(wf);
  renderWfDiagram(wf);
  renderWfJson(wf);
}

function renderWfSummary(wf) {
  const container = document.getElementById('wf-summary');
  container.innerHTML = `
    <div class="metric-card">
      <div class="metric-value">${wf.steps.length}</div>
      <div class="metric-label">Pipeline Steps</div>
    </div>
    <div class="metric-card">
      <div class="metric-value">${wf.total_estimated_latency.toFixed(2)}s</div>
      <div class="metric-label">Est. Latency</div>
    </div>
    <div class="metric-card">
      <div class="metric-value">$${wf.total_estimated_cost_per_run.toFixed(6)}</div>
      <div class="metric-label">Est. Cost / Run</div>
    </div>
    <div class="metric-card">
      <div class="metric-value" style="-webkit-text-fill-color: ${wf.status === 'completed' ? '#4ade80' : '#f87171'}">${wf.status}</div>
      <div class="metric-label">Status</div>
    </div>
  `;
}

function renderWfPipeline(wf) {
  const container = document.getElementById('wf-tab-pipeline');
  let html = `
    <div class="mb-6 p-4 rounded-xl bg-brand-500/5 border border-brand-500/10">
      <div class="text-xs text-brand-400 uppercase tracking-wider font-semibold mb-1">Task Analysis</div>
      <div class="text-gray-300 text-sm">${esc(wf.task_analysis)}</div>
    </div>
  `;

  wf.steps.forEach((step, i) => {
    const col = capColour(step.capability);
    html += `
      <div class="step-card animate-slide-up" style="animation-delay:${i * 0.1}s">
        <div class="flex items-start justify-between gap-4 mb-3">
          <div class="flex items-center gap-3">
            <div class="w-10 h-10 rounded-xl flex items-center justify-center font-bold text-sm" style="background:${col.bg};color:${col.fg};border:1px solid ${col.border}">
              ${step.step_number}
            </div>
            <div>
              <h4 class="font-semibold text-white">${esc(step.title)}</h4>
              <span class="cap-badge mt-1" style="background:${col.bg};color:${col.fg};border:1px solid ${col.border}">
                ${CAP_ICONS[step.capability] || '⚡'} ${step.capability.replace(/_/g, ' ')}
              </span>
            </div>
          </div>
          <div class="text-right text-xs text-gray-500 shrink-0">
            <div>Quality: <span class="text-gray-300 font-semibold">${step.avg_quality_score.toFixed(3)}</span></div>
            <div>Latency: <span class="text-gray-300 font-semibold">${step.avg_latency_seconds.toFixed(2)}s</span></div>
          </div>
        </div>
        <p class="text-sm text-gray-400 mb-3">${esc(step.description)}</p>
        <div class="flex flex-wrap items-center gap-3 text-xs">
          <div class="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-green-500/8 border border-green-500/15">
            <span class="text-green-400 font-semibold">✦ Recommended:</span>
            <span class="text-gray-300">${modelShort(step.recommended_model)}</span>
          </div>
          ${step.alternatives.length > 0 ? `
            <div class="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-white/3 border border-white/5">
              <span class="text-gray-500">Alternatives:</span>
              <span class="text-gray-400">${step.alternatives.map(modelShort).join(', ')}</span>
            </div>
          ` : ''}
        </div>
        ${step.input_description || step.output_description ? `
          <div class="grid grid-cols-2 gap-3 mt-3 text-xs">
            <div class="px-3 py-2 rounded-lg bg-white/2 border border-white/4">
              <span class="text-gray-500">Input:</span>
              <span class="text-gray-400 ml-1">${esc(step.input_description)}</span>
            </div>
            <div class="px-3 py-2 rounded-lg bg-white/2 border border-white/4">
              <span class="text-gray-500">Output:</span>
              <span class="text-gray-400 ml-1">${esc(step.output_description)}</span>
            </div>
          </div>
        ` : ''}
      </div>
    `;
    if (i < wf.steps.length - 1) {
      html += '<div class="step-connector"></div>';
    }
  });
  container.innerHTML = html;
}

function renderWfBenchmarks(wf) {
  const container = document.getElementById('wf-tab-benchmarks');
  let html = '';

  wf.step_benchmarks.forEach((bench, i) => {
    const col = capColour(bench.capability);
    html += `
      <div class="mb-8 animate-slide-up" style="animation-delay:${i * 0.1}s">
        <div class="flex items-center gap-3 mb-3">
          <span class="cap-badge" style="background:${col.bg};color:${col.fg};border:1px solid ${col.border}">
            ${CAP_ICONS[bench.capability] || '⚡'} ${bench.capability.replace(/_/g, ' ')}
          </span>
          <h4 class="font-semibold">Step ${bench.step_number}: ${esc(bench.step_title)}</h4>
          <span class="text-xs text-gray-500">(${bench.candidates_tested} candidates)</span>
        </div>
        <div class="text-sm text-gray-400 mb-4 pl-1">${esc(bench.recommendation_reason)}</div>
    `;

    if (bench.rankings.length > 0) {
      html += `<table class="rank-table"><thead><tr>
        <th></th><th>Model</th><th>Quality</th><th>Latency</th><th>Cost</th><th>Status</th>
      </tr></thead><tbody>`;

      bench.rankings.forEach(r => {
        const statusHtml = r.error
          ? `<span class="text-red-400 text-xs">${esc(r.error).slice(0, 40)}</span>`
          : '<span class="text-green-400">✓</span>';

        html += `<tr>
          <td class="medal">${MEDAL[r.rank] || ''}</td>
          <td class="font-medium text-gray-200">${modelShort(r.model_id)}</td>
          <td>
            <div class="flex items-center gap-2">
              <span>${r.avg_quality_score.toFixed(4)}</span>
              <div class="quality-bar-bg"><div class="quality-bar" style="width:${Math.min(r.avg_quality_score * 100, 100)}%"></div></div>
            </div>
          </td>
          <td>${r.avg_latency_seconds.toFixed(3)}s</td>
          <td>$${r.estimated_cost_usd.toFixed(6)}</td>
          <td>${statusHtml}</td>
        </tr>`;
      });
      html += '</tbody></table>';

      // Mini bar chart
      const valid = bench.rankings.filter(r => !r.error && r.avg_quality_score > 0);
      if (valid.length > 0) {
        const maxQ = Math.max(...valid.map(r => r.avg_quality_score), 0.01);
        html += '<div class="mt-4">';
        valid.forEach(r => {
          const pct = (r.avg_quality_score / maxQ) * 100;
          const c = r.rank === 1 ? 'background:linear-gradient(90deg,#3380ff,#7c3aed)' : 'background:rgba(255,255,255,0.08)';
          html += `
            <div class="bar-chart-row">
              <div class="bar-chart-label">${modelShort(r.model_id)}</div>
              <div class="bar-chart-bg">
                <div class="bar-chart-fill" style="width:${pct}%;${c}">${r.avg_quality_score.toFixed(4)}</div>
              </div>
            </div>
          `;
        });
        html += '</div>';
      }
    } else {
      html += '<div class="text-gray-500 text-sm">No benchmark data for this step.</div>';
    }

    html += '</div>';
    if (i < wf.step_benchmarks.length - 1) {
      html += '<hr class="border-white/5 my-6" />';
    }
  });

  container.innerHTML = html;
}

function renderWfDiagram(wf) {
  const container = document.getElementById('wf-tab-diagram');
  let html = '<div class="flex flex-col items-center py-4">';

  // Start node
  html += `
    <div class="px-4 py-2 rounded-full bg-green-500/10 border border-green-500/20 text-green-400 text-xs font-semibold mb-2">
      User Input
    </div>
    <div class="diagram-arrow"></div>
  `;

  wf.steps.forEach((step, i) => {
    const col = capColour(step.capability);
    html += `
      <div class="diagram-node animate-slide-up" style="animation-delay:${i * 0.15}s; border-color:${col.border}">
        <div class="flex items-center justify-center gap-2 mb-1">
          <span style="color:${col.fg}">${CAP_ICONS[step.capability] || '⚡'}</span>
          <span class="font-semibold text-sm">${esc(step.title)}</span>
        </div>
        <div class="cap-badge mx-auto mb-2" style="background:${col.bg};color:${col.fg};border:1px solid ${col.border}">
          ${step.capability.replace(/_/g, ' ')}
        </div>
        <div class="text-xs text-gray-400 mb-1">${modelShort(step.recommended_model)}</div>
        <div class="flex justify-center gap-3 text-[10px] text-gray-500">
          <span>Q: ${step.avg_quality_score.toFixed(3)}</span>
          <span>L: ${step.avg_latency_seconds.toFixed(2)}s</span>
        </div>
      </div>
    `;
    if (i < wf.steps.length - 1) {
      html += '<div class="diagram-arrow"></div>';
    }
  });

  // End node
  html += `
    <div class="diagram-arrow"></div>
    <div class="px-4 py-2 rounded-full bg-brand-500/10 border border-brand-500/20 text-brand-400 text-xs font-semibold mt-2">
      Final Output
    </div>
  `;

  html += '</div>';
  container.innerHTML = html;
}

function renderWfJson(wf) {
  document.getElementById('wf-tab-json').innerHTML =
    `<pre class="json-block">${syntaxHighlight(JSON.stringify(wf, null, 2))}</pre>`;
}

// ── Workflow tab switcher ────────────────────────────────────────────

function switchWfTab(tab) {
  document.querySelectorAll('.wf-tab').forEach(t => t.classList.toggle('active', t.dataset.tab === tab));
  document.querySelectorAll('.wf-tab-content').forEach(c => c.classList.add('hidden'));
  const el = document.getElementById('wf-tab-' + tab);
  if (el) { el.classList.remove('hidden'); el.style.animation = 'fadeIn 0.3s ease-out'; }
}

// ═══════════════════════════════════════════════════════════════════════
//  BENCHMARK
// ═══════════════════════════════════════════════════════════════════════

let currentBenchmark = null;

async function runBenchmark() {
  const selectedModels = [...document.querySelectorAll('#bm-model-checks input:checked')].map(c => c.value);
  if (selectedModels.length === 0) { showToast('Select at least one model', 'error'); return; }

  hide('bm-results'); hide('bm-error'); hide('bm-empty');
  show('bm-loading');
  document.getElementById('btn-benchmark').disabled = true;

  const body = {
    task_type: document.getElementById('bm-task').value,
    model_ids: selectedModels,
    max_samples: parseInt(document.getElementById('bm-samples').value) || 5,
  };

  try {
    const report = await apiPost('/experiments/sync', body);
    currentBenchmark = report;
    renderBenchmark(report);
    hide('bm-loading');
    show('bm-results');
    showToast('Benchmark complete!', 'success');
  } catch (err) {
    hide('bm-loading');
    document.getElementById('bm-error-msg').textContent = err.message;
    show('bm-error');
    showToast('Benchmark failed', 'error');
  } finally {
    document.getElementById('btn-benchmark').disabled = false;
  }
}

function renderBenchmark(report) {
  renderBmLeaderboard(report);
  renderBmCharts(report);
  renderBmJson(report);
}

function renderBmLeaderboard(report) {
  const container = document.getElementById('bm-tab-leaderboard');
  const lb = report.leaderboard || [];
  if (lb.length === 0) {
    container.innerHTML = '<div class="text-gray-500 text-center py-8">No results.</div>';
    return;
  }

  let html = `
    <div class="mb-6 grid grid-cols-3 gap-4">
      <div class="metric-card">
        <div class="metric-value">${report.task_type}</div>
        <div class="metric-label">Task Type</div>
      </div>
      <div class="metric-card">
        <div class="metric-value">${report.models.length}</div>
        <div class="metric-label">Models</div>
      </div>
      <div class="metric-card">
        <div class="metric-value" style="-webkit-text-fill-color:${report.status==='completed'?'#4ade80':'#f87171'}">${report.status}</div>
        <div class="metric-label">Status</div>
      </div>
    </div>
    <table class="rank-table"><thead><tr>
      <th></th><th>Model</th><th>Quality</th><th>Latency</th><th>Cost</th><th>Tasks</th>
    </tr></thead><tbody>
  `;

  lb.sort((a, b) => a.rank - b.rank).forEach(e => {
    html += `<tr>
      <td class="medal">${MEDAL[e.rank] || e.rank}</td>
      <td class="font-medium text-gray-200">${modelShort(e.model_id)}</td>
      <td>
        <div class="flex items-center gap-2">
          <span>${e.avg_quality_score.toFixed(4)}</span>
          <div class="quality-bar-bg"><div class="quality-bar" style="width:${Math.min(e.avg_quality_score * 100, 100)}%"></div></div>
        </div>
      </td>
      <td>${e.avg_latency_seconds.toFixed(3)}s</td>
      <td>$${e.total_cost_usd.toFixed(6)}</td>
      <td>${e.num_tasks}</td>
    </tr>`;
  });
  html += '</tbody></table>';
  container.innerHTML = html;
}

function renderBmCharts(report) {
  const container = document.getElementById('bm-tab-bm-charts');
  const lb = (report.leaderboard || []).sort((a, b) => a.rank - b.rank);
  if (lb.length === 0) { container.innerHTML = '<p class="text-gray-500">No data.</p>'; return; }

  const maxQ = Math.max(...lb.map(e => e.avg_quality_score), 0.01);
  const maxL = Math.max(...lb.map(e => e.avg_latency_seconds), 0.01);

  let html = '<h4 class="font-semibold mb-3">Quality Score</h4>';
  lb.forEach(e => {
    const pct = (e.avg_quality_score / maxQ) * 100;
    html += `<div class="bar-chart-row">
      <div class="bar-chart-label">${modelShort(e.model_id)}</div>
      <div class="bar-chart-bg">
        <div class="bar-chart-fill" style="width:${pct}%;background:linear-gradient(90deg,#3380ff,#7c3aed)">${e.avg_quality_score.toFixed(4)}</div>
      </div>
    </div>`;
  });

  html += '<h4 class="font-semibold mt-6 mb-3">Latency (seconds)</h4>';
  lb.forEach(e => {
    const pct = (e.avg_latency_seconds / maxL) * 100;
    html += `<div class="bar-chart-row">
      <div class="bar-chart-label">${modelShort(e.model_id)}</div>
      <div class="bar-chart-bg">
        <div class="bar-chart-fill" style="width:${pct}%;background:linear-gradient(90deg,#f472b6,#fb923c)">${e.avg_latency_seconds.toFixed(3)}s</div>
      </div>
    </div>`;
  });

  // Metric breakdown table
  if (lb.some(e => e.metric_breakdown && Object.keys(e.metric_breakdown).length > 0)) {
    const keys = new Set();
    lb.forEach(e => Object.keys(e.metric_breakdown || {}).forEach(k => keys.add(k)));
    const sortedKeys = [...keys].sort();

    html += '<h4 class="font-semibold mt-6 mb-3">Metric Breakdown</h4>';
    html += '<table class="rank-table"><thead><tr><th>Model</th>';
    sortedKeys.forEach(k => { html += `<th>${k}</th>`; });
    html += '</tr></thead><tbody>';
    lb.forEach(e => {
      html += `<tr><td class="font-medium text-gray-200">${modelShort(e.model_id)}</td>`;
      sortedKeys.forEach(k => {
        const v = (e.metric_breakdown || {})[k];
        html += `<td>${v !== undefined ? v.toFixed(4) : '—'}</td>`;
      });
      html += '</tr>';
    });
    html += '</tbody></table>';
  }

  container.innerHTML = html;
}

function renderBmJson(report) {
  document.getElementById('bm-tab-bm-json').innerHTML =
    `<pre class="json-block">${syntaxHighlight(JSON.stringify(report, null, 2))}</pre>`;
}

function switchBmTab(tab) {
  document.querySelectorAll('.bm-tab').forEach(t => t.classList.toggle('active', t.dataset.tab === tab));
  document.querySelectorAll('.bm-tab-content').forEach(c => c.classList.add('hidden'));
  const el = document.getElementById('bm-tab-' + tab);
  if (el) { el.classList.remove('hidden'); el.style.animation = 'fadeIn 0.3s ease-out'; }
}

// ═══════════════════════════════════════════════════════════════════════
//  Helpers
// ═══════════════════════════════════════════════════════════════════════

function show(id) { document.getElementById(id).classList.remove('hidden'); }
function hide(id) { document.getElementById(id).classList.add('hidden'); }

function esc(str) {
  const d = document.createElement('div');
  d.textContent = str || '';
  return d.innerHTML;
}

function modelShort(id) {
  return (id || '').split('/').pop();
}

function capColour(cap) {
  return CAP_COLOURS[cap] || CAP_COLOURS['text_generation'];
}

function syntaxHighlight(json) {
  return json.replace(/("(\\u[a-zA-Z0-9]{4}|\\[^u]|[^\\"])*"(\s*:)?|\b(true|false|null)\b|-?\d+(\.\d+)?([eE][+-]?\d+)?)/g, (match) => {
    let cls = 'text-amber-300';           // number
    if (/^"/.test(match)) {
      if (/:$/.test(match)) cls = 'text-blue-400'; // key
      else cls = 'text-green-400';        // string
    } else if (/true|false/.test(match)) {
      cls = 'text-purple-400';            // bool
    } else if (/null/.test(match)) {
      cls = 'text-gray-500';              // null
    }
    return `<span class="${cls}">${match}</span>`;
  });
}
