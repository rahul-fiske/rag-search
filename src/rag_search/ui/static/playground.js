'use strict';
/* Playground: a sandbox for trying different models/tunables and benchmarking, structurally separate from your
   real collections (see core/playground.py).  It is the same pipeline as production: the stage numbers (1 Discover ...
   6 Write, 3.1-3.5 inside Convert, S2-S6 for a search) are those of the Architecture tab, an experiment's settings are
   grouped by the stage that reads them, and an index or bench run is watched live through its job record and event log
   (core/playground_runs.py) exactly like an indexing run on the Indexing tab: every stage per document, every page
   with the reader that read it and how it ended, the workers, and for a bench every query.  Index and bench runs are
   started in a background process; search is short and answers at once.  The dashboard process never loads a model. */
(function () {
  let refs = {}, experiments = [], current = null, cfg = null, runs = [], busy = false;
  let doclingTunables = null, effective = null, runData = null, runId = '', history = [], polling = false;
  const expanded = new Set();     // documents whose page table is open
  const PIN_HINT = 'blank = production’s choice (Models tab)';

  /* The docling/OCR/table/PDF-backend tunables (spec.py's "indexer" section that have an environment variable), fetched
     once from /api/config -- the same registry the Settings tab uses.  Each is placed under the stage that owns it. */
  async function ensureDoclingTunables() {
    if (doclingTunables) return doclingTunables;
    const r = await api('config');
    doclingTunables = (r.ok !== false && r.tunables) ? r.tunables.filter(t => t.section === 'indexer' && t.env) : [];
    return doclingTunables;
  }

  const stageMeta = id => {
    const found = ((PL.last() || {}).stages || []).find(s => s.id === id);
    return found || { id, name: (STAGE_KEYS[Object.keys(STAGE_KEYS).find(k => STAGE_KEYS[k][0] === id)] || [id, id])[1], scope: '', where: '', what: '', settings: [] };
  };
  const stageOfTunable = t => {
    const s = ((PL.last() || {}).stages || []).find(x => (x.settings || []).some(r => r.id === 'indexer.' + t.key));
    return s ? s.id : '3.2';
  };

  function guardStages(bm25, dense) {
    if (!bm25.checked && !dense.checked) { bm25.checked = true; toast('At least one of BM25 or Dense must stay on', '', 1800); }
  }

  function stageBreakdown(x) {
    const chipOf = (label, rank, score) => h('span', { class: 'muted small' }, score == null ? `${label} —` : `${label} #${rank} (${score})`);
    const parts = [chipOf('S2 BM25', x.bm25_rank, x.bm25_score), chipOf('S3 Dense', x.dense_rank, x.dense_score)];
    if (x.rrf_score != null) parts.push(h('span', { class: 'muted small' }, `S4 RRF ${x.rrf_score}`));
    parts.push(h('span', { class: 'muted small' }, x.rerank_score != null ? `S5 Rerank ${x.rerank_score}` : 'S5 Rerank —'));
    return h('div', { class: 'row', style: { gap: '12px', marginTop: '2px' } }, parts);
  }

  async function refreshExperiments() {
    // api(path, body) only POSTs when *body* is passed: every /api/playground/* route is POST-only (see server.py).
    const r = await api('playground/list', {});
    experiments = r.ok ? (r.result || []) : [];
    renderExperimentList();
  }

  function renderExperimentList() {
    fill(refs.list, experiments.length ? experiments.map(e => h('div', { class: 'row', style: { justifyContent: 'space-between', padding: '6px 0', borderBottom: '1px solid var(--border, #333)' } },
      h('div', null, h('a', { href: '#', on: { click: ev => { ev.preventDefault(); selectExperiment(e.name); } } }, h('b', null, e.name)),
        h('div', { class: 'muted small' }, `${(e.sources || []).length} source folder(s) · collections ${e.indexed_collections.join(', ') || '(not indexed)'} · ${e.bench_runs} bench run(s) · ${e.config.embedding_model} + ${e.config.rerank_model || 'no reranker'}`
          + (e.config.reader_model ? ` · reader ${e.config.reader_model}` : ''))),
      h('div', { class: 'row', style: { gap: '8px' } },
        h('button', { class: 'btn small', on: { click: () => selectExperiment(e.name) } }, 'Open'),
        h('button', { class: 'btn small danger', on: { click: () => removeExperiment(e.name) } }, 'Delete'))))
      : empty('No playground experiments yet. Create one below.'));
  }

  async function createExperiment() {
    const name = refs.newName.value.trim(); if (!name) return;
    const body = { name };
    if (refs.fromProd.checked) body.from_production = true;
    const folder = refs.newFolder.value.trim();
    if (folder) body.folders = [folder];
    const r = await act('playground/create', body, `Created ${name}`);
    if (r.ok) { refs.newName.value = ''; refs.newFolder.value = ''; refs.fromProd.checked = false; await refreshExperiments(); selectExperiment(name); }
  }

  const fmtVal = v => (v === null || v === undefined || v === '' ? '(default)' : String(v));

  // "Promote to production": preview the diff first (a read, safe even in read-only mode), then confirm -- a change to the
  // embedding model or chunk size/overlap needs an extra explicit confirm (it leaves the existing index stale).
  async function promote() {
    const preview = await api('playground/preview', { name: current });
    if (!preview.ok) { toast(preview.error || 'preview failed', 'bad'); return; }
    const changes = preview.result.changes;
    if (!Object.keys(changes).length) { toast('Already matches production -- nothing to promote', '', 2200); return; }
    const rows = Object.entries(changes).map(([k, d]) =>
      h('div', { class: 'small mono' }, `${k}: ${fmtVal(d.from)} → ${fmtVal(d.to)}`));
    const body = [h('p', null, 'This experiment’s settings will replace production’s config.json:'),
      h('div', { style: { margin: '8px 0' } }, rows)];
    const needsReindex = preview.result.needs_reindex;
    if (needsReindex) {
      const est = preview.result.reindex_estimate || {};
      const cost = est.documents != null ? `~${est.documents} document(s)`
        + (est.estimated_s != null ? `, about ${est.estimated_s}s` : '') : 'unknown cost';
      body.push(h('p', { class: 'notice warn' },
        `This changes the embedding model and/or chunk size/overlap, which makes the existing `
        + `production index stale (${cost}) until you run a full reindex afterwards.`));
    }
    const ok = await confirmDialog('Promote to production', h('div', null, body), 'Promote', needsReindex);
    if (!ok) return;
    const r = await act('playground/promote', { name: current, confirm: needsReindex }, 'Promoted to production');
    if (r.ok) refreshExperiments();
  }

  async function removeExperiment(name) {
    const ok = await confirmDialog('Delete experiment', h('p', null, `Permanently delete the playground experiment "${name}" and all its bench runs? This cannot be undone.`), 'Delete', true);
    if (!ok) return;
    const r = await api('playground/rm', { name, confirm: true });
    if (!r.ok) { toast(r.error || 'failed', 'bad'); return; }
    toast(`Deleted ${name}`, 'ok');
    if (current === name) { current = null; runData = null; location.hash = '#/playground'; fill(refs.detail, empty('Select or create an experiment above.')); }
    await refreshExperiments();
  }

  // Keeping the open experiment in the URL (#/playground/NAME) means a page refresh reopens the same detail panel.
  async function selectExperiment(name) {
    current = name; runData = null; runId = ''; history = [];
    location.hash = '#/playground/' + encodeURIComponent(name);
    fill(refs.detail, empty('Loading…'));
    const [r] = await Promise.all([api('playground/config', { name }), ensureDoclingTunables(), PL.load(false)]);
    cfg = r.ok ? r.result : null;
    if (!r.ok) { toast(r.error || `could not load ${name}`, 'bad'); current = null; location.hash = '#/playground'; }
    runs = [];
    renderDetail();
    refreshCompare();
    loadEffective();
    pollRun(true);
  }

  async function loadEffective() {
    if (!current) return;
    const name = current;
    const r = await api('playground/settings', { name });
    if (name !== current) return;
    effective = r.ok ? r.result : null;
    renderEffective();
  }

  async function saveConfig() {
    const body = {
      name: current,
      embedding_model: refs.cEmb.value.trim() || undefined,
      rerank_model: refs.cRerank.value.trim() || undefined,
      reader_model: refs.cReader.value.trim(),            // "" = back to production's choice
      repair_model: refs.cRepair.value.trim(),
      chunk_size: parseInt(refs.cChunkSize.value, 10) || undefined,
      chunk_overlap: parseInt(refs.cChunkOverlap.value, 10) || undefined,
      rerank: refs.cNoRerank.checked ? false : true,
      stages: [refs.cBm25.checked && 'bm25', refs.cDense.checked && 'dense', refs.cRerankStage.checked && 'rerank'].filter(Boolean).join(','),
      retrieval_pool: parseInt(refs.cPoolR.value, 10) || undefined,
      rerank_pool: parseInt(refs.cPoolK.value, 10) || undefined,
      rrf_k: parseInt(refs.cRrfK.value, 10) || undefined,
      ...Object.assign({}, ...refs.doclingForms.map(f => f.collect())),
    };
    const r = await act('playground/config', body, 'Config saved');
    if (r.ok) { cfg = r.result; renderDetail(); refreshExperiments(); loadEffective(); pollRun(false); }
  }

  /* ---------- runs: start, watch, cancel ---------- */

  async function startRun(kind) {
    const body = { name: current };
    if (kind === 'index') Object.assign(body, { rebuild: refs.idxRebuild.checked, force_md: refs.idxForceMd.checked });
    else Object.assign(body, { k: Math.max(1, parseInt(refs.bK.value, 10) || 5), label: refs.bLabel.value.trim() || undefined });
    const r = await api('playground/' + kind, body);
    if (!r.ok) { toast(r.error || 'could not start the run', 'bad'); return; }
    runId = r.result.job; runData = null;
    toast(kind === 'index' ? 'Index build started' : 'Benchmark started', 'ok', 1800);
    await pollRun(true);
    if (refs.runCard && refs.runCard.scrollIntoView) refs.runCard.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
  }

  async function cancelRun() {
    const r = await api('playground/cancel', { name: current });
    if (!r.ok) { toast(r.error || 'cancel failed', 'bad'); return; }
    toast(r.result.cancelled ? 'Run cancelled' : 'Nothing is running', '', 1800);
    pollRun(false);
  }

  const running = () => runData && runData.job && ['queued', 'running'].includes(runData.job.status);

  async function pollRun(withHistory) {
    if (!current || polling) return;
    polling = true;
    const name = current, wasRunning = running();
    try {
      const r = await api('playground/run', { name, run: runId });
      if (name !== current) return;
      runData = r.ok ? r.result : null;
      if (runData && runData.job) runId = runData.job.id;
      if (withHistory || (wasRunning && !running())) {
        const h2 = await api('playground/runs', { name });
        history = h2.ok ? (h2.result || []) : [];
      }
      renderRun();
      if (wasRunning && !running()) { refreshExperiments(); refreshCompare(); }
    } finally { polling = false; }
  }

  /* what each stage has done so far, counted from the documents' stage events and pages */
  function documentsOf(d) {
    const items = ((d.documents && d.documents.items) || []).slice().reverse().map(i => ({
      file: (i.collection ? i.collection + '/' : '') + (i.path || i.source), status: i.status, message: i.message, item: i, tl: i.timeline }));
    const flying = (d.in_flight || []).map(t => ({ file: t.file, status: 'working', tl: t }));
    return [...flying, ...items];
  }

  const p7 = job => (job.progress || {}).phase === 'merge';

  function stageRows(job, docs) {
    const N = job.files != null ? job.files : docs.length;
    const st = (x, key) => x.tl && x.tl.stages && x.tl.stages[key];
    const n = (key, st2) => docs.filter(x => { const s = st(x, key); return s && s.status === (st2 || 'done'); }).length;
    const unchanged = docs.filter(x => { const s = st(x, 'fingerprint'); return s && s.unchanged; }).length;
    const M = Math.max(0, N - unchanged);                                   // documents that really went through 3-6
    const pages = docs.reduce((a, x) => a + ((x.tl && x.tl.pages) || []).length, 0);
    const repaired = docs.reduce((a, x) => a + ((x.tl && x.tl.pages) || []).filter(p => p.outcome === 'repaired').length, 0);
    const low = docs.reduce((a, x) => a + ((x.tl && x.tl.pages) || []).filter(p => p.outcome === 'low' || p.outcome === 'error').length, 0);
    // 6 Write happens twice per document: nodes.json before embedding, embeddings and index.meta.json after it
    const written = docs.filter(x => { const w = st(x, 'write'); return w && w.status === 'done' && String(w.part || '').includes('embeddings'); }).length;
    const convertDone = n('convert');
    const chunked = n('chunk'), toEmbed = Math.max(chunked, docs.filter(x => st(x, 'embed')).length);
    const rows = [
      { id: '1', key: 'discover', n: job.files != null ? 1 : 0, of: 1, text: job.files != null ? `${job.files} file(s)` : '…' },
      { id: '2', key: 'fingerprint', n: n('fingerprint'), of: N, text: `${n('fingerprint')} / ${N}` + (unchanged ? ` · ${unchanged} unchanged` : '') },
      { id: '3', key: 'convert', n: convertDone, of: M, text: M ? `${convertDone} / ${M} document(s)` : '–' },
      { id: '3.1', key: 'profile', n: n('profile'), of: M, text: M ? `${n('profile')} / ${M}` : '–', sub: true },
      { id: '3.2', key: 'read', n: convertDone, of: M, started: pages > 0, text: `${pages} page(s) read`, sub: true },
      { id: '3.3', key: 'gate', n: convertDone, of: M, started: pages > 0, text: pages ? `${pages} checked` + (low ? ` · ${low} low` : '') : '–', sub: true },
      { id: '3.4', key: 'repair', n: convertDone, of: M, started: pages > 0, text: pages ? `${repaired} repaired` : '–', sub: true },
      { id: '3.5', key: 'reconcile', n: convertDone, of: M, text: M ? `${convertDone} / ${M}` : '–', sub: true },
      { id: '4', key: 'chunk', n: chunked, of: M, text: M ? `${chunked} / ${M}` : '–' },
      { id: '5', key: 'embed', n: n('embed'), of: Math.max(toEmbed, chunked), text: chunked ? `${n('embed')} / ${Math.max(toEmbed, chunked)}` : '–' },
      { id: '6', key: 'write', n: written, of: Math.max(toEmbed, chunked), text: chunked ? `${written} / ${Math.max(toEmbed, chunked)}` : '–' },
      { id: '7', key: 'merge', n: job.status === 'done' || p7(job) ? 1 : 0, of: 1, text: job.status === 'done' ? 'index built' : p7(job) ? 'merging…' : '–' },
    ];
    for (const r of rows) {
      const needsPages = ['read', 'gate', 'repair'].includes(r.key);
      const complete = r.of > 0 && r.n >= r.of && (!needsPages || r.started);
      const begun = r.n > 0 || r.started || (r.key === 'discover' && running());
      r.state = complete ? 'done' : begun ? 'now' : '';
    }
    return rows;
  }

  function stagesStrip(rows) {
    const chipOf = r => h('div', { class: 'step ' + (r.state === 'done' ? 'done' : r.state === 'now' && running() ? 'now' : ''), title: PL.name(r.key) },
      (r.state === 'done' ? '✓ ' : '') + `${r.id} ${PL.name(r.key)}`, h('small', null, r.text));
    return h('div', null,
      h('div', { class: 'steps' }, rows.filter(r => !r.sub).map(chipOf)),
      h('div', { class: 'steps sub' }, rows.filter(r => r.sub).map(chipOf)));
  }

  const statusClass = { done: 'ok', failed: 'bad', cancelled: 'warn', running: 'warn', queued: 'warn' };
  const docClass = { indexed: 'ok', skipped: '', converted: 'warn', no_text: 'warn', error: 'bad', working: 'warn', unsupported: '', removed: '' };
  const docText = { indexed: 'indexed', skipped: 'unchanged', converted: 'converted, waiting to embed', no_text: 'no text', error: 'failed', working: 'working', unsupported: 'unsupported', removed: 'removed', known: 'not tried again' };

  function stageChips(tl) {
    if (!tl) return h('span', { class: 'muted small' }, '…');
    const order = ['fingerprint', 'convert', 'chunk', 'embed', 'write'];
    return h('span', { class: 'row', style: { gap: '4px', flexWrap: 'wrap' } }, order.filter(k => tl.stages[k]).map(k => {
      const s = tl.stages[k], done = s.status === 'done';
      const extra = [s.how, s.pages != null ? s.pages + ' pages' : '', s.chunks != null ? s.chunks + ' chunks' : '', s.unchanged ? 'unchanged' : '', s.part || ''].filter(Boolean).join(' · ');
      return h('span', { class: 'chip ' + (done ? '' : 'warn'), title: [PL.label(k), s.seconds != null ? s.seconds + ' s' : '', extra].filter(Boolean).join(' · ') },
        `${s.id} ${done ? '✓' : '●'}${s.seconds != null && done ? ' ' + dur(s.seconds) : ''}`);
    }));
  }

  function pageCells(pages) {
    return h('div', { class: 'cv-grid' }, (pages || []).map(p => h('span', {
      class: `cv-cell cvb-${p.branch} out-${p.outcome}`,
      title: [`page ${p.page} of ${p.of}`, `${PL.label('read')}: ${CV.label(p.branch)}`, `outcome: ${p.outcome}`, p.cache === 'hit' ? 'from the page cache' : '',
        p.read_s != null ? `read in ${p.read_s} s` : '', p.chars != null ? `${p.chars} characters` : '', p.model ? `model ${p.model}` : ''].filter(Boolean).join(' · '),
    }, p.outcome === 'low' || p.outcome === 'error' ? '!' : String(p.page))));
  }

  function pageTable(pages) {
    return h('div', { class: 'table-wrap' }, h('table', { class: 'plain' },
      h('thead', null, h('tr', null, ['Page', `${PL.label('read')} (branch)`, 'Outcome', 'Read', 'Characters', 'Tokens', 'Reader model', `${PL.label('gate')} checks`].map(t => h('th', null, t)))),
      h('tbody', null, (pages || []).map(p => h('tr', null,
        h('td', null, `${p.page} / ${p.of}`), h('td', null, CV.label(p.branch), p.cache === 'hit' ? h('span', { class: 'muted small' }, ' (cache)') : null),
        h('td', null, p.outcome), h('td', null, p.read_s != null ? dur(p.read_s) : '–'), h('td', null, p.chars != null ? num(p.chars) : '–'),
        h('td', null, p.tokens ? num(p.tokens) : '–'), h('td', { class: 'small mono' }, p.model || '–'),
        h('td', { class: 'small muted' }, (p.gate || []).join(', ') || '–'))))));
  }

  function documentRows(docs) {
    if (!docs.length) return empty(running() ? 'Looking at the documents…' : 'No documents in this run.');
    return h('div', { class: 'table-wrap' }, h('table', null,
      h('thead', null, h('tr', null, ['Document', 'Status', 'Stages (hover for detail)', 'Pages'].map(t => h('th', null, t)))),
      h('tbody', null, docs.flatMap(x => {
        const pages = (x.tl && x.tl.pages) || [], open = expanded.has(x.file);
        const conv = x.item && x.item.conversion, openable = !!(conv && conv.trace);
        const main = h('tr', { class: openable ? 'clickable' : '', title: openable ? 'Click for the page-by-page record, the source page and the converted Markdown' : '', on: openable ? { click: ev => { if (!ev.target.closest('a')) CV.openSummary(conv, current); } } : {} },
          h('td', { class: 'mono small' }, x.file),
          h('td', null, pill(docText[x.status] || x.status, docClass[x.status] || '', x.status === 'working'),
            x.message ? h('div', { class: 'small ' + (x.status === 'error' ? 'bad' : 'muted') }, x.message) : null),
          h('td', null, stageChips(x.tl)),
          h('td', null, pages.length ? h('div', null, pageCells(pages),
            h('a', { href: '#', class: 'small', on: { click: e => { e.preventDefault(); open ? expanded.delete(x.file) : expanded.add(x.file); renderRun(); } } }, open ? 'hide the page table' : 'page table')) : h('span', { class: 'muted small' }, x.tl && x.tl.stages.convert ? 'no per-page detail (read whole)' : '–')));
        return open ? [main, h('tr', null, h('td', { colspan: 4 }, pageTable(pages)))] : [main];
      }))));
  }

  function benchLive(d) {
    const qs = d.queries || [];
    if (!qs.length) return null;
    return h('div', { class: 'table-wrap' }, h('table', null,
      h('thead', null, h('tr', null, ['#', 'Query', 'Found at rank', 'S3 embed query', 'S2 Keyword', 'S3 Vectors', 'S5 Rerank', 'Total'].map(t => h('th', null, t)))),
      h('tbody', null, qs.map(q => { const t = q.timing || {}; return h('tr', null,
        h('td', null, `${q.i}/${q.of}`), h('td', null, q.query), h('td', null, q.hit_rank ? pill('#' + q.hit_rank, 'ok') : pill('not found', 'bad')),
        h('td', null, ms(t.embed_query_ms)), h('td', null, ms(t.keyword_ms)), h('td', null, ms(t.dense_ms)), h('td', null, t.rerank_ms != null ? ms(t.rerank_ms) : '–'), h('td', null, ms(q.total_ms))); }))));
  }

  function historyTable() {
    if (history.length < 2) return null;
    return h('details', { style: { marginTop: '12px' } }, h('summary', { class: 'small muted' }, `Earlier runs of this experiment (${history.length})`),
      h('div', { class: 'table-wrap' }, h('table', { class: 'plain' }, h('tbody', null, history.map(x => h('tr', null,
        h('td', null, h('a', { href: '#', class: 'mono small', on: { click: e => { e.preventDefault(); runId = x.id; runData = null; pollRun(false); } } }, x.id)),
        h('td', null, x.kind), h('td', null, pill(x.status, statusClass[x.status] || '')),
        h('td', { class: 'small muted' }, x.elapsed_s != null ? dur(x.elapsed_s) : '–'),
        h('td', { class: 'small muted' }, x.kind === 'index' && x.status === 'done' ? `${x.summary.indexed} indexed, ${x.summary.skipped_fresh} unchanged, ${x.summary.error_count || 0} failed` : (x.error || ''))))))));
  }

  function runView() {
    if (!runData || !runData.job) return empty('No run yet. “Build index” reads, chunks and embeds the sample documents; this panel then shows every stage, document and page as it happens.');
    const d = runData, job = d.job, isRunning = running(), p = job.progress || {};
    const docs = documentsOf(d), sum = job.summary || {};
    const pct = p.total ? Math.min(100, Math.round(100 * (p.done || 0) / p.total)) : (isRunning ? 0 : 100);
    const head = h('div', { class: 'card-head', style: { marginTop: 0 } },
      h('h4', null, (job.kind === 'bench' ? 'Benchmark' : 'Index build') + (isRunning ? ' (live)' : '')), pill(job.status, statusClass[job.status] || '', isRunning),
      h('span', { class: 'muted small mono' }, job.id),
      h('div', { class: 'spacer' }, h('span', { class: 'muted small' }, job.elapsed_s != null ? (isRunning ? 'running for ' : 'took ') + dur(job.elapsed_s) : ''),
        isRunning ? h('button', { class: 'btn small danger', style: { marginLeft: '10px' }, on: { click: cancelRun } }, 'Cancel') : null));
    const body = [head];
    if (job.kind === 'bench') {
      body.push(h('div', { class: 'bar ' + (job.status === 'failed' ? 'bad' : !isRunning ? 'ok' : ''), style: { marginTop: '8px' } }, h('i', { style: { width: pct + '%' } })),
        h('div', { class: 'small muted', style: { margin: '6px 0 8px' } }, p.phase === 'load' ? (p.message || 'loading the models') : p.message || (isRunning ? 'starting…' : '')),
        benchLive(d));
      if (!isRunning && sum.metrics) {
        const m = sum.metrics;
        body.push(h('div', { class: 'grid g4', style: { marginTop: '10px' } }, statCard(String(m.recall_at_k), `recall@${sum.combo.k}`), statCard(String(m.mrr), 'MRR'),
          statCard(String(m.ndcg_at_k), `nDCG@${sum.combo.k}`), statCard(`${m.latency_ms.mean} ms`, `latency (p95 ${m.latency_ms.p95} ms)`)));
      }
    } else {
      const rows = stageRows(job, docs);
      body.push(stagesStrip(rows));
      body.push(h('div', { class: 'bar ' + (job.status === 'failed' ? 'bad' : !isRunning ? 'ok' : ''), style: { marginTop: '10px' } }, h('i', { style: { width: pct + '%' } })),
        h('div', { class: 'row small muted', style: { marginTop: '6px' } },
          isRunning ? h('span', null, `${p.done || 0} / ${p.total || '?'} in this phase (${p.phase === 'embed' ? PL.label('embed') : p.phase === 'convert' ? PL.label('convert') : p.phase === 'merge' ? PL.label('merge') : p.phase || 'starting'})`) : null,
          isRunning && d.run && d.run.now && d.run.now.file ? h('span', null, '· now: ', h('b', { class: 'mono' }, d.run.now.file), d.run.now.since ? ` for ${since(d.run.now.since)}` : '', d.run.now.stage ? ` · stage ${d.run.now.stage}` : '') : null));
      if (isRunning && d.run) {
        body.push(CV.live(d.run.live) ? h('div', { style: { marginTop: '12px' } }, h('h4', null, 'Pages in active files'), CV.live(d.run.live)) : null,
          CV.lanes(d.run.lanes) ? h('div', { style: { marginTop: '12px' } }, h('h4', null, 'Workers'), CV.lanes(d.run.lanes, d.run.stall_limit_s)) : null);
      }
      if (!isRunning && sum.indexed !== undefined) {
        const ps = sum.phase_s || {};
        body.push(h('div', { class: 'grid g4', style: { marginTop: '12px' } },
          statCard(num(sum.indexed || 0), 'indexed'), statCard(num(sum.skipped_fresh || 0), `unchanged (stage ${PL.id('fingerprint')})`),
          statCard(num(sum.no_text_count || 0), 'no text'), statCard(num(sum.error_count || 0), 'failed'),
          ps.convert != null ? statCard(dur(ps.convert), `${PL.label('convert')} time`) : null, ps.embed != null ? statCard(dur(ps.embed), `${PL.label('embed')} time`) : null,
          ps.merge != null ? statCard(dur(ps.merge), `${PL.label('merge')} time`) : null,
          ...(sum.conversion && sum.conversion.pages ? CV.tiles(sum.conversion).slice(0, 2) : [])));
        if (sum.conversion && (sum.conversion.ok_branches || sum.conversion.branches) && CV.bands(sum.conversion.ok_branches || sum.conversion.branches))
          body.push(h('div', { style: { marginTop: '10px' } }, h('h4', null, 'Pages in successfully converted files'), CV.bands(sum.conversion.ok_branches || sum.conversion.branches),
            sum.conversion.ok_runways && Object.keys(sum.conversion.ok_runways).length ? h('div', { style: { marginTop: '8px' } }, h('div', { class: 'small muted', style: { marginBottom: '4px' } }, 'by the lane whose reader finished the page'), CV.runwayBar(sum.conversion.ok_runways, sum.conversion.moves)) : null));
      }
      body.push(h('h4', { style: { margin: '14px 0 6px' } }, 'Documents'), documentRows(docs));
    }
    if (job.error) body.push(h('div', { class: 'notice bad', style: { marginTop: '12px' } }, job.error));
    else if (job.status === 'failed' || job.status === 'cancelled') body.push(h('div', { class: 'notice warn', style: { marginTop: '12px' } }, job.status === 'cancelled' ? 'The run was cancelled.' : 'The run failed.'));
    body.push(historyTable());
    return h('div', null, body);
  }

  /* ---------- the experiment's source folders: registered where they are, like a production collection's ---------- */
  async function loadSources() {
    if (!current || !refs.srcBox) return;
    const r = await api('playground/sources', { name: current, op: 'list' });
    const st = (r.ok && r.result && r.result.status) || [];
    fill(refs.srcBox, st.length ? h('div', { class: 'table-wrap' }, h('table', null,
      h('thead', null, h('tr', null, ['Collection', 'Folder', ''].map(t => h('th', null, t)))),
      h('tbody', null, st.map(x => h('tr', null, h('td', { class: 'mono' }, x.collection),
        h('td', null, h('code', null, x.folder), x.reachable ? null : h('span', { class: 'bad small' }, '  unreachable')),
        h('td', null, h('button', { class: 'btn small danger', disabled: readOnly(), on: { click: async () => { const q = await act('playground/sources', { name: current, op: 'remove', collection: x.collection }, `Removed ${x.collection}`); if (q.ok) loadSources(); } } }, 'Remove'))))))) :
      empty('No source folders yet: add the folder that holds the sample documents. It is read where it is and never changed.'));
  }
  async function addSource() {
    const folder = refs.srcFolder.value.trim(); if (!folder || !current) return;
    const body = { name: current, op: 'add', folder };
    if (refs.srcName.value.trim()) body.collection = refs.srcName.value.trim();
    const r = await act('playground/sources', body, 'Source added');
    if (r.ok) { refs.srcFolder.value = ''; refs.srcName.value = ''; loadSources(); refreshExperiments(); }
  }

  function renderRun() {
    if (!refs.runBox) return;
    patch(refs.runBox, runView());
    const idle = !running();
    for (const b of [refs.idxBtn, refs.bGo]) if (b) b.disabled = !idle || readOnly();
    if (refs.idxBtn) refs.idxBtn.textContent = idle ? 'Build index' : 'Run in progress…';
    if (refs.bGo) refs.bGo.textContent = idle ? 'Run benchmark' : 'Run in progress…';
  }

  /* ---------- search (short: answers at once) ---------- */

  async function runSearch() {
    const q = refs.sQ.value.trim(); if (!q || busy) return;
    busy = true; refs.sGo.disabled = true; refs.sGo.textContent = 'Searching…';
    fill(refs.sOut, empty('Loading the models and searching (the first search can take a while)…'));
    const stages = [refs.sBm25.checked && 'bm25', refs.sDense.checked && 'dense', refs.sRerank.checked && 'rerank'].filter(Boolean);
    const body = { name: current, query: q, top_k: Math.max(1, Math.min(25, parseInt(refs.sK.value, 10) || 5)), stages: stages.join(',') };
    const rp = parseInt(refs.sPoolR.value, 10); if (rp > 0) body.retrieval_pool = rp;
    const kp = parseInt(refs.sPoolK.value, 10); if (kp > 0) body.rerank_pool = kp;
    const rk = parseInt(refs.sRrfK.value, 10); if (rk > 0) body.rrf_k = rk;
    const r = await api('playground/search', body);
    busy = false; refs.sGo.disabled = false; refs.sGo.textContent = 'Search';
    if (!r.ok) { fill(refs.sOut, h('div', { class: 'notice bad' }, r.error || 'search failed')); return; }
    const res = r.result;
    const hits = res.results || [];
    const t = res.timing || {};
    const out = [h('div', { class: 'notice', style: { marginBottom: '10px' } }, `${hits.length} result(s) in ${ms(t.total_ms)} · embedding ${res.models.embedding} · reranker ${res.models.reranker || 'off'}`),
      h('div', { class: 'row small muted', style: { gap: '14px', marginBottom: '10px' } },
        h('span', null, `S3 embed the query ${ms(t.embed_query_ms)}`), h('span', null, `S2 Keyword ${ms(t.keyword_ms)}`), h('span', null, `S3 Vectors ${ms(t.dense_ms)}`),
        t.candidates != null ? h('span', null, `S4 Fuse → ${t.candidates} candidates`) : null,
        h('span', null, t.reranked ? `S5 Rerank ${ms(t.rerank_ms)}` : 'S5 Rerank off'), h('span', null, 'S6 Top k'))];
    if (t.rerank_error) out.push(h('div', { class: 'notice warn' }, 'Rerank: ' + t.rerank_error));
    if (!hits.length) out.push(empty(res.note || 'No results.'));
    for (const x of hits) {
      out.push(h('div', { class: 'hit' },
        h('div', { class: 'hit-head' }, h('span', { class: 'rank' }, '#' + x.rank), h('b', null, x.file), h('span', null, 'p.' + x.page),
          x.heading ? h('span', { class: 'muted' }, '· ' + x.heading) : null, chip(x.collection),
          h('span', { class: 'muted small', style: { marginLeft: 'auto' } }, 'score ' + x.score)),
        stageBreakdown(x), h('div', { class: 'hit-text' }, x.text)));
    }
    fill(refs.sOut, out);
  }

  async function refreshCompare() {
    const r = await api('playground/compare', { name: current });
    runs = r.ok ? (r.result || []) : [];
    renderCompare();
  }

  function renderCompare() {
    if (!refs.compare) return;
    if (!runs.length) { fill(refs.compare, empty('No bench runs recorded yet for this experiment.')); return; }
    const rows = runs.slice().sort((a, b) => b.metrics.recall_at_k - a.metrics.recall_at_k);
    fill(refs.compare, h('div', { class: 'table-wrap' }, h('table', null,
      h('thead', null, h('tr', null, ['Run', 'Label', 'Recall', 'MRR', 'nDCG', 'Latency (ms)', 'Combo'].map(t => h('th', null, t)))),
      h('tbody', null, rows.map(r => h('tr', null,
        h('td', { class: 'small mono' }, r.run_id), h('td', null, r.label || '–'),
        h('td', null, String(r.metrics.recall_at_k)), h('td', null, String(r.metrics.mrr)), h('td', null, String(r.metrics.ndcg_at_k)),
        h('td', null, String(r.metrics.latency_ms.mean)),
        h('td', { class: 'small muted' }, `${r.combo.embedding_model}/${r.combo.rerank_model || 'no-rerank'} stages=${r.combo.stages.join(',')} pool=${r.combo.retrieval_pool || 'default'}/${r.combo.rerank_pool || 'default'} rrf_k=${r.combo.rrf_k || 'default'}`)))))));
  }

  /* ---------- the settings of the experiment, by pipeline stage ---------- */

  function renderEffective() {
    if (!refs.effBox) return;
    if (!effective) { fill(refs.effBox, empty('Could not work out the settings in effect.')); return; }
    const over = Object.entries(effective.overrides || {});
    fill(refs.effBox,
      over.length ? h('div', { class: 'notice warn' }, 'Environment variables the dashboard was started with win over an experiment’s own values: ',
        over.map(([k, v], i) => [i ? ' · ' : '', h('code', null, `${k}=${v}`)])) : null,
      (effective.errors || []).map(e => h('div', { class: 'notice bad' }, e)),
      (effective.stages || []).map(s => PL.stageBlock(s, null, null, true)),
      h('p', { class: 'small muted' }, effective.note + '. Hardware settings of the Models section (device, precision, batch sizes, memory limit) follow production, because they describe this computer.'));
  }

  function stageSection(id, title, ...fields) {
    const s = stageMeta(id);
    return h('div', { class: 'stage' + (s.parent ? ' sub' : ''), style: { marginBottom: '12px' } },
      id.startsWith('S') ? h('div', { class: 'stage-head' }, h('b', { class: 'stage-id' }, 'S2–S6'), h('b', null, title)) : PL.stageHead(s, !!s.parent),
      s.what && !id.startsWith('S') ? h('p', { class: 'small muted', style: { margin: '2px 0 6px' } }, s.what) : null,
      ...fields);
  }

  function renderDetail() {
    if (!current || !cfg) { fill(refs.detail, empty('Select or create an experiment above.')); return; }
    refs.cEmb = h('input', { type: 'text', value: cfg.embedding_model, style: { width: '260px' } });
    refs.cRerank = h('input', { type: 'text', value: cfg.rerank_model || '', style: { width: '260px' } });
    refs.cReader = h('input', { type: 'text', value: cfg.reader_model || '', placeholder: PIN_HINT, style: { width: '340px' }, title: 'a Hugging Face id such as mlx-community/Qwen3-VL-4B-Instruct-4bit' });
    refs.cRepair = h('input', { type: 'text', value: cfg.repair_model || '', placeholder: PIN_HINT, style: { width: '340px' } });
    refs.cChunkSize = h('input', { type: 'number', value: cfg.chunk_size, style: { width: '90px' } });
    refs.cChunkOverlap = h('input', { type: 'number', value: cfg.chunk_overlap, style: { width: '90px' } });
    refs.cNoRerank = h('input', { type: 'checkbox', checked: !cfg.rerank });
    const stages = new Set((cfg.stages ? cfg.stages.split(',') : ['bm25', 'dense', 'rerank']).map(s => s.trim()));
    refs.cBm25 = h('input', { type: 'checkbox', checked: stages.has('bm25'), on: { change: () => guardStages(refs.cBm25, refs.cDense) } });
    refs.cDense = h('input', { type: 'checkbox', checked: stages.has('dense'), on: { change: () => guardStages(refs.cBm25, refs.cDense) } });
    refs.cRerankStage = h('input', { type: 'checkbox', checked: stages.has('rerank') });
    refs.cPoolR = h('input', { type: 'number', value: cfg.retrieval_pool || '', placeholder: 'default', style: { width: '90px' } });
    refs.cPoolK = h('input', { type: 'number', value: cfg.rerank_pool || '', placeholder: 'default', style: { width: '90px' } });
    refs.cRrfK = h('input', { type: 'number', value: cfg.rrf_k || '', placeholder: '60', style: { width: '90px' } });
    refs.idxRebuild = h('input', { type: 'checkbox' });
    refs.idxForceMd = h('input', { type: 'checkbox' });
    refs.idxBtn = h('button', { class: 'btn primary', on: { click: () => startRun('index') } }, 'Build index');
    refs.promoteBtn = h('button', { class: 'btn small', on: { click: promote } }, 'Promote to production');
    // one tunablesForm per stage that owns docling tunables (the same widget as the Settings tab), loaded from this experiment
    const byStage = {};
    // a stage whose settings belong to several parts (the router and the lanes of 3.2) gets one form per part
    const partOf = t => {
      const sid = stageOfTunable(t);
      const g = (stageMeta(sid).groups || []).find(x => x.settings.includes('indexer.' + t.key));
      return g ? 'part:' + g.id : sid;
    };
    for (const t of doclingTunables || []) (byStage[partOf(t)] = byStage[partOf(t)] || []).push(t);
    refs.doclingForms = Object.entries(byStage).map(([id, ts]) => {
      const g = id.startsWith('part:') ? (stageMeta('3.2').groups || []).find(x => 'part:' + x.id === id) : null;
      if (g) ts.sort((a, b) => g.settings.indexOf('indexer.' + a.key) - g.settings.indexOf('indexer.' + b.key));
      const f = tunablesForm(ts); f.load(cfg); f.stage = id; return f;
    });
    const formOf = id => (refs.doclingForms.find(f => f.stage === id) || {}).root;
    const laneCards = (extraFor) => (stageMeta('3.2').groups || []).map(g => {
      const lane = /^[\d.]+([a-d])$/.exec(g.id), form = formOf('part:' + g.id), extra = extraFor[g.id] || null;
      return form || extra ? h('div', { class: 'set-group' + (lane ? ' rw-' + lane[1] : '') },
        h('div', { class: 'stage-head sub' }, h('b', { class: 'stage-id' }, g.id), h('b', null, g.name), PL.hw(g.where)),
        h('p', { class: 'small muted', style: { margin: '2px 0 8px' } }, g.what), extra, form || null) : null;
    });

    refs.sQ = h('input', { type: 'search', style: { flex: '1 1 260px' }, on: { keydown: e => { if (e.key === 'Enter') runSearch(); } } });
    refs.sGo = h('button', { class: 'btn primary small', on: { click: runSearch } }, 'Search');
    refs.sK = h('input', { type: 'number', min: 1, max: 25, value: 5, style: { width: '70px' } });
    refs.sBm25 = h('input', { type: 'checkbox', checked: true, on: { change: () => guardStages(refs.sBm25, refs.sDense) } });
    refs.sDense = h('input', { type: 'checkbox', checked: true, on: { change: () => guardStages(refs.sBm25, refs.sDense) } });
    refs.sRerank = h('input', { type: 'checkbox', checked: !!cfg.rerank });
    refs.sPoolR = h('input', { type: 'number', style: { width: '80px' } });
    refs.sPoolK = h('input', { type: 'number', style: { width: '80px' } });
    refs.sRrfK = h('input', { type: 'number', style: { width: '80px' } });
    refs.sOut = h('div', { style: { marginTop: '12px' } }, empty('Results appear here.'));

    refs.bK = h('input', { type: 'number', min: 1, value: 5, style: { width: '70px' } });
    refs.bLabel = h('input', { type: 'text', placeholder: 'baseline', style: { width: '160px' } });
    refs.bGo = h('button', { class: 'btn primary small', on: { click: () => startRun('bench') } }, 'Run benchmark');
    refs.compare = h('div', { style: { marginTop: '10px' } });
    refs.runBox = h('div', { style: { marginTop: '10px' } });
    refs.effBox = h('div');
    refs.srcBox = h('div');
    refs.srcFolder = h('input', { type: 'text', placeholder: '/path/to/sample/documents', style: { width: '340px' } });
    refs.srcName = h('input', { type: 'text', placeholder: 'collection name (optional)', style: { width: '200px' } });

    const home = RS.state.catalog && RS.state.catalog.home;
    refs.runCard = h('div', { class: 'card' },
      h('div', { class: 'card-head' }, h('h4', null, 'Run'), h('span', { class: 'muted small' }, 'same stages, same page-by-page reading as production; nothing is published')),
      h('div', { class: 'row', style: { gap: '12px', flexWrap: 'wrap' } }, refs.idxBtn,
        h('label', { class: 'check' }, refs.idxRebuild, ` re-embed even if unchanged (stage ${PL.id('embed')})`),
        h('label', { class: 'check' }, refs.idxForceMd, ` re-convert to Markdown (stage ${PL.id('convert')})`)),
      h('p', { class: 'small muted', style: { margin: '6px 0 0' } },
        `Stage ${PL.id('fingerprint')} skips a document whose file and settings are unchanged; changing a conversion setting re-converts on its own. The run continues if you leave this tab.`),
      refs.runBox);

    fill(refs.detail,
      h('h3', null, current),
      h('p', { class: 'small muted' }, `bench queries: ${home || '<home>'}/playground/${current}/bench/queries.jsonl.example`),
      h('div', { class: 'card' },
        h('div', { class: 'card-head' }, h('h4', null, 'Source folders'), h('span', { class: 'muted small' }, 'chosen by you, read in place, never copied or changed; each is one collection, as in production')),
        refs.srcBox,
        h('div', { class: 'row', style: { gap: '8px', marginTop: '10px', flexWrap: 'wrap' } }, refs.srcFolder, refs.srcName,
          h('button', { class: 'btn small', disabled: readOnly(), on: { click: addSource } }, 'Add folder'))),
      refs.runCard,
      h('div', { class: 'card' },
        h('div', { class: 'card-head' }, h('h4', null, 'Settings of this experiment, by pipeline stage'), h('span', { class: 'muted small' }, 'pinned to this experiment only; a blank field means the built-in default'),
          h('button', { class: 'btn small', style: { marginLeft: 'auto' }, on: { click: saveConfig } }, 'Save settings')),
        stageSection('3.2', 'Read',
          h('div', { class: 'set-groups' }, laneCards({ '3.2d':
            h('div', { class: 'row', style: { gap: '16px', flexWrap: 'wrap', marginBottom: '8px' } }, h('label', { class: 'field' }, 'Document reader model (lanes c and d)', refs.cReader)) })),
          formOf('3.2')),
        stageSection('3.4', 'Repair',
          h('div', { class: 'row', style: { gap: '16px', flexWrap: 'wrap', marginBottom: '8px' } }, h('label', { class: 'field' }, 'Repair model', refs.cRepair)),
          formOf('3.4')),
        ...Object.keys(byStage).filter(id => !['3.2', '3.4'].includes(id) && !id.startsWith('part:')).map(id => stageSection(id, stageMeta(id).name, formOf(id))),
        stageSection('4', 'Chunk', h('div', { class: 'row', style: { gap: '16px' } }, h('label', { class: 'field' }, 'Chunk size (tokens, estimated)', refs.cChunkSize), h('label', { class: 'field' }, 'Chunk overlap', refs.cChunkOverlap))),
        stageSection('5', 'Embed', h('div', { class: 'row', style: { gap: '16px' } }, h('label', { class: 'field' }, 'Embedding model', refs.cEmb))),
        stageSection('S2', 'Search defaults',
          h('div', { class: 'row', style: { gap: '16px', flexWrap: 'wrap' } },
            h('span', { class: 'muted small' }, 'Stages:'), h('label', { class: 'check' }, refs.cBm25, ' S2 BM25'), h('label', { class: 'check' }, refs.cDense, ' S3 Dense'), h('label', { class: 'check' }, refs.cRerankStage, ' S5 Rerank')),
          h('div', { class: 'row', style: { gap: '16px', marginTop: '8px', flexWrap: 'wrap' } },
            h('label', { class: 'field' }, 'S5 Reranker', refs.cRerank), h('label', { class: 'check' }, refs.cNoRerank, ' no reranker'),
            h('label', { class: 'field' }, 'S2/S3 Retrieval pool', refs.cPoolR), h('label', { class: 'field' }, 'S4 RRF k', refs.cRrfK), h('label', { class: 'field' }, 'S5 Rerank pool', refs.cPoolK))),
        h('div', { class: 'row', style: { marginTop: '12px', gap: '10px' } },
          h('button', { class: 'btn', on: { click: saveConfig } }, 'Save settings'), refs.promoteBtn),
        h('p', { class: 'small muted', style: { marginTop: '6px', marginBottom: 0 } },
          'Promoting writes this experiment’s models, chunk and search settings into production’s config.json (see the ', h('a', { href: '#/settings' }, 'Settings'),
          ' tab); it never touches an index by itself. Conversion settings are not promoted: they are set in production on the Settings tab.'),
        h('details', { style: { marginTop: '12px' }, on: { toggle: () => loadEffective() } }, h('summary', null, 'Settings in effect for the next run, by stage (what the pipeline will really use, and where each value comes from)'),
          h('div', { style: { marginTop: '8px' } }, refs.effBox))),
      h('div', { class: 'card' },
        h('div', { class: 'card-head' }, h('h4', null, 'Search'), h('span', { class: 'muted small' }, 'stages S1–S6 of the search pipeline')),
        h('div', { class: 'row' }, refs.sQ, refs.sGo),
        h('div', { class: 'row', style: { marginTop: '10px', gap: '16px' } },
          h('label', { class: 'field' }, 'Results', refs.sK),
          h('span', { class: 'muted small' }, 'Stages:'), h('label', { class: 'check' }, refs.sBm25, ' S2 BM25'), h('label', { class: 'check' }, refs.sDense, ' S3 Dense'), h('label', { class: 'check' }, refs.sRerank, ' S5 Rerank')),
        h('details', { style: { marginTop: '8px' } }, h('summary', { class: 'small muted' }, 'Advanced: pool sizes & RRF k'),
          h('div', { class: 'row', style: { marginTop: '8px', gap: '16px' } },
            h('label', { class: 'field' }, 'Retrieval pool', refs.sPoolR), h('label', { class: 'field' }, 'Rerank pool', refs.sPoolK), h('label', { class: 'field' }, 'RRF k', refs.sRrfK))),
        refs.sOut),
      h('div', { class: 'card' },
        h('div', { class: 'card-head' }, h('h4', null, 'Benchmark'),
          h('span', { class: 'muted small' }, `labeled queries at <home>/playground/${current}/bench/queries.jsonl`)),
        h('div', { class: 'row', style: { gap: '16px' } },
          h('label', { class: 'field' }, 'k', refs.bK), h('label', { class: 'field' }, 'Label', refs.bLabel), refs.bGo),
        h('p', { class: 'small muted', style: { margin: '6px 0 0' } }, 'The run is shown live in the Run card above, query by query, with the time each search stage took.'),
        h('h4', { style: { marginTop: '16px' } }, 'Every recorded run'),
        refs.compare));
    renderRun();
    loadSources();
  }

  // ── conversion benchmarks: gold sets and stored runs (listing and comparing only; runs are made
  //    with `rag-search bench run`, because they load document readers and take minutes) ──
  let bench = { sets: [], runs: [], engines: [] }, picked = new Set();
  const pct = v => v == null ? '–' : (100 * v).toFixed(1) + '%';

  async function refreshBench() {
    const r = await api('conversion/bench');
    bench = r.ok ? r : { sets: [], runs: [], engines: [] };
    renderBench();
  }

  async function compareBench() {
    const [a, b] = [...picked];
    const runA = bench.runs.find(x => x.id === a), runB = bench.runs.find(x => x.id === b);
    if (!runA || !runB || runA.set !== runB.set) { toast('Pick two runs of the same gold set', '', 2500); return; }
    // older run first: the second is compared against it
    const [first, second] = (runA.started_at <= runB.started_at) ? [runA, runB] : [runB, runA];
    const r = await api('conversion/bench-compare?set=' + encodeURIComponent(first.set) + '&a=' + encodeURIComponent(first.id) + '&b=' + encodeURIComponent(second.id));
    if (!r.ok) { toast(r.error || 'compare failed', 'err', 3000); return; }
    const c = r.result;
    const fmt = (m, v) => v == null ? '–' : (m === 's_per_page' ? v.toFixed(2) : pct(v));
    const cell = (m, x) => h('td', { class: x.better === true ? 'bench-better' : x.better === false ? 'bench-worse' : '' }, `${fmt(m, x.a)} → ${fmt(m, x.b)}`);
    const metrics = ['cell_exact', 'cell_bag', 'cer', 'table_sim', 'balance_ok', 'query_hit', 's_per_page'];
    const names = { cell_exact: 'Numeric cells exact', cell_bag: 'Found anywhere', cer: 'CER', table_sim: 'Table similarity', balance_ok: 'Balance passes', query_hit: 'Phrases found', s_per_page: 's / page' };
    const rows = [['all', c.all], ...Object.entries(c.by_class)];
    fill(refs.benchCmp, h('p', { class: 'small muted' }, `${c.a.id} (${c.a.engine}) → ${c.b.id} (${c.b.engine})`),
      h('div', { class: 'table-wrap' }, h('table', null,
        h('thead', null, h('tr', null, [h('th', null, 'Class'), ...metrics.map(m => h('th', null, names[m]))])),
        h('tbody', null, rows.map(([k, d]) => h('tr', null, h('td', null, k), ...metrics.map(m => cell(m, d[m]))))))),
      c.worse.length ? h('p', { class: 'small muted' }, 'Pages that got worse: ' + c.worse.slice(0, 8).map(w => `${w.file} p.${w.page}`).join(', ')) : null);
  }

  function renderBench() {
    if (!refs.benchBox) return;
    const sets = bench.sets.length ? h('div', { class: 'table-wrap' }, h('table', null,
      h('thead', null, h('tr', null, ['Gold set', 'Pages', 'Verified', 'Classes', 'Runs'].map(t => h('th', null, t)))),
      h('tbody', null, bench.sets.map(s => h('tr', null, h('td', null, h('b', null, s.name)), h('td', null, String(s.pages)),
        h('td', null, `${s.verified} / ${s.pages}`), h('td', { class: 'small muted' }, Object.entries(s.classes).map(([k, v]) => `${k} ${v}`).join(', ')),
        h('td', null, String(s.runs))))))) : empty('No gold sets yet. Create one from the pages you have already converted: rag-search bench gold init NAME');
    const runs = bench.runs.length ? h('div', { class: 'table-wrap' }, h('table', null,
      h('thead', null, h('tr', null, ['', 'Run', 'Set', 'Engine', 'Cells exact', 'CER', 'Balance', 's/page', 'Pages'].map(t => h('th', null, t)))),
      h('tbody', null, bench.runs.map(r => h('tr', null,
        h('td', null, h('input', { type: 'checkbox', checked: picked.has(r.id), on: { change: e => { e.target.checked ? picked.add(r.id) : picked.delete(r.id); if (picked.size > 2) { picked.delete(picked.values().next().value); renderBench(); } refs.benchCmpBtn.disabled = picked.size !== 2; } } })),
        h('td', { class: 'small mono' }, r.id), h('td', null, r.set), h('td', null, r.engine),
        h('td', null, pct(r.summary.cell_exact)), h('td', null, pct(r.summary.cer)), h('td', null, pct(r.summary.balance_ok)),
        h('td', null, r.summary.s_per_page == null ? '–' : String(r.summary.s_per_page)), h('td', null, String(r.summary.pages))))))) : empty('No benchmark runs yet: rag-search bench run SET');
    fill(refs.benchBox, sets, h('h4', { style: { marginTop: '14px' } }, 'Runs'), runs,
      h('div', { class: 'row', style: { marginTop: '8px', gap: '10px' } }, refs.benchCmpBtn,
        h('span', { class: 'small muted' }, 'tick two runs of the same set')),
      refs.benchCmp);
    refs.benchCmpBtn.disabled = picked.size !== 2;
  }

  RS.views.playground = {
    show() { refreshExperiments(); if (current) pollRun(true); },
    tick() { if (current && running()) pollRun(false); },
    init(root) {
      refs.newName = h('input', { type: 'text', placeholder: 'experiment name', style: { width: '200px' } });
      refs.fromProd = h('input', { type: 'checkbox' });
      refs.newFolder = h('input', { type: 'text', placeholder: 'source folder (optional)', style: { width: '260px' } });
      refs.list = h('div', { style: { marginTop: '10px' } });
      refs.detail = h('div', { style: { marginTop: '16px' } }, empty('Select or create an experiment above.'));
      refs.benchBox = h('div', { style: { marginTop: '10px' } });
      refs.benchCmp = h('div', { style: { marginTop: '10px' } });
      refs.benchCmpBtn = h('button', { class: 'btn small', disabled: true, on: { click: compareBench } }, 'Compare');
      root.append(h('h2', null, 'Playground'),
        h('p', { class: 'small muted' }, 'A sandbox for trying different embedding/reranker models and tunables, and for '
          + 'benchmarking them against a small sample of documents -- structurally separate from your real collections. '
          + 'Nothing here is published or served in production, and nothing in production is ever read by it.'),
        h('div', { class: 'card' },
          h('div', { class: 'card-head' }, h('h3', null, 'Experiments')),
          refs.list,
          h('div', { class: 'row', style: { marginTop: '12px', gap: '8px', flexWrap: 'wrap' } }, refs.newName, refs.newFolder,
            h('label', { class: 'check' }, refs.fromProd, ' copy from production'),
            h('button', { class: 'btn primary small', on: { click: createExperiment } }, 'New experiment')),
          h('p', { class: 'small muted', style: { marginTop: '8px', marginBottom: 0 } },
            'A new experiment reads its documents from folders you choose, exactly like a production collection: '
            + 'give one here or add folders after opening it, or from a terminal: rag-search playground create NAME --from FOLDER. '
            + '"Copy from production" seeds the embedding/reranker model, chunk size/overlap and search '
            + 'tunables from today’s production settings instead of this tool’s own defaults.')),
        refs.detail,
        h('div', { class: 'card', style: { marginTop: '16px' } },
          h('div', { class: 'card-head' }, h('h3', null, 'Conversion benchmarks'),
            h('button', { class: 'btn small', on: { click: refreshBench } }, 'Refresh')),
          h('p', { class: 'small muted' }, 'How well pages are read: pages with checked text (a gold set) are read by an engine and '
            + 'scored on numeric table cells, character errors, running-balance checks and search phrases. This list is read-only; '
            + 'create sets and run them with rag-search bench (see Help). Runs never touch your collections.'),
          refs.benchBox));
      refreshExperiments();
      refreshBench();
      // #/playground/NAME (set by selectExperiment) reopens that experiment's detail panel on
      // load/refresh instead of always landing on "select an experiment above".
      const parts = location.hash.replace(/^#\/?/, '').split('/');
      if (parts[0] === 'playground' && parts[1]) selectExperiment(decodeURIComponent(parts[1]));
    },
  };
})();
