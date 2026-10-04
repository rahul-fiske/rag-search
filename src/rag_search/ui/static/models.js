'use strict';
/* Models: which embedding model and reranker are in use, what else is available, download and switch. */
(function () {
  let refs = {}, data = null, loading = false, lastLoad = 0, error = '';
  const FIT = { ok: null, tight: ['tight on memory', 'warn'], too_large: ['too large for the memory budget', 'bad'] };

  const taskActive = () => !!(data && data.task && (data.task.status === 'running' || data.task.status === 'queued'));

  async function load(force) {
    if (loading || (!force && Date.now() - lastLoad < 900)) return;
    loading = true;
    const r = await api('models');
    loading = false; lastLoad = Date.now();
    if (r.ok === false) { error = r.error || 'cannot read the model state'; } else { error = ''; data = r; }
    render();
  }

  function gb(n) { return (Number(n) || 0).toFixed(1) + ' GB'; }
  function fmtEta(s) { return s ? 'roughly ' + dur(s) : ''; }

  /* ---------- actions ---------- */
  function planBody(pl, opts) {
    const rows = [];
    const kind = pl.kind === 'embedding' ? 'embedding model' : 'reranker';
    rows.push(h('p', { style: { marginTop: 0 } }, 'Switch the ', kind, ' from ', h('code', null, pl.current), ' to ', h('code', null, pl.model), pl.custom ? '' : ` (${pl.label}, ${pl.license})`, '.'));
    const ul = h('ul', { class: 'tight' });
    ul.append(h('li', null, pl.cached ? 'Already downloaded.' : (pl.weights_gb ? `Downloads about ${gb(pl.weights_gb)} of weights from Hugging Face.` : 'Downloads the model from Hugging Face (size not known in advance).')));
    if (pl.fit !== 'unknown') ul.append(h('li', null, `Memory: about ${gb(pl.estimate_gb)} for the search daemon together with the other model (budget ${gb(pl.budget_gb)}).`));
    ul.append(h('li', null, 'A quick test loads the model once and checks it ranks a relevant passage first; nothing is changed if it fails.'));
    if (pl.kind === 'embedding') {
      const rx = pl.reindex;
      if (rx && rx.documents) ul.append(h('li', null, h('b', null, `${num(rx.documents)} of ${num(rx.total_documents)} document(s) will be embedded again`), rx.estimated_s ? ', ' + fmtEta(rx.estimated_s) + ' of embedding' : '', '. Conversion is not repeated. Search keeps using the current index and model until all of it is done, then switches at once.'));
      else ul.append(h('li', null, 'No documents are indexed yet, so nothing needs re-embedding.'));
    } else {
      ul.append(h('li', null, 'No re-indexing. The running search daemon loads the new reranker and swaps it in.'));
    }
    for (const w of pl.warnings || []) ul.append(h('li', { class: 'muted' }, w));
    rows.push(ul);
    if (pl.blocking && pl.blocking.length) {
      rows.push(h('div', { class: 'notice bad' }, pl.blocking.join('; ')));
      opts.force = h('input', { type: 'checkbox' });
      rows.push(h('label', { class: 'check', style: { display: 'flex', margin: '8px 0 0' } }, opts.force, 'Try anyway (it may not load, or may make the computer swap heavily)'));
    }
    if (pl.kind === 'embedding' && pl.reindex && pl.reindex.documents) {
      opts.reindex = h('input', { type: 'checkbox', checked: true });
      rows.push(h('label', { class: 'check', style: { display: 'flex', margin: '8px 0 0' } }, opts.reindex, 'Start re-embedding as soon as the model is ready'));
    }
    return h('div', null, rows);
  }

  async function switchTo(kind, model) {
    model = (model || '').trim();
    if (!model) { toast('Enter a Hugging Face model id, for example BAAI/bge-m3', 'bad'); return; }
    const r = await api('models/plan?kind=' + encodeURIComponent(kind) + '&model=' + encodeURIComponent(model));
    if (r.ok === false) { toast(r.error || 'cannot plan this switch', 'bad'); return; }
    const pl = r.plan;
    if (pl.same) { toast('That model is already in use', ''); return; }
    const opts = {};
    const okText = pl.kind === 'embedding' && pl.reindex && pl.reindex.documents ? 'Download, switch and re-embed' : 'Download and switch';
    if (!await confirmDialog('Switch model?', planBody(pl, opts), okText)) return;
    const force = !!(opts.force && opts.force.checked);
    if (pl.blocking && pl.blocking.length && !force) { toast('Not switched: ' + pl.blocking.join('; '), 'bad'); return; }
    const res = await act('models/switch', { kind, model, force, reindex: opts.reindex ? opts.reindex.checked : true });
    if (res.ok !== false) { toast('Started: watch the progress below', 'ok'); load(true); }
  }

  async function download(model) {
    const res = await act('models/download', { models: [model] });
    if (res.ok !== false) { toast('Download started', 'ok'); load(true); }
  }
  async function verifyActive(kind) {
    const res = await act('models/verify', { kind });
    if (res.ok !== false) { toast('Testing the models in use', 'ok'); load(true); }
  }
  async function cancelTask() {
    if (!await confirmDialog('Cancel this task?', 'A partly downloaded model is kept and continues next time. Nothing is switched.', 'Cancel task', true)) return;
    await act('models/cancel', {}, 'Cancelling…');
    setTimeout(() => load(true), 1500);
  }
  async function saveLimit() {
    const v = ($('#models-limit') || { value: '' }).value.trim();
    const gbv = v === '' || /^(off|none)$/i.test(v) ? 0 : Number(v);
    if (isNaN(gbv) || gbv < 0) { toast('Enter a number of GB, or leave it empty for no limit', 'bad'); return; }
    const r = await act('models/limit', { gb: gbv }, gbv ? `Memory limit ${gbv} GB` : 'Memory limit removed');
    if (r.ok !== false) load(true);
  }
  async function reembed() {
    const r = await act('index/start', { mode: 'new', restart: true });
    if (r.ok !== false) { toast('Re-embedding started (see Indexing)', 'ok'); }
  }

  /* ---------- rendering ---------- */
  /* "is it on disk": a copy that is on disk but unusable says why, instead of "not downloaded" */
  function dlChip(r) {
    if (r.partial) return chip('partly downloaded', 'warn');
    if (r.cached) return chip('downloaded', 'ok');
    if (r.downloaded_bytes && r.why_not) return h('span', { class: 'chip warn', title: r.why_not }, 'on disk, not usable');
    return chip('not downloaded');
  }
  /* One chip for the chosen model: "in use" only when it is really usable, otherwise what is missing. */
  const SELECTED = {
    in_use: ['in use', 'accent'],
    selected_download: ['selected · not downloaded yet', 'warn'],
    selected_partial: ['selected · download incomplete', 'warn'],
    selected_runtime: ['selected · runtime not installed', 'warn'],
    selected_platform: ['selected · needs an Apple Silicon Mac', 'bad'],
    selected_off: ['selected · reader switched off', 'warn'],
  };
  function selectedChip(r) {
    if (!r.active) return null;
    const c = SELECTED[r.state] || ['selected', 'accent'];
    return chip(c[0], c[1]);
  }

  function statusChips(r, kind) {
    const c = [];
    if (r.active) c.push(selectedChip(r));
    if (kind === 'embedding' && r.serving) c.push(chip('serving the current index', 'ok'));
    c.push(dlChip(r));
    const f = FIT[r.fit]; if (f) c.push(chip(f[0], f[1]));
    if (r.missing && r.missing.length) c.push(chip('needs newer libraries', 'bad'));
    if (r.custom) c.push(chip('custom'));
    return c;
  }

  function modelTable(kind) {
    const sec = data[kind], ro = readOnly(), busy = taskActive();
    const rows = sec.models.map(r => h('tr', null,
      h('td', { style: { whiteSpace: 'normal', minWidth: '260px' } }, h('b', null, r.label), h('div', { class: 'mono small muted' }, r.id),
        r.note ? h('div', { class: 'small muted' }, r.note) : null,
        r.missing && r.missing.length ? h('div', { class: 'small', style: { color: 'var(--bad)' } }, r.missing.join('; ')) : null),
      h('td', { class: 'num nowrap' }, r.params_m ? num(r.params_m) + 'M' : '–'),
      h('td', { class: 'nowrap' }, r.license),
      h('td', { class: 'num nowrap' }, gb(r.estimate_gb)),
      h('td', null, statusChips(r, kind)),
      h('td', { class: 'nowrap' },
        r.active ? null : h('button', { class: 'btn small primary', disabled: ro || busy, title: 'Download (if needed), test and switch to this model', on: { click: () => switchTo(kind, r.id) } }, 'Use this'), ' ',
        r.cached ? null : h('button', { class: 'btn small', disabled: ro || busy, title: 'Download only; nothing is switched', on: { click: () => download(r.id) } }, r.partial ? 'Resume download' : 'Download'))));
    return h('div', { class: 'table-wrap' }, h('table', null,
      h('thead', null, h('tr', null, ['Model', 'Params', 'Licence', 'Memory*', 'State', ''].map((t, i) => h('th', { class: i === 1 || i === 3 ? 'num' : '' }, t)))),
      h('tbody', null, rows)));
  }

  function reindexNotice() {
    const rx = data.reindex, ws = data.workspace || {};
    if (!rx || !rx.needed) return null;
    const done = (ws.by_model || {})[data.embedding.active] || 0, total = ws.documents || 0;
    const pct = total ? Math.round(100 * done / total) : 0;
    const job = activeJob();
    return h('div', { class: 'notice warn', style: { margin: '10px 0' } },
      h('b', null, 'Re-embedding is not finished. '),
      `${num(done)} of ${num(total)} document(s) are on ${data.embedding.active}; the published index (${data.serving || 'none yet'}) keeps serving searches until all of them are.`,
      h('div', { class: 'bar', style: { margin: '8px 0' } }, h('i', { style: { width: pct + '%' } })),
      job ? h('span', { class: 'small' }, 'An indexing run is active: ', h('a', { href: '#/indexing' }, 'follow it in Indexing'), '.')
        : h('button', { class: 'btn small primary', disabled: readOnly() || taskActive(), on: { click: reembed } }, 'Finish re-embedding now'));
  }

  function embeddingCard() {
    const sec = data.embedding;
    return h('div', { class: 'card' },
      h('div', { class: 'card-head' }, h('h2', null, 'Embedding model'),
        h('span', { class: 'muted small' }, 'turns text into vectors · changing it embeds every document again (conversion is reused)')),
      h('p', { class: 'small muted', style: { margin: '0 0 8px' } }, (sec.models.some(m => m.active && m.state === 'in_use') ? 'In use: ' : 'Selected (not usable yet, see below): '), h('code', null, sec.active), sec.source === 'environment' ? '  (set by $RAG_SEARCH_MODEL, which overrides this page)' : sec.source === 'default' ? '  (default)' : ''),
      reindexNotice(), modelTable('embedding'));
  }

  function rerankerCard() {
    const sec = data.reranker;
    return h('div', { class: 'card', style: { marginTop: '16px' } },
      h('div', { class: 'card-head' }, h('h2', null, 'Reranker'),
        h('span', { class: 'muted small' }, 'orders the best hits · switching is immediate and needs no re-indexing')),
      h('p', { class: 'small muted', style: { margin: '0 0 8px' } }, (sec.models.some(m => m.active && m.state === 'in_use') ? 'In use: ' : 'Selected (not usable yet, see below): '), h('code', null, sec.active), sec.source === 'environment' ? '  (set by $RAG_SEARCH_RERANK_MODEL, which overrides this page)' : sec.source === 'default' ? '  (default)' : ''),
      modelTable('reranker'));
  }

  async function chooseVlm(kind, model) {
    const r = await act('models/' + kind, { model }, kind === 'reader' ? 'Document reader chosen' : 'Repair model chosen');
    if (r.ok !== false) load(true);
  }

  function readerTable(kind) {
    const sec = data.vlm[kind], ro = readOnly(), busy = taskActive();
    const rows = sec.models.map(r => h('tr', null,
      h('td', { style: { whiteSpace: 'normal', minWidth: '260px' } }, h('b', null, r.label), h('div', { class: 'mono small muted' }, r.id),
        r.note ? h('div', { class: 'small muted' }, r.note) : null),
      h('td', { class: 'nowrap' }, r.license),
      h('td', { class: 'num nowrap' }, gb(r.need_gb)),
      h('td', null, [selectedChip(r),
        dlChip(r),
        r.fit === 'tight' ? chip('tight on memory now', 'warn') : null, r.custom ? chip('custom') : null]),
      h('td', { class: 'nowrap' },
        r.active ? null : h('button', { class: 'btn small primary', disabled: ro, title: 'Use this model (nothing is downloaded by this)', on: { click: () => chooseVlm(kind, r.id) } }, 'Use this'), ' ',
        r.cached ? null : h('button', { class: 'btn small', disabled: ro || busy, title: 'Download only', on: { click: () => download(r.id) } }, r.partial ? 'Resume download' : 'Download'))));
    return h('div', { class: 'table-wrap' }, h('table', null,
      h('thead', null, h('tr', null, ['Model', 'Licence', 'Memory needed', 'State', ''].map((t, i) => h('th', { class: i === 2 ? 'num' : '' }, t)))),
      h('tbody', null, rows)));
  }

  async function installRuntime() {
    const rt = data.vlm.runtime;
    const ok = await confirmDialog('Install the document reader runtime?',
      h('div', null,
        h('p', { style: { marginTop: 0 } }, 'This installs ', rt.packages.filter(x => !x.installed).map(x => x.package).join(', ') || 'nothing new', ' into the Python environment rag-search runs in (',
          h('code', null, rt.installer === 'uv' ? 'uv pip install' : 'pip install'), '). It downloads a few hundred MB of Python packages, no model.'),
        h('p', { class: 'small muted', style: { marginBottom: 0 } }, 'Same as ', h('code', null, 'rag-search models runtime install'), ' in a terminal.')), 'Install');
    if (!ok) return;
    const res = await act('models/runtime', {});
    if (res.ok !== false) { toast('Installing: watch the progress at the top of this page', 'ok'); load(true); }
  }

  function checkFix(c) {
    const ro = readOnly(), busy = taskActive();
    if (c.fix === 'install_runtime') return h('button', { class: 'btn small primary', disabled: ro || busy, on: { click: installRuntime } }, 'Install');
    if (c.fix && c.fix.startsWith('download:')) return h('button', { class: 'btn small', disabled: ro || busy, on: { click: () => download(c.fix.slice(9)) } }, 'Download');
    return null;
  }

  function readinessList(v) {
    return h('div', { class: 'table-wrap', style: { margin: '0 0 12px' } }, h('table', null,
      h('tbody', null, v.checks.map(c => h('tr', null,
        h('td', { class: 'nowrap' }, c.ok ? chip('ok', 'ok') : chip(c.required ? 'missing' : 'optional', c.required ? 'bad' : 'warn')),
        h('td', { class: 'nowrap' }, h('b', null, c.label)),
        h('td', { style: { whiteSpace: 'normal' }, class: 'small muted' }, c.detail),
        h('td', { class: 'nowrap' }, c.ok ? null : checkFix(c)))))));
  }

  function readerCard() {
    const v = data.vlm;
    if (!v) return null;
    const rd = v.reading;
    return h('div', { class: 'card', style: { marginTop: '16px' } },
      h('div', { class: 'card-head' }, h('h2', null, 'Document reader'),
        h('span', { class: 'muted small' }, 'a vision model reads scanned pages, pictures in PDFs and image files · docling OCR is the fallback')),
      h('div', { class: 'notice ' + (rd.by === 'reader' ? 'ok' : 'warn'), style: { margin: '0 0 10px' } }, rd.text),
      h('p', { class: 'small muted', style: { margin: '0 0 8px' } },
        'Mode: ', h('b', null, v.mode === 'off' ? 'off' : 'auto'), v.mode === 'off' ? ' (RAG_SEARCH_VLM)' : ' (change it in Settings, stage 3.2 · Read)',
        v.free_gb != null ? ` · ${v.free_gb.toFixed(1)} GB of memory free now` : '',
        ' · it runs in its own process, only while pages are being read, and never downloads anything by itself.'),
      readinessList(v),
      h('details', { class: 'cmd', style: { margin: '0 0 12px' } },
        h('summary', null, h('span', { class: 'name' }, 'Why is the reader installed differently from the other models?')),
        h('div', { class: 'body small' },
          h('p', { style: { marginTop: 0 } }, 'The embedding model and the reranker run on libraries every installation has (PyTorch, sentence-transformers), so they only need their weights, which a download here fetches. The document reader runs on ', h('b', null, 'MLX'), ', Apple’s GPU framework: ', h('code', null, 'mlx-vlm'), ' exists for Apple Silicon only, and ', h('code', null, 'ocrmac'), ' (Apple Vision) only for macOS. They are therefore an optional extra, ', h('code', null, 'rag-search[mac-vlm]'), ', not part of the base install, so that the package still installs on Linux and Intel machines.'),
          h('p', null, 'Two steps, then: install the runtime once (the button above, or ', h('code', null, 'rag-search models runtime install'), '), and download the model like any other (the table below). Without them docling OCR reads scans, which is fine for clean scans and weak for photographs.'),
          h('p', { style: { marginBottom: 0 } }, 'Installed in: ', h('code', null, v.runtime.python), ' with ', h('code', null, v.runtime.installer), '.'))),
      h('p', { class: 'small muted', style: { margin: '0 0 8px' } }, 'Reads pages: ', h('code', null, v.reader.active), v.reader.source === 'environment' ? '  (set by $RAG_SEARCH_VLM_MODEL)' : v.reader.source === 'default' ? '  (default)' : ''),
      readerTable('reader'),
      h('p', { class: 'small muted', style: { margin: '12px 0 8px' } }, 'Re-reads suspect table cells (stage 3.4 Repair): ', h('code', null, v.repair.active)),
      readerTable('repair'));
  }

  function machineCard() {
    const m = data.machine;
    return h('div', { class: 'card' },
      h('div', { class: 'card-head' }, h('h2', null, 'This computer')),
      kv([['Memory', `${m.ram_gb} GB · compute device ${m.device}`], ['Budget for the models', `${gb(data.budget_gb)} ${data.memory_limit_gb ? '(your limit)' : '(60% of the memory)'}`]]),
      h('p', { class: 'small muted', style: { margin: '8px 0 6px' } }, '*Memory = what the search daemon needs with this model and the one in use in the other role (weights, index, about 1.5 GB of overhead). It is an estimate; it is used to warn, and does not cap anything.'),
      h('div', { class: 'row', style: { margin: '0 0 10px' } }, h('button', { class: 'btn small', disabled: readOnly() || taskActive(), title: 'Loads the two models in use in a separate process and checks they rank a relevant passage first', on: { click: () => verifyActive('') } }, 'Test the models in use')),
      h('div', { class: 'row' }, h('label', { class: 'small' }, 'Memory limit for the models (GB) '), h('input', { type: 'text', id: 'models-limit', size: 5, style: { width: '70px' }, placeholder: 'GB', value: data.memory_limit_gb || '' }), h('button', { class: 'btn small', disabled: readOnly(), on: { click: saveLimit } }, 'Save'),
        h('span', { class: 'small muted' }, ' empty = no limit of your own')));
  }

  function presetsCard() {
    const ro = readOnly(), busy = taskActive();
    const names = Object.keys(data.presets);
    return h('div', { class: 'card' },
      h('div', { class: 'card-head' }, h('h2', null, 'Presets')),
      h('p', { class: 'small muted', style: { marginTop: 0 } }, 'Both models at once. A preset changes both, so the embedding part re-embeds the documents.'),
      names.map(n => {
        const p = data.presets[n];
        const active = p.embedding === data.embedding.active && p.reranker === data.reranker.active;
        return h('div', { class: 'row', style: { margin: '4px 0' } },
          h('b', null, n), active ? chip('in use', 'accent') : null,
          h('span', { class: 'small muted' }, `${p.embedding.split('/').pop()} + ${p.reranker.split('/').pop()}`),
          active ? null : h('button', { class: 'btn small', disabled: ro || busy, on: { click: () => usePreset(p) } }, 'Use…'));
      }));
  }

  async function usePreset(p) {
    // reranker first (cheap), then the embedding model; each is confirmed and shown on its own
    if (p.reranker !== data.reranker.active) { await switchTo('reranker', p.reranker); }
    else if (p.embedding !== data.embedding.active) { await switchTo('embedding', p.embedding); }
    if (p.reranker !== data.reranker.active && p.embedding !== data.embedding.active) toast('When this finishes, choose the preset again to switch the embedding model too.', '', 8000);
  }

  function taskCard() {
    const t = data.task;
    if (!t) return null;
    const active = t.status === 'running' || t.status === 'queued';
    if (!active && (Date.now() / 1000 - (t.finished_at || t.updated_at || 0)) > 3600) return null;
    const p = t.progress || {};
    const pct = p.total ? Math.min(100, Math.round(100 * (p.done || 0) / p.total)) : null;
    const cls = t.status === 'failed' ? 'bad' : t.status === 'succeeded' ? 'ok' : t.status === 'cancelled' ? '' : 'warn';
    const title = { switch: 'Switching', download: 'Downloading', verify: 'Testing', runtime: 'Installing the document reader runtime' }[t.op] || t.op;
    const res = t.result || {};
    return h('div', { class: 'card', style: { marginBottom: '16px' } },
      h('div', { class: 'card-head' }, h('h2', null, title + ' ' + (t.model || '')), pill(t.status, cls, active),
        h('div', { class: 'spacer' }, active ? h('button', { class: 'btn small danger', disabled: readOnly(), on: { click: cancelTask } }, 'Cancel') : null)),
      active ? h('p', { class: 'small muted', style: { margin: '0 0 6px' } }, `${t.phase || ''} · running for ${since(t.started_at)}`) : null,
      p.model && (p.done || p.total) ? h('div', null, h('div', { class: 'small' }, `${p.model}: ${bytes(p.done)}${p.total ? ' of ' + bytes(p.total) : ''}${pct !== null ? ' (' + pct + '%)' : ''}`),
        h('div', { class: 'bar', style: { margin: '6px 0 10px' } }, h('i', { style: { width: (pct === null ? 8 : pct) + '%' } }))) : null,
      t.error ? h('div', { class: 'notice bad' }, t.error) : null,
      t.status === 'succeeded' && res.reindex === 'started' ? h('div', { class: 'notice ok' }, `Re-embedding ${num(res.documents)} document(s) has started. Follow it in `, h('a', { href: '#/indexing' }, 'Indexing'), '; search switches to the new model when it is complete.') : null,
      t.status === 'succeeded' && res.reindex === 'not started' ? h('div', { class: 'notice warn' }, 'Saved. Documents are embedded with the new model the next time they are indexed; use “Finish re-embedding now” below.') : null,
      t.status === 'succeeded' && res.applied && !res.reindex ? h('div', { class: 'notice ok' }, res.unchanged ? 'Nothing to change.' : 'Switched.') : null,
      (t.log && t.log.length) ? h('div', { class: 'log', style: { maxHeight: '140px', marginTop: '8px' } }, t.log.slice(-8).map(l => h('div', { class: 'ln' }, l.msg))) : null);
  }

  function render() {
    if (!refs.body) return;
    if (!data) { patch(refs.body, error ? h('div', { class: 'notice bad' }, error) : empty('loading…')); return; }
    patch(refs.body,
      error ? h('div', { class: 'notice bad', style: { marginBottom: '12px' } }, error) : null,
      taskCard(),
      embeddingCard(), rerankerCard(), readerCard(),
      h('div', { class: 'grid g2', style: { marginTop: '16px' } }, presetsCard(), machineCard()));
    refs.customBtn.disabled = readOnly() || taskActive();
  }

  RS.views.models = {
    init(root) {
      refs.body = h('div');
      refs.customKind = h('select', null, h('option', { value: 'embedding' }, 'embedding model'), h('option', { value: 'reranker' }, 'reranker'));
      refs.customId = h('input', { type: 'text', placeholder: 'ORG/NAME, for example BAAI/bge-reranker-base', style: { minWidth: '320px' } });
      refs.customBtn = h('button', { class: 'btn primary', on: { click: () => switchTo(refs.customKind.value, refs.customId.value) } }, 'Check and switch…');
      const custom = h('div', { class: 'card', style: { marginTop: '16px' } },
        h('div', { class: 'card-head' }, h('h2', null, 'Any other Hugging Face model')),
        h('p', { class: 'small muted', style: { marginTop: 0 } }, 'Loaded as a standard sentence-transformers model (embedding) or cross-encoder (reranker). It is downloaded and tested first; nothing changes if the test fails. Models that need special prompts or a different loader are not supported this way.'),
        h('div', { class: 'row' }, refs.customKind, refs.customId, refs.customBtn));
      const advanced = h('p', { class: 'small muted', style: { marginTop: '16px' } },
        'How the models run on this computer (batch sizes, precision, device, longest passage) is set in ',
        h('a', { href: '#/settings' }, 'Settings'), ', next to the stage that uses it: ', h('b', null, '5 · Embed'), ' and ', h('b', null, 'S5 · Rerank'), '.');
      root.append(h('h2', null, 'Models'),
        h('p', { class: 'muted', style: { marginTop: 0 } }, 'Models are downloaded from Hugging Face into its local cache, only when you ask here (or with ', h('code', null, 'rag-search models'), '). Search itself never uses the network. Apart from the two defaults, the models listed are untested on your documents: try one, and switch back if it is not better.'),
        refs.body, custom, advanced);
    },
    show() { load(true); },
    update() {},
    tick() {
      const every = taskActive() ? 2000 : 15000;
      if (Date.now() - lastLoad > every) load(false);
    },
  };
})();
