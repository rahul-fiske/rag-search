'use strict';
/* Indexing: start / cancel / publish, the live run (phases, progress, ETA), per-document timings, history. */
(function () {
  let refs = {};
  // the run's phases, with the numbers of the pipeline stages each one covers (see Architecture)
  const STEPS = [
    { key: 'convert', title: 'Convert', ids: '1–4', note: 'discover, fingerprint, convert, chunk' },
    { key: 'embed', title: 'Embed', ids: '5–6', note: 'vectors from the embedding model, written' },
    { key: 'merge', title: 'Merge', ids: '7', note: 'one index per collection' },
    { key: 'publish', title: 'Publish', ids: '8', note: 'new generation → search daemon' },
  ];

  const EMPTY_DOCS = { total: 0, matched: 0, by_status: {}, by_collection: {}, items: [] };
  const DOC_STATUS_FILTERS = [
    ['', 'All'], ['indexed', 'Indexed'], ['converted', 'Converted'],
    ['skipped', 'Unchanged'], ['no_text', 'No text'], ['error', 'Failed'],
    ['unsupported', 'Skipped — unsupported format'],
  ];
  // Filter/UI state persists across re-renders (the view object itself is a singleton).
  let docFilter = { status: '', collection: '', q: '', branch: '', outcome: '' };
  let estimate = { busy: false, data: null, error: '' };
  let docFetch = { sig: '', data: null, at: 0, loading: false };
  let collapsedGroups = new Set();
  let searchTimer = null;

  function statusPill(st) {
    const cls = st === 'succeeded' ? 'ok' : (st === 'failed' || st === 'error') ? 'bad' : (st === 'running' || st === 'queued') ? 'warn' : '';
    return pill(st, cls, st === 'running');
  }

  async function start(mode) {
    const body = { mode, path: refs.path.value.trim(), rebuild: refs.rebuild.checked, force_md: refs.forceMd.checked, restart: refs.restart.checked };
    if (mode === 'all') {
      const scope = body.path ? `“${body.path}”` : 'every document';
      if (!await confirmDialog('Rebuild from scratch?', `This wipes the index of ${scope} and re-converts and re-embeds it. It can take a long time. Searches keep using the current generation until the new one is published.`, 'Rebuild', true)) return;
      body.confirm = true;
    }
    const r = await act('index/start', body);
    if (r.ok === false) return;
    toast(r.already_running ? 'An indexing run is already active (see below)' : 'Indexing started', r.already_running ? '' : 'ok');
  }
  async function cancel() {
    if (!await confirmDialog('Cancel the running indexing run?', 'Documents that already finished are kept; nothing new is published.', 'Cancel run', true)) return;
    const r = await act('index/cancel', {});
    if (r.ok !== false) toast(r.cancelled ? 'Cancel requested' : (r.note || 'Nothing was running'), 'ok');
  }
  async function publishNow() {
    const r = await act('index/publish', {});
    if (r.ok !== false) toast(r.changed === false ? 'Nothing new to publish' : 'Published' + (r.generation ? ' generation ' + r.generation : ''), 'ok');
  }

  function controls() {
    const ro = readOnly(), job = activeJob();
    refs.btnEstimate.disabled = estimate.busy; refs.btnStart.disabled = ro; refs.btnAll.disabled = ro; refs.btnCancel.disabled = ro || !job; refs.btnPublish.disabled = ro;
  }

  function stepper(job, running) {
    const p = job.progress || {};
    let idx = STEPS.findIndex(s => s.key === p.phase);
    const finished = !running;
    if (p.phase === 'done' || finished) idx = job.status === 'succeeded' || job.status === 'partial' ? STEPS.length : Math.max(idx, 0);
    return h('div', { class: 'steps' }, STEPS.map((s, i) => h('div', { class: 'step ' + (i < idx ? 'done' : i === idx && running ? 'now' : '') },
      (i < idx ? '✓ ' : '') + s.title, h('small', null, `stages ${s.ids} · ${s.note}`))));
  }

  function eta(job) {
    const p = job.progress || {};
    if (!p.total || !(p.done > 0) || !p.phase_started_at || !['convert', 'embed'].includes(p.phase)) return null;
    const t0 = toDate(p.phase_started_at); if (!t0) return null;
    const per = (Date.now() - t0.getTime()) / 1000 / p.done;
    return per * (p.total - p.done);
  }

  // A failed document needs attention: the list is open (up to 5 shown; the filtered list below has the rest).
  function failedNotice(sum) {
    const shown = (sum.errors || []);
    return h('div', { class: 'notice bad', style: { marginTop: '12px' } },
      h('b', null, `${sum.error_count} document(s) failed`),
      h('ul', { class: 'tight' }, shown.slice(0, 5).map(e => h('li', null, h('span', { class: 'mono' }, e.src || ''), e.message ? ': ' + e.message : ''))),
      h('p', { class: 'small', style: { margin: '4px 0 0' } },
        sum.error_count > Math.min(5, shown.length) ? `showing ${Math.min(5, shown.length)} of ${sum.error_count} -- ` : null,
        h('a', { href: '#/indexing', on: { click: e => { e.preventDefault(); jumpToStatus('error'); } } }, 'see every failed document below, already filtered')));
  }

  // Files of a kind the pipeline cannot read are only counted here, by extension; the names are one click away
  // (they are rows of the Documents list below, filtered) and collapsed so a big folder does not push the page down.
  function unsupportedNotice(sum) {
    const listed = sum.unsupported_extension || [];
    const byExt = {};
    for (const e of listed) { const k = e.extension || '(no extension)'; byExt[k] = (byExt[k] || 0) + 1; }
    const exts = Object.entries(byExt).sort((a, b) => b[1] - a[1]);
    return h('details', { class: 'notice warn', style: { marginTop: '12px' } },
      h('summary', { style: { cursor: 'pointer' } }, h('b', null, `${num(sum.unsupported_count)} file(s) skipped — format not supported`),
        h('span', { class: 'small muted' }, exts.length ? '  ' + exts.slice(0, 6).map(([k, n]) => `${k} ×${n}`).join(', ') + (exts.length > 6 ? ', …' : '') + (sum.unsupported_count > listed.length ? ` (of the first ${listed.length} listed)` : '') : '')),
      h('p', { class: 'small', style: { margin: '6px 0' } }, 'Counted, never changed or deleted. Accepted formats are listed under Start. ',
        h('a', { href: '#/indexing', on: { click: e => { e.preventDefault(); jumpToStatus('unsupported'); } } }, 'Show all of them in the Documents list below')),
      h('ul', { class: 'tight', style: { maxHeight: '180px', overflow: 'auto' } }, listed.map(e => h('li', null, h('span', { class: 'mono' }, e.src || ''), ' (', e.extension || 'no extension', ')'))));
  }

  function runCard() {
    const idx = RS.state.live && RS.state.live.index; const job = idx && idx.job;
    if (!job) return h('div', { class: 'card' }, h('div', { class: 'card-head' }, h('h2', null, 'Current run')),
      empty('No indexing run yet. Start one above.'));
    const running = job.status === 'running' || job.status === 'queued';
    const p = job.progress || {};
    const pct = p.total ? Math.min(100, Math.round(100 * (p.done || 0) / p.total)) : (running ? 0 : 100);
    const left = running ? eta(job) : null;
    const docs = idx.documents || { total: 0, by_status: {} };
    const by = docs.by_status || {};
    const sum = job.summary;
    return h('div', { class: 'card' },
      h('div', { class: 'card-head' }, h('h2', null, running ? 'Current run' : 'Last run'), statusPill(job.status),
        h('span', { class: 'muted small mono' }, job.id),
        h('div', { class: 'spacer' }, h('span', { class: 'muted small' }, `${job.mode || ''}${job.path ? ' · ' + job.path : ''}`))),
      stepper(job, running),
      h('div', { class: 'bar ' + (job.status === 'failed' ? 'bad' : !running ? 'ok' : ''), style: { marginTop: '10px' } }, h('i', { style: { width: pct + '%' } })),
      h('div', { class: 'row small muted', style: { marginTop: '6px' } },
        running ? h('span', null, `${p.done || 0} / ${p.total || '?'} document(s) in this phase`) : null,
        p.current ? h('span', null, '· now: ', h('b', { class: 'mono' }, p.current), p.current_since ? ` for ${since(p.current_since)}` : '', p.message ? ` (${p.message})` : '') : null,
        left !== null ? h('span', { style: { marginLeft: 'auto' } }, `≈ ${dur(left)} left in this phase (estimate)`) : null),
      h('div', { class: 'grid g4', style: { marginTop: '14px' } },
        statCard(dur(job.elapsed_s), running ? 'running for' : 'took'),
        statCard(num(by.indexed || 0), 'documents indexed'), by.converted ? statCard(num(by.converted), 'converted, waiting to embed') : null,
        statCard(num(by.skipped || 0), 'unchanged (skipped)'), by.no_text ? statCard(num(by.no_text), 'no text (skipped)') : null,
        statCard(num(by.error || 0), 'failed'), by.unsupported ? statCard(num(by.unsupported), 'unsupported format (skipped)') : null),
      job.error ? h('div', { class: 'notice bad', style: { marginTop: '12px' } }, job.error) : null,
      sum && sum.error_count ? failedNotice(sum) : null,
      sum && sum.unsupported_count ? unsupportedNotice(sum) : null,
      job.publish ? h('p', { class: 'small muted', style: { marginBottom: 0 } }, 'Published: ' + (job.publish.generation ? `generation ${job.publish.generation}` : (job.publish.changed === false ? 'nothing changed' : JSON.stringify(job.publish)))) : null);
  }

  // ---------- Documents: filterable, grouped-by-collection list ----------
  // The search box / collection select / status buttons are built ONCE in init() and mutated in
  // place (never rebuilt inside a patch()'d subtree) -- rebuilding an <input> every tick would
  // reparent it through a detached scratch node and silently drop focus while someone is typing.

  function filterActive() { return !!(docFilter.status || docFilter.collection || docFilter.q || docFilter.branch || docFilter.outcome); }

  function docsSource(idx) {
    if (filterActive()) return (docFetch.data && docFetch.data.documents) || EMPTY_DOCS;
    return (idx && idx.documents) || EMPTY_DOCS;
  }

  function setFilter(patchObj) {
    Object.assign(docFilter, patchObj);
    docFetch.sig = ''; // force a refetch (or an immediate switch back to the live SSE data)
    RS.views.indexing.update();
  }

  function clearFilters() {
    docFilter = { status: '', collection: '', q: '', branch: '', outcome: '' };
    docFetch = { sig: '', data: null, at: 0, loading: false };
    refs.docQ.value = ''; refs.docColl.value = ''; refs.docBranch.value = ''; refs.docOutcome.value = '';
    RS.views.indexing.update();
  }

  // Jumps from the "Last run" card's failed/skipped-doc preview straight to the full,
  // already-filtered list below, instead of leaving people to scroll down and filter themselves.
  function jumpToStatus(status) {
    setFilter({ status });
    refs.docs.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }

  async function maybeFetchDocs(job) {
    if (!filterActive() || !job) { docFetch.sig = ''; return; }
    const running = job.status === 'running' || job.status === 'queued';
    const sig = [job.id, docFilter.status, docFilter.collection, docFilter.q, docFilter.branch, docFilter.outcome].join('|');
    const now = Date.now();
    if (docFetch.loading || (sig === docFetch.sig && !(running && now - docFetch.at > 3000))) return;
    docFetch.sig = sig; docFetch.at = now; docFetch.loading = true;
    const qs = new URLSearchParams({ job_id: job.id, limit: '500' });
    if (docFilter.status) qs.set('status', docFilter.status);
    if (docFilter.collection) qs.set('collection', docFilter.collection);
    if (docFilter.q) qs.set('q', docFilter.q);
    if (docFilter.branch) qs.set('branch', docFilter.branch);
    if (docFilter.outcome) qs.set('outcome', docFilter.outcome);
    let r;
    try { r = await api('conversion/documents?' + qs.toString()); } finally { docFetch.loading = false; }
    if (r && r.ok !== false) { docFetch.data = r; updateDocsCard(); }
  }

  function docRow(d, maxT) {
    const t = d.total_s || 0;
    const seg = (d.status === 'indexed' || d.status === 'converted') && t > 0 ? h('div', { style: { width: Math.max(10, Math.round(100 * t / maxT)) + '%' } }, h('div', { class: 'seg' },
      ['convert', 'chunk', 'embed'].map(k => h('i', { class: k, title: `${PL.label(k)} ${dur(d[k + '_s'])}`, style: { width: (100 * (d[k + '_s'] || 0) / t) + '%' } })))) : null;
    const conv = d.conversion;
    return h('tr', { class: conv && conv.trace ? 'clickable' : '', title: conv && conv.trace ? 'Click for the page-by-page record' : '', on: conv && conv.trace ? { click: () => CV.openSummary(conv) } : {} },
      h('td', null, d.status === 'indexed' ? chip('indexed', 'ok') : d.status === 'converted' ? chip('converted', 'accent') : d.status === 'skipped' ? chip('unchanged') : d.status === 'no_text' ? chip('skipped · no text', 'warn') : d.status === 'unsupported' ? chip('skipped · unsupported format', 'warn') : chip('failed', 'bad')),
      h('td', null, h('span', { class: 'muted' }, d.collection + '/'), d.source, d.status === 'error' && d.message ? h('div', { class: 'small', style: { color: 'var(--bad)' } }, d.message) : d.status === 'no_text' ? h('div', { class: 'small muted' }, d.message || 'no text to index') : d.status === 'unsupported' ? h('div', { class: 'small muted' }, 'extension: ' + (d.extension || '(none)')) : null),
      h('td', { style: { minWidth: '120px' } }, conv && conv.pages ? h('div', null, CV.strip(conv), h('div', { class: 'small muted' }, `${num(conv.pages)} p.` + (conv.outcomes && (conv.outcomes.low || conv.outcomes.error) ? ' · ' + [conv.outcomes.low ? conv.outcomes.low + ' low' : '', conv.outcomes.error ? conv.outcomes.error + ' error' : ''].filter(Boolean).join(', ') : ''))) : ''),
      h('td', { class: 'num' }, d.chunks !== undefined ? num(d.chunks) : ''),
      h('td', { style: { minWidth: '160px' } }, seg, d.status === 'indexed' ? h('div', { class: 'small muted' }, `${PL.label('convert')} ${dur(d.convert_s)} · ${PL.label('chunk')} ${dur(d.chunk_s)} · ${PL.label('embed')} ${dur(d.embed_s)}`, conv && CV.costText(conv.cost) ? ' · ' + CV.costText(conv.cost) : '') : d.status === 'converted' ? h('div', { class: 'small muted' }, `${PL.label('convert')} ${dur(d.convert_s)} · ${PL.label('chunk')} ${dur(d.chunk_s)} · waiting for ${PL.label('embed')}`) : null),
      h('td', { class: 'num nowrap' }, t ? dur(t) : ''), h('td', { class: 'nowrap muted' }, clock(d.finished_at)));
  }

  function docsTable(items) {
    const maxT = Math.max(1, ...items.map(d => d.total_s || 0));
    return h('div', { class: 'table-wrap' }, h('table', null,
      h('thead', null, h('tr', null, ['Result', 'Document', 'Pages · branch', 'Chunks', 'Time split · cost', 'Total', 'Finished'].map((t, i) => h('th', { class: i === 3 || i === 5 ? 'num' : '' }, t)))),
      h('tbody', null, items.map(d => docRow(d, maxT)))));
  }

  function groupBadges(stats) {
    const s = stats || {}; const out = [];
    if (s.indexed) out.push(chip(`${num(s.indexed)} indexed`, 'ok'));
    if (s.converted) out.push(chip(`${num(s.converted)} converted`, 'accent'));
    if (s.skipped) out.push(chip(`${num(s.skipped)} unchanged`));
    if (s.no_text) out.push(chip(`${num(s.no_text)} no text`, 'warn'));
    if (s.error) out.push(chip(`${num(s.error)} failed`, 'bad'));
    if (s.unsupported) out.push(chip(`${num(s.unsupported)} unsupported format`, 'warn'));
    return out;
  }

  function groupByCollection(items) {
    const groups = new Map();
    for (const d of items) {
      const c = d.collection || '';
      if (!groups.has(c)) groups.set(c, []);
      groups.get(c).push(d);
    }
    return groups;
  }

  function docsBody(docs) {
    const items = (docs.items || []).slice().reverse();
    const active = filterActive();
    if (!items.length) return empty(active ? 'No documents match this filter.' : 'Per-document timings appear here while a run is going.');
    const byColl = docs.by_collection || {};
    const collNames = Object.keys(byColl);
    if (docFilter.collection || collNames.length <= 1) return docsTable(items);
    const groups = groupByCollection(items);
    return Array.from(groups.keys()).sort().map(name => {
      const grpItems = groups.get(name);
      const open = !collapsedGroups.has(name);
      return h('details', {
        class: 'cmd', open,
        on: { toggle: e => { if (e.target.open) collapsedGroups.delete(name); else collapsedGroups.add(name); } },
      },
        h('summary', null, h('span', { class: 'name mono' }, name || '(default)'),
          h('span', { class: 'small muted', style: { marginLeft: '8px' } }, `${num(grpItems.length)} shown`),
          h('span', { style: { marginLeft: 'auto' } }, groupBadges(byColl[name]))),
        h('div', { class: 'body' }, docsTable(grpItems)));
    });
  }

  function updateDocsCard() {
    const idx = RS.state.live && RS.state.live.index;
    const docs = docsSource(idx);
    const active = filterActive();
    const shown = Math.min((docs.items || []).length, active ? docs.matched : docs.total);
    refs.docHeadline.textContent = active
      ? `${num(docs.matched)} of ${num(docs.total)} match this filter` + (shown < docs.matched ? ` (showing ${num(shown)})` : '')
      : (docs.total > shown ? `latest ${shown} of ${docs.total}` : `${num(docs.total)} handled`);

    const byStatus = docs.by_status || {};
    fill(refs.docStatusRow, DOC_STATUS_FILTERS.map(([key, label]) => {
      const count = key === '' ? docs.total : (byStatus[key] || 0);
      return h('button', {
        type: 'button', class: docFilter.status === key ? 'on' : '',
        on: { click: () => setFilter({ status: docFilter.status === key ? '' : key }) },
      }, `${label} (${num(count)})`);
    }));

    const names = Object.keys(docs.by_collection || {}).sort();
    const sig = names.join('|');
    if (refs.docColl.dataset.sig !== sig) {
      refs.docColl.dataset.sig = sig;
      const cur = refs.docColl.value;
      fill(refs.docColl, h('option', { value: '' }, 'All collections'),
        names.map(c => h('option', { value: c }, `${c} (${num(Object.values(docs.by_collection[c]).reduce((a, b) => a + b, 0))})`)));
      if (names.includes(cur)) refs.docColl.value = cur;
    }
    syncSelect(refs.docBranch, 'All branches', docs.by_branch, CV.label, docFilter.branch);
    syncSelect(refs.docOutcome, 'All outcomes', docs.by_outcome, k => k.replace('_', ' '), docFilter.outcome);
    refs.docClear.classList.toggle('hidden', !active);

    patch(refs.docsBody, docsBody(docs));
  }

  // a <select> whose options (and counts) follow the data; rebuilt only when they change
  function syncSelect(sel, allText, counts, nameOf, current) {
    const keys = Object.keys(counts || {}).filter(k => counts[k]);
    const sig = keys.map(k => k + ':' + counts[k]).join('|') + '|' + current;
    if (sel.dataset.sig === sig) return;
    sel.dataset.sig = sig;
    fill(sel, h('option', { value: '' }, allText), keys.map(k => h('option', { value: k }, `${nameOf(k)} (${num(counts[k])})`)));
    sel.value = keys.includes(current) ? current : '';
  }

  // ---------- Conversion: what the run did to the pages (live) ----------
  function conversionCard() {
    const live = RS.state.live && RS.state.live.conversion;
    const idx = RS.state.live && RS.state.live.index; const job = idx && idx.job;
    if (!job) return null;
    const totals = (live && live.totals) || {};
    const running = job.status === 'running' || job.status === 'queued';
    const p = job.progress || {};
    return h('div', { class: 'card', style: { marginTop: '16px' } },
      h('div', { class: 'card-head' }, h('h2', null, 'Conversion'), running ? pill('live', 'warn', true) : null,
        h('div', { class: 'spacer' }, h('span', { class: 'muted small' }, 'colour = how the page was read · chips = where it runs'))),
      CV.flow(totals),
      CV.live(live && live.live) ? h('div', { style: { marginTop: '14px' } }, h('h4', null, 'Pages read right now'), CV.live(live.live)) : null,
      totals.pages ? h('div', { style: { marginTop: '14px' } },
        h('h4', null, 'How the finished documents’ pages were read'), CV.bands(totals.branches),
        h('p', { class: 'small muted', style: { margin: '6px 0 0' } }, 'Text-layer pages are read by docling without forced OCR, scanned pages with full-page OCR, repeats come from the page cache; every page then passes the quality gate.')) : null,
      CV.lanes(live && live.lanes) ? h('div', { style: { marginTop: '14px' } }, h('h4', null, 'Workers'), CV.lanes(live.lanes)) : null,
      totals.pages ? h('div', { class: 'grid g4', style: { marginTop: '14px' } }, CV.tiles(totals)) : (running && p.phase === 'convert' ? h('p', { class: 'small muted', style: { marginTop: '12px' } }, 'Page figures appear as the first documents finish.') : null),
      totals.outcomes && Object.keys(totals.outcomes).length ? h('div', { style: { marginTop: '8px' } }, CV.outcomeChips(totals.outcomes)) : null);
  }

  // ---------- Sources: one per collection (the docs folder's sub-folders, registered folders, imports) ----------
  function sourcesView(idx) {
    const src = idx && idx.sources;
    const docsFolder = (src && src.docs_folder) || (idx && idx.docs_folder) || '';
    if (!src) return docsFolder ? h('p', { class: 'small muted', style: { margin: 0 } }, 'Docs folder: ', h('code', null, docsFolder)) : '';
    const dc = src.docs_collections || [], locs = src.locations || [], imp = src.imported || [];
    const n = dc.length + locs.length + imp.length;
    const parts = [dc.length ? `${dc.length} from the docs folder` : '', locs.length ? `${locs.length} registered folder${locs.length > 1 ? 's' : ''}` : '', imp.length ? `${imp.length} imported` : ''].filter(Boolean);
    const rows = [
      ...dc.map(c => [c, h('span', null, h('code', null, docsFolder + '/' + c), h('span', { class: 'muted small' }, '  docs folder'))]),
      ...locs.map(l => [l.collection, h('span', null, h('code', null, l.folder), h('span', { class: 'muted small' }, '  registered folder'))]),
      ...imp.map(c => [c, h('span', { class: 'muted' }, 'imported collection: no source folder, never re-indexed here')]),
    ];
    return h('details', { class: 'cmd' },
      h('summary', null, h('span', { class: 'name' }, n ? `Sources · ${num(n)} collection${n > 1 ? 's' : ''}` : 'Sources · none yet'),
        h('span', { class: 'muted small' }, parts.join(' · ') || 'put documents in the docs folder, or register a folder on the Collections tab')),
      h('div', { class: 'body' },
        h('p', { class: 'small muted', style: { marginTop: 0 } }, 'Each collection has its own source. Sources are only ever read, never changed. The docs folder: ', h('code', null, docsFolder), '. Add or remove sources on the ', h('a', { href: '#/collections' }, 'Collections & access'), ' tab.'),
        rows.length ? h('div', { class: 'table-wrap' }, h('table', null, h('thead', null, h('tr', null, h('th', null, 'Collection'), h('th', null, 'Source'))),
          h('tbody', null, rows.map(([c, where]) => h('tr', null, h('td', { class: 'mono' }, c), h('td', null, where)))))) : null));
  }

  // ---------- the settings the pipeline runs with (read-only; edit them on the Settings tab) ----------
  let plSig = '';
  async function loadPipeline() {
    const data = await PL.load(false);
    if (!data) return;
    const sig = JSON.stringify([data.stages.map(s => s.settings.map(x => [x.value, x.source])), data.overrides, data.errors]);
    if (sig === plSig) return;
    plSig = sig;
    const stages = PL.stages(data, 'indexing');
    const ov = Object.keys((data.overrides && data.overrides.indexer) || {});
    patch(refs.plcard, h('details', { class: 'cmd' },
      h('summary', null, h('span', { class: 'name' }, 'Settings in effect, by pipeline stage'), h('span', { class: 'muted small' }, 'read-only · what the next run uses · change them on the Settings tab')),
      h('div', { class: 'body' },
        h('p', { class: 'small muted', style: { marginTop: 0 } }, 'A value from ', h('b', null, 'config.json'), ' was set on the Settings or Models tab; one from the ', h('b', null, 'environment'), ' was set in the environment the indexer daemon started with and wins over config.json until it is unset. Hover a row to see which function reads it and when it matters. Stage numbers are the ones of the ', h('a', { href: '#/architecture' }, 'Architecture'), ' diagram and of the page traces.'),
        ov.length ? h('div', { class: 'notice warn' }, 'The indexer daemon was started with: ', ov.join(', '), '. Those win over config.json.') : null,
        (data.errors || []).length ? h('div', { class: 'notice bad' }, data.errors.join('; ')) : null,
        stages.map(s => PL.stageBlock(s)),
        h('p', { class: 'small', style: { marginBottom: 0 } }, h('a', { href: '#/settings' }, 'Change these settings')))));
  }

  async function runEstimate() {
    estimate = { busy: true, data: null, error: '' }; RS.views.indexing.update();
    const r = await api('conversion/estimate', { target: refs.path.value.trim() });
    estimate = { busy: false, data: r.ok === false ? null : r.result, error: r.ok === false ? (r.error || 'failed') : '' };
    RS.views.indexing.update();
  }

  function historyCard() {
    const idx = RS.state.live && RS.state.live.index; const hist = (idx && idx.history) || [];
    return h('div', { class: 'card' }, h('div', { class: 'card-head' }, h('h2', null, 'Run history')),
      hist.length ? h('div', { class: 'table-wrap' }, h('table', null,
        h('thead', null, h('tr', null, ['Run', 'Mode', 'Result', 'Started', 'Took', 'Documents'].map(t => h('th', null, t)))),
        h('tbody', null, hist.map(j => h('tr', null, h('td', { class: 'mono' }, j.id), h('td', null, (j.mode || '') + (j.path ? ' · ' + j.path : '')),
          h('td', null, statusPill(j.status)), h('td', { class: 'nowrap muted' }, clock(j.started_at || j.created_at)),
          h('td', { class: 'nowrap' }, j.elapsed_s !== undefined ? dur(j.elapsed_s) : ''),
          h('td', { class: 'muted' }, j.summary ? `${j.summary.conversion && j.summary.conversion.pages ? num(j.summary.conversion.pages) + ' pages, ' : ''}${j.summary.indexed ?? 0} indexed, ${(j.summary.skipped_fresh ?? j.summary.skipped) ?? 0} unchanged, ${j.summary.no_text_count ? j.summary.no_text_count + ' without text, ' : ''}${j.summary.unsupported_count ? j.summary.unsupported_count + ' unsupported format, ' : ''}${j.summary.error_count || 0} failed` : ''))))))
        : empty('No runs yet.'));
  }

  RS.views.indexing = {
    init(root) {
      refs.path = h('input', { type: 'text', placeholder: 'everything (or a collection / file)', list: 'ix-colls', style: { minWidth: '260px' } });
      refs.list = h('datalist', { id: 'ix-colls' });
      refs.rebuild = h('input', { type: 'checkbox' }); refs.forceMd = h('input', { type: 'checkbox' }); refs.restart = h('input', { type: 'checkbox' });
      refs.btnStart = h('button', { class: 'btn primary', on: { click: () => start('new') } }, 'Index new & changed');
      refs.btnAll = h('button', { class: 'btn danger', on: { click: () => start('all') } }, 'Rebuild everything…');
      refs.btnEstimate = h('button', { class: 'btn', title: 'Look at the sources (nothing is converted) and estimate pages per branch and time', on: { click: runEstimate } }, 'Estimate');
      refs.est = h('div');
      refs.btnCancel = h('button', { class: 'btn', on: { click: cancel } }, 'Cancel run');
      refs.btnPublish = h('button', { class: 'btn', title: 'Publish the workspace now (normally automatic after a run)', on: { click: publishNow } }, 'Publish now');
      refs.sources = h('div', { style: { marginTop: '10px' } });
      refs.supportedExt = h('span', { class: 'muted small' });
      refs.plcard = h('div', { style: { marginTop: '16px' } });
      refs.run = h('div'); refs.conv = h('div'); refs.hist = h('div', { style: { marginTop: '16px' } });

      refs.docHeadline = h('span', { class: 'muted small' });
      refs.docQ = h('input', {
        type: 'search', placeholder: 'filter by filename…', style: { minWidth: '220px' },
        on: { input: e => { const v = e.target.value; clearTimeout(searchTimer); searchTimer = setTimeout(() => setFilter({ q: v.trim() }), 300); } },
      });
      refs.docColl = h('select', { on: { change: e => setFilter({ collection: e.target.value }) } }, h('option', { value: '' }, 'All collections'));
      refs.docBranch = h('select', { title: 'documents with at least one page read by this branch', on: { change: e => setFilter({ branch: e.target.value }) } }, h('option', { value: '' }, 'All branches'));
      refs.docOutcome = h('select', { title: 'documents with at least one page with this outcome', on: { change: e => setFilter({ outcome: e.target.value }) } }, h('option', { value: '' }, 'All outcomes'));
      refs.docClear = h('button', { class: 'btn small hidden', on: { click: clearFilters } }, 'Clear filters');
      refs.docStatusRow = h('div', { class: 'subnav', style: { marginTop: '10px', marginBottom: 0 } });
      refs.docsBody = h('div', { style: { marginTop: '12px' } });
      refs.docs = h('div', { style: { marginTop: '16px' } },
        h('div', { class: 'card' },
          h('div', { class: 'card-head' }, h('h2', null, 'Documents'), refs.docHeadline,
            h('div', { class: 'spacer' }, h('span', { class: 'lg convert' }, PL.label('convert')), h('span', { class: 'lg chunk' }, PL.label('chunk')), h('span', { class: 'lg embed' }, PL.label('embed')))),
          h('div', { class: 'row', style: { marginTop: '10px' } }, refs.docQ, refs.docColl, refs.docBranch, refs.docOutcome, refs.docClear),
          refs.docStatusRow, refs.docsBody));

      root.append(h('h2', null, 'Indexing'),
        h('div', { class: 'card' },
          h('div', { class: 'card-head' }, h('h3', null, 'Start'), h('span', { class: 'muted small' }, 'indexes every source of every collection; new documents become searchable when the run finishes'), refs.supportedExt),
          refs.sources,
          h('div', { class: 'row' },
            h('label', { class: 'field' }, 'Limit to (optional)', refs.path, refs.list),
            h('label', { class: 'check' }, refs.rebuild, 're-embed even if unchanged'),
            h('label', { class: 'check' }, refs.forceMd, 're-convert to Markdown'),
            h('label', { class: 'check' }, refs.restart, 'restart an active run')),
          h('div', { class: 'row', style: { marginTop: '12px' } }, refs.btnStart, refs.btnAll, refs.btnEstimate, refs.btnCancel, refs.btnPublish,
            h('span', { class: 'muted small' }, 'Only one run exists at a time. New documents become searchable automatically when it finishes.')),
          refs.est),
        h('div', { style: { marginTop: '16px' } }, refs.run), refs.conv, refs.docs, refs.plcard, refs.hist);
    },
    update() {
      const idx = RS.state.live && RS.state.live.index;
      const job = idx && idx.job;
      patch(refs.sources, sourcesView(idx));
      const supExt = idx && idx.supported_extensions;
      refs.supportedExt.textContent = supExt && supExt.length
        ? 'accepted formats: ' + supExt.map(e => e.replace(/^\./, '')).join(', ') : '';
      loadPipeline();
      const names = ((RS.state.catalog && RS.state.catalog.list && RS.state.catalog.list.collections) || []).map(c => c.collection);
      if (refs.list.dataset.sig !== names.join('|')) { refs.list.dataset.sig = names.join('|'); fill(refs.list, names.map(n => h('option', { value: n }))); }
      controls();
      maybeFetchDocs(job);
      patch(refs.run, runCard());
      patch(refs.conv, conversionCard() || '');
      patch(refs.est, estimate.busy ? h('p', { class: 'small muted', style: { marginTop: '10px' } }, 'Profiling the sources…') : estimate.error ? h('div', { class: 'notice bad', style: { marginTop: '12px' } }, estimate.error) : estimate.data ? CV.estimateView(estimate.data) : '');
      updateDocsCard();
      patch(refs.hist, historyCard());
    },
    tick() { this.update(); },
  };
})();
