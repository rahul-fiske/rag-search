'use strict';
/* System: health check (doctor), where things live, and the daemon logs. */
(function () {
  let refs = {}, doctor = null, doctorBusy = false, arch = null, pageCache = null, logKind = 'search', logTimer = null, logAuto = true;

  async function runDoctor() {
    if (doctorBusy) return;
    doctorBusy = true; renderDoctor();
    const r = await api('doctor', {});
    doctorBusy = false;
    doctor = r.ok === false ? { error: r.error || 'doctor failed' } : r;
    doctor.at = Date.now() / 1000;
    renderDoctor();
  }

  function renderDoctor() {
    const ro = readOnly(); let body;
    if (doctorBusy) body = h('p', { class: 'muted' }, pill('checking', 'warn', true), ' Running the checks (this can take a few seconds)…');
    else if (!doctor) body = h('p', { class: 'muted' }, 'Checks the installation: Python and libraries, models on disk, folders, daemons, index, access rules. Nothing is changed.');
    else if (doctor.error) body = h('div', { class: 'notice bad' }, doctor.error);
    else body = h('div', null,
      h('p', { class: 'small muted', style: { margin: '0 0 8px' } }, `${doctor.failed ? 'Some checks failed.' : 'All required checks passed.'} Ran ${ago(doctor.at)}.`),
      h('div', { class: 'table-wrap' }, h('table', null, h('tbody', null, (doctor.rows || []).map(r => h('tr', null,
        h('td', { style: { width: '70px' } }, pill(r.status, { ok: 'ok', fail: 'bad', warn: 'warn' }[String(r.status).toLowerCase()] || '')),
        h('td', { class: 'nowrap' }, r.check), h('td', { class: 'muted', style: { whiteSpace: 'normal', overflowWrap: 'anywhere', minWidth: '260px' } }, r.detail || '')))))));
    fill(refs.doctor, h('div', { class: 'card-head' }, h('h2', null, 'Health check'),
      h('div', { class: 'spacer' }, h('button', { class: 'btn small', disabled: doctorBusy, on: { click: runDoctor } }, doctor ? 'Run again' : 'Run checks'))), body);
  }

  function renderPaths() {
    const p = (arch && arch.paths) || {}; const cat = RS.state.catalog || {};
    const rows = [['Home (indexes, logs, sockets)', h('code', null, p.home || cat.home || '–')]];
    if (arch) {
      rows.push(['Version', `rag-search ${arch.version} on Python ${arch.python} (${arch.platform})`]);
      rows.push(['Embedding model', h('code', null, arch.models.embedding.name)]);
      rows.push(['Reranker', h('code', null, arch.models.reranker.name), arch.models.reranker.enabled ? '' : ' (switched off)']);
    }
    if (pageCache) rows.push(['Page cache', `${num(pageCache.entries)} page(s), ${bytes(pageCache.bytes)} — pages already read, kept so an interrupted run resumes and no page is read twice (rag-search index cache --clear empties it)`]);
    fill(refs.paths, h('div', { class: 'card-head' }, h('h2', null, 'Installation')), kv(rows),
      h('p', { class: 'small muted', style: { marginBottom: 0 } }, 'Change the home folder with ', h('code', null, 'RAG_SEARCH_HOME'), ' or ', h('code', null, '--home'), '. Source folders are registered per collection on the Collections tab.'));
  }

  /* Memory of the search daemon, and what each collection takes of it. */
  function renderMemory() {
    const d = (((RS.state.live || {}).daemons || {}).search) || {}; const m = d.memory || {};
    const colls = Object.entries(m.collections || {}).filter(([, e]) => e.ram_bytes !== undefined).sort((a, b) => b[1].ram_bytes - a[1].ram_bytes);
    let body;
    if (!d.state || !m.rss_bytes) body = h('p', { class: 'muted' }, 'The search daemon is not running, so nothing is held in memory (it starts on demand).');
    else {
      const models = Object.entries(m.models_bytes || {}).map(([k, v]) => `${k} ${bytes(v)}`).join(', ');
      const sum = k => colls.reduce((n, [, e]) => n + (e[k] || 0), 0);
      body = h('div', null,
        kv([['Search daemon', `${bytes(m.rss_bytes)} resident (peak ${bytes(m.peak_rss_bytes)})`],
          ['Models', models || null],
          ['Indexes, all collections', `~${bytes(m.indexes_bytes !== undefined ? m.indexes_bytes : sum('ram_bytes'))} (estimated)`]]),
        colls.length ? h('div', { class: 'table-wrap' }, h('table', { class: 'plain cost' },
          h('thead', null, h('tr', null, ['Collection', 'Chunks', 'Vectors', 'Passages', 'Keyword index', 'Total'].map((t, n) => h('th', { class: n ? 'num' : '' }, t)))),
          h('tbody', null, colls.map(([n, e]) => h('tr', null, h('td', null, n), h('td', { class: 'num' }, num(e.chunks)), h('td', { class: 'num' }, bytes(e.embeddings_bytes)),
            h('td', { class: 'num' }, '~' + bytes(e.passages_bytes)), h('td', { class: 'num' }, '~' + bytes(e.keyword_bytes)), h('td', { class: 'num' }, h('b', null, '~' + bytes(e.ram_bytes))))),
            h('tr', { class: 'total' }, h('td', null, h('b', null, 'All collections')), h('td', { class: 'num' }, num(colls.reduce((n, [, e]) => n + e.chunks, 0))),
              h('td', { class: 'num' }, bytes(sum('embeddings_bytes'))), h('td', { class: 'num' }, '~' + bytes(sum('passages_bytes'))), h('td', { class: 'num' }, '~' + bytes(sum('keyword_bytes'))),
              h('td', { class: 'num' }, h('b', null, '~' + bytes(sum('ram_bytes')))))))) : null,
        h('p', { class: 'small muted', style: { marginBottom: 0 } }, 'Vectors are exact; passages (Python objects) and the keyword index (built in memory from the passages, never stored) are estimates. The daemon\u2019s resident figure is what the operating system reports.'));
    }
    fill(refs.memory, h('div', { class: 'card-head' }, h('h2', null, 'Memory'), h('span', { class: 'muted small' }, 'what the search daemon holds, per collection')), body);
  }

  async function loadLogs() {
    if (RS.current !== 'system') return;
    const r = await api('logs?kind=' + logKind + '&lines=300');
    if (r.ok === false) { fill(refs.log, r.error || 'cannot read the log'); return; }
    const box = refs.log; const stick = box.scrollTop + box.clientHeight >= box.scrollHeight - 30;
    const lines = r.lines && r.lines.length ? r.lines : ['(the log is empty)'];
    const text = lines.join('\n');
    if (box.dataset.text !== text) {
      box.dataset.text = text;
      const cls = l => /\b(FAILED|ERROR|Traceback)\b|WARNING/.test(l) ? 'ln-bad' : /\bINDEXED\b|finished:/.test(l) ? 'ln-ok' : /\b(convert|chunk|embed)\s{1,4}(started|done|\d)/.test(l) ? 'ln-run' : '';
      box.replaceChildren(...lines.map(l => h('div', { class: 'ln ' + cls(l) }, l)));
      if (stick) box.scrollTop = box.scrollHeight;
    }
    refs.logfile.textContent = r.file || '';
  }
  function setKind(k) {
    logKind = k; refs.log.dataset.text = ''; refs.log.textContent = 'loading…';
    for (const b of $$('button[data-k]', refs.kinds)) b.classList.toggle('on', b.dataset.k === k);
    loadLogs().then(() => { refs.log.scrollTop = refs.log.scrollHeight; });
  }

  RS.views.system = {
    init(root) {
      refs.doctor = h('div', { class: 'card' }); refs.paths = h('div', { class: 'card' }); refs.memory = h('div', { class: 'card', style: { marginTop: '16px' } });
      refs.log = h('div', { class: 'log', tabindex: 0, role: 'log' }, 'loading…');
      refs.logfile = h('span', { class: 'muted small mono' });
      refs.kinds = h('div', { class: 'subnav', style: { margin: 0 } },
        [['search', 'Search daemon'], ['indexer', 'Indexer daemon (files & pipeline)'], ['ui', 'Dashboard']].map(([k, t]) =>
          h('button', { type: 'button', 'data-k': k, class: k === logKind ? 'on' : '', on: { click: () => setKind(k) } }, t)));
      const auto = h('input', { type: 'checkbox', checked: true, on: { change: e => { logAuto = e.target.checked; } } });
      refs.logcard = h('div', { class: 'card', style: { marginTop: '16px' } },
        h('div', { class: 'card-head' }, h('h2', null, 'Logs'), refs.kinds,
          h('div', { class: 'spacer row' }, h('label', { class: 'check' }, auto, 'follow'), h('button', { class: 'btn small', on: { click: loadLogs } }, 'Refresh'))),
        refs.log, h('div', { style: { marginTop: '6px' } }, refs.logfile));
      root.append(h('h2', null, 'System'), h('div', { class: 'grid g2' }, refs.doctor, refs.paths), refs.memory, refs.logcard);
      renderDoctor(); renderPaths(); renderMemory();
      api('architecture').then(a => { if (a.ok !== false) { arch = a; renderPaths(); } });
      api('conversion/page-cache').then(r => { if (r.ok !== false) { pageCache = r.result; renderPaths(); } });
    },
    show() { loadLogs(); clearInterval(logTimer); logTimer = setInterval(() => { if (logAuto) loadLogs(); }, 3000); },
    leave() { clearInterval(logTimer); logTimer = null; },
    update() { renderMemory(); },
    refreshSoon() { setTimeout(() => { if (RS.current === 'system') loadLogs(); }, 1200); },
  };
})();
