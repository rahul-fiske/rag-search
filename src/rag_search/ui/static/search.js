'use strict';
/* Search playground: semantic search and regex grep, optionally "as" another client, with a
   latency breakdown, per-hit stage scores, and pipeline stage/pool-size overrides for
   troubleshooting the BM25 -> dense -> RRF -> rerank pipeline. */
(function () {
  let refs = {}, mode = 'search', busy = false, arch = null, cfgDefaults = null;

  function poolDefaults(topK) {
    return (arch && arch.pools && arch.pools[String(Math.max(1, Math.min(25, topK || 5)))]) || null;
  }

  // Placeholders show the *current production default* for each field: the configured
  // value from Settings when one is set, falling back to the built-in formula/constant
  // otherwise -- never a value that actually gets sent unless the user types into the field.
  function refreshAdvancedPlaceholders() {
    const kVal = parseInt(refs.k.value, 10);
    const topK = kVal > 0 ? kVal : (cfgDefaults && cfgDefaults.top_k) || 5;
    const d = poolDefaults(topK);
    const cfgRP = cfgDefaults && cfgDefaults.retrieval_pool;
    const cfgKP = cfgDefaults && cfgDefaults.rerank_pool;
    const cfgRK = cfgDefaults && cfgDefaults.rrf_k;
    refs.poolRetrieval.placeholder = cfgRP ? String(cfgRP) : (d ? String(d.per_retriever) : 'default');
    refs.poolRerank.placeholder = cfgKP ? String(cfgKP) : (d ? String(d.to_reranker) : 'default');
    refs.rrfK.placeholder = cfgRK ? String(cfgRK) : (arch && arch.fusion ? String(arch.fusion.k) : '60');
    if (arch && arch.pool_overrides) {
      refs.poolRetrieval.max = arch.pool_overrides.retrieval_pool_max;
      refs.poolRerank.max = arch.pool_overrides.rerank_pool_max;
    }
    if (arch && arch.fusion) { refs.rrfK.min = arch.fusion.k_min; refs.rrfK.max = arch.fusion.k_max; }
    if (mode === 'search') refs.k.placeholder = (cfgDefaults && cfgDefaults.top_k) ? String(cfgDefaults.top_k) : '5';
  }

  async function loadDefaults() {
    const [a, c] = await Promise.all([api('architecture'), api('config')]);
    if (a && a.ok !== false) arch = a;
    if (c && c.ok !== false) cfgDefaults = (c.values && c.values.search) || null;
    refreshAdvancedPlaceholders();
  }

  function guardStages() {
    if (!refs.stBm25.checked && !refs.stDense.checked) {
      refs.stBm25.checked = true;
      toast('At least one of BM25 or Dense must stay on', '', 1800);
    }
  }

  function visibleCollections(client) {
    const a = RS.state.catalog && RS.state.catalog.access; if (!a) return [];
    if (client === 'cli') return a.collections.filter(c => c.indexed).map(c => c.collection);
    const who = a.clients.find(c => c.client === client);
    const usable = who ? who.can_use : a.collections.map(c => c.collection);
    return a.collections.filter(c => c.indexed && usable.includes(c.collection)).map(c => c.collection);
  }

  function highlight(text, terms) {
    if (!terms.length) return [text];
    const rx = new RegExp('(' + terms.map(t => t.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')).join('|') + ')', 'gi');
    return text.split(rx).map((part, i) => i % 2 ? h('mark', null, part) : part);
  }
  const termsOf = q => Array.from(new Set(q.toLowerCase().match(/[\p{L}\p{N}][\p{L}\p{N}._-]{2,}/gu) || [])).slice(0, 12);

  function waterfall(t) {
    if (!t) return null;
    const embed = t.embed_query_ms || 0, kw = t.keyword_ms || 0, dn = t.dense_ms || 0, rr = t.rerank_ms || 0;
    const fuse = Math.max(0, (t.retrieve_ms || 0) - embed - kw - dn);
    const parts = [['embed', 'embed query', embed], ['keyword', 'BM25', kw], ['dense', 'vector scan', dn], ['fuse', 'fusion', fuse], ['rerank', 'rerank', rr]];
    const sum = Math.max(1, parts.reduce((a, p) => a + p[2], 0));
    const stages = (t.stages || []).join('+') || '?';
    const gapPart = (label, key) => t[key + '_gap'] != null ? `${label} gap ${t[key + '_gap']} (top1 ${t[key + '_top1']})` : null;
    const diag = [
      `stages ${stages}`,
      t.retrieval_pool != null ? `pool ${t.retrieval_pool}→${t.rerank_pool}` : null,
      t.rrf_k != null ? `rrf k=${t.rrf_k}` : null,
      t.bm25_candidates != null ? `BM25 ${t.bm25_candidates}/Dense ${t.dense_candidates} (overlap ${t.overlap_count || 0}, BM25-only ${t.bm25_only_count || 0}, Dense-only ${t.dense_only_count || 0})` : null,
      gapPart('BM25', 'bm25'), gapPart('Dense', 'dense'), gapPart('Rerank', 'rerank'),
      t.rerank_error ? `rerank: ${t.rerank_error}` : null,
    ].filter(Boolean);
    return h('div', { class: 'card', style: { marginBottom: '12px' } },
      h('div', { class: 'card-head' }, h('h3', null, 'Where the time went'), h('span', { class: 'muted small' }, `total ${ms(t.total_ms)} in the engine`
        + (t.queue_ms ? ` · waited ${ms(t.queue_ms)}` : '') + (t.round_trip_ms !== undefined ? ` · round trip ${ms(t.round_trip_ms)}` : '')
        + ` · generation ${t.generation} · ${t.collections} collection(s) · ${t.candidates || 0} candidates${t.reranked ? ' reranked' : ', no rerank'}`)),
      h('div', { class: 'seg', style: { height: '16px' } }, parts.map(p => h('i', { class: p[0], title: `${p[1]} ${p[2]} ms`, style: { width: (100 * p[2] / sum) + '%' } }))),
      h('div', { style: { marginTop: '8px' } }, parts.map(p => h('span', { class: 'lg ' + p[0] }, `${p[1]} ${p[2]} ms`))),
      diag.length ? h('div', { class: 'muted small', style: { marginTop: '8px' } }, diag.join(' · ')) : null);
  }

  function scoreChip(label, rank, score) {
    return h('span', { class: 'muted small' }, score == null ? `${label} —` : `${label} #${rank} (${score})`);
  }

  function stageBreakdown(x) {
    const parts = [scoreChip('BM25', x.bm25_rank, x.bm25_score), scoreChip('Dense', x.dense_rank, x.dense_score)];
    if (x.rrf_score != null) parts.push(h('span', { class: 'muted small' }, `RRF ${x.rrf_score}`));
    parts.push(h('span', { class: 'muted small' }, x.rerank_score != null ? `Rerank ${x.rerank_score}` : 'Rerank —'));
    return h('div', { class: 'row', style: { gap: '12px', marginTop: '2px' } }, parts);
  }

  function renderSearch(res, q) {
    const r = res.result || {}; const hits = r.results || []; const terms = termsOf(q);
    const out = [waterfall(r.timing)];
    if (r.note) out.push(h('div', { class: 'notice', style: { marginBottom: '10px' } }, r.note));
    if (!hits.length && !r.note) out.push(empty('No results.'));
    for (const x of hits) {
      out.push(h('div', { class: 'hit' },
        h('div', { class: 'hit-head' },
          h('span', { class: 'rank' }, '#' + x.rank), h('b', null, x.file), h('span', null, 'p.' + x.page),
          x.heading ? h('span', { class: 'muted' }, '· ' + x.heading) : null, chip(x.collection),
          x.confidence === 'low' ? chip('low-confidence page', 'warn') : null,
          h('span', { class: 'muted small', style: { marginLeft: 'auto' }, title: x.rrf_score != null ? `RRF fusion score ${x.rrf_score}` : 'single-stage score (no fusion)' }, 'score ' + x.score, ' ',
            h('span', { class: 'bar score' }, h('i', { style: { width: Math.round(Math.max(0, Math.min(1, x.score)) * 100) + '%' } }))),
          h('button', { class: 'btn small', on: { click: () => { navigator.clipboard && navigator.clipboard.writeText(`${x.source || x.file}, p.${x.page}`); toast('Citation copied', 'ok', 1500); } } }, 'Copy citation')),
        stageBreakdown(x),
        h('div', { class: 'hit-text' }, highlight(x.text, terms))));
    }
    return out;
  }

  function renderGrep(res, q) {
    const r = res.result || {}; const m = r.matches || [];
    if (r.error) return [h('div', { class: 'notice bad' }, r.error)];
    const t = r.timing || {};
    const out = [h('div', { class: 'notice', style: { marginBottom: '10px' } }, `${r.total_matches} match(es)${r.truncated ? ' (truncated)' : ''}${r.timed_out ? ' (time budget reached)' : ''} · scanned ${t.files_scanned || 0} file(s) in ${ms(t.scan_ms)}${t.round_trip_ms !== undefined ? ' · round trip ' + ms(t.round_trip_ms) : ''}`)];
    if (!m.length) out.push(empty('No matches.'));
    let rx = null; try { rx = new RegExp('(' + q + ')', 'gi'); } catch (e) { /* invalid in JS flavour: no highlight */ }
    for (const x of m) {
      out.push(h('div', { class: 'hit' },
        h('div', { class: 'hit-head' }, h('b', null, x.doc), h('span', null, 'p.' + x.page), h('span', { class: 'muted' }, 'line ' + x.line), chip(x.collection)),
        h('div', { class: 'hit-text mono' }, (x.context && x.context.length ? x.context.join('\n') : x.text).split('\n').map((line, i) => [i ? '\n' : '', rx && line.length < 2000 ? line.split(rx).map((p, j) => j % 2 ? h('mark', null, p) : p) : line]))));
    }
    return out;
  }

  async function run() {
    const q = refs.q.value.trim(); if (!q || busy) return;
    busy = true; refs.go.disabled = true; refs.go.textContent = 'Searching…';
    fill(refs.out, h('div', { class: 'empty' }, mode === 'search' ? 'Searching (the first search after a start loads the models and can take a while)…' : 'Scanning…'));
    const colls = $$('input[name=coll]:checked', refs.colls).map(i => i.value);
    const body = { client: refs.as.value, collections: colls };
    let res;
    if (mode === 'search') {
      const stages = [refs.stBm25.checked && 'bm25', refs.stDense.checked && 'dense', refs.stRerank.checked && 'rerank'].filter(Boolean);
      const extra = {};
      const rp = parseInt(refs.poolRetrieval.value, 10); if (rp > 0) extra.retrieval_pool = rp;
      const kp = parseInt(refs.poolRerank.value, 10); if (kp > 0) extra.rerank_pool = kp;
      const rk = parseInt(refs.rrfK.value, 10); if (rk > 0) extra.rrf_k = rk;
      // A blank Results field omits top_k entirely, so the daemon falls back to the
      // production default in Settings -- it does NOT get silently pinned to 5.
      const kVal = parseInt(refs.k.value, 10); if (kVal > 0) extra.top_k = Math.max(1, Math.min(25, kVal));
      res = await api('search', { ...body, query: q, stages, ...extra });
    }
    else {
      const mVal = parseInt(refs.k.value, 10);
      res = await api('grep', { ...body, pattern: q, context_lines: 2, max_matches: mVal > 0 ? Math.max(1, Math.min(100, mVal)) : 20 });
    }
    busy = false; refs.go.disabled = false; refs.go.textContent = mode === 'search' ? 'Search' : 'Grep';
    if (!res.ok) {
      const warming = res.code === 'warming_up';
      fill(refs.out, h('div', { class: 'notice ' + (warming ? 'warn' : 'bad') }, warming ? 'The search engine is still loading its models. Try again in a moment.' : (res.error || 'search failed')));
      return;
    }
    fill(refs.out, mode === 'search' ? renderSearch(res, q) : renderGrep(res, q));
  }

  function setMode(m) {
    mode = m;
    for (const b of $$('button[data-mode]', refs.modes)) b.classList.toggle('on', b.dataset.mode === m);
    refs.go.textContent = m === 'search' ? 'Search' : 'Grep';
    refs.q.placeholder = m === 'search' ? 'Ask a question or type keywords…' : 'Regular expression, e.g. error code \\d+';
    refs.kLabel.firstChild.nodeValue = m === 'search' ? 'Results' : 'Max matches';
    refs.k.value = ''; refs.k.max = m === 'search' ? 25 : 100;
    refs.k.placeholder = m === 'search' ? '5' : '20';
    refs.stagesRow.style.display = m === 'search' ? '' : 'none';
    refs.advanced.style.display = m === 'search' ? '' : 'none';
    if (m === 'search') refreshAdvancedPlaceholders();
  }

  RS.views.search = {
    init(root) {
      refs.q = h('input', { type: 'search', style: { flex: '1 1 320px', fontSize: '15px', padding: '9px 12px' }, on: { keydown: e => { if (e.key === 'Enter') run(); } } });
      refs.go = h('button', { class: 'btn primary', style: { padding: '9px 20px' }, on: { click: run } }, 'Search');
      refs.as = h('select', { on: { change: () => this.update() } }, h('option', { value: 'cli' }, 'cli (administrator: everything)'));
      refs.k = h('input', { type: 'number', min: 1, max: 25, placeholder: '5', style: { width: '80px' }, on: { input: refreshAdvancedPlaceholders } });
      refs.kLabel = h('label', { class: 'field' }, 'Results', refs.k);
      refs.colls = h('div', { class: 'row' });
      refs.modes = h('div', { class: 'subnav', style: { margin: 0 } },
        h('button', { type: 'button', 'data-mode': 'search', class: 'on', on: { click: () => setMode('search') } }, 'Semantic search'),
        h('button', { type: 'button', 'data-mode': 'grep', on: { click: () => setMode('grep') } }, 'Regex grep'));
      refs.stBm25 = h('input', { type: 'checkbox', checked: true, on: { change: guardStages } });
      refs.stDense = h('input', { type: 'checkbox', checked: true, on: { change: guardStages } });
      refs.stRerank = h('input', { type: 'checkbox', checked: true });
      refs.stagesRow = h('div', { class: 'row', style: { gap: '16px' } },
        h('span', { class: 'muted small' }, 'Pipeline stages:'),
        h('label', { class: 'check' }, refs.stBm25, ' BM25'),
        h('label', { class: 'check' }, refs.stDense, ' Dense'),
        h('label', { class: 'check' }, refs.stRerank, ' Rerank'));
      refs.poolRetrieval = h('input', { type: 'number', min: 1, style: { width: '90px' } });
      refs.poolRerank = h('input', { type: 'number', min: 1, style: { width: '90px' } });
      refs.rrfK = h('input', { type: 'number', min: 1, style: { width: '90px' } });
      refs.advanced = h('details', { style: { marginTop: '10px' } },
        h('summary', { class: 'small muted', style: { cursor: 'pointer' } }, 'Advanced: pool sizes & RRF k (blank = current default)'),
        h('div', { class: 'row', style: { marginTop: '8px', gap: '16px', flexWrap: 'wrap' } },
          h('label', { class: 'field' }, 'Retrieval pool (per retriever)', refs.poolRetrieval),
          h('label', { class: 'field' }, 'Rerank pool', refs.poolRerank),
          h('label', { class: 'field' }, 'RRF k', refs.rrfK)),
        h('p', { class: 'small muted', style: { marginTop: '6px', marginBottom: 0 } },
          'Placeholders show the production default from ', h('a', { href: '#/settings' }, 'Settings'),
          '. Type a value to override it for this one search only -- it isn’t saved. See the ',
          h('a', { href: '#/architecture' }, 'Architecture'), ' tab for what each of these means.'));
      refs.out = h('div', { style: { marginTop: '16px' } }, h('div', { class: 'empty' }, 'Results appear here. Semantic search finds passages by meaning; grep finds exact text.'));
      root.append(h('h2', null, 'Search'),
        h('div', { class: 'card' },
          h('div', { class: 'row' }, refs.modes), h('div', { class: 'row', style: { marginTop: '12px' } }, refs.q, refs.go),
          h('div', { class: 'row', style: { marginTop: '12px', alignItems: 'flex-end' } },
            h('label', { class: 'field' }, 'View as client', refs.as), refs.kLabel,
            h('div', { class: 'field' }, 'Collections (none ticked = all it may use)', refs.colls)),
          h('div', { style: { marginTop: '12px' } }, refs.stagesRow),
          refs.advanced,
          h('p', { class: 'small muted', style: { marginBottom: 0, marginTop: '10px' } }, 'Searching as a host shows exactly what that host would get. These test searches are not counted as that host being connected.')),
        refs.out);
      loadDefaults();
    },
    show() { loadDefaults(); },
    update() {
      const a = RS.state.catalog && RS.state.catalog.access; if (!a) return;
      const names = a.clients.map(c => c.client);
      if (refs.as.dataset.sig !== names.join('|')) {
        const cur = refs.as.value; refs.as.dataset.sig = names.join('|');
        fill(refs.as, names.map(n => h('option', { value: n }, n === 'cli' ? 'cli (administrator: everything)' : n)));
        refs.as.value = names.includes(cur) ? cur : 'cli';
      }
      const colls = visibleCollections(refs.as.value); const sig = refs.as.value + ':' + colls.join('|');
      if (refs.colls.dataset.sig !== sig) {
        const keep = new Set($$('input[name=coll]:checked', refs.colls).map(i => i.value));
        refs.colls.dataset.sig = sig;
        fill(refs.colls, colls.length ? colls.map(c => h('label', { class: 'check' }, h('input', { type: 'checkbox', name: 'coll', value: c, checked: keep.has(c) }), c)) : h('span', { class: 'muted small' }, 'nothing indexed yet'));
      }
    },
  };
})();
