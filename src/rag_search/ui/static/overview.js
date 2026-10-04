'use strict';
/* Overview: the answer to "is it working, and what should I do next?".
   A status banner with the next action, the whole path of a document drawn once -- documents -> index (stages 1-6) ->
   publish (7-8) -> search daemon (S1-S6) -> clients -- with the live numbers of each step, a short list of what needs
   attention (every item says what is wrong and links to where it is fixed), the collections, and the daemons with
   their controls.  The stage numbers are those of the Architecture tab. */
(function () {
  let refs = {}, models = null, modelsAt = 0, showDaemons = false;

  async function loadModels() {
    if (Date.now() - modelsAt < 20000) return;
    modelsAt = Date.now();
    const r = await api('models');
    if (r && r.ok !== false) { models = r; RS.views.overview.update(); }
  }

  const pl = (n, word, plural) => `${num(n)} ${n === 1 ? word : (plural || word + 's')}`;
  const phaseStage = { convert: 'convert', embed: 'embed', merge: 'merge' };

  function daemonControls(which, up) {
    const ro = readOnly();
    const run = (action, label) => async () => {
      if (action !== 'start' && which === 'indexer' && activeJob()) {
        if (!await confirmDialog(label + ' the indexer?', 'An indexing run is active. It will be cancelled (finished documents are kept).', label, true)) return;
      }
      const r = await act('daemon', { which, action }, `${which} daemon: ${action} requested`);
      if (r.ok !== false) RS.views.system && RS.views.system.refreshSoon && RS.views.system.refreshSoon();
    };
    return h('div', { class: 'row' },
      h('button', { class: 'btn small', disabled: ro || up, on: { click: run('start', 'Start') } }, 'Start'),
      h('button', { class: 'btn small', disabled: ro || !up, on: { click: run('restart', 'Restart') } }, 'Restart'),
      h('button', { class: 'btn small danger', disabled: ro || !up, on: { click: run('stop', 'Stop') } }, 'Stop'));
  }

  async function indexNew() {
    const r = await act('index/start', { mode: 'new' });
    if (r.ok !== false) toast(r.already_running ? 'A run is already active' : 'Indexing started', r.already_running ? '' : 'ok');
  }

  async function publishNow() {
    const r = await act('index/publish', {});
    if (r.ok !== false) toast('Published', 'ok');
  }

  /* ---------- what needs attention ---------- */

  function attention() {
    const live = RS.state.live || {}, cat = RS.state.catalog || {}, list = cat.list || {};
    const items = [];
    const search = (live.daemons || {}).search || {};
    const idx = live.index || {}, job = idx.job, sum = job && job.summary;
    const add = (sev, text, link, linkText, action) => items.push({ sev, text, link, linkText, action });

    if (search.error) add('bad', 'The search daemon reports an error: ' + search.error, '#/system', 'Open System');
    if (search.index_error) add('bad', 'The published index cannot be loaded: ' + search.index_error, '#/collections', 'Open Collections');
    if (search.config_error) add('warn', search.config_error, '#/settings', 'Open Settings');
    if (search.access_error) add('warn', search.access_error, '#/collections', 'Open Collections & access');
    if (job && job.status === 'failed') add('bad', `The last indexing run failed: ${job.error || 'see the run for the reason'}`, '#/indexing', 'Open the run');
    if (sum && sum.error_count) add('bad', `${sum.error_count} document(s) failed in the last run.`, '#/indexing', 'See which, and why');
    if (sum && sum.no_text_count) add('warn', `${sum.no_text_count} document(s) contain no text and were skipped (scanned pages need the document reader).`, '#/indexing', 'Show them');
    if (job && job.status === 'succeeded' && sum && sum.indexed > 0 && !job.publish && !activeJob())
      add('warn', 'The last run finished but its result is not published yet, so search still serves the old index.', null, null, { label: 'Publish now', run: publishNow });
    const brief = cat.models || {};
    if (brief.serving && brief.embedding && brief.serving !== brief.embedding)
      add('warn', `The published index was built with ${brief.serving}, but ${brief.embedding} is selected: re-index to use it.`, '#/models', 'Open Models');
    if (models) {
      for (const kind of ['embedding', 'reranker']) {
        const row = ((models[kind] || {}).models || []).find(r => r.active);
        if (row && ['selected_download', 'selected_partial'].includes(row.state))
          add('warn', `The ${kind} model ${row.label || row.id} is chosen but not downloaded yet.`, '#/models', 'Open Models');
      }
      const rd = models.vlm && models.vlm.reading;
      if (rd && rd.by !== 'reader' && models.vlm.mode !== 'off') {
        const t = ((live.conversion || {}).totals || {}).branches || {};
        const scans = (t.raster || 0) + (t.image || 0) + (t.fallback || 0);
        add(scans ? 'warn' : 'info', rd.text.split(' (')[0] + (rd.text.includes('Photographed') ? ' Photographed pages and pictures usually need it.' : ''), '#/models', 'See why, and fix it');
      }
    }
    if (idx.job && (idx.job.summary || {}).unsupported_count)
      add('info', `${idx.job.summary.unsupported_count} file(s) were skipped because their format is not supported.`, '#/indexing', 'Show them');
    if (!list.generation && !activeJob()) add('info', 'Nothing is published yet: add documents to a collection and index them.', '#/indexing', 'Open Indexing');
    const order = { bad: 0, warn: 1, info: 2 };
    return items.sort((a, b) => order[a.sev] - order[b.sev]);
  }

  function attentionCard(items) {
    const worst = items.length ? items[0].sev : 'ok';
    return h('div', { class: 'card ov-attn' },
      h('div', { class: 'card-head' }, h('h2', null, 'Needs attention'),
        items.length ? pill(String(items.length), worst === 'bad' ? 'bad' : worst === 'warn' ? 'warn' : '') : pill('nothing', 'ok')),
      items.length ? h('ul', { class: 'ov-list' }, items.map(i => h('li', { class: 'sev-' + i.sev },
        h('i', { class: 'ov-dot' }), h('span', null, i.text),
        i.link ? h('a', { href: i.link, class: 'ov-fix' }, i.linkText || 'Open') : null,
        i.action ? h('button', { class: 'btn small primary ov-fix', disabled: readOnly(), on: { click: i.action.run } }, i.action.label) : null)))
        : h('p', { class: 'muted', style: { margin: 0 } }, 'Everything the dashboard can check looks fine.'));
  }

  /* ---------- the banner ---------- */

  function hero(items) {
    const live = RS.state.live; if (!live) return h('div', { class: 'ov-hero' }, h('h2', null, 'Connecting…'));
    const list = (RS.state.catalog || {}).list || {}, t = list.totals || {};
    const job = activeJob(), bad = items.find(i => i.sev === 'bad');
    const ro = readOnly();
    let cls = 'ok', title, line, actions;
    if (job) {
      const p = job.progress || {};
      cls = 'busy'; title = 'Indexing is running';
      line = `${p.phase ? PL.label(phaseStage[p.phase] || p.phase) : 'starting'} · ${p.done || 0} of ${p.total || '?'} document(s)` + (p.current ? ` · ${p.current}` : '') + ` · ${dur(job.elapsed_s)} so far`;
      actions = [h('a', { class: 'btn primary', href: '#/indexing' }, 'Watch the run')];
    } else if (bad) {
      cls = 'bad'; title = 'Needs attention'; line = bad.text;
      actions = [bad.link ? h('a', { class: 'btn primary', href: bad.link }, bad.linkText || 'Open') : null];
    } else if (list.generation) {
      title = 'Ready to search';
      line = `${pl(t.documents, 'document')} in ${pl(t.collections, 'collection')}, ${pl(t.chunks, 'passage')} · generation ${list.generation}, published ${ago(list.published_at)}`;
      actions = [h('a', { class: 'btn primary', href: '#/search' }, 'Search'),
        h('button', { class: 'btn', disabled: ro, on: { click: indexNew } }, 'Index new and changed documents')];
    } else {
      cls = 'idle'; title = 'Nothing to search yet';
      line = 'Put documents in a collection, then index them. Unchanged documents are skipped, so it is safe to run again.';
      actions = [h('button', { class: 'btn primary', disabled: ro, on: { click: indexNew } }, 'Index now'), h('a', { class: 'btn', href: '#/indexing' }, 'Where do documents go?')];
    }
    return h('div', { class: 'ov-hero ' + cls },
      h('div', { class: 'ov-hero-main' }, h('h2', null, title), h('p', null, line)),
      h('div', { class: 'ov-hero-actions' }, actions));
  }

  /* ---------- the path of a document ---------- */

  function node(tag, title, href, state, lines, extra) {
    return h('a', { class: 'ov-node ' + (state || ''), href },
      h('span', { class: 'ov-tag' }, tag), h('h4', null, title),
      lines.filter(Boolean).map(l => h('p', null, l)), extra || null);
  }

  function flow() {
    const live = RS.state.live || {}, cat = RS.state.catalog || {}, list = cat.list || {}, t = list.totals || {};
    const idx = live.index || {}, d = live.daemons || {}, job = idx.job, src = idx.sources || {};
    const ist = indexerStatus(), sst = searchStatus();
    const nColl = (src.docs_collections || []).length + (src.locations || []).length + (src.imported || []).length;
    const running = activeJob(), p = (running && running.progress) || {};
    const pct = p.total ? Math.min(100, Math.round(100 * (p.done || 0) / p.total)) : 0;
    const arrow = text => h('div', { class: 'ov-arrow', title: text }, h('span', null, text), h('i', null, '→'));
    const sum = job && job.summary;
    const lastRun = job && !running ? `last run ${job.status} ${ago(job.finished_at || job.started_at || job.created_at)}` + (sum && sum.indexed !== undefined ? ` · ${sum.indexed} indexed, ${sum.skipped_fresh || 0} unchanged` : '') : null;
    const clients = ((cat.access || {}).clients || []);
    const seen = clients.filter(c => c.last_seen);
    const mem = ((d.search || {}).memory || {});
    return h('div', { class: 'ov-flow' },
      node('1', 'Documents', '#/indexing', nColl ? 'ok' : 'idle',
        [nColl ? pl(nColl, 'collection') : 'no collection yet', (src.locations || []).length ? `${pl(src.locations.length, 'registered folder')}` : null,
          'read-only: never changed']),
      arrow('read'),
      node('1–6', 'Index', '#/indexing', running ? 'busy' : ist.cls === 'bad' ? 'bad' : ist.up ? 'ok' : 'idle',
        [h('span', null, pill(ist.label, ist.cls, ist.busy)),
          running ? `${PL.label(phaseStage[p.phase] || p.phase || 'convert')} · ${p.done || 0}/${p.total || '?'}` : lastRun || 'no run yet'],
        running ? h('div', { class: 'bar', style: { marginTop: '6px' } }, h('i', { style: { width: pct + '%' } })) : null),
      arrow('vectors'),
      node('7 · 8', 'Publish', '#/collections', list.generation ? 'ok' : 'idle',
        [list.generation ? `generation ${list.generation}` : 'nothing published', list.generation ? `${pl(t.documents, 'document')} · ${pl(t.chunks, 'passage')}` : null,
          list.generation ? `since ${ago(list.published_at)}` : null]),
      arrow('serve'),
      node('S1–S6', 'Search', '#/search', sst.cls === 'bad' ? 'bad' : sst.up ? 'ok' : 'idle',
        [h('span', null, pill(sst.label, sst.cls, sst.busy)), mem.rss_bytes ? `${bytes(mem.rss_bytes)} in memory` : 'starts on demand',
          cat.models && cat.models.embedding ? cat.models.embedding.split('/').pop() : null]),
      arrow('queries'),
      node('', 'Clients', '#/collections', seen.length ? 'ok' : 'idle',
        [`${pl(seen.length, 'client')} used it since the daemon started`, seen.length ? `last ${ago(Math.max(...seen.map(c => c.last_seen)))}` : 'MCP tools, the CLI and this dashboard'], null));
  }

  /* ---------- the last run's pages ---------- */

  function lastRunCard() {
    const live = RS.state.live || {}, conv = live.conversion || {}, totals = conv.totals || {};
    if (!totals.pages || activeJob()) return null;
    return h('div', { class: 'card' },
      h('div', { class: 'card-head' }, h('h2', null, 'How the last run read its pages'),
        h('div', { class: 'spacer' }, h('a', { href: '#/indexing' }, 'Details'))),
      CV.bands(totals.branches),
      h('div', { class: 'row', style: { marginTop: '8px', gap: '6px' } }, CV.outcomeChips(totals.outcomes)),
      h('div', { class: 'grid g4', style: { marginTop: '12px' } }, CV.tiles(totals).slice(0, 4)));
  }

  /* ---------- collections ---------- */

  function collectionsCard() {
    const list = (RS.state.catalog || {}).list || {}, cols = list.collections || [];
    return h('div', { class: 'card' },
      h('div', { class: 'card-head' }, h('h2', null, 'Collections'),
        list.generation ? h('span', { class: 'muted small' }, `generation ${list.generation}, published ${ago(list.published_at)}`) : null,
        h('div', { class: 'spacer' }, h('a', { href: '#/collections' }, 'Collections & access'))),
      cols.length ? h('div', { class: 'ov-colls' }, cols.slice(0, 12).map(c => h('a', { class: 'ov-coll', href: '#/collections' },
        h('b', null, c.collection), c.origin === 'imported' ? chip('imported') : null,
        h('div', { class: 'ov-coll-n' }, h('span', null, h('b', null, num(c.document_count)), c.document_count === 1 ? ' document' : ' documents'), h('span', null, h('b', null, num(c.chunks)), c.chunks === 1 ? ' passage' : ' passages'),
          h('span', null, bytes(c.index_bytes))),
        c.description ? h('p', { class: 'small muted' }, c.description) : null,
        h('p', { class: 'small muted' }, c.built_at ? 'built ' + ago(c.built_at) : '', c.model ? ' · ' + c.model.split('/').pop() : ''))))
        : h('p', { class: 'muted' }, 'Nothing is published yet. ' + (list.hint || '')),
      cols.length > 12 ? h('p', { class: 'small muted', style: { marginBottom: 0 } }, `+ ${cols.length - 12} more`) : null,
      list.model ? h('p', { class: 'small muted', style: { margin: '10px 0 0' } }, 'Embedding model: ', h('code', null, list.model)) : null);
  }

  /* ---------- daemons (details, with controls) ---------- */

  function searchRows(d) {
    const w = (d && d.warmup) || {}, m = (d && d.memory) || {}, lr = (d && d.last_reload) || null;
    const rows = [];
    if (d && d.state) {
      rows.push(['Process', `pid ${d.pid}, up ${dur(d.uptime_s)}`]);
      rows.push(['Serving', d.generation ? `generation ${d.generation}: ${num(d.collections)} collection(s), ${num(d.chunks)} chunks` : 'nothing published yet']);
      rows.push(['Warm-up', w.status === 'warm' ? `finished in ${dur(w.total_s)} (models ${dur(w.models_s)}, indexes ${dur(w.index_s)})`
        : w.status === 'warming_up' ? `running for ${dur(w.elapsed_s)}` : w.status || '–']);
      rows.push(['Memory', m.rss_bytes ? `${bytes(m.rss_bytes)} resident (peak ${bytes(m.peak_rss_bytes)}); vectors ${bytes(m.embeddings_bytes)}, chunk text ${bytes(m.text_bytes)}` : '–']);
      if (m.models_bytes && Object.keys(m.models_bytes).length)
        rows.push(['Models in memory', Object.entries(m.models_bytes).map(([k, v]) => `${k} ${bytes(v)}`).join(', ')]);
      if (lr) rows.push(['Last reload', `${clock(lr.at || lr.time)} → generation ${lr.generation}` + (lr.loaded && lr.loaded.length ? `, loaded ${lr.loaded.join(', ')}` : '') + (lr.reused && lr.reused.length ? `, reused ${lr.reused.join(', ')}` : '')]);
    } else rows.push(['Process', 'not running. It starts on demand, or with Start.']);
    return rows;
  }

  function indexerRows(d) {
    const job = (RS.state.live.index || {}).job;
    const rows = [];
    if (d && d.state) {
      rows.push(['Process', `pid ${d.pid}, up ${dur(d.uptime_s)}`]);
      rows.push(['Current run', d.running ? `${job ? job.id : d.job_id} (${job && job.mode || ''})` : 'none, waiting for work']);
    } else rows.push(['Process', 'not running. It starts when you index.']);
    if (job) rows.push(['Last run', `${job.status} ${ago(job.finished_at || job.started_at || job.created_at)}` + (job.elapsed_s !== undefined ? `, took ${dur(job.elapsed_s)}` : '')]);
    return rows;
  }

  function daemonsCard() {
    const d = (RS.state.live || {}).daemons || {};
    const st = searchStatus(), ist = indexerStatus();
    return h('details', { class: 'card ov-daemons', open: showDaemons, on: { toggle: e => { showDaemons = e.target.open; } } },
      h('summary', null, h('h2', { style: { display: 'inline' } }, 'Daemons'), ' ',
        h('span', { class: 'small muted' }, 'search '), pill(st.label, st.cls, st.busy), ' ', h('span', { class: 'small muted' }, 'indexer '), pill(ist.label, ist.cls, ist.busy)),
      h('div', { class: 'grid g2', style: { marginTop: '12px' } },
        h('div', null, h('div', { class: 'card-head' }, h('h3', null, 'Search daemon'), h('div', { class: 'spacer' }, daemonControls('search', st.up))),
          kv(searchRows(d.search))),
        h('div', null, h('div', { class: 'card-head' }, h('h3', null, 'Indexer daemon'), h('div', { class: 'spacer' }, daemonControls('indexer', ist.up))),
          kv(indexerRows(d.indexer)))));
  }

  RS.views.overview = {
    init(root) {
      refs = { hero: h('div'), flow: h('div'), attn: h('div'), cols: h('div', { style: { marginTop: '16px' } }), last: h('div', { style: { marginTop: '16px' } }), daemons: h('div', { style: { marginTop: '16px' } }) };
      root.append(h('h2', null, 'Overview'), refs.hero, h('div', { class: 'card', style: { marginTop: '16px' } }, h('div', { class: 'card-head' }, h('h2', null, 'The path of a document'),
        h('span', { class: 'muted small' }, 'numbers are the pipeline stages (Architecture tab)')), refs.flow),
        h('div', { style: { marginTop: '16px' } }, refs.attn), refs.cols, refs.last, refs.daemons);
    },
    show() { modelsAt = 0; loadModels(); },
    update() {
      if (!RS.state.live) return;
      const items = attention();
      patch(refs.hero, hero(items));
      patch(refs.flow, flow());
      patch(refs.attn, attentionCard(items));
      patch(refs.cols, collectionsCard());
      patch(refs.last, lastRunCard());
      patch(refs.daemons, daemonsCard());
    },
    tick() { loadModels(); this.update(); },
  };
})();
