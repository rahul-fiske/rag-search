'use strict';
/* Settings: the production defaults of the indexing and the search pipeline (config.json's `indexer`, `search` and
   `models` sections), arranged by pipeline stage: the numbers are those of the Architecture tab, the page traces and
   the logs.  Each field shows the value that is really in effect and where it comes from (default, config.json, or an
   environment variable the daemon was started with, which wins).  Choosing a model is done on the Models tab. */
(function () {
  let refs = {};
  const forms = [];                 // {stage, section, tunables, form, rows}: one tunablesForm per stage and config section
  let tunables = [], built = false;

  const tkey = t => t.section + '.' + t.key;

  function whereToChange(row) {
    if (row.id.startsWith('models.') && ['models.embedding', 'models.reranker', 'models.reader', 'models.repair', 'models.memory_limit_gb'].includes(row.id))
      return h('span', { class: 'small muted' }, 'chosen on the ', h('a', { href: '#/models' }, 'Models'), ' tab');
    return h('span', { class: 'small muted' }, 'set in config.json as ', h('code', null, row.id));
  }

  function effectiveLine(row) {
    if (!row) return '';
    const note = row.source === 'environment'
      ? h('span', { class: 'small', style: { color: 'var(--warn)' } }, ` · the daemon was started with ${row.env}, which wins over what you save here until it is unset`)
      : null;
    return h('div', { class: 'small muted tunable-effective' }, 'in effect: ', h('b', { class: 'mono' }, PL.valueText(row)), ' ', PL.sourceChip(row), note);
  }

  /* The settings of one block (a stage, or one part of a stage): read-only rows for what is chosen elsewhere, an
     editable field for every tunable, and the environment-only variables that reach it.  *rows* are the resolved
     settings (`/api/pipeline`), *envs* the environment variables with their values. */
  function settingsBlock(stageId, rows, envs) {
    const ids = new Set(rows.map(r => r.id));
    const order = rows.map(r => r.id);                       // the block's own order, not the registry's
    const mine = tunables.filter(t => ids.has(tkey(t))).sort((a, b) => order.indexOf(tkey(a)) - order.indexOf(tkey(b)));
    const bySection = {};
    for (const t of mine) (bySection[t.section] = bySection[t.section] || []).push(t);
    const fields = Object.entries(bySection).map(([section, ts]) => {
      const form = tunablesForm(ts);
      const rec = { stage: stageId, section, tunables: ts, form };
      forms.push(rec);
      // an "in effect" line under every field; refreshed with the pipeline data, the inputs are never rebuilt
      rec.lines = ts.map(() => h('div'));
      Array.from(form.root.children).forEach((row, i) => row.append(rec.lines[i]));
      return form.root;
    });
    const editable = new Set(mine.map(tkey));
    const others = rows.filter(x => !editable.has(x.id));
    const readonly = others.length ? h('div', { class: 'table-wrap' }, h('table', { class: 'plain' }, h('tbody', null,
      others.map(r => h('tr', null, h('td', null, r.label), h('td', { class: 'mono' }, PL.valueText(r)), h('td', null, PL.sourceChip(r)), h('td', null, whereToChange(r))))))) : null;
    const envNote = envs && envs.length ? h('p', { class: 'small muted set-env' },
      'Environment variables only (set where the daemon is started, not saved here): ',
      envs.flatMap((e, i) => [i ? ' · ' : '', h('code', { title: e.set ? 'set for this daemon' : 'not set: the built-in default applies' }, e.name + (e.set ? '=' + e.value : '')), e.set ? null : h('span', { class: 'faint' }, ' not set')])) : null;
    return { nodes: [readonly, fields.length ? h('div', { style: { marginTop: others.length ? '10px' : 0 } }, fields) : null, envNote], count: rows.length + (envs || []).length };
  }

  /* One part of a stage with settings of its own (a lane of 3.2): its own card, in the lane's colour. */
  function groupCard(s, g) {
    const rows = g.settings.map(id => s.settings.find(r => r.id === id)).filter(Boolean);
    const envs = g.env_only.map(n => (s.env_only || []).find(e => e.name === n)).filter(Boolean);
    const lane = /^[\d.]+([a-d])$/.exec(g.id);
    const block = settingsBlock(s.id, rows, envs);
    return h('div', { class: 'set-group' + (lane ? ' rw-' + lane[1] : ''), id: 'set-part-' + g.id },
      h('div', { class: 'stage-head sub' }, h('b', { class: 'stage-id' }, g.id), h('b', null, g.name), PL.hw(g.where)),
      h('p', { class: 'small muted', style: { margin: '2px 0 8px' } }, g.what),
      block.nodes,
      !block.count ? h('p', { class: 'small muted', style: { margin: 0 } }, 'Nothing to configure.') : null);
  }

  /* One stage = one card.  Its sub-stages (3.1 - 3.5 of 3 Convert) are cards inside it, and a stage whose settings
     belong to several parts (3.2: the router and the four lanes) shows one card per part. */
  function stageCard(s, all) {
    const kids = all.filter(x => x.parent === s.id);
    const groups = s.groups || [];
    const block = groups.length ? null : settingsBlock(s.id, s.settings, s.env_only);
    const consts = (s.constants || []).map(c => `${c.label} ${c.value}`).join(' · ');
    const empty = !groups.length && !block.count;
    return h('div', { class: 'set-card' + (s.parent ? ' sub' : '') + (empty && !kids.length ? ' bare' : ''), id: 'set-' + s.id },
      PL.stageHead(s, !!s.parent),
      h('p', { class: 'small muted', style: { margin: '2px 0 8px' } }, s.what),
      groups.length ? h('div', { class: 'set-groups' }, groups.map(g => groupCard(s, g))) : block.nodes,
      empty ? h('p', { class: 'small muted', style: { margin: 0 } }, 'Nothing to configure' + (consts ? '; fixed: ' + consts : '') + '.') : null,
      !empty && consts ? h('p', { class: 'small muted', style: { margin: '8px 0 0' } }, 'Fixed, not configurable: ' + consts) : null,
      kids.length ? h('div', { class: 'set-kids' },
        h('div', { class: 'small muted set-kids-head' }, `The steps of ${s.id} · ${s.name}, each with its own settings:`),
        kids.map(k => stageCard(k, all))) : null);
  }

  async function save(pipeline) {
    const bySection = {};
    for (const rec of forms.filter(f => (pipeline === 'search') === f.stage.startsWith('S'))) {
      const vals = rec.form.collect();
      bySection[rec.section] = { ...(bySection[rec.section] || {}), ...vals };
    }
    for (const [section, values] of Object.entries(bySection)) {
      const r = await act('config/set', { section, values });
      if (r.ok === false) return;
    }
    toast('Saved', 'ok');
    await load();
  }

  function controls() {
    const ro = readOnly();
    refs.saveIdx.disabled = ro; refs.saveSearch.disabled = ro;
  }

  function showEffective(data) {
    const byId = {};
    for (const s of data.stages) for (const r of s.settings) byId[r.id] = r;
    for (const rec of forms) rec.tunables.forEach((t, i) => patch(rec.lines[i], effectiveLine(byId[tkey(t)])));
  }

  async function load() {
    const [r, data] = await Promise.all([api('config'), PL.load(true)]);
    if (r.ok === false) { toast(r.error || 'failed to load settings', 'bad'); return; }
    if (!built && data) {
      built = true; tunables = r.tunables;
      for (const [pipeline, body] of [['indexing', refs.idxBody], ['search', refs.searchBody]]) {
        const all = PL.stages(data, pipeline);
        all.filter(s => !s.parent).forEach(s => body.append(stageCard(s, all)));
      }
    }
    for (const rec of forms) rec.form.load(r.values[rec.section]);
    if (data) showEffective(data);
    refs.errorNote.textContent = r.error || '';
    refs.errorNote.classList.toggle('hidden', !r.error);
    controls();
  }

  async function refreshEffective() {
    const data = await PL.load(false);
    if (data && built) showEffective(data);
  }

  RS.views.settings = {
    init(root) {
      refs.idxBody = h('div'); refs.searchBody = h('div');
      refs.errorNote = h('div', { class: 'notice bad hidden' });
      refs.saveIdx = h('button', { class: 'btn primary', on: { click: () => save('indexing') } }, 'Save indexing settings');
      refs.saveSearch = h('button', { class: 'btn primary', on: { click: () => save('search') } }, 'Save search settings');
      const bar = (btn) => h('div', { class: 'row', style: { marginTop: '14px' } }, btn,
        h('button', { class: 'btn', on: { click: load } }, 'Discard'),
        h('span', { class: 'muted small' }, 'A blank field (or 0) clears that setting back to its built-in default.'));
      root.append(
        h('h2', null, 'Settings'),
        h('p', { class: 'muted small' },
          'Production defaults, arranged by the stage of the pipeline that uses them (the numbers are those of the Architecture tab and of every page trace). '
          + 'They apply to every client, the MCP tools included, when it does not ask for something different itself; one search or run can still override them. '
          + 'Each field says which value is really in effect and where it comes from. Which models are used is chosen on the ',
          h('a', { href: '#/models' }, 'Models'), ' tab.'),
        refs.errorNote,
        h('div', { class: 'set-pipeline' },
          h('div', { class: 'card-head' }, h('h3', null, 'Indexing pipeline'), h('span', { class: 'applies-tag' }, 'applies to the next indexing run, unless a field says otherwise'),
            h('span', { class: 'small muted' }, 'one card per stage; the steps of 3 Convert and the lanes of 3.2 Read have cards of their own')),
          refs.idxBody, bar(refs.saveIdx)),
        h('div', { class: 'set-pipeline', style: { marginTop: '22px' } },
          h('div', { class: 'card-head' }, h('h3', null, 'Search pipeline'), h('span', { class: 'applies-tag' }, 'applies immediately, to the very next search, unless a field says otherwise')),
          refs.searchBody, bar(refs.saveSearch)));
      load();
    },
    show() { load(); },
    update() { controls(); },
    tick() { controls(); refreshEffective(); },
  };
})();
