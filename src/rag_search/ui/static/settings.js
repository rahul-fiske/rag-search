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

  function stageTunables(s) {
    const mine = new Set(s.settings.map(x => x.id));
    return tunables.filter(t => mine.has(tkey(t)));
  }

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

  function stageSection(s) {
    const mine = stageTunables(s);
    const bySection = {};
    for (const t of mine) (bySection[t.section] = bySection[t.section] || []).push(t);
    const fields = Object.entries(bySection).map(([section, ts]) => {
      const form = tunablesForm(ts);
      const rec = { stage: s.id, section, tunables: ts, form };
      forms.push(rec);
      // an "in effect" line under every field; refreshed with the pipeline data, the inputs are never rebuilt
      rec.lines = ts.map(() => h('div'));
      Array.from(form.root.children).forEach((row, i) => row.append(rec.lines[i]));
      return form.root;
    });
    const ids = new Set(mine.map(tkey));
    const others = s.settings.filter(x => !ids.has(x.id));
    const readonly = others.length ? h('div', { class: 'table-wrap' }, h('table', { class: 'plain' }, h('tbody', null,
      others.map(r => h('tr', null, h('td', null, r.label), h('td', { class: 'mono' }, PL.valueText(r)), h('td', null, PL.sourceChip(r)), h('td', null, whereToChange(r))))))) : null;
    const wrap = h('div', { class: 'stage' + (s.parent ? ' sub' : ''), id: 'set-' + s.id },
      PL.stageHead(s, !!s.parent),
      h('p', { class: 'small muted', style: { margin: '2px 0 8px' } }, s.what),
      readonly, fields.length ? h('div', { style: { marginTop: others.length ? '10px' : 0 } }, fields) : null,
      !s.settings.length ? h('p', { class: 'small muted', style: { margin: 0 } }, 'Nothing to configure' + ((s.constants || []).length ? '; fixed: ' + s.constants.map(c => `${c.label} ${c.value}`).join(' · ') : '') + '.') : null,
      s.settings.length && (s.constants || []).length ? h('p', { class: 'small muted', style: { margin: '8px 0 0' } }, 'Fixed, not configurable: ' + s.constants.map(c => `${c.label} ${c.value}`).join(' · ')) : null);
    return wrap;
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
      PL.stages(data, 'indexing').forEach(s => refs.idxBody.append(stageSection(s)));
      PL.stages(data, 'search').forEach(s => refs.searchBody.append(stageSection(s)));
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
        h('div', { class: 'card' },
          h('div', { class: 'card-head' }, h('h3', null, 'Indexing pipeline'), h('span', { class: 'applies-tag' }, 'applies to the next indexing run, unless a field says otherwise')),
          refs.idxBody, bar(refs.saveIdx)),
        h('div', { class: 'card', style: { marginTop: '16px' } },
          h('div', { class: 'card-head' }, h('h3', null, 'Search pipeline'), h('span', { class: 'applies-tag' }, 'applies immediately, to the very next search, unless a field says otherwise')),
          refs.searchBody, bar(refs.saveSearch)));
      load();
    },
    show() { load(); },
    update() { controls(); },
    tick() { controls(); refreshEffective(); },
  };
})();
