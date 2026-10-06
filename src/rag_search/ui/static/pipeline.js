'use strict';
/* The numbered pipeline in the browser (rag_search/stages.py is the source; /api/pipeline serves it with the
   value each setting really has now).  Stage numbers are the same everywhere: Architecture, Indexing, Settings,
   page traces, logs, the Playground.  STAGE_KEYS repeats the registry's keys for the places that only know a
   key (time breakdowns, events); tests/test_stages.py keeps the two equal. */
const STAGE_KEYS = {
  discover: ['1', 'Discover'], fingerprint: ['2', 'Fingerprint'], convert: ['3', 'Convert'],
  profile: ['3.1', 'Profile'], read: ['3.2', 'Read'], gate: ['3.3', 'Gate'], repair: ['3.4', 'Repair'],
  reconcile: ['3.5', 'Reconcile'], chunk: ['4', 'Chunk'], embed: ['5', 'Embed'], write: ['6', 'Write'],
  merge: ['7', 'Merge'], publish: ['8', 'Publish'],
  access: ['S1', 'Access'], keyword: ['S2', 'Keyword'], vectors: ['S3', 'Vectors'], fuse: ['S4', 'Fuse'],
  rerank: ['S5', 'Rerank'], top_k: ['S6', 'Top k'],
};

const PL = (function () {
  const SCOPE = { run: 'once per run', document: 'per document', collection: 'per collection', query: 'per query' };
  const APPLIES = { immediate: 'applies at once', 'next-run': 'next indexing run', restart: 'after a daemon restart',
    're-embed': 'every document is embedded again' };
  let cache = null, at = 0, pending = null;

  const id = key => (STAGE_KEYS[key] || [key])[0];
  const name = key => (STAGE_KEYS[key] || [key, key])[1];
  /* "3.1 Profile" for a key such as "profile"; unknown keys are returned as they are */
  const label = key => STAGE_KEYS[key] ? `${STAGE_KEYS[key][0]} ${STAGE_KEYS[key][1]}` : String(key).replace(/_/g, ' ');
  const hw = where => where ? h('span', { class: 'hw ' + (where === 'gpu' ? 'gpu' : 'cpu'),
    title: where === 'gpu' ? 'runs on the GPU' : where === 'cpu+gpu' ? 'CPU and GPU' : 'runs on the CPU' }, where === 'cpu+gpu' ? 'CPU+GPU' : where.toUpperCase()) : null;

  /* the pipeline with values, cached for a few seconds (several tabs ask) */
  async function load(force) {
    if (!force && cache && Date.now() - at < 4000) return cache;
    if (pending) return pending;
    pending = api('pipeline').then(r => { pending = null; if (r.ok !== false) { cache = r; at = Date.now(); } return cache; });
    return pending;
  }
  const last = () => cache;

  function valueText(row) {
    let v = row.value;
    if (v === null || v === undefined || v === '') return '–';
    if (typeof v === 'boolean') return v ? 'on' : 'off';
    if (Array.isArray(v)) return v.join(', ') || '–';
    if (typeof v === 'object') return Object.entries(v).map(([k, x]) => `${k} ${x}`).join(', ');
    return String(v);
  }

  /* where the value comes from: the daemon's environment wins over config.json, which wins over the default */
  function sourceChip(row) {
    if (row.source === 'experiment') return chip('this experiment', 'accent');
    if (row.source === 'production config') return chip('production config.json', 'accent');
    if (row.source === 'environment') return chip('environment ' + (row.env || ''), 'warn');
    if (row.source === 'config') return chip('config.json', 'accent');
    return chip('default');
  }

  function appliesText(row) { return APPLIES[row.applies] || row.applies || ''; }

  /* one setting, read-only: label, value, source, when a change applies, which code reads it */
  function settingRow(row, quiet) {
    const tip = [row.read_by ? 'read by ' + row.read_by : '', row.when || '', row.also_used_by && row.also_used_by.length ? 'also used by ' + row.also_used_by.join(', ') : ''].filter(Boolean).join('\n');
    return h('tr', { title: tip },
      h('td', null, row.label, row.editable || quiet ? null : h('span', { class: 'muted small' }, row.id.startsWith('models.') ? '  (chosen on the Models tab)' : '  (config.json only)')),
      h('td', { class: 'mono' }, valueText(row)),
      h('td', null, sourceChip(row)),
      h('td', { class: 'small muted' }, appliesText(row)));
  }

  function stageHead(s, level) {
    return h('div', { class: 'stage-head' + (level ? ' sub' : '') },
      h('b', { class: 'stage-id' }, s.id), h('b', null, s.name), hw(s.where),
      h('span', { class: 'chip' }, SCOPE[s.scope] || s.scope), s.optional ? h('span', { class: 'chip' }, 'optional') : null);
  }

  /* a stage and what it uses; `extra` is appended inside (the Settings tab puts the editable fields there) */
  function stageBlock(s, extra, only, quiet) {
    const shown = (s.settings || []).filter(r => !only || only(r));
    const rows = shown.map(r => settingRow(r, quiet));
    const envs = (s.env_only || []).filter(e => e.set);
    // a stage whose settings belong to several parts (the router and the lanes of 3.2): one small table per part
    const parts = (s.groups || []).map(g => ({ g, rows: g.settings.map(id => shown.find(r => r.id === id)).filter(Boolean) })).filter(p => p.rows.length);
    const table = rs => h('div', { class: 'table-wrap' }, h('table', { class: 'plain' }, h('tbody', null, rs)));
    return h('div', { class: 'stage' + (s.parent ? ' sub' : ''), id: 'stage-' + s.id },
      stageHead(s, !!s.parent),
      h('p', { class: 'small muted', style: { margin: '2px 0 6px' } }, s.what),
      parts.length ? parts.map(p => h('div', { class: 'stage-part', title: p.g.what },
        h('div', { class: 'small' }, h('b', { class: 'mono' }, p.g.id), ' ', h('b', null, p.g.name)), table(p.rows.map(r => settingRow(r, quiet)))))
        : rows.length ? table(rows) : null,
      envs.length ? h('p', { class: 'small', style: { margin: '4px 0' } }, 'Environment variables set for this daemon: ', envs.flatMap((e, i) => [i ? ' · ' : '', h('code', { title: e.value }, e.name + '=' + e.value)])) : null,
      !rows.length && !extra && !(s.settings || []).length && !(s.constants || []).length ? h('p', { class: 'small muted', style: { margin: 0 } }, 'Nothing to configure here.') : null,
      (s.constants || []).length ? h('p', { class: 'small muted', style: { margin: '4px 0 0' } }, 'Fixed, not configurable: ', s.constants.map(c => `${c.label} ${c.value}`).join(' · ')) : null,
      extra || null);
  }

  /* every stage of one pipeline ("indexing" = ids without S, "search" = S1..S6) */
  function stages(data, pipeline) {
    return ((data && data.stages) || []).filter(s => (pipeline === 'search') === s.id.startsWith('S'));
  }

  return { id, name, label, hw, load, last, valueText, sourceChip, appliesText, settingRow, stageHead, stageBlock, stages, SCOPE };
})();
