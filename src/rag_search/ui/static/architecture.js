'use strict';
/* Architecture: how indexing, search, the daemons and the clients fit together.
   Every number on this page is fetched from /api/architecture, i.e. from the constants the engine itself uses. */
(function () {
  let A = null, refs = {}, section = 'system', docLoaded = false;
  const SECTIONS = [['system', 'System overview'], ['indexing', 'Indexing pipeline'], ['format', 'Index files'], ['search', 'Search pipeline'],
    ['models', 'Models'], ['daemons', 'Daemons & clients'], ['doc', 'Full document']];

  const node = (title, text, opts) => {
    opts = opts || {};
    return h('div', { class: 'node' + (opts.hl ? ' hl' : '') + (opts.store ? ' store' : '') },
      opts.tag ? h('span', { class: 'tag' }, opts.tag) : null,
      opts.live ? h('span', { class: 'live', 'data-live': opts.live }) : null,
      h('h4', null, title), text ? h('p', null, text) : null, opts.extra || null);
  };
  const flow = (...kids) => h('div', { class: 'flow' }, kids);
  const arrow = t => h('div', { class: 'arrow', title: t || '' }, '→');
  const down = t => h('div', { class: 'between' }, '▼ ' + (t || ''));
  const card = (title, sub, ...kids) => h('div', { class: 'card', style: { marginBottom: '16px' } },
    h('div', { class: 'card-head' }, h('h3', null, title), sub ? h('span', { class: 'muted small' }, sub) : null), kids);
  const code = t => h('code', null, t);
  const isProps = x => x && typeof x === 'object' && !(x instanceof Node) && !Array.isArray(x);
  const p = (...k) => isProps(k[0]) ? h('p', k[0], k.slice(1)) : h('p', null, k);
  const table = (head, rows, cls) => h('div', { class: 'table-wrap' }, h('table', { class: cls || '' },
    h('thead', null, h('tr', null, head.map(t => h('th', null, t)))),
    h('tbody', null, rows.map(r => h('tr', null, r.map((c, i) => h('td', { class: i === 0 ? 'nowrap' : '' }, c)))))));
  const fixed = n => (Math.round(n * 1e6) / 1e6).toString();

  /* ---------- pictures ----------
     Hand-drawn SVG built with the DOM (no library, nothing external, CSP-safe).  Every colour
     comes from the theme variables in style.css (.dg rules), so light and dark both work, and
     the viewBox scales the drawing to the card. */
  const SVGNS = 'http://www.w3.org/2000/svg';
  function sv(tag, attrs, ...kids) {
    const el = document.createElementNS(SVGNS, tag);
    for (const [k, v] of Object.entries(attrs || {})) if (v != null && v !== false) el.setAttribute(k, String(v));
    for (const k of kids.flat(Infinity)) if (k != null && k !== false) el.append(k instanceof Node ? k : document.createTextNode(String(k)));
    return el;
  }
  /* A box: bold title, then short lines.  cls: hl (a daemon / the key step), store (files), dim. */
  function dbox(x, y, w, hh, title, lines, cls, live) {
    return sv('g', { class: 'box ' + (cls || '') },
      sv('rect', { x, y, width: w, height: hh, rx: 10 }),
      sv('text', { x: x + 14, y: y + 24, class: 't' }, title),
      (lines || []).filter(l => l != null).map((l, i) => sv('text', { x: x + 14, y: y + 44 + i * 17, class: 's' }, l)),
      live ? sv('circle', { cx: x + w - 18, cy: y + 19, r: 6, class: 'dot', 'data-live-dot': live }, sv('title', null, '')) : null);
  }
  /* An arrow along path d, with an optional label at (lx, ly). */
  const dedge = (id, d, label, lx, ly, anchor, dashed) => [
    sv('path', { d, class: 'edge' + (dashed ? ' dash' : ''), 'marker-end': `url(#ah-${id})` }),
    label ? sv('text', { x: lx, y: ly, class: 'lbl', 'text-anchor': anchor || 'middle' }, label) : null];
  const dhead = (x, y, t) => sv('text', { x, y, class: 'hd' }, t);
  function dsvg(id, w, hh, aria, ...kids) {
    return sv('svg', { class: 'dg', viewBox: `0 0 ${w} ${hh}`, role: 'img', 'aria-label': aria, preserveAspectRatio: 'xMidYMin meet' },
      sv('defs', null, sv('marker', { id: 'ah-' + id, viewBox: '0 0 10 10', refX: 9, refY: 5, markerWidth: 7, markerHeight: 7, orient: 'auto-start-reverse' },
        sv('path', { d: 'M0,0 L10,5 L0,10 z', class: 'ah' }))),
      kids);
  }
  const figure = (svg, caption) => h('figure', { class: 'dg-fig' }, h('div', { class: 'dg-wrap' }, svg), h('figcaption', null, caption));
  const shortName = s => String(s || '?').split('/').pop();

  function systemDiagram() {
    const hosts = ((A && A.hosts) || []).map(x => x.label);
    const hostLine = hosts.length ? (hosts.join(', ').length > 26 ? hosts.length + ' more hosts' : hosts.join(', ')) : 'any MCP host';
    const E = (...a) => dedge('sys', ...a);
    return dsvg('sys', 1180, 530,
      'System overview: clients call rag_search.api, which talks to the search daemon for queries and the indexer daemon for jobs; the indexer spawns a worker that reads the source folders and writes indexer_workspace; publishing hard-links the workspace into serving, which the search daemon loads.',
      dhead(20, 20, 'CLIENTS · SHORT-LIVED'), dhead(320, 20, 'SHARED FRONT DOOR'), dhead(660, 20, 'DAEMONS · ONE OF EACH'), dhead(990, 20, 'FILES ON DISK'),
      // clients
      dbox(20, 60, 210, 64, 'rag-search CLI', ['administrator (“cli”)', 'search · index · access']),
      dbox(20, 138, 210, 64, 'Claude Desktop / Code', ['MCP · --profile claude', 'seven rag_* tools']),
      dbox(20, 216, 210, 64, 'Other MCP hosts', [hostLine, 'same tools, own identity']),
      dbox(20, 294, 210, 64, 'This dashboard', ['same api calls as the CLI', 'loopback + token'], 'hl'),
      E('M230,92 H316', 'commands', 274, 86), E('M230,170 H316', 'MCP · stdio', 274, 164),
      E('M230,248 H316', 'MCP · stdio', 274, 242), E('M230,326 H316', 'HTTP', 274, 320),
      // front door
      dbox(320, 60, 240, 298, 'rag_search.api', ['the only code that knows', 'the daemon sockets', '', '• starts a daemon on demand', '• one protocol for all clients:', '  JSON lines, Unix sockets', '• list and grep still work', '  when the search daemon', '  is down (dashed path)', '', 'loads no model itself'], 'hl'),
      dbox(320, 400, 240, 108, 'Settings & rules', ['config.json · access.json', 'locations.json', 'written by CLI + dashboard'], 'store'),
      E('M440,358 V396', 'settings · rules', 448, 383, 'start'),
      // daemons
      dbox(660, 60, 230, 110, 'Search daemon', ['both models + the published', 'index, held in memory', 'checks access rules', 'search · grep · list · reload'], 'hl', 'search'),
      dbox(660, 246, 230, 96, 'Indexer daemon', ['one job at a time', 'start · status · cancel', 'stdlib only, tiny'], 'hl', 'indexer'),
      dbox(660, 412, 230, 96, 'Worker process', ['one per job, killable', 'convert · chunk · embed', 'docling + embedding model']),
      E('M560,115 H656', 'queries', 608, 109), E('M560,294 H656', 'jobs', 608, 288),
      E('M760,246 V174', 'reload', 768, 214, 'start'), E('M775,342 V408', 'spawns · can kill', 783, 380, 'start'),
      // files
      dbox(990, 60, 170, 110, 'serving/', ['gen-N ← current', 'published, never', 'edited in place'], 'store'),
      dbox(990, 246, 170, 96, 'indexer_workspace/', ['markup/ + index/', 'per collection'], 'store'),
      dbox(990, 412, 170, 96, 'Source folders', ['docs/ + locations', 'read, never changed'], 'store'),
      E('M990,115 H894', 'loads', 940, 109), E('M1075,246 V174', 'publish', 1083, 214, 'start'),
      E('M890,440 H948 V320 H986', 'writes', 956, 385, 'start'), E('M990,480 H894', 'read only', 940, 474),
      // fallback
      E('M440,60 V44 H1075 V56', 'list · grep read the published files directly when the search daemon is down', 760, 39, 'middle', true));
  }

  /* hardware chip drawn on a box: dark GPU, outlined CPU */
  const hwchip = (x, y, kind) => sv('g', { class: 'hwc ' + kind },
    sv('rect', { x, y, width: 36, height: 17, rx: 5 }), sv('text', { x: x + 18, y: y + 12.5, 'text-anchor': 'middle' }, kind.toUpperCase()));

  /* a setting's value in effect, from the numbered pipeline (/api/architecture -> stages) */
  const stageOf = id => ((A && A.stages) || []).find(s => s.id === id) || { settings: [] };
  const setting = (id, key) => { const r = stageOf(id).settings.find(x => x.id === key); return r ? PL.valueText(r) : '?'; };

  /* ONE pipeline: every box carries the number it has everywhere else (Indexing tab, Settings, page traces, logs). */
  function pipelineDiagram() {
    const live = RS.state.live && RS.state.live.conversion, t = (live && live.totals) || {};
    const br = t.branches || {}, out = t.outcomes || {};
    const n = (...ks) => ks.reduce((a, k) => a + (br[k] || 0), 0);
    const e = A.models.embedding, c = A.chunking;
    const E = (...a) => dedge('idx', ...a);
    return dsvg('idx', 1180, 660,
      'The indexing pipeline as one numbered list. 1 Discover lists the files of every collection; 2 Fingerprint skips unchanged documents; 3 Convert turns a file into page-marked Markdown through five steps: 3.1 Profile looks at every page, 3.2 Read sends each page to the reader it needs (a: docling for text pages, b: the document reader for scans and photos, c: the same reader for large pictures), 3.3 Gate checks the result, 3.4 Repair re-reads a suspect table cell of a scanned page, 3.5 Reconcile joins tables that run across pages. Then 4 Chunk, 5 Embed with the model loaded once per run, 6 Write, 7 Merge per collection and 8 Publish.',
      dhead(20, 24, 'PER RUN, THEN PER DOCUMENT'),
      dbox(20, 40, 170, 90, '1 · Discover', ['every collection:', 'registered folders', 'and imports'], 'store'),
      dbox(240, 40, 180, 90, '2 · Fingerprint', ['SHA-256 of the file +', 'chunk, model and', 'conversion settings'], 'hl'),
      sv('g', { class: 'box hl' }, sv('polygon', { points: '515,40 575,85 515,130 455,85' }),
        sv('text', { x: 515, y: 82, class: 't', 'text-anchor': 'middle' }, 'same as'), sv('text', { x: 515, y: 99, class: 't', 'text-anchor': 'middle' }, 'last time?')),
      dbox(640, 55, 170, 60, 'Skip: keep it', ['seconds for a collection'], 'dim'),
      dbox(880, 40, 280, 90, 'One numbered pipeline', ['the numbers are the same in the Indexing', 'tab, Settings, page traces and logs', 'CPU outlined · GPU dark'], 'store'),
      E('M190,85 H236', 'each file', 213, 79), E('M420,85 H451', null), E('M575,85 H636', 'yes', 605, 79), E('M515,130 V166', 'no', 523, 154, 'start'),
      // 3 · Convert: the container
      sv('rect', { x: 20, y: 170, width: 1140, height: 310, rx: 14, class: 'grp' }),
      sv('text', { x: 36, y: 192, class: 'hd' }, '3 · CONVERT, PAGE BY PAGE · several documents side by side'),
      dbox(44, 288, 150, 76, '3.1 · Profile', [t.pages ? `${num(t.pages)} pages this run` : 'text layer? scan? photo?', 'pictures · ink · script'], 'hl'), hwchip(152, 294, 'cpu'),
      dbox(250, 204, 250, 76, '3.2a · docling', [br.digital || br.office ? `${num(n('digital', 'office', 'copy'))} pages: text, Office, HTML` : 'text pages · Office · HTML', `OCR ${setting('3.2', 'indexer.ocr')} · tables ${setting('3.2', 'indexer.table_mode')}`], 'hl'), hwchip(458, 210, 'cpu'),
      dbox(250, 288, 250, 76, '3.2b · document reader', [br.raster || br.image || br.fallback ? `${num(n('raster', 'image'))} scanned · ${num(br.fallback || 0)} by docling OCR` : 'scans and photos · image files', 'docling OCR when it cannot run'], 'hl'), hwchip(458, 294, 'gpu'),
      dbox(250, 372, 250, 76, '3.2c · pictures', [br.embedded ? `${num(br.embedded)} pages with a picture read` : 'large pictures on text pages', 'same reader as 3.2b'], 'hl'), hwchip(458, 378, 'gpu'),
      dbox(560, 288, 170, 76, '3.3 · Gate', ['coverage · script ·', 'tables · balances'], 'hl'), hwchip(688, 294, 'cpu'),
      dbox(820, 204, 170, 76, '3.4 · Repair', [t.repair_tried ? `${num(t.repaired_cells || 0)} of ${num(t.repair_tried)} cells fixed` : 'suspect cell of a scan', 're-read; sums must agree'], 'hl'), hwchip(948, 210, 'gpu'),
      dbox(1000, 288, 140, 76, '3.5 · Reconcile', [t.merged_tables ? `${num(t.merged_tables)} tables joined` : 'tables across', 'a page break'], 'hl'), hwchip(1098, 340, 'cpu'),
      E('M194,326 H246', null), E('M222,326 V242 H246', null), E('M222,326 V410 H246', null),
      E('M500,242 H530 V326 H556', null), E('M500,326 H556', null), E('M500,410 H530 V326 H556', null),
      E('M730,326 H770 V242 H816', 'suspect', 793, 236, 'middle'), E('M730,346 H996', 'pass', 860, 340, 'middle'),
      E('M990,242 H1070 V284', 'repaired', 1076, 264, 'start'),
      sv('text', { x: 36, y: 472, class: 's' }, 'Outcome per page: pass · repaired · low confidence (kept, flagged) · no text · error.  Colours on the Indexing tab show how each page was read.'),
      sv('text', { x: 126, y: 500, class: 's' }, 'page-marked Markdown + a trace of every page'), E('M115,480 V526', null),
      // 4 .. 8
      dhead(132, 520, 'THEN: 4 PER DOCUMENT · 5 ONCE PER RUN, MODEL LOADED A SINGLE TIME · 6 PER DOCUMENT · 7–8 PER COLLECTION'),
      dbox(20, 536, 190, 100, '4 · Chunk', [`~${c.size} tokens, ~${c.overlap} overlap`, 'never across a page', 'page + heading kept'], null), hwchip(168, 542, 'cpu'),
      dbox(257, 536, 190, 100, '5 · Embed', [shortName(e.name), `${e.dim || 1024}-d vectors, batch ${e.batch}`, `≤ ${e.max_seq} tokens per chunk`], 'hl'), hwchip(405, 542, 'gpu'),
      dbox(494, 536, 190, 100, '6 · Write', ['nodes.json', 'embeddings.npy', 'index.meta.json last'], 'store'),
      dbox(731, 536, 190, 100, '7 · Merge', ['_all/ per collection', 'concatenate, no', 're-embedding']),
      dbox(968, 536, 190, 100, '8 · Publish', ['hard-link → serving/gen-N', 'switch “current” symlink', `keep ${A.index.generations_kept}; search reloads`], 'hl'),
      E('M210,586 H253', 'new chunks', 232, 580), E('M447,586 H490', null), E('M684,586 H727', null), E('M921,586 H964', null));
  }

  function searchDiagram() {
    const m = A.models, f = A.fusion;
    const E = (...a) => dedge('srch', ...a);
    return dsvg('srch', 1180, 300,
      'Search pipeline: the query passes an access check, then a keyword (BM25) and a vector (cosine) branch each rank chunks in every allowed collection; reciprocal rank fusion merges the lists into one pool; the reranker re-orders the best candidates and the top k are returned.',
      dbox(20, 105, 130, 90, 'Query', ['text + top_k', 'collections'], 'store'),
      dbox(190, 105, 150, 90, 'S1 · Access check', ['forbidden =', 'unknown, like', 'a typo'], 'hl'),
      dbox(390, 30, 220, 100, 'S2 · Keyword · BM25', ['exact terms, IDs, versions', `k1 ${A.keyword.k1} · b ${A.keyword.b}`, 'best pool_n per collection'], null),
      dbox(390, 170, 220, 100, 'S3 · Vectors · cosine', ['meaning, paraphrases', shortName(m.embedding.name), 'best pool_n per collection'], null),
      dbox(660, 105, 150, 90, 'S4 · Fuse · RRF', [`Σ 1/(${f.k} + rank)`, 'one pool across', 'all collections'], 'hl'),
      dbox(850, 105, 150, 90, 'S5 · Rerank', [shortName(m.reranker.name), m.reranker.enabled ? 'query + chunk read' : 'switched OFF:', m.reranker.enabled ? 'together, 0 – 1' : 'RRF order used'], m.reranker.enabled ? 'hl' : 'dim'),
      dbox(1040, 105, 120, 90, 'S6 · Top k', ['page, heading,', 'snippet, score'], 'store'),
      E('M150,150 H186', null),
      E('M340,150 H365 V80 H386', null), E('M340,150 H365 V220 H386', null),
      E('M610,80 H635 V135 H656', null), E('M610,220 H635 V165 H656', null),
      E('M810,150 H846', `≤ ${m.reranker.cap}`, 828, 144), E('M1000,150 H1036', 'top_k', 1018, 144),
      sv('text', { x: 20, y: 292, class: 's' }, 'grep is a separate path: a regular-expression scan of the converted Markdown, no models, answers even while they load.'));
  }

  /* Where the pipeline tunables (Settings tab, `rag-search config
     set`) actually persist -- one file, one section per daemon, environment variables as the
     final override layer.  Data comes straight from spec.TUNABLES via /api/architecture's
     config_storage, so this never drifts from what config.py/spec.py actually do; the
     what-it-means/impact text for each tunable lives on the Settings/Models tabs, not here. */
  function configStorageCard() {
    const cs = A.config_storage; if (!cs) return null;
    const sectionRows = Object.entries(cs.sections).map(([section, keys]) =>
      [code(section), h('span', null, keys.flatMap((k, i) => i ? [', ', code(k.key)] : [code(k.key)]))]);
    return card('Where the tunables are stored', cs.file,
      p('Every pipeline tunable -- the ', h('a', { href: '#/settings' }, 'Settings'), ' tab (arranged by pipeline stage), or ',
        code('rag-search config set'), ' -- lives in this one JSON file, one section per daemon:'),
      table(['config.json section', 'tunables stored there'], sectionRows),
      h('h4', { style: { marginTop: '14px' } }, 'What wins when a tunable is set in more than one place'),
      h('ol', { class: 'tight' }, cs.precedence.map(x => h('li', null, x))),
      h('h4', { style: { marginTop: '14px' } }, 'Who reads each section, and when'),
      table(['Section', 'Read by'], Object.entries(cs.who_reads_it).map(([k, v]) => [code(k), v])));
  }

  /* Where a collection's documents come from: registered locations and imports. */
  function sourcesCard() {
    const S = A.sources || { locations: [], imported: [] };
    const rows = [
      ...S.locations.map(l => [code(l.collection), code(l.folder), 'registered location']),
      ...S.imported.map(n => [code(n), '–', 'imported bundle: ready-made index, no source documents here'])];
    return card('Where documents come from', 'the worker reads these, never writes them',
      table(['Collection', 'Folder', 'Kind'], rows),
      p({ class: 'small muted' }, 'Add a location, import or export a bundle, or delete a collection on the ',
        h('a', { href: '#/collections' }, 'Collections'), ' tab. Delete removes only the workspace (converted Markdown and index); a collection whose documents are still in place is built again on the next indexing run.'));
  }

  /* ---------- 1. system overview ---------- */
  function sysView() {
    return [
      card('How the pieces fit', 'the dots on the daemons are live',
        figure(systemDiagram(), 'Reads and writes never meet: indexing writes only to indexer_workspace/, searches read only the published serving/ generation, and “publish” is the one step that moves data from one side to the other. Source folders are only ever read.')),
      sourcesCard(),
      card('Design rules that shape everything else', null, h('ul', { class: 'tight' },
        h('li', null, 'Front-ends never load a model. Only the search daemon (queries) and the worker (indexing) do.'),
        h('li', null, 'Only the indexer writes indexes; only ', code('publish'), ' makes them visible; searches always see one consistent generation.'),
        h('li', null, 'Source documents are never changed or deleted. Deleting a collection removes only its converted Markdown and index from the workspace.'),
        h('li', null, 'Everything is local: no network calls at run time. Models are downloaded only when you ask (Models tab, or rag-search models / setup) and then come from the local cache.'),
        h('li', null, 'Restricting a collection changes what a client can list, search or grep immediately; a restricted collection looks like a missing one.'))),
      configStorageCard()];
  }

  /* The stage reference: one row per stage, from the registry (stages.py) with the values in effect. */
  function stageReferenceCard(pipeline, title, sub) {
    const list = ((A.stages) || []).filter(s => (pipeline === 'search') === s.id.startsWith('S'));
    return card(title, sub, h('div', { class: 'table-wrap' }, h('table', null,
      h('thead', null, h('tr', null, ['#', 'Stage', 'Runs', 'What it does', 'Settings it uses (value in effect)'].map(t => h('th', null, t)))),
      h('tbody', null, list.map(s => h('tr', { id: 'arch-' + s.id, class: s.parent ? 'sub-row' : '' },
        h('td', { class: 'nowrap' }, h('b', { class: 'stage-id' }, s.id)),
        h('td', { class: 'nowrap' }, s.parent ? '\u00a0\u00a0' + s.name : h('b', null, s.name), s.optional ? h('span', { class: 'muted small' }, ' (optional)') : null),
        h('td', { class: 'nowrap small' }, PL.hw(s.where), h('div', { class: 'muted' }, PL.SCOPE[s.scope] || '')),
        h('td', { class: 'small', style: { whiteSpace: 'normal', minWidth: '260px' } }, s.what),
        h('td', { class: 'small', style: { whiteSpace: 'normal' } }, (s.settings || []).length ? s.settings.map(r => h('div', null, r.label + ': ', h('b', { class: 'mono' }, PL.valueText(r)), ' ', PL.sourceChip(r))) : h('span', { class: 'muted' }, (s.constants || []).length ? 'fixed: ' + s.constants.map(c => `${c.label} ${c.value}`).join(' · ') : '–'))))))));
  }

  /* ---------- 2. indexing ---------- */
  function idxView() {
    const c = A.chunking, conv = A.models.conversion;
    const convText = conv.error ? conv.error : Object.entries(conv).filter(([k]) => k !== 'tool').map(([k, v]) => `${k}: ${typeof v === 'object' ? JSON.stringify(v) : v}`).join(' · ');
    return [
      card('The indexing pipeline, one numbered list', 'the same numbers are used in the Indexing tab, Settings, page traces and logs',
        figure(pipelineDiagram(), 'Each document is handled on its own: fingerprinted, then converted (3.1–3.5) and chunked only if something changed. Conversion is a stage of the pipeline like any other: its steps have numbers, settings, events and timings of their own. The embedding model is loaded once per run for all new chunks. index.meta.json is written last, so a half-written document is never mistaken for a finished one, and nothing becomes searchable until publish.')),
      stageReferenceCard('indexing', 'Indexing stage reference', 'what each stage does, where it runs, which settings it owns and the value in effect now'),
      h('div', { class: 'split' },
        card('Chunking in detail', null, h('ul', { class: 'tight' },
          h('li', null, 'Split at the page markers first, so a chunk never straddles two pages.'),
          h('li', null, 'Blocks are paragraphs; code fences and tables are kept whole where they fit.'),
          h('li', null, 'Oversized blocks are split: tables by rows (header row repeated in each piece), prose by sentences, then words.'),
          h('li', null, `Greedy packing to about ${c.size} estimated tokens with about ${c.overlap} tokens of overlap between neighbours.`),
          h('li', null, 'Size estimate: ', code(c.estimator), ' (no tokenizer needed).'),
          h('li', null, 'Each chunk records page, nearest heading, file, collection and source path.'))),
        card('Keyword index (BM25)', 'built in memory, not stored', h('ul', { class: 'tight' },
          h('li', null, 'The search daemon builds it when it loads a collection, from the chunk text in nodes.json.'),
          h('li', null, 'Tokens are lower-cased words. Compound tokens such as ', code('svm-name'), ', ', code('9.16.1'), ' or ', code('a/b'), ' are kept whole and also split into parts, so exact identifiers and their pieces both match.'),
          h('li', null, `Okapi BM25, k1 = ${A.keyword.k1}, b = ${A.keyword.b}. A tokenizer change (${A.keyword.tokenizer}) re-indexes.`)))),
      card('When is a document re-indexed?', 'the redone stages are numbered',
        table(['Changes', 'What is redone'], [
          ['the source file (SHA-256)', '3 convert · 4 chunk · 5 embed · 6 write'],
          ['chunk size / overlap, chunker version, tokenizer version, index format', '4 chunk · 5 embed · 6 write (the Markdown is reused)'],
          ['embedding model', '5 embed · 6 write (chunks are reused)'],
          ['conversion settings (OCR, tables, routing, document reader or repair switched off)', '3 convert · 4 chunk · 5 embed · 6 write'],
          ['nothing', '2 fingerprint only: skipped, seconds for a whole collection']])),
      card('Conversion profile in effect (stage 3)', null, p(convText || '–'),
        p({ class: 'small muted' }, 'A change here makes documents count as stale, so the next “Index new & changed” run redoes them.'))];
  }

  /* ---------- 3. files ---------- */
  function fieldTable(fields) { return table(['Field', 'Meaning'], Object.entries(fields).map(([k, v]) => [code(k), v])); }
  function fmtView() {
    const ix = A.index;
    return [
      card('Where the index lives', 'under the home folder: ' + A.paths.home,
        h('pre', null, `indexer_workspace/                       written by the worker (and collection import / delete)
  markup/<collection>/<doc>.md            page-annotated Markdown (+ .md.sha256 of the source)
  index/<collection>/<doc>/
    nodes.json                            the chunks: id, text, metadata
    embeddings.npy                        float32 matrix, one row per chunk, same order as nodes.json
    index.meta.json                       written LAST: fingerprint + sizes + timings
  index/<collection>/_all/                merged view of the collection
    nodes.json  embeddings.npy  merge.manifest.json
  index/<collection>/collection.origin.json   only for imported collections

serving/                                  what the search daemon reads
  gen-000042/                             immutable, hard-linked; last ${ix.generations_kept} kept
    catalog.json                          generation, content_sha, model, collections, documents
    index/<collection>/_all/…   markup/…
  current -> gen-000042                   symlink, switched atomically`)),
      card('Vector format', null, h('ul', { class: 'tight' },
        h('li', null, code('embeddings.npy'), ': NumPy ', code('float32'), ' array of shape (chunks, ' + (A.models.embedding.dim || 'dim') + '). Row i belongs to node i of nodes.json.'),
        h('li', null, 'Vectors are L2-normalised, so the dot product of two vectors is their cosine similarity.'),
        h('li', null, 'There is no ANN structure (no HNSW/IVF): the whole matrix is held in memory and searched exactly with one matrix-vector product. At this scale that is both exact and fast, and there is nothing to tune or rebuild.'),
        h('li', null, 'The search daemon loads text and vectors into memory once per collection and builds the keyword index from the text; memory use is on the Overview tab.'))),
      h('div', { class: 'split' },
        card('nodes.json', 'format ' + ix.format, p(code('{"format": 1, "nodes": [ … ]}'), ' one entry per chunk:'), fieldTable(ix.node_fields),
          h('pre', null, `{ "id": "…", "text": "## Enabling MFA\\n…",
  "metadata": { "page_label": "12", "heading": "Enabling MFA", /* "confidence": "low" appears only on flagged pages */
    "file_name": "Authentication_and_access_control",
    "source_name": "Authentication_and_access_control.pdf",
    "collection": "security", "doc_path": "…", "src_path": "/…/docs/security/…pdf" } }`)),
        card('index.meta.json', 'the completeness marker', fieldTable(ix.meta_fields))),
      card('catalog.json (per generation)', null, fieldTable(ix.catalog_fields),
        p({ class: 'small muted' }, 'The search daemon compares each collection’s manifest_sha and the model with what it already holds; matching collections are reused from memory and only changed ones are loaded.'))];
  }

  /* ---------- 4. search ---------- */
  function rrfCalc() {
    const K = A.fusion.k;
    const rows = [['exact term in the text', 1, 9], ['paraphrase, no shared words', '', 1], ['fairly good in both', 3, 4], ['in another collection, good keyword hit', 2, '']];
    const wrap = h('div');
    const inputs = rows.map(r => [h('input', { type: 'number', min: 1, value: r[1], placeholder: '–', style: { width: '64px' } }),
      h('input', { type: 'number', min: 1, value: r[2], placeholder: '–', style: { width: '64px' } })]);
    const out = h('tbody');
    const calc = () => {
      const vals = rows.map((r, i) => {
        const a = parseInt(inputs[i][0].value, 10), d = parseInt(inputs[i][1].value, 10);
        const pa = a > 0 ? 1 / (K + a) : 0, pd = d > 0 ? 1 / (K + d) : 0;
        return { name: r[0], a, d, pa, pd, s: pa + pd };
      });
      const order = vals.slice().sort((x, y) => y.s - x.s);
      fill(out, vals.map(v => h('tr', null, h('td', null, v.name), h('td', null, inputs[vals.indexOf(v)][0]), h('td', null, inputs[vals.indexOf(v)][1]),
        h('td', { class: 'num mono' }, v.a > 0 ? `1/(${K}+${v.a}) = ${fixed(v.pa)}` : '–'), h('td', { class: 'num mono' }, v.d > 0 ? `1/(${K}+${v.d}) = ${fixed(v.pd)}` : '–'),
        h('td', { class: 'num mono' }, h('b', null, fixed(v.s))), h('td', { class: 'num' }, chip('#' + (order.indexOf(v) + 1), order.indexOf(v) === 0 ? 'ok' : '')))));
    };
    for (const pair of inputs) for (const i of pair) i.addEventListener('input', calc);
    wrap.append(h('div', { class: 'table-wrap' }, h('table', null,
      h('thead', null, h('tr', null, ['Chunk', 'Rank in keyword list', 'Rank in vector list', 'Keyword share', 'Vector share', 'RRF score', 'Fused'].map((t, i) => h('th', { class: i > 0 ? 'num' : '' }, t)))), out)));
    calc();
    return wrap;
  }
  function searchView() {
    const f = A.fusion, m = A.models, L = A.limits, pools = A.pools;
    return [
      card('One query, step by step', 'every collection the client may use is searched; all share one candidate pool',
        figure(searchDiagram(), `Keyword search finds exact identifiers, vector search finds paraphrases; fusion needs no score calibration because it uses only ranks. Each branch keeps max(4 × top_k, 20) chunks per collection; at most ${m.reranker.cap} reach the reranker, which reads query and chunk together. Default top_k is ${L.top_k_default}, at most ${L.top_k_max}; snippets are cut to ${L.snippet_chars} characters.`)),
      stageReferenceCard('search', 'Search stage reference', 'S1–S6: what each stage does, where it runs, which settings it uses and the value in effect now'),
      card('Why fuse ranks instead of scores?', null,
        p('BM25 scores are unbounded and depend on the collection; cosine similarities sit in a narrow band. Adding them needs careful calibration. RRF only uses ', h('i', null, 'positions'), ', so the two lists combine without tuning: a chunk near the top of both lists beats one that is first in only one, and the constant ',
          code('k = ' + f.k), ' keeps a single first place from dominating.'),
        h('div', { class: 'formula' }, `RRF(chunk) = 1/(${f.k} + rank_keyword) + 1/(${f.k} + rank_vector)   (rank starts at 1; a missing list adds 0)`),
        h('h4', { style: { marginTop: '14px' } }, 'Try it: change the ranks'), rrfCalc()),
      card('Pool sizes', 'how many candidates each stage works on',
        table(['top_k', 'per retriever per collection', 'into the reranker'], Object.entries(pools).map(([k, v]) => [k, num(v.per_retriever), num(v.to_reranker)])),
        p({ class: 'small muted' }, 'per retriever = max(4 × top_k, 20). Into the reranker = min(max(3 × top_k, 15), ' + m.reranker.cap + '). The cross-encoder is too slow to read every chunk, so it only re-orders the survivors of fusion.')),
      card('Exact-text search: grep', null, p(code('rag_grep'), ' / ', code('rag-search grep'), ' is a separate path: a regular-expression scan of the converted Markdown, run in an isolated child process with a time budget. Use it for exact strings, error codes and IDs; it needs no models and answers even while they load.'))];
  }

  /* ---------- 5. models ---------- */
  function modelsView() {
    const m = A.models, e = m.embedding, r = m.reranker, conv = m.conversion;
    const d = RS.state.live && RS.state.live.daemons && RS.state.live.daemons.search; const mem = (d && d.memory && d.memory.models_bytes) || {};
    const memText = Object.keys(mem).length ? Object.entries(mem).map(([k, v]) => `${k} ${bytes(v)}`).join(', ') : 'loaded when the search daemon is running';
    const models = [
      ['Embedding', e.name, e.kind, [['Role', 'turns text into a vector so similar meaning = nearby vectors. Used for every chunk at indexing time and for every query.'],
        ['Dimensions', String(e.dim || 1024) + ', L2-normalised float32'], ['Input length', `up to ${e.max_seq} tokens per chunk (chunks are ~${A.chunking.size})`], ['Batch size', String(e.batch)],
        ['Search', 'exact cosine (matrix product), no approximate index'], ['Environment', 'the Models tab or rag-search models set; RAG_SEARCH_MODEL overrides; RAG_SEARCH_MAX_SEQ, RAG_SEARCH_EMBED_BATCH']]],
      ['Reranker', r.name, r.kind, [['Role', 'second opinion on the best candidates: reads query and chunk together, so it catches what the two cheap first-stage scores miss.'],
        ['Output', 'relevance in [0, 1] (a sigmoid of the logit, or the probability of “yes” for an LLM reranker)'], ['Candidates per query', `max(3 × top_k, 15), at most ${r.cap}`], ['Input length', r.max_len ? `query + chunk up to ${r.max_len} tokens` : 'model default'], ['Batch size', String(r.batch)],
        ['Status', r.enabled ? 'on' : 'off (RAG_SEARCH_RERANK=0)'], ['Environment', 'the Models tab or rag-search models set; RAG_SEARCH_RERANK_MODEL overrides; RAG_SEARCH_RERANK']]],
      ['Document conversion', conv.tool, 'layout + table + OCR pipeline (not a search model)', [['Role', 'turns PDFs, Office files, HTML and images into page-annotated Markdown.'],
        ['Settings', conv.error || Object.entries(conv).filter(([k]) => k !== 'tool').map(([k, v]) => `${k}: ${typeof v === 'object' ? JSON.stringify(v) : v}`).join(' · ') || 'defaults'],
        ['Runs in', 'the indexing worker only; the search daemon never loads it']]]];
    return [
      card('Models in use', 'read from the running configuration'),
      ...models.map(([t, name, kind, rows]) => card(t, kind, h('p', null, h('b', null, name)), kv(rows))),
      card('Memory in the search daemon', null, p(memText)),
      p({ class: 'small muted' }, 'Models are downloaded once into the local Hugging Face cache; nothing is sent anywhere at run time. Switch them in the Models tab or with rag-search models: a new reranker is used at once; a new embedding model means every document is embedded again (chunks are reused, and search keeps using the old index until the new one is complete).')];
  }

  /* ---------- 6. daemons ---------- */
  function daemonsView() {
    return [
      card('Who talks to whom', 'live status on the daemon boxes',
        h('div', { class: 'lanes' },
          h('div', { class: 'lane' }, h('h4', null, 'Clients'),
            node('CLI', 'client “cli”: administrator, may use every collection, the only writer of access.json besides this dashboard.'),
            node('MCP adapter: Claude', 'one process per host; identity fixed by --profile claude.'),
            ((A && A.hosts) || []).map(x => node('MCP adapter: ' + x.label, `identity fixed by --profile ${x.name}; tools identical to Claude’s.`)),
            node('Dashboard', 'this page. Runs its own searches as “cli” or “as” another client, flagged so they are not counted as that client connecting.', { hl: true })),
          h('div', { class: 'lane' }, h('h4', null, 'Search daemon'),
            node('Search daemon', 'single instance (flock on run/search.alive). Socket run/search.sock. Reloads generations off to the side and swaps them in with one pointer change.', { live: 'search', hl: true }),
            h('div', { class: 'between' }, 'state: starting → loading_models → loading_index → ready (or error)'),
            node('What it keeps in memory', 'per collection: chunk text, vectors, BM25 postings; both models; the access rules; the set of client names it has served.', { store: true })),
          h('div', { class: 'lane' }, h('h4', null, 'Indexer daemon'),
            node('Indexer daemon', 'single instance (flock on run/indexer.alive). Socket run/indexer.sock. Stdlib only, so it stays tiny. Only writer of job records.', { live: 'indexer', hl: true }),
            h('div', { class: 'between' }, 'spawns, tails events, can kill the whole process group'),
            node('Worker', 'python -m rag_search.core.worker <job-id>. Reads jobs/<id>.json, writes indexer_workspace/ and jobs/<id>.events.jsonl.', { tag: 'own process group' })))),
      h('div', { class: 'split' },
        card('Lifecycle', null, h('ul', { class: 'tight' },
          h('li', null, 'Both daemons start on demand the first time a client needs them (serialised by a start lock, so no duplicates), or at login via ', code('rag-search service install'), '.'),
          h('li', null, 'Default idle exit is 0 = never; ', code('idle_exit_seconds'), ' in config.json changes it.'),
          h('li', null, 'If the search daemon is down, list and grep read the published files directly; search needs the daemon (it holds the models).'),
          h('li', null, 'Index status and cancel never start the indexer just to report “idle”.'),
          h('li', null, 'Stopping the indexer cancels the active run; finished documents are kept.'))),
        card('Indexing job states', null, table(['State', 'Meaning'], [
          [pill('queued', ''), 'accepted, worker not started yet'], [pill('running', 'warn'), 'worker active: convert → chunk → embed → merge'],
          [pill('succeeded', 'ok'), 'everything indexed and published'], [pill('partial', 'warn'), 'some documents failed; the rest is published'],
          [pill('failed', 'bad'), 'nothing could be published'], [pill('cancelled', ''), 'cancelled by a user'], [pill('interrupted', 'warn'), 'daemon or machine stopped mid-run']]))),
      card('Wire protocol', 'protocol v1: one request per connection, newline-delimited JSON',
        h('pre', null, `→ {"v":1, "client":"claude", "action":"search", "query":"…", "top_k":5, "collections":[…], "wait_s":40}
← {"ok":true, "result":{ … results, timing … }}
← {"event":"progress", …} … {"event":"end", …}     (streaming actions such as follow)
← {"ok":false, "code":"warming_up", "error":"…"}`),
        p({ class: 'small muted' }, 'Error codes: bad_request, protocol_mismatch, warming_up, unavailable, forbidden, model_error, model_mismatch, busy, internal. A daemon rejects clients speaking a newer protocol and tells them to upgrade.')),
      card('Clients, identity and access', null, h('ul', { class: 'tight' },
        h('li', null, 'A client’s identity is declared, not authenticated: MCP hosts pass ', code('--profile NAME'), ' (any valid name; “cli”, “unknown” and “all” are reserved). It is a guard against mix-ups, not a security boundary between users on one machine.'),
        h('li', null, 'By default every collection is open to every client. ', code('rag-search access restrict hr claude'), ' limits “hr” to claude; no names = nobody; ', code('grant'), ' adds a client; ', code('restrict hr all'), ' opens it again.'),
        h('li', null, 'The MCP tools cannot change access. A restricted collection is invisible to other clients: it is missing from listings and searches for it fail with the same “unknown collection” error as a typo.'),
        h('li', null, 'A damaged access.json fails closed: every collection name still readable in it stays restricted to nobody, and the problem is reported by doctor and on the Overview tab.'))),
      card('What happens on a search from Claude Desktop', null, h('ol', { class: 'tight' },
        h('li', null, 'The host calls the MCP tool ', code('rag_search'), '; the adapter (identity “claude”) runs api.search in a worker thread.'),
        h('li', null, 'api connects to run/search.sock, starting the daemon if needed, and sends the request with client “claude”.'),
        h('li', null, 'The daemon applies the access rules, waits if models are still loading, then runs the pipeline on the current generation.'),
        h('li', null, 'The reply carries hits and timings; the adapter formats them as text with page numbers and headings for the model to cite.')))];
  }

  /* ---------- 7. full document ---------- */
  async function docView() {
    const box = refs.body;
    fill(box, empty('Loading…'));
    const r = await api('help');
    if (r.ok === false) { fill(box, h('div', { class: 'notice bad' }, r.error || 'cannot load')); return; }
    const d = r.architecture_doc || {}; const md = h('div', { class: 'md' });
    md.innerHTML = d.html || '<p>The architecture document is not packaged with this install.</p>';
    const toc = h('nav', { class: 'toc', 'aria-label': 'Contents' }, h('h4', null, 'Contents'), (d.toc || []).map(t => h('a', { href: '#/architecture/doc', class: 'l' + t.level, on: { click: e => { e.preventDefault(); const el = md.querySelector('#' + CSS.escape(t.id)); if (el) el.scrollIntoView({ behavior: 'smooth', block: 'start' }); } } }, t.title)));
    fill(box, h('div', { class: 'with-toc' }, toc, md));
    docLoaded = true;
  }

  const BUILD = { system: sysView, indexing: idxView, format: fmtView, search: searchView, models: modelsView, daemons: daemonsView };

  function refreshLive() {
    const s = searchStatus(), i = indexerStatus();
    for (const el of $$('[data-live]', refs.body)) {
      const st = el.dataset.live === 'search' ? s : i;
      patch(el, pill(st.label, st.cls, st.busy));
    }
    for (const el of $$('[data-live-dot]', refs.body)) {
      const st = el.dataset.liveDot === 'search' ? s : i;
      el.setAttribute('class', 'dot ' + (st.cls || ''));
      el.firstChild.textContent = (el.dataset.liveDot === 'search' ? 'search daemon: ' : 'indexer daemon: ') + st.label;
    }
  }

  function render() {
    for (const b of $$('button[data-s]', refs.nav)) b.classList.toggle('on', b.dataset.s === section);
    if (section === 'doc') { docView(); return; }
    if (!A) { fill(refs.body, empty('Loading…')); return; }
    fill(refs.body, h('div', { class: 'explain-wrap' }, BUILD[section]()));
    refreshLive();
  }

  function go(s) {
    section = s;
    if (location.hash !== '#/architecture/' + s) history.replaceState(null, '', '#/architecture/' + s);
    render();
  }

  RS.views.architecture = {
    init(root) {
      refs.nav = h('div', { class: 'subnav' }, SECTIONS.map(([k, t]) => h('button', { type: 'button', 'data-s': k, on: { click: () => go(k) } }, t)));
      refs.body = h('div');
      root.append(h('h2', null, 'Architecture'),
        p({ class: 'muted' }, 'How documents become searchable, how a query is answered, and how the daemons and clients cooperate. Numbers come from the running engine, so this page cannot drift from the code.'),
        refs.nav, refs.body);
      api('architecture').then(r => { if (r.ok === false) { fill(refs.body, h('div', { class: 'notice bad' }, r.error || 'cannot load')); return; } A = r; if (RS.current === 'architecture') render(); });
    },
    show() {
      const want = location.hash.split('/')[2];
      const next = SECTIONS.some(s => s[0] === want) ? want : section;
      if (next !== section || !refs.body.childNodes.length || refs.body.dataset.s !== next) { section = next; refs.body.dataset.s = next; render(); }
      else for (const b of $$('button[data-s]', refs.nav)) b.classList.toggle('on', b.dataset.s === section);
    },
    update() { if (A && section !== 'doc') refreshLive(); },
    tick() { if (A && section === 'models') { /* memory figures are static text; re-render only on entry */ } },
  };
})();
