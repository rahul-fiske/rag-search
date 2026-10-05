'use strict';
/* Collections & access: what is published, and which client may use which collection. */
(function () {
  let refs = {};
  const open = new Set();          // expanded collection rows

  function rows() {
    const cat = RS.state.catalog; if (!cat) return [];
    const info = {}; for (const c of (cat.list && cat.list.collections) || []) info[c.collection] = c;
    return ((cat.access && cat.access.collections) || []).map(r => ({ ...r, info: info[r.collection] || null }));
  }

  async function setAccess(collection, clients) {
    const r = await act('access', { action: 'restrict', collection, clients });
    if (r.ok === false) return;
    const res = r.result || {};
    toast(res.clients === null ? `${collection}: open to every client` : `${collection}: ${res.clients.length ? 'restricted to ' + res.clients.join(', ') : 'nobody (only your terminal and this dashboard)'}`, 'ok');
  }

  async function describeDialog(row) {
    const current = (row.info && row.info.description) || '';
    const ta = h('textarea', { rows: 5, maxlength: 500, style: { width: '100%' }, value: current,
      placeholder: 'What is in this collection, in a sentence or two (500 characters at most)' });
    const body = h('div', null, ta,
      h('p', { class: 'small muted' }, 'Shown to every client that can use this collection, in ', h('code', null, 'rag_list_collections'),
        ', so an agent knows what it holds without listing every document. Leave it empty to remove it. Same as ', h('code', null, 'rag-search describe COLLECTION "text"'), '.'));
    if (!await dialog(`Description of “${row.collection}”`, body, { okText: 'Save' })) return;
    const r = await act('describe', { collection: row.collection, description: ta.value });
    if (r.ok === false) return;
    toast(r.result && r.result.description ? `${row.collection}: description saved` : `${row.collection}: description removed`, 'ok');
  }

  function kindChip(r) {
    if (r.kind === 'location') return h('span', { class: 'chip accent', style: { marginLeft: '8px' }, title: r.folder || '' }, 'location');
    if (r.kind === 'imported') return h('span', { class: 'chip', style: { marginLeft: '8px' }, title: 'unpacked from a collection export: no source documents, never re-indexed' }, 'imported');
    return null;
  }

  /* ---------- add / import / export / delete ---------- */
  const field = (label, input, help) => h('label', { class: 'cd-field' }, h('span', null, label), input, help ? h('span', { class: 'muted small' }, help) : null);
  const textInput = (placeholder, value) => h('input', { type: 'text', placeholder, value: value || '', style: { width: '100%' }, spellcheck: 'false', autocomplete: 'off' });

  async function addDialog() {
    const name = textInput('e.g. notes'), folder = textInput('/Users/you/Documents/Notes');
    const now = h('input', { type: 'checkbox', checked: true });
    const body = h('div', null,
      h('p', { class: 'small muted', style: { marginTop: 0 } }, 'Index a folder where it is: its whole tree becomes one collection. rag-search only ever reads it -- nothing in it is changed, moved or deleted.'),
      field('Collection name', name, 'letters, digits, - and _; not used by another collection. Leave it empty to use the folder\u2019s own name'),
      field('Folder', folder, 'the full path (a browser cannot hand a page a folder you pick, so paste it -- in Finder: right-click the folder, hold ⌥ Option, “Copy … as Pathname”). Spaces and characters like @ are fine; quotes around it are ignored)'),
      h('label', { class: 'check', style: { display: 'flex', margin: '8px 0 0' } }, now, h('span', null, 'Index it now')));
    if (!await dialog('Add a collection', body, { okText: 'Add' })) return;
    const r = await act('collection/add-location', { name: name.value, folder: folder.value });
    if (r.ok === false) return;
    toast(`${r.result.collection}: ${r.result.folder} registered`, 'ok');
    if (now.checked) {
      const x = await act('index/start', { mode: 'new', path: r.result.collection });
      if (x.ok !== false) toast(x.already_running ? 'A run is already active: index it when it finishes' : 'Indexing started', x.already_running ? '' : 'ok');
    }
  }

  async function importDialog() {
    const file = textInput('/Users/you/Downloads/manuals.rag.tgz'), asName = textInput('(keep the name in the export)');
    const replace = h('input', { type: 'checkbox' });
    const body = h('div', null,
      h('p', { class: 'small muted', style: { marginTop: 0 } }, 'Add a collection someone exported with “Export” or rag-search collection export. It must have been embedded with this installation’s embedding model, or it is refused. It arrives open to every client: set Access… afterwards if needed.'),
      field('Export file (.rag.tgz)', file, 'the full path of the file on this computer'),
      field('Import as', asName, 'optional: another name, e.g. when the name is already taken here'),
      h('label', { class: 'check', style: { display: 'flex', margin: '8px 0 0' } }, replace, h('span', null, 'Replace an earlier import of the same collection')));
    if (!await dialog('Import a collection', body, { okText: 'Import' })) return;
    toast('Importing…');
    const r = await act('collection/import', { file: file.value, as_name: asName.value, replace: replace.checked });
    if (r.ok === false) return;
    const res = r.result;
    toast(`${res.replaced ? 'Replaced' : 'Imported'} ${res.collection}: ${num(res.documents)} documents, ${num(res.chunks)} chunks${res.note ? ' (' + res.note + ')' : ''}`, 'ok', 8000);
  }

  async function exportDialog(row) {
    const home = (RS.state.catalog && RS.state.catalog.home) || '';
    const folder = textInput(home ? home + '/exports' : 'folder');
    const body = h('div', null,
      h('p', { class: 'small muted', style: { marginTop: 0 } }, 'One .rag.tgz file with the collection’s index, converted Markdown and build details -- what someone else needs to import it. Source documents and access rules are not included; the converted text is, so share it only with people who may read these documents.'),
      field('Save in folder', folder, `leave empty for ${home ? home + '/exports' : 'the exports folder in the data folder'}`));
    if (!await dialog(`Export “${row.collection}”`, body, { okText: 'Export' })) return;
    toast('Exporting…');
    const r = await act('collection/export', { name: row.collection, folder: folder.value });
    if (r.ok === false) return;
    const res = r.result;
    await dialog('Export ready', h('div', null,
      h('p', null, `${num(res.documents)} documents, ${num(res.chunks)} chunks, ${bytes(res.bytes)}.`),
      h('div', { class: 'cd-path' }, h('span', { class: 'muted' }, 'File'), h('code', null, res.file),
        h('button', { class: 'btn small', on: { click: () => copyText(res.file) } }, 'Copy')),
      h('p', null, h('a', { class: 'btn', href: '/api/' + res.download, download: '' }, 'Download a copy'))), { okText: 'Close', noCancel: true });
  }

  async function deleteDialog(row, info) {
    const typed = textInput(row.collection);
    const src = info && info.source && info.source.folder;
    const rebuilt = row.kind !== 'imported' && (row.kind === 'location' || (info ? info.source.files > 0 : row.exists));
    const body = h('div', null,
      h('p', { style: { marginTop: 0 } }, 'This deletes the collection’s ', h('b', null, 'converted Markdown and index'), ' from rag-search’s workspace, and removes it from search.'),
      h('ul', { class: 'small' },
        h('li', null, 'Source documents are never touched', src ? h('span', null, ' (', h('code', null, src), ')') : null, '.'),
        h('li', null, 'Its access rule and description are kept.'),
        rebuilt ? h('li', null, h('b', null, 'Its documents are still in place, so the next indexing run builds it again'), ' -- use this to start its index over. To stop indexing it, ', row.kind === 'location' ? 'use “Remove location” instead.' : 'its folder is not registered, so nothing updates it.') : null,
        row.kind === 'imported' ? h('li', null, 'An imported collection has no source here: to get it back, import the export again.') : null),
      field(`Type “${row.collection}” to confirm`, typed));
    if (!await dialog(`Delete “${row.collection}”`, body, { okText: 'Delete', danger: true })) return;
    if (typed.value.trim().toLowerCase() !== row.collection.toLowerCase()) return toast('The name did not match: nothing was deleted', 'bad');
    const r = await act('collection/delete', { name: row.collection, confirm: typed.value });
    if (r.ok === false) return;
    infoCache.delete(row.collection);
    toast(`${row.collection}: Markdown and index deleted${r.result.note ? ' -- ' + r.result.note : ''}`, 'ok', 8000);
  }

  async function removeLocationDialog(row) {
    const typed = textInput(row.collection);
    const body = h('div', null,
      h('p', { style: { marginTop: 0 } }, 'Unregister this location and delete its converted Markdown and index. The folder ', row.folder ? h('code', null, row.folder) : null, ' and its documents are not touched; its access rule and description are kept.'),
      field(`Type “${row.collection}” to confirm`, typed));
    if (!await dialog(`Remove location “${row.collection}”`, body, { okText: 'Remove', danger: true })) return;
    if (typed.value.trim().toLowerCase() !== row.collection.toLowerCase()) return toast('The name did not match: nothing was changed', 'bad');
    const r = await act('location/remove', { name: row.collection, confirm: typed.value });
    if (r.ok === false) return;
    open.delete(row.collection);
    infoCache.delete(row.collection);
    toast(`${row.collection}: location removed`, 'ok');
  }

  function toggleCell(row, client) {
    const clients = knownClients();
    const allowed = row.clients === null ? clients.slice() : row.clients.slice();
    const next = allowed.includes(client) ? allowed.filter(c => c !== client) : [...allowed, client];
    return setAccess(row.collection, next);
  }

  async function editDialog(row) {
    const known = knownClients();
    const current = row.clients === null ? null : row.clients;
    const radios = ['everyone', 'only'].map(v => h('input', { type: 'radio', name: 'acc-mode', value: v, checked: (v === 'everyone') === (current === null) }));
    const boxes = known.map(c => h('input', { type: 'checkbox', value: c, checked: current !== null && current.includes(c) }));
    const others = h('input', { type: 'text', placeholder: 'other client names, comma separated', style: { width: '100%' },
      value: current ? current.filter(c => !known.includes(c)).join(', ') : '' });
    const body = h('div', null,
      h('label', { class: 'check', style: { display: 'flex', margin: '4px 0' } }, radios[0], h('span', null, h('b', null, 'Everyone'), ' – every client may use it')),
      h('label', { class: 'check', style: { display: 'flex', margin: '4px 0' } }, radios[1], h('span', null, h('b', null, 'Only these clients'))),
      h('div', { style: { margin: '6px 0 6px 26px' } }, known.map((c, i) => h('label', { class: 'check', style: { marginRight: '14px' } }, boxes[i], c)), h('div', { style: { marginTop: '8px' } }, others)),
      h('p', { class: 'small muted' }, 'Your terminal (cli) and this dashboard can always use every collection. A client is the name a host passes as ', h('code', null, 'rag-search-mcp --profile NAME'), '.'));
    if (!await dialog(`Access to “${row.collection}”`, body, { okText: 'Save' })) return;
    if (radios[0].checked) return setAccess(row.collection, ['all']);
    const chosen = boxes.filter(b => b.checked).map(b => b.value).concat(others.value.split(',').map(s => s.trim().toLowerCase()).filter(Boolean));
    return setAccess(row.collection, chosen);
  }

  function accessChips(row) {
    if (row.clients === null) return chip('everyone', 'ok');
    if (!row.clients.length) return chip('nobody', 'bad');
    return row.clients.map(c => chip(c, 'accent'));
  }

  /* ---------- per-collection details (fetched on demand when a row is expanded) ---------- */
  const infoCache = new Map();      // collection -> { data, error, at, key, loading }
  const sectionsOpen = new Set();   // "collection:section" the user opened
  const sectionsShut = new Set();   // ... or closed (overrides a section's default)
  const docFilter = new Map();      // collection -> text typed into the documents filter

  function freshnessKey() {
    const c = RS.state.catalog, job = RS.state.live && RS.state.live.index && RS.state.live.index.job;
    return [c && c.list && c.list.generation, job && job.id, job && job.status].join('|');
  }

  function loadInfo(name, force) {
    const ent = infoCache.get(name), key = freshnessKey();
    if (!force && ent && (ent.loading || (ent.key === key && Date.now() - ent.at < 60000))) return;
    infoCache.set(name, { ...(ent || {}), loading: true });
    api('collection/info?name=' + encodeURIComponent(name)).then(r => {
      infoCache.set(name, { data: r.ok === false ? null : r.result, error: r.ok === false ? (r.error || 'failed') : '',
        at: Date.now(), key, loading: false });
      RS.views.collections.update();
    });
  }

  function copyText(t) {
    const ok = () => toast('Copied to the clipboard', 'ok');
    if (navigator.clipboard && window.isSecureContext) navigator.clipboard.writeText(t).then(ok, () => toast(t));
    else toast(t);
  }

  function section(coll, id, title, badge, body, openByDefault) {
    const k = coll + ':' + id;
    const isOpen = sectionsOpen.has(k) || (openByDefault && !sectionsShut.has(k));
    return h('details', { class: 'cd-sec', open: isOpen, on: { toggle: e => {
        if (e.target.open) { sectionsOpen.add(k); sectionsShut.delete(k); } else { sectionsOpen.delete(k); sectionsShut.add(k); } } } },
      h('summary', null, h('b', null, title), badge ? h('span', { class: 'muted small' }, badge) : null),
      h('div', { class: 'cd-body' }, body));
  }

  function pathRow(label, path, note) {
    if (!path) return null;
    return h('div', { class: 'cd-path' }, h('span', { class: 'muted' }, label), h('code', null, path),
      h('button', { class: 'btn small', title: 'Copy this path', on: { click: e => { e.stopPropagation(); copyText(path); } } }, 'Copy'),
      note ? h('span', { class: 'muted small' }, note) : null);
  }

  const STATE = {
    ok: ['up to date', 'ok'], pending: ['needs indexing', 'warn'], unpublished: ['not published yet', 'warn'],
    unreachable: ['source unreachable', 'bad'], not_indexed: ['not indexed', ''], imported: ['imported', 'accent'],
  };

  function tiles(i) {
    const ws = i.workspace, src = i.source, b = i.build || {};
    const docLabel = i.kind === 'imported' ? 'documents (imported)'
      : src.files === null || src.files === undefined ? 'documents indexed' : `indexed of ${num(src.files)} in the source folder`;
    return h('div', { class: 'cd-tiles' },
      statCard(num(ws.documents), docLabel),
      statCard(num(b.chunks || 0), 'chunks'),
      statCard(bytes(i.disk.total_bytes), `on disk: Markdown ${bytes(ws.markdown_bytes)} + index ${bytes(ws.index_bytes)}`),
      src.bytes !== null && src.bytes !== undefined ? statCard(bytes(src.bytes), 'source documents') : null,
      h('div', { class: 'stat', title: b.last_indexed ? clock(b.last_indexed) : '' }, h('b', null, b.last_indexed ? ago(b.last_indexed) : '–'), h('span', null, 'last indexed')),
      b.build_seconds !== null && b.build_seconds !== undefined ? statCard(dur(b.build_seconds), 'total build time') : null);
  }

  function attention(i) {
    const a = i.attention, groups = [];
    const list = (label, grp, cls) => {
      if (!grp.count) return;
      groups.push(h('div', { class: 'cd-att' }, chip(`${num(grp.count)} ${label}`, cls),
        h('ul', null, grp.names.map(n => h('li', null, h('code', null, n),
          grp.reasons && grp.reasons[n] ? h('span', { class: 'muted small' }, '  ' + grp.reasons[n]) : null))),
        grp.count > grp.names.length ? h('div', { class: 'muted small' }, `… and ${num(grp.count - grp.names.length)} more (rag-search collection info ${i.collection} --json lists them)`) : null));
    };
    list('not indexed yet', a.not_indexed, 'warn');
    list('changed since indexed', a.modified_since_indexed, 'warn');
    list('incomplete (interrupted run)', a.incomplete, 'bad');
    return groups;
  }

  function lastRun(run) {
    if (!run) return h('span', { class: 'muted' }, 'no recent run touched this collection');
    const order = ['indexed', 'skipped', 'removed', 'no_text', 'unsupported', 'error', 'converted'];
    const cls = { indexed: 'ok', error: 'bad', removed: 'warn', no_text: '', unsupported: '' };
    const counts = Object.entries(run.counts).sort((x, y) => order.indexOf(x[0]) - order.indexOf(y[0]));
    return h('div', null,
      h('div', null, h('code', null, run.job), ' ', chip(run.status, run.status === 'succeeded' ? 'ok' : run.status === 'failed' ? 'bad' : ''),
        h('span', { class: 'muted small' }, ` ${run.mode || ''}${run.path ? ' ' + run.path : ''} · finished ${run.finished_at ? clock(run.finished_at) : '–'}`)),
      h('div', { style: { marginTop: '4px' } }, counts.map(([k, v]) => chip(`${k.replace('_', ' ')} ${num(v)}`, cls[k] || ''))),
      run.errors.length ? h('ul', { class: 'cd-errors' }, run.errors.map(e => h('li', null, h('code', null, e.source), h('span', { class: 'muted small' }, '  ' + e.message)))) : null);
  }

  function documentsTable(row, coll) {
    const c = row.info;
    const docs = (c && c.documents) || [];
    if (!docs.length) return h('div', { class: 'muted small' }, 'Nothing published for this collection yet.');
    const f = (docFilter.get(coll) || '').toLowerCase();
    const shown = f ? docs.filter(d => (d.name + ' ' + (d.source || '')).toLowerCase().includes(f)) : docs;
    return h('div', null,
      docs.length > 10 ? h('input', { type: 'search', placeholder: `filter ${docs.length} documents…`, value: docFilter.get(coll) || '', style: { margin: '0 0 8px', width: 'min(320px, 100%)' },
        on: { input: e => { docFilter.set(coll, e.target.value); RS.views.collections.update(); }, click: e => e.stopPropagation() } }) : null,
      h('div', { class: 'table-wrap' }, h('table', null,
        h('thead', null, h('tr', null, ['Document', 'Chunks', 'Indexed', 'Convert', 'Embed', 'Total', 'Index size'].map((t, i) => h('th', { class: i > 0 && i !== 2 ? 'num' : '' }, t)))),
        h('tbody', null, shown.map(d => h('tr', null,
          h('td', null, d.name, d.source && d.source !== d.name ? h('span', { class: 'muted small' }, '  ' + d.source) : null), h('td', { class: 'num' }, num(d.chunks)),
          h('td', { class: 'nowrap muted' }, clock(d.indexed_at)), h('td', { class: 'num' }, d.convert_s !== null && d.convert_s !== undefined ? dur(d.convert_s) : '–'),
          h('td', { class: 'num' }, d.embed_s !== null && d.embed_s !== undefined ? dur(d.embed_s) : '–'), h('td', { class: 'num' }, d.build_s !== null && d.build_s !== undefined ? dur(d.build_s) : '–'),
          h('td', { class: 'num' }, bytes(d.index_bytes))))))),
      f && shown.length !== docs.length ? h('div', { class: 'muted small' }, `${shown.length} of ${docs.length} shown`) : null);
  }

  function conversionBody(coll, c) {
    const docList = (grp, text) => grp && grp.count ? h('div', { style: { marginTop: '10px' } },
      h('b', null, `${num(grp.count)} document(s) ${text}`),
      h('ul', { class: 'tight' }, grp.items.map(it => h('li', null,
        h('a', { href: '#/collections', on: { click: e => { e.preventDefault(); e.stopPropagation(); CV.open(coll, it.doc); } } }, it.doc),
        h('span', { class: 'muted small' }, ` · page${it.pages.length > 1 ? 's' : ''} ${it.pages.join(', ')}`))))) : null;
    return h('div', null,
      CV.bands(c.branches),
      h('div', { style: { marginTop: '6px' } }, CV.outcomeChips(c.outcomes)),
      kv([
        ['Documents', num(c.documents)],
        ['Time', Object.entries(c.time_s || {}).map(([k, v]) => `${k} ${dur(v)}`).join(' · ') || null],
        ['Cost', CV.costText(c.cost) || null],
        ['Scripts', Object.keys(c.scripts || {}).length ? Object.entries(c.scripts).map(([k, v]) => `${k} ${v}`).join(', ') : null],
        ['Docling grades', Object.keys(c.docling_grades || {}).length ? Object.entries(c.docling_grades).map(([k, v]) => `${k} ${v}`).join(', ') : null],
        ['Tables / big pictures', c.tables || c.big_pictures ? `${num(c.tables)} / ${num(c.big_pictures)}` : null],
        ['Repaired cells', c.repair_tried ? `${num(c.repaired_cells || 0)} of ${num(c.repair_tried)} suspect cells` : null],
        ['Tables across pages', c.merged_tables ? num(c.merged_tables) : null],
        ['Without a record', c.no_record ? `${num(c.no_record)} (indexed before conversion tracking; re-convert to record)` : null],
      ]),
      docList(c.low_documents, 'need attention: low-confidence pages (a search hit on them says so; check the source page)'),
      docList(c.poor_documents, 'with pages docling itself graded poor'));
  }

  function details(row) {
    const coll = row.collection;
    if (!row.exists && !row.info) return h('div', { class: 'muted small', style: { padding: '8px 4px' } }, 'This rule is for a collection that has no folder or index yet; it applies as soon as one exists.');
    loadInfo(coll);
    const ent = infoCache.get(coll) || {};
    const i = ent.data;
    const head = h('div', { class: 'cd-head' },
      i ? chip(...(STATE[i.state] || [i.state, ''])) : null,
      h('span', { class: 'muted small' }, i ? i.state_detail : ent.error ? ent.error : 'loading details…'),
      h('button', { class: 'btn small', style: { marginLeft: 'auto' }, title: 'Read the folders and counts again', disabled: !!ent.loading,
        on: { click: e => { e.stopPropagation(); loadInfo(coll, true); RS.views.collections.update(); } } }, ent.loading ? 'Refreshing…' : 'Refresh'));
    const ro = readOnly();
    const hasIndex = i ? (i.workspace.index_exists || i.workspace.markdown_exists || !!i.published) : !!row.info;
    const actions = h('div', { class: 'cd-actions' },
      h('button', { class: 'btn small', disabled: ro || !hasIndex, title: 'Write this collection to one .rag.tgz file for someone else to import', on: { click: e => { e.stopPropagation(); exportDialog(row); } } }, 'Export…'),
      h('button', { class: 'btn small danger', disabled: ro || !hasIndex, title: 'Delete its converted Markdown and index (never the documents)', on: { click: e => { e.stopPropagation(); deleteDialog(row, i); } } }, 'Delete…'),
      row.kind === 'location' ? h('button', { class: 'btn small', disabled: ro, title: 'Unregister this folder (and delete its index)', on: { click: e => { e.stopPropagation(); removeLocationDialog(row); } } }, 'Remove location…') : null);
    const desc = (row.info && row.info.description) || (i && i.description);
    const parts = [head, actions, desc ? h('p', { style: { margin: '6px 0' } }, desc) : h('p', { class: 'small muted', style: { margin: '6px 0' } }, 'No description yet (Describe… sets one).')];
    if (i) {
      const ws = i.workspace, src = i.source, pub = i.published, b = i.build || {};
      parts.push(tiles(i));
      const where = [
        i.kind === 'imported' ? null : pathRow('Source documents', src.folder, src.reachable === false ? 'not reachable now' : src.unsupported ? `${num(src.unsupported)} file(s) in formats that are not indexed` : ''),
        pathRow('Markdown (workspace)', ws.markdown_folder, `${num(ws.markdown_files)} file(s), ${bytes(ws.markdown_bytes)}`),
        pathRow('Index (workspace)', ws.index_folder, `${bytes(ws.index_bytes)}: merged ${bytes(ws.merged_index_bytes)} + per-document ${bytes(ws.per_document_index_bytes)}`),
        pub ? pathRow('Published copy', pub.index_folder, `generation ${pub.generation}; hard links, no extra space`) : null,
      ];
      parts.push(section(coll, 'where', 'Where it lives', `  ${i.kind === 'location' ? 'location' : 'imported'} · ${bytes(i.disk.total_bytes)} on disk`, where, false));
      parts.push(section(coll, 'indexing', 'Indexing & publishing', b.last_indexed ? `  last indexed ${clock(b.last_indexed)}` : '', h('div', null,
        kv([
          ['Model', b.model ? h('span', null, h('code', null, String(b.model)), b.model_revision ? h('span', { class: 'muted small' }, ` @ ${String(b.model_revision).slice(0, 12)}`) : null) : null],
          ['Vectors', b.dim ? `${b.dim} dimensions` : null],
          ['Chunking', b.chunk_size ? `${b.chunk_size} tokens, ${b.chunk_overlap} overlap` : null],
          ['First indexed', b.first_indexed ? clock(b.first_indexed) : null],
          ['Last indexed', b.last_indexed ? `${clock(b.last_indexed)} (${ago(b.last_indexed)})` : null],
          ['Build time', b.build_seconds !== null && b.build_seconds !== undefined ? `${dur(b.build_seconds)} (convert ${dur(b.convert_seconds)}, embed ${dur(b.embed_seconds)})` : null],
          ['Merged index', ws.merged ? `${num(ws.merged.documents)} documents, ${num(ws.merged.chunks)} chunks, ${clock(ws.merged.built_at)}` : 'none'],
          ['Published', pub ? `generation ${pub.generation}, ${clock(pub.published_at)}: ${num(pub.documents)} documents, ${num(pub.chunks)} chunks` : 'not published'],
          ['Access', i.access === 'everyone' ? 'every client' : (i.access.length ? i.access.join(', ') : 'nobody (only your terminal and this dashboard)')],
          ['Last run', lastRun(i.last_run)],
        ])), false));
      if (i.origin) {
        const o = i.origin;
        parts.push(section(coll, 'origin', 'Origin', `  export of “${o.source_collection}”`, kv([
          ['Exported from', o.source_collection], ['Export file', o.file], ['Exported', o.exported_at ? clock(o.exported_at) : null],
          ['By rag-search', o.exported_by_version], ['Imported', o.imported_at ? clock(o.imported_at) : null],
          ['Model', o.model ? `${o.model}${o.model_revision ? ' @ ' + String(o.model_revision).slice(0, 12) : ''}` : null]]), false));
      }
      if (i.conversion && i.conversion.pages) parts.push(section(coll, 'conversion', 'Conversion', `  ${num(i.conversion.pages)} pages, as last converted`, conversionBody(coll, i.conversion), false));
      const att = attention(i);
      if (att.length) parts.push(section(coll, 'attention', 'Needs attention', `  ${att.length} group(s)`, att, true));
    }
    const nDocs = ((row.info && row.info.documents) || []).length;
    parts.push(section(coll, 'documents', `Documents (${num(nDocs)})`, '  published, with per-document timings', documentsTable(row, coll), nDocs > 0 && nDocs <= 10));
    return h('div', { class: 'cd' }, parts);
  }

  function listCard() {
    const rs = rows(); const ro = readOnly();
    const body = [];
    for (const r of rs) {
      const c = r.info; const isOpen = open.has(r.collection);
      body.push(h('tr', { class: 'clickable', on: { click: () => { isOpen ? open.delete(r.collection) : open.add(r.collection); RS.views.collections.update(); } } },
        h('td', null, (isOpen ? '▾ ' : '▸ '), h('b', null, r.collection), kindChip(r), !r.exists ? h('span', { class: 'chip warn', style: { marginLeft: '8px' } }, 'rule only') : !r.indexed ? h('span', { class: 'chip', style: { marginLeft: '8px' } }, 'not indexed') : null),
        h('td', { class: 'num' }, c ? num((c.documents || []).length) : '–'), h('td', { class: 'num' }, c ? num(c.chunks) : '–'),
        h('td', { class: 'num' }, c ? bytes(c.index_bytes) : '–'), h('td', { class: 'num' }, c ? bytes(c.markdown_bytes) : '–'), h('td', { class: 'num' }, c ? bytes(c.source_bytes) : '–'),
        h('td', { class: 'nowrap muted' }, c ? ago(c.built_at) : '–'), h('td', { class: 'num' }, c && c.build_seconds !== null && c.build_seconds !== undefined ? dur(c.build_seconds) : '–'),
        h('td', null, accessChips(r)),
        h('td', { class: 'nowrap', on: { click: e => e.stopPropagation() } },
          h('button', { class: 'btn small', disabled: ro, title: 'Choose who may use it', on: { click: () => editDialog(r) } }, 'Access…'), ' ',
          h('button', { class: 'btn small', disabled: ro || !r.exists, title: 'Say in a sentence what this collection holds', on: { click: () => describeDialog(r) } }, 'Describe…'), ' ',
          h('button', { class: 'btn small', disabled: ro || !r.exists || r.kind === 'imported', title: 'Index new and changed documents of this collection', on: { click: async () => { const x = await act('index/start', { mode: 'new', path: r.collection }); if (x.ok !== false) toast(x.already_running ? 'A run is already active' : 'Indexing started', x.already_running ? '' : 'ok'); } } }, 'Index'))));
      if (isOpen) body.push(h('tr', { class: 'cd-row' }, h('td', { colspan: 10 }, details(r))));
    }
    return h('div', { class: 'card' },
      h('div', { class: 'card-head' }, h('h2', null, 'Collections'), h('span', { class: 'muted small' }, 'a registered folder or an imported export; click a row for its details'),
        h('span', { class: 'cd-head-actions' },
          h('button', { class: 'btn small primary', disabled: readOnly(), title: 'Index a folder elsewhere on this computer as a collection', on: { click: () => addDialog() } }, 'Add collection…'), ' ',
          h('button', { class: 'btn small', disabled: readOnly(), title: 'Add a collection someone exported', on: { click: () => importDialog() } }, 'Import…'))),
      rs.length ? h('div', { class: 'table-wrap' }, h('table', null,
        h('thead', null, h('tr', null, ['Collection', 'Docs', 'Chunks', 'Index', 'Markdown', 'Sources', 'Built', 'Took', 'Who can use it', ''].map((t, i) => h('th', { class: i > 0 && i < 6 ? 'num' : '' }, t)))),
        h('tbody', null, body))) : empty('No collections yet. Register a folder on this tab, then start indexing.'));
  }

  function matrixCard() {
    const rs = rows(); const ro = readOnly(); const clients = knownClients();
    return h('div', { class: 'card' },
      h('div', { class: 'card-head' }, h('h2', null, 'Who can use what'),
        h('span', { class: 'muted small' }, 'click a cell to allow or block a client; changes apply immediately, no restart')),
      rs.length ? h('div', { class: 'table-wrap' }, h('table', { class: 'matrix' },
        h('thead', null, h('tr', null, h('th', null, 'Collection'), clients.map(c => h('th', null, c)), h('th', null, ''))),
        h('tbody', null, rs.map(r => h('tr', null, h('td', null, h('b', null, r.collection), r.clients === null ? h('span', { class: 'muted small' }, '  open to everyone') : null),
          clients.map(c => {
            const yes = r.clients === null || r.clients.includes(c);
            return h('td', null, h('button', { class: 'cell ' + (yes ? 'yes' : 'no'), disabled: ro, title: `${c} ${yes ? 'may' : 'may not'} use ${r.collection}. Click to ${yes ? 'block' : 'allow'}.`,
              'aria-label': `${c} ${yes ? 'allowed' : 'blocked'} for ${r.collection}`, on: { click: () => toggleCell(r, c) } }, yes ? '✓' : '✕'));
          }),
          h('td', { class: 'nowrap' }, r.clients !== null ? h('button', { class: 'btn small', disabled: ro, on: { click: () => setAccess(r.collection, ['all']) } }, 'Open to everyone') : null)))))) : empty('Nothing to show yet.'),
      h('p', { class: 'small muted', style: { marginBottom: 0 } }, 'Columns are the clients known so far (hosts registered with ', h('code', null, 'rag-search register'), ', clients named in a rule, and clients seen by the search daemon). Add another one with Access… → “other client names”. Same as ',
        h('code', null, 'rag-search access restrict COLLECTION CLIENT…'), '.'));
  }

  RS.views.collections = {
    init(root) {
      refs.list = h('div'); refs.matrix = h('div', { style: { marginTop: '16px' } });
      root.append(h('h2', null, 'Collections & access'), refs.list, refs.matrix);
    },
    update() {
      const cat = RS.state.catalog; if (!cat) return;
      patch(refs.list, listCard());
      patch(refs.matrix, matrixCard());
    },
    tick() { this.update(); },
  };
})();
