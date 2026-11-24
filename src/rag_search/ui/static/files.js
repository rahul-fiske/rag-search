'use strict';
/* Files: browse a collection's folders and see everything rag-search recorded about one file (core/dossier.py).
   Read-only: a folder is listed a level at a time, a search walks the collection, and a file's "dossier" puts the records of every
   step side by side (facts, description, faces, a ledger of what produced what and whether it is still current).
   The address is #/files/<collection>/<path of the file>, so a file can be linked to (the Indexing tab and the People viewer do). */
(function () {
  const S = { collections: null, collection: '', prefix: '', q: '', list: null, file: null, dossier: null, tab: 'overview', showFaces: true, page: 1, busy: 0 };
  let refs = {}, timer = null;

  const qs = o => new URLSearchParams(o).toString();
  const hashOf = (coll, path) => '#/files/' + encodeURIComponent(coll) + (path ? '/' + path.split('/').map(encodeURIComponent).join('/') : '');
  RS.fileLink = (coll, path) => hashOf(coll, path);
  function fromHash() {
    if (!/^#\/?files(\/|$)/.test(location.hash)) return { coll: '', path: '' };       // another tab's address (#/collections): keep what is shown
    const rest = location.hash.replace(/^#\/?files\/?/, '');
    if (!rest) return { coll: '', path: '' };
    const parts = rest.split('/').map(p => { try { return decodeURIComponent(p); } catch (e) { return p; } });
    return { coll: parts[0], path: parts.slice(1).join('/') };
  }
  const parentOf = p => p.includes('/') ? p.slice(0, p.lastIndexOf('/')) : '';
  const KIND = { photo: 'photograph', video: 'video', timeline: 'Timeline', pages: 'document' };
  const when = t => t ? stamp(t) : '–';

  async function loadCollections() {
    const r = await api('files/collections');
    S.collections = r.ok === false ? [] : r.collections || [];
  }
  async function loadList() {
    const mine = ++S.busy;
    const r = await api('files/list?' + qs({ collection: S.collection, prefix: S.prefix, q: S.q }));
    if (mine !== S.busy) return;                        // a newer request took over
    S.list = r.ok === false ? { error: r.error, folders: [], files: [] } : r;
    paintBrowser();
  }
  async function openFile(coll, path) {
    S.collection = coll; S.file = path; S.dossier = null; S.page = 1;
    S.prefix = parentOf(path); S.q = ''; if (refs.search) refs.search.value = '';
    paintBrowser(); paintInspector();
    await Promise.all([loadList(), (async () => {
      const r = await api('files/dossier?' + qs({ collection: coll, path }));
      if (S.file !== path || S.collection !== coll) return;
      S.dossier = r.ok === false ? { error: r.error } : r;
      paintInspector();
    })()]);
  }

  /* ---- the browser (left) ---- */
  const dot = e => e.state === 'indexed' ? '' : h('span', { class: 'chip ' + (e.state === 'failed' ? 'bad' : 'warn'), style: { marginLeft: '6px' } }, e.state);
  function fileRow(e) {
    const sel = S.file === e.path && S.collection === S.list.collection;
    const thumb = e.kind === 'photo' || e.kind === 'video'
      ? h('img', { class: 'fthumb', loading: 'lazy', alt: '', src: 'api/files/original?' + qs({ collection: S.collection, path: e.path, w: 256 }) })
      : h('span', { class: 'fthumb doc' }, e.kind === 'pages' ? '\u{1F4C4}' : '•');
    const dir = S.q ? parentOf(e.path) : '';
    return h('a', { class: 'frow' + (sel ? ' sel' : ''), href: hashOf(S.collection, e.path), title: e.path }, thumb,
      h('span', { class: 'fname' }, e.name, dot(e), dir ? h('span', { class: 'small muted', style: { display: 'block' } }, dir) : null),
      h('span', { class: 'small muted nowrap' }, bytes(e.size)));
  }
  function crumbs() {
    const parts = S.prefix ? S.prefix.split('/') : [];
    const out = [h('a', { href: '#', class: 'crumb', on: { click: ev => { ev.preventDefault(); S.prefix = ''; S.q = ''; refs.search.value = ''; loadList(); } } }, S.collection || 'collection')];
    parts.forEach((p, i) => {
      const to = parts.slice(0, i + 1).join('/');
      out.push(h('span', { class: 'muted' }, ' / '), h('a', { href: '#', class: 'crumb', on: { click: ev => { ev.preventDefault(); S.prefix = to; S.q = ''; refs.search.value = ''; loadList(); } } }, p));
    });
    return out;
  }
  function paintBrowser() {
    if (!refs.list) return;
    const names = (S.collections || []);
    fill(refs.pick, ...names.map(c => h('option', { value: c.name, selected: c.name === S.collection }, c.name + (c.indexed ? '' : ' (not indexed)'))));
    const l = S.list;
    fill(refs.crumbs, ...(S.q ? [h('span', { class: 'muted' }, `search in ${S.collection}`)] : crumbs()));
    if (!l) { fill(refs.list, h('p', { class: 'muted' }, 'loading…')); return; }
    if (l.error) { fill(refs.list, h('div', { class: 'notice bad' }, l.error)); return; }
    const up = !S.q && S.prefix ? [h('a', { class: 'frow folder', href: '#', on: { click: ev => { ev.preventDefault(); S.prefix = parentOf(S.prefix); loadList(); } } }, h('span', { class: 'fthumb doc' }, '←'), h('span', { class: 'fname' }, '..'))] : [];
    fill(refs.list, ...up,
      l.folders.map(f => h('a', { class: 'frow folder', href: '#', on: { click: ev => { ev.preventDefault(); S.prefix = f.path; loadList(); } } },
        h('span', { class: 'fthumb doc' }, '\u{1F4C1}'), h('span', { class: 'fname' }, f.name))),
      l.files.map(fileRow),
      !l.folders.length && !l.files.length ? empty(S.q ? 'Nothing matches.' : 'This folder is empty.') : null,
      l.more ? h('p', { class: 'small muted' }, `Only the first ${num(l.files.length)} files are shown: ${S.q ? 'make the search more specific' : 'open a subfolder'}.`) : null);
  }

  /* ---- the inspector (right) ---- */
  const HIDDEN_FACTS = /^(Looks like|Things seen|People in the picture|Described by|Reference data|Faces detected)/;
  const colour = f => f.how === 'you' || f.how === 'auto' ? 'var(--accent)' : 'var(--muted)';
  const faceName = f => f.how === 'you' || f.how === 'auto' ? f.name : f.how === 'suggested' ? 'unnamed (guess: ' + f.name + ')' : 'unnamed';
  const openFace = id => { RS.openFace = id; location.hash = '#/people'; };
  const HOW = { you: 'named by you', auto: 'matched automatically', suggested: 'a weaker match: a guess', '': 'nobody matched' };

  function factsCard(d) {
    const rows = d.facts.filter(f => !HIDDEN_FACTS.test(f.label)).map(f => {
      const val = /^Position/.test(f.label) && d.geo
        ? h('span', null, f.value.replace(/\s*\(geo:[^)]*\)/, ''), ' ', h('a', { href: `https://www.openstreetmap.org/?mlat=${d.geo.lat}&mlon=${d.geo.lon}#map=15/${d.geo.lat}/${d.geo.lon}`, target: '_blank', rel: 'noopener noreferrer' }, 'map'),
            h('span', { class: 'small muted' }, ` geo:${d.geo.lat},${d.geo.lon}`))
        : f.value;
      return h('tr', null, h('th', { class: 'kvk' }, f.label), h('td', null, val));
    });
    return h('div', { class: 'card' }, h('div', { class: 'card-head' }, h('h3', null, d.kind === 'video' ? 'Video details' : 'Details')),
      rows.length ? h('table', { class: 'kvt' }, h('tbody', null, rows)) : h('p', { class: 'muted' }, 'No details recorded in the Markdown.'));
  }
  function describeCard(d) {
    const ds = d.descriptions || [];
    if (d.kind !== 'photo') return null;
    const cur = ds.find(x => x.current) || ds[0];
    return h('div', { class: 'card' }, h('div', { class: 'card-head' }, h('h3', null, 'Description'), cur ? h('span', { class: 'muted small' }, `${cur.id} · ${cur.at}`) : null),
      cur ? [h('p', { style: { margin: '0 0 8px' } }, cur.description),
        h('div', null, cur.tags.map(t => chip(t))),
        h('div', { class: 'small muted' }, `people counted by the model: ${cur.people}`, cur.current ? '' : ' · made by another model or prompt than the current one'),
        ds.length > 1 ? h('details', { style: { marginTop: '6px' } }, h('summary', { class: 'small' }, `${ds.length - 1} other description(s)`),
          ds.filter(x => x !== cur).map(x => h('div', { class: 'small', style: { marginTop: '6px' } }, h('b', null, x.id), ` · ${x.at}`, h('div', null, x.description)))) : null]
        : h('p', { class: 'muted' }, 'Not described yet.'));
  }
  function facesCard(d) {
    if (d.kind !== 'photo') return null;
    const fs = d.faces || [];
    return h('div', { class: 'card' }, h('div', { class: 'card-head' }, h('h3', null, 'Faces'), h('span', { class: 'muted small' }, fs.length ? `${fs.length} found` : 'none found or not looked at')),
      fs.length ? h('div', { class: 'ftiles' }, fs.map(f => h('button', { type: 'button', class: 'ftile', title: 'open this photograph in the People viewer', on: { click: () => openFace(f.id) } },
        h('img', { class: 'face', src: 'api/people/face?id=' + f.id, alt: '', width: 72, height: 72, loading: 'lazy' }),
        h('b', null, faceName(f)),
        h('span', { class: 'small muted' }, HOW[f.how || ''] + (f.similarity ? ` · similarity ${f.similarity}` : '')),
        h('span', { class: 'small muted' }, `score ${f.score} · ${f.px} px`)))) : null);
  }
  function ledgerCard(d) {
    const cls = { done: 'ok', missing: 'warn', failed: 'bad', 'n/a': '' };
    return h('div', { class: 'card' }, h('div', { class: 'card-head' }, h('h3', null, 'What was done'), h('span', { class: 'muted small' }, 'one row per step, from the records the steps left')),
      h('div', { class: 'table-wrap' }, h('table', null,
        h('thead', null, h('tr', null, ['Step', 'Result', 'Produced by', 'When', 'Still current?'].map(t => h('th', null, t)))),
        h('tbody', null, d.ledger.map(r => h('tr', null,
          h('td', null, h('b', null, r.step)),
          h('td', null, chip(r.state, cls[r.state]), ' ', r.result, r.note ? h('div', { class: 'small muted' }, r.note) : null),
          h('td', { class: 'small', style: { wordBreak: 'break-word' } }, r.by || ''),
          h('td', { class: 'nowrap' }, when(r.when)),
          h('td', null, r.current === true ? chip('yes', 'ok') : r.current === false ? chip('no', 'warn') : h('span', { class: 'muted' }, '–')))))
      )));
  }
  function overview(d) {
    return h('div', { class: 'grid', style: { gap: '16px' } },
      d.changed ? h('div', { class: 'notice warn' }, 'The file has changed since it was indexed (its size is not the one recorded): the records below describe the old version.') : null,
      d.shares_folder.length ? h('div', { class: 'notice warn' }, `${d.shares_folder.join(', ')} ${d.shares_folder.length === 1 ? 'has' : 'have'} the same name in this folder, so they share one set of records (a known limit): what is shown may belong to the other file.`) : null,
      !d.indexed && d.outcome ? h('div', { class: 'notice bad' }, 'It could not be indexed and the failure is remembered: ' + (d.outcome.error || d.outcome.reason || JSON.stringify(d.outcome))) : null,
      !d.indexed && !d.outcome ? h('div', { class: 'notice' }, 'Not indexed yet: nothing is recorded for this file.') : null,
      ledgerCard(d), factsCard(d), describeCard(d), facesCard(d));
  }

  function original(d) {
    const base = { collection: d.collection, path: d.path };
    if (!d.exists) return h('div', { class: 'notice warn' }, 'The file is not in the source folder any more.');
    if (d.kind === 'video') {
      return h('div', null, h('video', { controls: true, preload: 'none', style: { maxWidth: '100%', maxHeight: '70vh' }, poster: 'api/files/original?' + qs(base), src: 'api/files/raw?' + qs(base) }),
        h('p', { class: 'small muted' }, 'If it does not play, this browser cannot decode the video; the picture above is its first frame.'));
    }
    if (d.kind === 'timeline') return h('div', { class: 'notice' }, 'A Timeline file has no picture to show.');
    const isImg = /\.(jpe?g|png|tiff?|bmp|webp|heic|heif)$/i.test(d.name);
    const pages = d.pages || 1;
    const img = h('img', { alt: d.name, src: 'api/files/original?' + qs({ ...base, w: 1600, ...(isImg ? {} : { page: S.page }) }),
      on: { error: ev => ev.target.replaceWith(h('div', { class: 'notice warn' }, 'This file cannot be shown as a picture here.')) } });
    const boxes = d.kind === 'photo' && S.showFaces ? (d.faces || []).map(f => h('div', { class: 'pv-box' + (f.score < 0.8 ? ' weak' : ''), title: faceName(f) + ' · score ' + f.score,
      style: { left: 100 * f.x + '%', top: 100 * f.y + '%', width: 100 * f.w + '%', height: 100 * f.h + '%', borderColor: colour(f) }, on: { click: () => openFace(f.id) } },
      h('span', { class: 'pv-tag', style: { background: colour(f) } }, f.how === 'you' || f.how === 'auto' ? f.name : '?'))) : [];
    return h('div', null,
      h('div', { class: 'row', style: { gap: '12px', alignItems: 'center', marginBottom: '8px', flexWrap: 'wrap' } },
        d.kind === 'photo' && (d.faces || []).length ? h('label', { class: 'check' }, h('input', { type: 'checkbox', checked: S.showFaces, on: { change: ev => { S.showFaces = ev.target.checked; paintInspector(); } } }), 'face boxes') : null,
        !isImg && pages > 1 ? [h('button', { class: 'btn small', disabled: S.page <= 1, on: { click: () => { S.page--; paintInspector(); } } }, '←'), h('span', null, `page ${S.page} of ${pages}`),
          h('button', { class: 'btn small', disabled: S.page >= pages, on: { click: () => { S.page++; paintInspector(); } } }, '→')] : null),
      h('div', { class: 'pv-stage', style: { margin: '0' } }, img, boxes));
  }
  function markdown(d) {
    return h('div', { class: 'grid g2' }, h('div', null, original(d)),
      h('div', null, d.markdown ? [h('pre', { class: 'mdpre' }, d.markdown), d.markdown_cut ? h('p', { class: 'small muted' }, 'The Markdown is longer: only the start is shown.') : null]
        : h('p', { class: 'muted' }, 'No Markdown was made for this file.')));
  }
  /* what each record is for and what it holds, shown under its name */
  const WHAT = [
    [/^index\.meta\.json$/, 'The index card of the file: its checksum and size, when it was indexed, the embedding model and settings, the conversion settings, how many passages it has and how long each step took. Its checksum is how an unchanged file is skipped.'],
    [/^nodes\.json$/, 'The passages (chunks) the Markdown was cut into, with the text and the metadata of each: what search reads back as results.'],
    [/^embeddings\.npy$/, 'One vector per passage from the embedding model: what meaning-based search compares your question with.'],
    [/^pictures\.json$/, 'The key of the picture vector: which picture model made it and for which checksum, so a stale or half-written vector is never used.'],
    [/^pictures\.npy$/, 'The picture vector: the photograph itself as numbers (not its text), what searching by what a picture looks like compares against.'],
    [/^outcome\.json$/, 'A remembered failure: why the file could not be indexed, so it is not tried again until it changes.'],
    [/\.md\.sha256$/, 'The checksum of the original file the Markdown was made from (first line) and the conversion settings (second): tells whether the Markdown is still current.'],
    [/\.md\.geo\.json$/, 'The map stamp: which map data (places, towns, areas) described the position in this file. When the map data changes, the file is described again.'],
    [/\.trace\.json$/, 'The conversion trace: what each reader did with the file, per page (reader, outcome, time and cost). The Indexing tab\'s page-by-page record is made from it.'],
    [/\.md$/, 'The Markdown: the text rag-search made of the file (details, description, faces and text for a photograph; the page text for a document). What is chunked, searched and read back.'],
  ];
  const whatIs = name => { const w = WHAT.find(([re]) => re.test(name)); return w ? w[1] : ''; };
  function records(d) {
    if (!d.records.length) return empty('Nothing on disk for this file.');
    const groups = [...new Set(d.records.map(r => r.group))];
    return h('div', null,
      d.index ? h('p', { class: 'small muted' }, `Embedded with ${d.index.model} (${d.index.dim} dimensions, ${d.index.nodes} passage(s)); converted with ${d.index.convert}.`) : null,
      groups.map(g => h('div', { style: { marginBottom: '14px' } }, h('h4', { style: { margin: '0 0 6px' } }, g),
        h('div', { class: 'table-wrap' }, h('table', null, h('tbody', null, d.records.filter(r => r.group === g).map(r => h('tr', null,
          h('td', { style: { maxWidth: '420px' } }, h('code', null, r.name), whatIs(r.name) ? h('div', { class: 'small muted', style: { margin: '3px 0' } }, whatIs(r.name)) : null, r.note ? h('div', { class: 'small muted' }, r.note) : null), h('td', { class: 'num nowrap' }, bytes(r.size)), h('td', { class: 'nowrap' }, when(r.mtime)),
          h('td', { class: 'small muted', style: { wordBreak: 'break-all' } }, r.path)))))))));
  }

  function paintInspector() {
    if (!refs.inspector) return;
    const d = S.dossier;
    if (!S.file) { fill(refs.inspector, h('div', { class: 'card' }, empty('Pick a file on the left to see everything rag-search recorded about it.'))); return; }
    if (!d) { fill(refs.inspector, h('div', { class: 'card' }, h('h3', null, S.file.split('/').pop()), h('p', { class: 'muted' }, 'loading…'))); return; }
    if (d.error) { fill(refs.inspector, h('div', { class: 'card' }, h('h3', null, S.file.split('/').pop()), h('div', { class: 'notice bad' }, d.error))); return; }
    const tabs = [['overview', 'Overview'], ['original', 'Original'], ['markdown', 'Markdown'], ['records', 'Records']];
    const body = { overview, original, markdown, records }[S.tab](d);
    fill(refs.inspector, h('div', { class: 'card' },
      h('div', { class: 'card-head' }, h('h2', { style: { wordBreak: 'break-word' } }, d.name),
        chip(KIND[d.kind] || d.kind), d.indexed ? chip('indexed', 'ok') : chip(d.outcome ? 'failed' : 'not indexed', d.outcome ? 'bad' : 'warn'),
        d.size != null ? h('span', { class: 'muted small' }, bytes(d.size)) : null, d.mtime ? h('span', { class: 'muted small' }, 'file changed ', stamp(d.mtime)) : null),
      h('div', { class: 'small muted', style: { marginBottom: '8px', wordBreak: 'break-all' } }, d.collection + '/' + d.path),
      h('div', { class: 'subtabs' }, tabs.map(([k, t]) => h('button', { type: 'button', class: 'btn small' + (S.tab === k ? ' primary' : ''), on: { click: () => { S.tab = k; paintInspector(); } } }, t))),
      h('div', { style: { marginTop: '12px' } }, body)));
  }

  /* ---- the view ---- */
  async function sync() {
    if (!S.collections) await loadCollections();
    const { coll, path } = fromHash();
    if (coll && path) { await openFile(coll, path); return; }
    S.collection = coll || S.collection || ((S.collections.find(c => c.indexed) || S.collections[0] || {}).name || '');
    if (coll && coll !== S.list?.collection) { S.prefix = ''; S.file = null; S.dossier = null; }
    paintBrowser(); paintInspector();
    if (S.collection) await loadList(); else paintBrowser();
  }

  /* The Files section sits below the collections on the Collections & access tab (the address #/files/... opens that tab). */
  RS.files = {
    mount(root) {
      refs.pick = h('select', { on: { change: ev => { S.q = ''; refs.search.value = ''; S.prefix = ''; S.file = null; S.dossier = null; location.hash = hashOf(ev.target.value, ''); } } });
      refs.search = h('input', { type: 'search', placeholder: 'Search names and paths', style: { width: '100%' },
        on: { input: () => { clearTimeout(timer); timer = setTimeout(() => { S.q = refs.search.value.trim(); loadList(); }, 300); } } });
      refs.crumbs = h('div', { class: 'small', style: { margin: '8px 0' } });
      refs.list = h('div', { class: 'flist' });
      refs.inspector = h('div', null);
      root.append(h('h2', null, 'Files'),
        h('p', { class: 'muted' }, 'Browse a collection and see everything rag-search recorded about one file. Nothing here changes anything.'),
        h('div', { class: 'files' }, h('div', { class: 'card fbrowser' }, refs.pick, h('div', { style: { height: '8px' } }), refs.search, refs.crumbs, refs.list), refs.inspector));
    },
    sync,
  };
})();
