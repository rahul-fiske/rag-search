'use strict';
/* Conversion: how each page of each document was read.  Shared by the Indexing tab (live run, document
   table, drawer) and the Collections tab (per-collection totals).  One vocabulary everywhere:
   a page takes one BRANCH (which path of the flow chart read it) and ends with one OUTCOME.
   Colour = what happened to the page; the CPU / GPU chips say where the work runs.
   With page routing on (the default) the branch is the path that really read the page: digital pages
   by docling without forced OCR, scanned pages by docling's full-page OCR (fallback), repeats from the
   page cache; the quality gate gives every page its outcome. */
const CV = (function () {
  const BRANCHES = {
    copy: ['text', 'Markdown / text, used as is'],
    office: ['office', 'Office / HTML file, docling reads the structure'],
    digital: ['digital', 'PDF page with a text layer'],
    raster: ['scanned', 'scanned or photographed PDF page, read as an image by the document VLM'],
    image: ['image', 'image file (or one frame of a multi-page TIFF), read by the document VLM'],
    embedded: ['embedded', 'large picture inside a PDF page, read by the document VLM'],
    fallback: ['scanned (OCR)', 'scanned page read by docling with full-page OCR (the document VLM is off, not installed, or could not read this page)'],
    cached: ['cached', 'page result reused from the page cache'],
    unknown: ['unprofiled', 'not profiled (the profiling time limit was reached)'],
  };
  const LETTERS = { c: 'copy', o: 'office', d: 'digital', r: 'raster', i: 'image', e: 'embedded', f: 'fallback', k: 'cached', '?': 'unknown' };
  const OUTCOMES = {
    pass: 'passed the quality checks', repaired: 'passed after repair', low: 'low confidence: kept, but flagged',
    no_text: 'no text on the page', error: 'could not be read',
  };
  const label = b => (BRANCHES[b] || [b])[0];
  const branchTitle = b => (BRANCHES[b] || [b, b])[1];
  const order = Object.keys(BRANCHES);
  const sortedBranches = o => Object.keys(o || {}).filter(k => o[k]).sort((a, b) => order.indexOf(a) - order.indexOf(b));

  function parseStrip(s) {
    const out = []; const re = /([a-z?])(\d*)/g; let m;
    while ((m = re.exec(s || ''))) out.push({ branch: LETTERS[m[1]] || 'unknown', n: parseInt(m[2] || '1', 10) });
    return out;
  }

  const hw = kind => h('span', { class: 'hw ' + kind, title: kind === 'gpu' ? 'runs on the GPU' : 'runs on the CPU' }, kind.toUpperCase());

  /* A document's pages as a run-length strip, one colour per branch. */
  function strip(conv) {
    if (!conv || !conv.strip) return h('span', { class: 'muted small' }, '–');
    const runs = parseStrip(conv.strip);
    return h('div', { class: 'cv-strip', title: runs.map(r => `${r.n} × ${label(r.branch)}`).join(', ') },
      runs.map(r => h('i', { class: 'cvb-' + r.branch, style: { flexGrow: String(r.n) } })));
  }

  function branchChips(branches) {
    return sortedBranches(branches).map(b => h('span', { class: 'chip cv-chip', title: branchTitle(b) },
      h('i', { class: 'cv-dot cvb-' + b }), `${label(b)} ${num(branches[b])}`));
  }
  function outcomeChips(outcomes) {
    const cls = { pass: 'ok', repaired: 'ok', low: 'bad', no_text: 'warn', error: 'bad' };
    return Object.keys(outcomes || {}).filter(k => outcomes[k]).map(k =>
      chip(`${k.replace('_', ' ')} ${num(outcomes[k])}`, cls[k] || ''));
  }

  /* Pages per branch as one proportional bar, with the figures beside it. */
  function bands(branches) {
    const keys = sortedBranches(branches); const total = keys.reduce((a, k) => a + branches[k], 0);
    if (!total) return null;
    return h('div', null,
      h('div', { class: 'cv-bands' }, keys.map(k => h('i', { class: 'cvb-' + k, title: `${label(k)}: ${num(branches[k])} pages (${Math.round(100 * branches[k] / total)}%)`, style: { flexGrow: String(branches[k]) } }))),
      h('div', { style: { marginTop: '6px' } }, branchChips(branches)));
  }

  function costText(cost) {
    const bits = [];
    if (cost && cost.cpu_s) bits.push(`CPU ${dur(cost.cpu_s)}`);
    if (cost && cost.peak_mb) bits.push(`peak ${bytes(cost.peak_mb * 1048576)}`);
    return bits.join(' · ');
  }

  /* The four lanes of step 3.2.  A page goes down exactly one lane; a page a cheaper lane doubts is handed on to lane d. */
  const RUNWAYS = {
    a: { name: 'Text layer', tool: 'docling reads the text layer and the tables', hw: ['cpu'],
      tip: 'Lane 3.2a: a page with a text layer is read by docling without forced OCR. Its text is then compared with the PDF\'s own text layer, and what docling left out is added.' },
    b: { name: 'OCR', tool: 'docling OCR · Tesseract (skewed pages, image files)', hw: ['cpu'], env: 'RAG_SEARCH_OCR_FIRST',
      tip: 'Lane 3.2b: a clean scan is read by an OCR engine (docling\'s, or Tesseract for a skewed page and for image files) and checked by the gate. A page the gate doubts goes on to lane d.' },
    c: { name: 'Text layer + pictures', tool: 'docling + document reader on pictures', hw: ['cpu', 'gpu'], env: 'RAG_SEARCH_RESIDUE',
      tip: 'Lane 3.2c: docling reads the text layer and the document reader reads the large pictures (and, with RAG_SEARCH_RESIDUE=auto, the regions of ink the text layer does not explain). What it finds that the page does not already say is added.' },
    d: { name: 'Document reader', tool: 'vision model · repair · Tesseract as last resort', hw: ['gpu'],
      tip: 'Lane 3.2d: a vision-language model reads the whole page image. Suspect table cells are re-read by the repair model; Tesseract is the last resort when a reader runs away.' },
  };
  const RW_ORDER = ['a', 'b', 'c', 'd'];
  const OUTCOME_CLS = { pass: 'ok', repaired: 'fix', low: 'bad', no_text: 'warn', error: 'bad' };

  /* the value a lane switch has in the daemon's environment (from the numbered pipeline), or '' when it is not known */
  function switchValue(name) {
    const st = ((PL.last && PL.last()) || {}).stages || [];
    const s = st.find(x => x.id === '3.2');
    const e = s && (s.env_only || []).find(x => x.name === name);
    return e ? String(e.value || '') : '';
  }
  function switchChip(k) {
    const env = RUNWAYS[k].env;
    if (!env) return h('span', { class: 'chip', title: 'always on' }, 'always on');
    const v = switchValue(env).toLowerCase(), on = v === 'auto';
    const text = k === 'c' ? (on ? 'pictures + regions' : 'large pictures') : (on ? 'on' : 'off');
    return h('span', { class: 'chip ' + (on || k === 'c' ? 'ok' : ''), title: `${env}=${v || (k === 'c' || k === 'b' ? 'off' : '')}${on ? '' : ' (default)'}` }, text);
  }
  const sumOf = o => Object.values(o || {}).reduce((a, b) => a + b, 0);

  function laneCard(k, rw, total, mv, lo) {
    const d = RUNWAYS[k], n = rw[k] || 0, pct = total ? Math.round(100 * n / total) : 0;
    const out = lo[k] || {}, outN = sumOf(out);
    const handed = Object.entries(mv).filter(([m]) => m.startsWith(k + '>')).reduce((a, [, v]) => a + v, 0);
    return h('div', { class: `cv-rw rw-${k}${n ? '' : ' idle'}`, title: d.tip },
      h('div', { class: 'cv-rw-head' }, h('span', { class: 'cv-rw-badge' }, k), h('b', null, d.name),
        h('span', { class: 'cv-rw-hw' }, d.hw.map(hw)), switchChip(k)),
      h('div', { class: 'cv-rw-tool' }, d.tool),
      h('div', { class: 'cv-rw-row' },
        h('span', { class: 'cv-rw-meter', title: `${num(n)} of ${num(total)} pages ended in this lane` }, h('i', { style: { width: pct + '%' } })),
        h('span', { class: 'cv-rw-n' }, n ? `${num(n)} pages · ${pct}%` : 'no pages')),
      outN ? h('div', { class: 'cv-rw-row' },
        h('span', { class: 'cv-rw-out', title: Object.entries(out).map(([o, v]) => `${o.replace('_', ' ')} ${num(v)}`).join(' · ') },
          Object.keys(out).filter(o => out[o]).map(o => h('i', { class: 'o-' + (OUTCOME_CLS[o] || 'ok'), style: { flexGrow: String(out[o]) } }))),
        h('span', { class: 'cv-rw-n small muted' }, `${num(out.low || 0)} low · ${num(out.repaired || 0)} repaired (this run)`)) : null,
      handed ? h('div', { class: 'cv-rw-hand' }, `↳ ${num(handed)} page${handed === 1 ? '' : 's'} handed on to the document reader`) : null);
  }

  /* The pipeline of step 3 as one picture: Profile and router on the left, the four lanes of 3.2 in the middle (how many pages
     ended in each, how they fared, how many were handed on), Gate, Repair, Reconcile and the outcome on the right.
     *t* = the run's totals, *l* = the live view (pages finished this run, by lane and outcome). */
  function flow(t, l) {
    t = t || {}; l = l || {};
    const out = t.outcomes || {}, mv = t.moves || l.moves || {}, gf = t.gate_failed || {};
    let rw = t.runways || l.runways || {}, approx = false;
    if (!sumOf(rw) && t.pages && t.branches) {            // a run made before the lanes were recorded: the branches say it nearly as well
      const b = t.branches;
      rw = { a: (b.digital || 0) + (b.office || 0) + (b.copy || 0), b: b.fallback || 0, c: b.embedded || 0, d: (b.raster || 0) + (b.image || 0) };
      approx = true;
    }
    const total = sumOf(rw), lo = l.runway_outcomes || {};
    const node = (title, sub, kind, state, extra) => h('div', { class: 'cv-node ' + (state || ''), title: extra || '' },
      h('div', { class: 'cv-node-h' }, title, kind ? hw(kind) : null), h('small', null, sub));
    const gateFails = Object.keys(gf).length ? Object.entries(gf).map(([k, v]) => `${k.replace('_', ' ')} ${num(v)}`).join(' · ') : (t.pages ? 'no check failed' : 'is the text trustworthy?');
    const moved = Object.entries(mv).map(([k, v]) => `${k.replace('>', ' → ')} ${num(v)}`).join(' · ');
    const eng = l.engines && Object.keys(l.engines).length ? Object.entries(l.engines).map(([k, v]) => `${k} ${num(v)}`).join(' · ') : '';
    return h('div', { class: 'cv-pipe' },
      h('div', { class: 'cv-col' },
        node('3.1 · Profile', t.pages ? `${num(t.pages)} pages looked at` : 'what is on each page', 'cpu', 'on', 'Text layer? scan? photo? pictures, ink, resolution, script'),
        h('span', { class: 'cv-arrow down', 'aria-hidden': 'true' }, '↓'),
        node('3.2 · Router', total ? `${num(total)} pages sent down a lane` + (moved ? ` · handed on: ${moved}` : '') : 'one lane per page', 'cpu', 'on',
          'The router picks the cheapest lane that is sure enough: a text layer goes to a (or c when it has large pictures); a clean scan to b when OCR first is on; everything else to d. A page the gate doubts after a cheap lane is handed on to d.')),
      h('span', { class: 'cv-arrow', 'aria-hidden': 'true' }, '→'),
      h('div', { class: 'cv-rws' }, RW_ORDER.map(k => laneCard(k, rw, total, mv, lo)),
        eng ? h('div', { class: 'small muted cv-rw-eng' }, `Lane b engines this run: ${eng}`) : null,
        approx ? h('div', { class: 'small muted cv-rw-eng' }, 'This run did not record lanes; the counts are worked out from how the pages were read (cached pages count by what they were).') : null),
      h('span', { class: 'cv-arrow', 'aria-hidden': 'true' }, '→'),
      h('div', { class: 'cv-col' },
        node('3.3 · Gate', gateFails, 'cpu', 'on', 'Coverage, garbled text, docling grade, table shape, balance checks; for OCR the amount and plausibility of the text; for pictures that all were read'),
        h('span', { class: 'cv-arrow down', 'aria-hidden': 'true' }, '↓'),
        node('3.4 · Repair', t.repair_tried ? `${num(t.repaired_cells || 0)} of ${num(t.repair_tried)} suspect cells fixed` : 'suspect table cells re-read', 'gpu', 'on', 'A table cell that breaks the table\'s arithmetic is cut out of the scan, read again by the repair model and replaced only when a second, independent reader and the arithmetic agree. Pages read as images only'),
        h('span', { class: 'cv-arrow down', 'aria-hidden': 'true' }, '↓'),
        node('3.5 · Reconcile', t.merged_tables ? `${num(t.merged_tables)} tables joined across pages` : 'tables across a page break', 'cpu', 'on', 'A table that continues on the next page is joined (the continuation gets the header) and checked across the break'),
        h('span', { class: 'cv-arrow down', 'aria-hidden': 'true' }, '↓'),
        node('Outcome', [`pass ${num(out.pass || 0)}`, out.repaired ? `repaired ${num(out.repaired)}` : null, out.no_text ? `no text ${num(out.no_text)}` : null, out.error ? `error ${num(out.error)}` : null, out.low ? `low ${num(out.low)}` : null].filter(Boolean).join(' · ') || 'pass / low confidence', null, 'on')));
  }

  /* Pages per lane as one proportional bar with the figures beside it (what each lane's reader finished). */
  function runwayBar(rw, mv) {
    const total = sumOf(rw);
    if (!total) return null;
    const moved = Object.entries(mv || {}).map(([k, v]) => `${k.replace('>', ' → ')} ${num(v)}`).join(' · ');
    return h('div', null,
      h('div', { class: 'cv-bands' }, RW_ORDER.filter(k => rw[k]).map(k => h('i', { class: 'rwb-' + k, title: `lane ${k} (${RUNWAYS[k].name}): ${num(rw[k])} pages (${Math.round(100 * rw[k] / total)}%)`, style: { flexGrow: String(rw[k]) } }))),
      h('div', { style: { marginTop: '6px' } }, RW_ORDER.filter(k => rw[k]).map(k => h('span', { class: 'chip cv-chip', title: RUNWAYS[k].tip },
        h('i', { class: 'cv-dot rwb-' + k }), `${k} · ${RUNWAYS[k].name} ${num(rw[k])}`)),
      moved ? h('span', { class: 'small muted', style: { marginLeft: '8px' } }, `handed on to the document reader: ${moved}`) : null));
  }

  /* What the run is doing right now, from the page events: pages read per branch, per open document. */
  /* Pages in the files being converted now (what kinds of pages they hold, whatever the format), the progress of
     each, and one line about the run so far: pages finished, how many were read now and how many were reused
     from the page cache (a reused page keeps its kind: it is not a kind of its own). */
  function live(l) {
    const a = (l && l.active) || {};
    if (!l || (!l.pages && !a.files)) return null;
    const docs = Object.entries(l.open || {}).map(([file, d]) => h('div', { class: 'cv-live-doc' },
      h('span', { class: 'mono' }, file), h('span', { class: 'bar cv-lane-bar', title: `${d.done} of ${d.of} pages` }, h('i', { style: { width: Math.round(100 * d.done / Math.max(1, d.of)) + '%' } })),
      h('span', { class: 'muted small nowrap' }, `${d.done}/${d.of} pages`)));
    const read = l.read !== undefined ? l.read : l.pages - (l.cached || 0);
    return h('div', { class: 'cv-live' },
      a.files ? h('div', { class: 'small muted' }, `${num(a.files)} file${a.files === 1 ? '' : 's'} being converted · ${num(a.pages)} pages (${num(a.done)} finished)`
        + (a.unprofiled ? ` · ${num(a.unprofiled)} not profiled yet` : '')) : null,
      a.pages ? bands(a.branches) : null,
      docs.length ? h('div', { style: { marginTop: '6px' } }, docs) : null,
      l.pages ? h('div', { class: 'small muted', style: { marginTop: '8px' } },
        `This run so far: ${num(l.pages)} pages finished · ${num(read)} read now`
        + (l.cached ? ` · ${num(l.cached)} reused from the page cache` : '')
        + (l.pages_per_min ? ` · ${num(l.pages_per_min)} pages read per minute` : '')
        + (l.tokens ? ` · document reader: ${num(l.tokens)} tokens in ${dur(l.gpu_s || 0)}` + (l.tokens_per_s ? ` (${num(l.tokens_per_s)} tokens/s)` : '') : '')) : null);
  }

  function lanes(list) {
    if (!list || !list.length) return null;
    return h('div', { class: 'cv-lanes' }, list.map(l => h('div', { class: 'cv-lane ' + l.state },
      h('span', { class: 'cv-lane-name' }, hw(l.kind), l.name),
      h('span', { class: 'cv-lane-state' }, l.state === 'working'
        ? h('span', null, l.phase ? h('span', { class: 'chip' }, l.phase) : null, ' ', h('b', { class: 'mono' }, l.file || '…'),
          l.stage ? ` · stage ${l.stage}${l.stage_name ? ' ' + (PL.name ? PL.name(l.stage_name) : l.stage_name) : ''}` : '',
          l.progress && l.progress.of ? ` · page ${l.progress.done} of ${l.progress.of}` : '', l.since ? ' for ' + since(l.since) : '')
        : h('span', { class: 'muted' }, 'idle')),
      h('span', { class: 'bar cv-lane-bar', title: `busy ${l.busy_pct}% of the time since it started` }, h('i', { style: { width: l.busy_pct + '%' } })),
      h('span', { class: 'muted small nowrap' }, `${l.busy_pct}% busy · ` + (l.docs_by_phase && Object.keys(l.docs_by_phase).length ? Object.entries(l.docs_by_phase).map(([k, n]) => `${num(n)} ${k}`).join(', ') : `${num(l.docs)} docs`)))));
  }

  function tiles(t) {
    t = t || {};
    if (!t.pages) return [];
    const time = t.time_s || {}, tot = Object.values(time).reduce((a, b) => a + b, 0);
    return [
      statCard(num(t.pages), 'pages converted'),
      t.pages_per_min ? statCard(num(t.pages_per_min), 'pages read per minute (reused pages not counted)') : null,
      statCard(dur(tot), PL.label('convert').toLowerCase() + ' time' + (time.profile ? ` (${PL.label('profile')} ${dur(time.profile)})` : '')),
      t.cost && t.cost.cpu_s ? statCard(dur(t.cost.cpu_s), 'CPU time' + (t.cost.peak_mb ? ` · peak ${bytes(t.cost.peak_mb * 1048576)}` : '')) : null,
      t.cached_pages ? statCard(num(t.cached_pages), 'pages from the page cache (not read again)') : null,
      t.step_s && t.step_s.read ? statCard(dur(t.step_s.read), PL.label('read') + ' pages' + (t.step_s.gate ? ` · ${PL.label('gate')} ${dur(t.step_s.gate)}` : '')) : null,
      t.low_docs ? statCard(num(t.low_docs), 'documents with low-confidence pages') : null,
    ];
  }

  /* ---------- the document drawer ---------- */
  let drawer = null, st = null;
  function ensureDrawer() {
    if (drawer) return drawer;
    drawer = h('aside', { class: 'cv-drawer hidden', role: 'dialog', 'aria-label': 'Document conversion details' });
    document.body.append(drawer);
    document.addEventListener('keydown', e => { if (e.key === 'Escape' && !drawer.classList.contains('hidden')) close(); });
    return drawer;
  }
  function close() { if (drawer) drawer.classList.add('hidden'); st = null; }
  function split(tracePath) {
    const m = String(tracePath || '').replace(/\.trace\.json$/, '');
    const i = m.indexOf('/');
    return i < 0 ? null : [m.slice(0, i), m.slice(i + 1)];
  }
  const qs = (coll, doc, extra) => new URLSearchParams({ collection: coll, doc, ...(st && st.exp ? { exp: st.exp } : {}), ...(extra || {}) }).toString();

  /* *exp*: a playground experiment's name; its documents are read from the experiment's own workspace. */
  async function open(coll, doc, exp) {
    ensureDrawer();
    st = { exp: exp || '', coll, doc, data: null, error: '', page: 0, detail: null, image: false, md: null };
    render(); drawer.classList.remove('hidden');
    const r = await api('conversion/trace?' + qs(coll, doc));
    if (!st || st.coll !== coll || st.doc !== doc) return;
    if (r.ok === false) st.error = r.error || 'no trace'; else st.data = r.result;
    render();
  }
  function openSummary(conv, exp) {
    const parts = split(conv && conv.trace);
    if (parts) open(parts[0], parts[1], exp);
    else toast('No page record for this document (indexed before conversion tracking). Re-convert it to get one.', '');
  }
  async function pick(n) {
    if (!st) return;
    st.page = n; st.detail = null; st.image = false; st.md = null; render();
    const cur = st; const r = await api('conversion/trace?' + qs(st.coll, st.doc, { page: n }));
    if (st !== cur || st.page !== n) return;
    st.detail = r.ok === false ? { error: r.error } : (r.result.pages[0] || {});
    render();
  }

  /* The converted Markdown of the selected page, as the chunker read it. */
  async function showMarkdown() {
    if (!st) return;
    const cur = st, n = st.page;
    st.md = { loading: true }; render();
    const r = await api('conversion/markdown?' + qs(st.coll, st.doc, { page: n }));
    if (st !== cur || st.page !== n) return;
    st.md = r.ok === false ? { error: r.error || 'no converted text' } : r.result;
    render();
  }
  /* A source page that cannot be shown: ask the server why instead of guessing. */
  async function imageError(e, url) {
    const box = h('div', { class: 'notice warn' }, 'The source page cannot be shown.');
    e.target.replaceWith(box);
    const r = await api(url.replace(/^\/api\//, ''));
    if (r && r.error) box.textContent = 'The source page cannot be shown: ' + r.error;
  }
  function markdownBlock() {
    const m = st.md;
    if (!m) return null;
    if (m.loading) return h('p', { class: 'small muted' }, 'Loading the converted text…');
    if (m.error) return h('div', { class: 'notice warn', style: { marginTop: '10px' } }, m.error);
    return h('div', null,
      h('div', { class: 'small muted', style: { marginTop: '10px' } }, `Converted Markdown of page ${m.page} · ${num(m.chars)} characters` + (m.truncated ? ' (cut: open the whole document for the rest)' : '')),
      h('pre', { class: 'cv-page-md' }, m.markdown || '(empty)'));
  }

  function pageGrid(pages) {
    return h('div', { class: 'cv-grid' }, pages.map(p => h('button', {
      type: 'button',
      class: `cv-cell cvb-${p.branch} out-${p.outcome}` + (st.page === p.page ? ' sel' : ''),
      title: `page ${p.page} · ${label(p.branch)} · ${p.outcome}` + (p.grade ? ` · docling: ${p.grade}` : '') + (p.chars !== undefined && p.chars !== null ? ` · ${p.chars} characters` : ''),
      on: { click: () => pick(p.page) },
    }, p.outcome === 'low' || p.outcome === 'error' ? '!' : String(p.page))));
  }

  function detailBlock() {
    const d = st.detail;
    if (!st.page) return h('p', { class: 'small muted' }, 'Click a page to see why it took this branch and what docling reported.');
    if (!d) return h('p', { class: 'small muted' }, 'Loading page ' + st.page + '…');
    if (d.error) return h('div', { class: 'notice bad' }, d.error);
    const kvs = o => Object.entries(o || {}).filter(([, v]) => v !== null && v !== undefined && v !== '' && typeof v !== 'object')
      .map(([k, v]) => [k.replace(/_/g, ' '), String(typeof v === 'number' && !Number.isInteger(v) ? Math.round(v * 100) / 100 : v)]);
    const img = '/api/conversion/page-image?' + qs(st.coll, st.doc, { page: st.page, width: 700 });
    return h('div', null,
      h('h4', null, `Page ${d.page}`),
      kv([
        ['Branch', h('span', null, h('i', { class: 'cv-dot cvb-' + d.branch }), ' ' + label(d.branch), h('span', { class: 'muted small' }, ' · ' + branchTitle(d.branch)))],
        ['Outcome', `${d.outcome}: ${OUTCOMES[d.outcome] || ''}`], ['Why', d.why],
        ...kvs(d.profile).map(([k, v]) => ['profile · ' + k, v]),
        ...kvs(d.docling).map(([k, v]) => ['docling · ' + k, v]),
        ...kvs(d.reader).map(([k, v]) => ['reader · ' + k, v]),
        ...kvs(d.time_s).map(([k, v]) => ['time · ' + PL.label(k), v + ' s']),
        d.cache ? ['Page cache', d.cache === 'hit' ? 'reused: ' + (d.note || 'not read again') : d.cache] : null,
        ...kvs(d.out).map(([k, v]) => ['result · ' + k, v]),
        d.gate && d.gate.checks && d.gate.checks.length ? ['Failed checks', h('div', null, d.gate.checks.map(c => h('div', { class: 'small' }, h('b', null, c.name.replace('_', ' ')), c.detail ? ': ' + c.detail : '')))] : null,
        d.gate && d.gate.violations && d.gate.violations.length ? ['Suspect cells', h('div', null, d.gate.violations.map(v => h('div', { class: 'small mono' }, `row ${v.row}, col ${v.col + 1}: ${v.found || '(empty)'} → ${v.expected}`)))] : null,
        d.repair ? ['Repair', h('div', null,
          h('div', { class: 'small' }, `${d.repair.fixed || 0} of ${d.repair.tried || 0} suspect cell(s) fixed` + (d.repair.tier === 'page' ? ' · page read again by the repair model' : '') + (d.repair.model ? ` · ${d.repair.model}` : '') + (d.repair.second ? ` · second reader ${d.repair.second}` : '')),
          (d.repair.cells || []).map(c => h('div', { class: 'small mono' }, `row ${c.row + 1}, col ${c.col + 1}: ${c.before || '(empty)'} → ${c.status === 'fixed' ? c.after : '(kept)'}  [${c.status}${c.status !== 'fixed' && c.why ? ': ' + c.why : ''}]`)))] : null,
        d.reconcile ? ['Table across pages', h('div', { class: 'small' }, (d.reconcile.role === 'starts' ? `continues on page ${d.reconcile.with}` : `continues the table of page ${d.reconcile.with}`) + ` · ${d.reconcile.rows} rows in all` + (d.reconcile.header === 'added' ? ' · header copied onto this page' : ' · header repeated'),
          (d.reconcile.violations || []).map(v => h('div', { class: 'small mono' }, `row ${v.row + 1}, col ${v.col + 1}: ${v.found || '(empty)'} → ${v.expected} (${v.why})`)))] : null,
        d.note && d.cache !== 'hit' ? ['Note', d.note] : null].filter(Boolean)),
      h('div', { class: 'row', style: { marginTop: '10px' } },
        st.image ? null : h('button', { class: 'btn small', on: { click: () => { st.image = true; render(); } } }, 'Show the source page'),
        st.md ? null : h('button', { class: 'btn small', on: { click: showMarkdown } }, 'Show the converted Markdown')),
      st.image ? h('img', { class: 'cv-page-img', src: img, alt: `page ${d.page} of the source`, on: { error: e => imageError(e, img) } }) : null,
      markdownBlock());
  }

  function render() {
    if (!st) return;
    const d = st.data, s = d && d.summary;
    const body = !d && !st.error ? h('p', { class: 'muted' }, 'Loading…')
      : st.error ? h('div', { class: 'notice warn' }, st.error)
      : h('div', null,
        h('div', { class: 'row small muted' }, d.source ? h('span', null, 'source ', h('b', { class: 'mono' }, d.source)) : null,
          d.convert ? h('span', null, '· ' + d.convert) : null, d.written_at ? h('span', null, '· recorded ' + clock(d.written_at)) : null),
        s ? h('div', { style: { margin: '10px 0' } }, strip(s), h('div', { style: { marginTop: '8px' } }, branchChips(s.branches), outcomeChips(s.outcomes))) : null,
        s ? kv([
          ['Time', Object.entries(s.time_s || {}).map(([k, v]) => `${PL.label(k)} ${dur(v)}`).join(' · ') || null],
          ['Reading', s.step_s && s.step_s.read !== undefined ? `${PL.label('read')} ${dur(s.step_s.read)} · ${PL.label('gate')} ${dur(s.step_s.gate || 0)}` + (s.cached_pages ? ` · ${s.cached_pages} from the page cache` : '') : null],
          ['Failed checks', Object.keys(s.gate_failed || {}).length ? Object.entries(s.gate_failed).map(([k, v]) => `${k.replace(/_/g, ' ')} ${v}`).join(', ') : null],
          ['Repair', s.repair_tried ? `${num(s.repaired_cells || 0)} of ${num(s.repair_tried)} suspect cell(s) fixed` : null],
          ['Tables across pages', s.merged_tables ? num(s.merged_tables) : null],
          ['Cost', costText(s.cost) || null],
          ['Scripts', Object.keys(s.scripts || {}).length ? Object.entries(s.scripts).map(([k, v]) => `${k} ${v}`).join(', ') : null],
          ['Tables / big pictures', s.tables || s.big_pictures ? `${num(s.tables)} / ${num(s.big_pictures)}` : null],
          ['Docling grades', Object.keys(s.docling_grades || {}).length ? Object.entries(s.docling_grades).map(([k, v]) => `${k} ${v}`).join(', ') : null],
          ['Readers', (s.readers || []).join(', ') || null],
        ]) : null,
        d.note ? h('p', { class: 'small muted' }, d.note) : null,
        h('p', { class: 'small' }, h('a', { href: '/api/conversion/markdown?' + qs(st.coll, st.doc, { raw: 1 }), target: '_blank', rel: 'noopener' }, 'Open the whole converted Markdown'), h('span', { class: 'muted' }, ' · the text that was chunked and indexed, in a new tab')),
        (d.pages || []).length ? h('div', null, h('h4', null, `Pages (${d.pages.length})`), pageGrid(d.pages)) : null,
        h('div', { style: { marginTop: '12px' } }, detailBlock()));
    fill(drawer,
      h('div', { class: 'cv-drawer-head' }, h('h3', null, h('span', { class: 'muted' }, st.coll + '/'), st.doc),
        h('button', { class: 'btn small', on: { click: close } }, 'Close')),
      body);
  }

  /* Estimate (dry run) result as a notice. */
  function estimateView(e) {
    const rows = [
      ['Files', `${num(e.profiled)} of ${num(e.files)} profiled in ${dur(e.profile_s)}` + (e.partial ? ' (time limit reached: the rest is not counted)' : '')],
      ['Pages', num(e.pages)],
      ['Branches', e.branches && Object.keys(e.branches).length ? branchChips(e.branches) : null],
      ['By file type', Object.entries(e.by_extension || {}).map(([k, v]) => `${k || 'none'} ${v}`).join(', ') || null],
      ['Scripts', Object.entries(e.scripts || {}).map(([k, v]) => `${k} ${v}`).join(', ') || null],
      ['docling today', `about ${dur(e.docling.seconds)} (${e.docling.s_per_page} s per page, ${e.docling.basis})`],
      e.planned_vlm && e.planned_vlm.pages ? ['Document reader', e.planned_vlm.reader && e.planned_vlm.reader.usable
        ? `${num(e.planned_vlm.pages)} scanned pages / images with ${e.planned_vlm.reader.model}: about ${dur(e.planned_vlm.seconds_low)} to ${dur(e.planned_vlm.seconds_high)} (estimate)`
        : `${num(e.planned_vlm.pages)} scanned pages / images read by docling OCR: the reader cannot run now (${(e.planned_vlm.reader && e.planned_vlm.reader.why) || 'unknown'})`] : null,
    ].filter(Boolean);
    return h('div', { class: 'notice', style: { marginTop: '12px' } }, h('b', null, 'Estimate (nothing was converted)'), kv(rows),
      e.error_count ? h('div', { class: 'small', style: { color: 'var(--bad)' } }, `${e.error_count} file(s) could not be profiled: ` + (e.errors || []).slice(0, 3).map(x => x.src).join(', ')) : null);
  }

  return { label, branchTitle, strip, branchChips, outcomeChips, bands, runwayBar, RUNWAYS, flow, lanes, tiles, costText, hw, open, openSummary, estimateView, sortedBranches, live, BRANCHES, OUTCOMES };
})();
