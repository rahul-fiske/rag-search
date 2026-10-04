'use strict';
/* rag-search dashboard core: helpers, live state (Server-Sent Events), router, header. */

const RS = { state: { live: null, catalog: null }, views: {}, current: '', conn: 'connecting' };

/* ---------- tiny DOM helper: h('div', {class:'x', on:{click:fn}}, 'text', child, [more]) ---------- */
function h(tag, props, ...kids) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(props || {})) {
    if (k === 'open' && tag === 'details') el.__open = !!v;   // the caller tracks this twisty's state (see morphChildren)
    if (v === null || v === undefined || v === false) continue;
    if (k === 'class') el.className = v;
    else if (k === 'on') { el.__l = el.__l || {}; for (const [ev, fn] of Object.entries(v)) { el.__l[ev] = fn; el.addEventListener(ev, fn); } }
    else if (k === 'dataset') Object.assign(el.dataset, v);
    else if (k === 'style' && typeof v === 'object') Object.assign(el.style, v);
    else if (k === 'value' || k === 'checked' || k === 'disabled' || k === 'selected') el[k] = v;
    else el.setAttribute(k, v === true ? '' : v);
  }
  add(el, kids);
  return el;
}
function add(el, kids) {
  for (const k of kids.flat(Infinity)) {
    if (k === null || k === undefined || k === false) continue;
    el.append(k instanceof Node ? k : document.createTextNode(String(k)));
  }
  return el;
}
function fill(el, ...kids) { el.replaceChildren(); add(el, kids); return el; }

/* patch(): update a live area in place. The new content is built off-screen and merged into the
   existing nodes, so buttons keep focus and clicks are not lost while numbers tick, and an
   expanded <details> stays expanded. */
function patch(container, ...kids) {
  const tmp = document.createElement('div');
  add(tmp, kids);
  morphChildren(container, tmp);
  return container;
}
function morphChildren(oldParent, newParent) {
  const o = Array.from(oldParent.childNodes), n = Array.from(newParent.childNodes);
  for (let i = 0; i < Math.max(o.length, n.length); i++) {
    const a = o[i], b = n[i];
    if (!b) { a.remove(); continue; }
    if (!a) { oldParent.append(b); continue; }
    if (a.nodeType !== b.nodeType || a.nodeName !== b.nodeName) { oldParent.replaceChild(b, a); continue; }
    if (a.nodeType === 3) { if (a.nodeValue !== b.nodeValue) a.nodeValue = b.nodeValue; continue; }
    if (a.nodeType !== 1) continue;
    // A <details> the caller does not track (no `open` prop given to h()) keeps whatever the user
    // did to it: without this every live tick removed `open` and the section fell shut again.
    const keepOpen = a.nodeName === 'DETAILS' && b.__open === undefined;
    for (const at of Array.from(a.attributes)) if (!b.hasAttribute(at.name) && !(keepOpen && at.name === 'open')) a.removeAttribute(at.name);
    for (const at of Array.from(b.attributes)) if (a.getAttribute(at.name) !== at.value) a.setAttribute(at.name, at.value);
    if ('disabled' in a) a.disabled = b.disabled;
    for (const [ev, fn] of Object.entries(a.__l || {})) a.removeEventListener(ev, fn);
    a.__l = b.__l || {};
    for (const [ev, fn] of Object.entries(a.__l)) a.addEventListener(ev, fn);
    morphChildren(a, b);
  }
}
const $ = (sel, root) => (root || document).querySelector(sel);
const $$ = (sel, root) => Array.from((root || document).querySelectorAll(sel));

/* ---------- formatting ---------- */
function bytes(n) {
  if (n === null || n === undefined || isNaN(n)) return '–';
  const u = ['B', 'KB', 'MB', 'GB', 'TB']; let i = 0; n = Number(n);
  while (n >= 1024 && i < u.length - 1) { n /= 1024; i++; }
  return (i === 0 ? n.toFixed(0) : n < 10 ? n.toFixed(1) : n.toFixed(0)) + ' ' + u[i];
}
function dur(s) {
  if (s === null || s === undefined || isNaN(s)) return '–';
  s = Number(s);
  if (s < 1) return Math.round(s * 1000) + ' ms';
  if (s < 60) return s.toFixed(s < 10 ? 1 : 0) + ' s';
  const m = Math.floor(s / 60), sec = Math.round(s % 60);
  if (m < 60) return m + 'm ' + String(sec).padStart(2, '0') + 's';
  return Math.floor(m / 60) + 'h ' + String(m % 60).padStart(2, '0') + 'm';
}
function ms(v) { return v === null || v === undefined ? '–' : v + ' ms'; }
function toDate(t) {
  if (t === null || t === undefined || t === '') return null;
  const d = typeof t === 'number' ? new Date(t * 1000) : new Date(t);
  return isNaN(d) ? null : d;
}
function clock(t) {
  const d = toDate(t); if (!d) return '–';
  return d.toLocaleString(undefined, { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit', second: '2-digit' });
}
function ago(t) {
  const d = toDate(t); if (!d) return '–';
  const s = Math.max(0, (Date.now() - d.getTime()) / 1000);
  return s < 5 ? 'just now' : dur(s) + ' ago';
}
function since(t) { const d = toDate(t); return d ? dur(Math.max(0, (Date.now() - d.getTime()) / 1000)) : '–'; }
function num(n) { return n === null || n === undefined ? '–' : Number(n).toLocaleString(); }
function pill(text, cls, busy) { return h('span', { class: 'pill ' + (cls || '') + (busy ? ' busy' : '') }, text); }
function chip(text, cls) { return h('span', { class: 'chip ' + (cls || '') }, text); }
function statCard(value, label) { return h('div', { class: 'stat' }, h('b', null, value), h('span', null, label)); }
function kv(rows) {
  const dl = h('dl', { class: 'kv' });
  for (const [k, v] of rows) { if (v === null || v === undefined) continue; dl.append(h('dt', null, k), h('dd', null, v)); }
  return dl;
}
function empty(text) { return h('div', { class: 'empty' }, text); }

/* ---------- shared pipeline-tunables form (Settings tab) ----------
   One <input>/<select> per tunable, built ONCE (never rebuilt on a live tick -- see indexing.js's
   docsCard for why that matters for anything the user might be typing into). `label` and `what`
   are always visible; `impact` and "when does this take effect" sit behind an info icon's native
   title tooltip, so the screen stays uncluttered.  A single "Save" button per section POSTs
   every field at once via /api/config/set.

   Blank/0 in the underlying value always means "no override" (see spec.py) -- but a bare "0" or
   an empty box reads as "the value IS zero/nothing", not "nothing is set". So every field also
   shows `default_label` (the actual resolved value/formula that applies today): as the number
   placeholder, as the text placeholder, or as the blank <option>'s own label for a dropdown.
   Each dropdown option also gets a `choice_help` one-liner as its native title tooltip, so what
   each value means is a hover away without adding any new visible UI. */
function tunableField(t) {
  const id = 'tun-' + t.section + '-' + t.key;
  const dflt = t.default_label || 'built-in default';
  let input;
  if (t.kind === 'choice') {
    const help = t.choice_help || {};
    input = h('select', { id },
      h('option', { value: '', title: `Use the built-in default: ${dflt}` }, `(default: ${dflt})`),
      t.choices.map(c => h('option', { value: c, title: help[c] || '' }, c)));
  } else if (t.kind === 'int') {
    input = h('input', { id, type: 'number', min: '0', placeholder: `0 = ${dflt}` });
  } else {
    input = h('input', { id, type: 'text', placeholder: `(default: ${dflt})` });
  }
  const reset = h('button', {
    type: 'button', class: 'btn small tunable-reset',
    title: `Clear this field back to the built-in default (${dflt})`,
    on: {
      click: () => {
        input.value = '';
        reset.disabled = true;
        input.focus();
      },
    },
  }, 'Reset');
  // An int tunable's sentinel is 0 (not ''), so "0" in the box is still "no override" -- only
  // disable the reset button when there is truly nothing to clear.
  const syncReset = () => {
    const v = input.value;
    reset.disabled = t.kind === 'int' ? (v === '' || Number(v) === 0) : v === '';
  };
  input.addEventListener('input', syncReset);
  input.addEventListener('change', syncReset);
  const row = h('div', { class: 'tunable-row' },
    h('div', { class: 'tunable-head' },
      h('label', { for: id }, t.label),
      h('span', { class: 'info-icon', tabindex: '0',
                 title: `${t.impact}\n\n${t.applies_label}${t.env ? ' · env: ' + t.env : ''}` }, 'ⓘ')),
    h('div', { class: 'small muted' }, t.what),
    h('div', { class: 'tunable-input-row' }, input, reset));
  row.__input = input;
  row.__syncReset = syncReset;
  return row;
}

/* Builds the section's fields once and returns {root, load(values), collect()}. `root` is meant
   to be appended once (e.g. in a view's init()); `load` sets each field's current value (call on
   fetch/after save) without ever recreating the <input> elements; `collect` reads them back into
   a {key: value} object suitable for POSTing to /api/config/set. */
function tunablesForm(tunables) {
  const rows = tunables.map(tunableField);
  const root = h('div', { class: 'tunables-grid' }, rows);
  return {
    root,
    load(values) {
      rows.forEach((row, i) => {
        const t = tunables[i], v = (values || {})[t.key];
        // 0 (int) / "" (choice, text, stages) is the "no override" sentinel -- show the box
        // genuinely blank (so its placeholder/blank-option can say what's actually in effect)
        // rather than a literal "0" that reads as "the value is zero".
        const blank = v === undefined || v === null || v === '' || (t.kind === 'int' && Number(v) === 0);
        row.__input.value = blank ? '' : v;
        row.__syncReset();
      });
    },
    collect() {
      const out = {};
      rows.forEach((row, i) => { out[tunables[i].key] = row.__input.value; });
      return out;
    },
  };
}

/* ---------- messages ---------- */
function toast(msg, kind, ttl) {
  const t = h('div', { class: 'toast ' + (kind || '') }, msg);
  $('#toasts').append(t);
  setTimeout(() => t.remove(), ttl || (kind === 'bad' ? 9000 : 4500));
}
function dialog(title, body, opts) {
  opts = opts || {};
  const dlg = $('#dlg');
  $('#dlg-title').textContent = title;
  fill($('#dlg-body'), body);
  const ok = $('#dlg-ok'), cancel = $('#dlg-cancel');
  ok.textContent = opts.okText || 'OK';
  ok.className = 'btn ' + (opts.danger ? 'danger' : 'primary');
  cancel.classList.toggle('hidden', opts.noCancel === true);
  return new Promise(resolve => {
    dlg.addEventListener('close', () => resolve(dlg.returnValue === 'ok'), { once: true });
    dlg.returnValue = 'cancel';
    dlg.showModal();
  });
}
const confirmDialog = (title, body, okText, danger) => dialog(title, body, { okText, danger });

/* ---------- API ---------- */
async function api(path, body) {
  const opt = body === undefined ? {} : { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) };
  let r;
  try { r = await fetch('/api/' + path, opt); } catch (e) { return { ok: false, error: 'cannot reach the dashboard server', network: true }; }
  if (r.status === 401) { showBanner('Session not authorised. Start the dashboard with `rag-search ui` and use the link it prints.'); }
  let data = {};
  try { data = await r.json(); } catch (e) { data = { ok: false, error: 'bad reply from the server' }; }
  if (!r.ok && data.ok === undefined) data.ok = false;
  return data;
}
async function act(path, body, okMsg) {
  const r = await api(path, body);
  if (r.ok === false) toast(r.error || 'failed', 'bad');
  else if (okMsg) toast(okMsg, 'ok');
  return r;
}
function showBanner(text) { const b = $('#banner'); b.textContent = text; b.classList.toggle('hidden', !text); }
const readOnly = () => !!(RS.state.catalog && RS.state.catalog.read_only);

/* ---------- derived status ---------- */
function searchStatus() {
  const live = RS.state.live; const d = live && live.daemons && live.daemons.search;
  if (!live) return { label: 'connecting…', cls: '', up: false };
  if (!d || !d.state) return d && d.starting ? { label: 'starting', cls: 'warn', busy: true, up: false } : { label: 'stopped', cls: '', up: false };
  const w = d.warmup || {};
  if (d.state === 'error' || d.error || w.status === 'error') return { label: 'error', cls: 'bad', up: true };
  if (w.status === 'warming_up' || d.state === 'loading_models' || d.state === 'starting')
    return { label: 'warming up' + (w.elapsed_s ? ' ' + dur(w.elapsed_s) : ''), cls: 'warn', busy: true, up: true };
  return { label: 'warm', cls: 'ok', up: true };
}
function indexerStatus() {
  const live = RS.state.live; const d = live && live.daemons && live.daemons.indexer;
  if (!live) return { label: 'connecting…', cls: '', up: false };
  if (!d || !d.state) return d && d.starting ? { label: 'starting', cls: 'warn', busy: true, up: false } : { label: 'stopped', cls: '', up: false };
  const job = live.index && live.index.job;
  if (d.running) {
    const p = job && job.progress;
    return { label: 'indexing' + (p && p.total ? ` ${p.done || 0}/${p.total}` : ''), cls: 'warn', busy: true, up: true };
  }
  return { label: 'idle', cls: 'ok', up: true };
}
function activeJob() {
  const live = RS.state.live; const job = live && live.index && live.index.job;
  return job && (job.status === 'running' || job.status === 'queued') ? job : null;
}
function knownClients() {
  const a = RS.state.catalog && RS.state.catalog.access;
  return a ? a.clients.map(c => c.client).filter(c => c !== 'cli') : ['claude'];
}

/* ---------- header ---------- */
function updateHeader() {
  const s = searchStatus(), i = indexerStatus();
  const conn = RS.conn === 'live' ? pill('live', 'ok') : RS.conn === 'connecting' ? pill('connecting', 'warn', true) : pill('reconnecting', 'bad', true);
  fill($('#pills'), h('span', { class: 'small muted' }, 'search daemon'), pill(s.label, s.cls, s.busy),
    h('span', { class: 'small muted' }, 'indexer'), pill(i.label, i.cls, i.busy), conn,
    readOnly() ? pill('read-only', 'plain') : null);
  const v = RS.state.catalog && RS.state.catalog.version;
  if (v) { $('#ver').textContent = 'v' + v; $('#ver2').textContent = 'v' + v; }
  const iv = RS.state.catalog && RS.state.catalog.installed_version;
  if (iv && v && iv !== v && RS.conn === 'live') {
    showBanner(`rag-search ${iv} is installed, but this dashboard is still running ${v}. Restart it: run "rag-search ui" in a terminal (it replaces the old one), then reload this page.`);
  }
}

/* ---------- live connection ---------- */
function onData(kind, data) {
  RS.state[kind] = data;
  updateHeader();
  const v = RS.views[RS.current];
  if (v && v.update) { try { v.update(RS.state); } catch (e) { console.error(e); } }
}
function connect() {
  if (!window.EventSource) { poll(); return; }
  const es = new EventSource('/api/events');
  es.addEventListener('live', e => onData('live', JSON.parse(e.data)));
  es.addEventListener('catalog', e => onData('catalog', JSON.parse(e.data)));
  es.onopen = () => { RS.conn = 'live'; showBanner(''); updateHeader(); };
  es.onerror = () => {
    RS.conn = RS.conn === 'live' ? 'lost' : RS.conn; updateHeader();
    setTimeout(async () => { const r = await fetch('/api/ping'); if (r.ok) { const s = await fetch('/api/state'); if (s.status === 401) showBanner('Session not authorised. Start the dashboard with `rag-search ui` and use the link it prints.'); } }, 1500);
  };
}
async function poll() {
  const r = await api('state');
  if (r.live) { RS.conn = 'live'; onData('live', r.live); }
  if (r.catalog) onData('catalog', r.catalog);
  setTimeout(poll, 2000);
}

/* ---------- router ---------- */
function route() {
  const name = (location.hash.replace(/^#\/?/, '').split('/')[0]) || 'overview';
  const view = RS.views[name] ? name : 'overview';
  if (RS.current && RS.views[RS.current] && RS.views[RS.current].leave) RS.views[RS.current].leave();
  RS.current = view;
  for (const a of $$('#tabs a')) a.toggleAttribute('aria-current', a.dataset.tab === view), a.setAttribute('aria-current', a.dataset.tab === view ? 'page' : 'false');
  for (const s of $$('.view')) s.classList.toggle('active', s.id === 'v-' + view);
  const v = RS.views[view], root = $('#v-' + view);
  if (!root.dataset.ready) { v.init(root); root.dataset.ready = '1'; }
  if (v.show) v.show();
  if (v.update && (RS.state.live || RS.state.catalog)) v.update(RS.state);
  window.scrollTo(0, 0);
}

/* ---------- theme ---------- */
function initTheme() {
  let saved = null; try { saved = localStorage.getItem('rs-theme'); } catch (e) { /* private mode */ }
  if (saved) document.documentElement.dataset.theme = saved;
  $('#theme').addEventListener('click', () => {
    const cur = document.documentElement.dataset.theme || (matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light');
    const next = cur === 'dark' ? 'light' : 'dark';
    document.documentElement.dataset.theme = next;
    try { localStorage.setItem('rs-theme', next); } catch (e) { /* ignore */ }
  });
}

async function boot() {
  initTheme();
  const first = await api('state');
  if (first.live) RS.state.live = first.live;
  if (first.catalog) RS.state.catalog = first.catalog;
  window.addEventListener('hashchange', route);
  route(); updateHeader();
  connect();
  setInterval(() => { updateHeader(); const v = RS.views[RS.current]; if (v && v.tick) v.tick(); }, 1000);
}
window.addEventListener('DOMContentLoaded', () => setTimeout(boot, 0));
