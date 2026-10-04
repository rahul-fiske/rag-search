'use strict';
/* Help: the command-line reference (generated from the real parser) and the README guide. */
(function () {
  let refs = {}, data = null, tab = 'cli', loading = false;

  const QUICK = [
    ['Start the dashboard', 'rag-search ui'],
    ['Index new and changed documents', 'rag-search index new'],
    ['Search from the terminal', 'rag-search search "how do I enable MFA?" -k 5'],
    ['Find exact text', 'rag-search grep "error code \\d+"'],
    ['Who may use which collection', 'rag-search access'],
    ['Only claude may use “hr”', 'rag-search access restrict hr claude'],
    ['Check the installation', 'rag-search doctor'],
    ['Daemon status', 'rag-search daemon status'],
    ['Indexing progress', 'rag-search index status']];

  function copyBtn(text) {
    return h('button', { class: 'btn small', title: 'Copy', on: { click: e => { e.stopPropagation(); navigator.clipboard && navigator.clipboard.writeText(text); toast('Copied', 'ok', 1200); } } }, 'Copy');
  }

  function cmdCard(c, open) {
    const body = h('div', { class: 'body' },
      c.description && c.description !== c.help ? h('p', null, c.description) : null,
      h('pre', null, c.usage),
      c.args.length ? h('div', { class: 'table-wrap' }, h('table', null,
        h('thead', null, h('tr', null, ['Argument', 'What it does', 'Default'].map(t => h('th', null, t)))),
        h('tbody', null, c.args.map(a => h('tr', null, h('td', { class: 'nowrap' }, h('code', null, a.flags)), h('td', null, a.help || ''), h('td', { class: 'muted' }, a.default || '')))))) : h('p', { class: 'muted small' }, 'No arguments.'));
    const d = h('details', { class: 'cmd', 'data-cmd': c.command }, h('summary', null, h('span', { class: 'name' }, 'rag-search ' + c.command), h('span', { class: 'muted' }, c.help || ''), h('span', { style: { marginLeft: 'auto' } }, copyBtn('rag-search ' + c.command))), body);
    if (open) d.open = true;
    return d;
  }

  function renderCli() {
    const cli = data.cli; const f = (refs.filter.value || '').trim().toLowerCase();
    const cmds = cli.commands.filter(c => !f || (c.command + ' ' + c.help + ' ' + c.args.map(a => a.flags + ' ' + a.help).join(' ')).toLowerCase().includes(f));
    fill(refs.cli,
      h('div', { class: 'card', style: { marginBottom: '16px' } }, h('div', { class: 'card-head' }, h('h3', null, 'Quick reference')),
        h('div', { class: 'table-wrap' }, h('table', null, h('tbody', null, QUICK.map(([t, c]) => h('tr', null, h('td', null, t), h('td', null, h('code', null, c)), h('td', { class: 'num' }, copyBtn(c)))))))),
      h('div', { class: 'card-head' }, h('h3', null, `All commands (${cmds.length}${f ? ' of ' + cli.commands.length : ''})`), h('span', { class: 'muted small' }, 'generated from the installed command line, so it always matches this version')),
      cmds.length ? cmds.map(c => cmdCard(c, !!f && cmds.length <= 4)) : empty('No command matches.'),
      cli.global.length ? h('div', { class: 'card', style: { marginTop: '16px' } }, h('h3', null, 'Options for every command'),
        h('div', { class: 'table-wrap' }, h('table', null, h('tbody', null, cli.global.map(a => h('tr', null, h('td', { class: 'nowrap' }, h('code', null, a.flags)), h('td', null, a.help)))))),
        h('p', { class: 'small muted' }, 'Most commands also accept ', h('code', null, '--json'), ' for machine-readable output, ', h('code', null, '--home DIR'), ' to use another data folder and ', h('code', null, '--client NAME'), ' to run as another identity (administrator only).')) : null);
  }

  function renderGuide() {
    const d = data.readme || {}; const md = h('div', { class: 'md' });
    md.innerHTML = d.html || '<p>The README is not packaged with this install.</p>';
    const toc = h('nav', { class: 'toc', 'aria-label': 'Contents' }, h('h4', null, 'Contents'),
      (d.toc || []).map(t => h('a', { href: '#/help', class: 'l' + t.level, on: { click: e => { e.preventDefault(); const el = md.querySelector('#' + CSS.escape(t.id)); if (el) el.scrollIntoView({ behavior: 'smooth', block: 'start' }); } } }, t.title)));
    fill(refs.guide, h('div', { class: 'with-toc' }, toc, md));
  }

  function setTab(t) {
    tab = t;
    for (const b of $$('button[data-t]', refs.nav)) b.classList.toggle('on', b.dataset.t === t);
    refs.cli.classList.toggle('hidden', t !== 'cli'); refs.guide.classList.toggle('hidden', t !== 'guide');
    refs.filterBox.classList.toggle('hidden', t !== 'cli');
  }

  async function load() {
    if (data || loading) return;
    loading = true;
    const r = await api('help'); loading = false;
    if (r.ok === false) { fill(refs.cli, h('div', { class: 'notice bad' }, r.error || 'cannot load help')); return; }
    data = r; renderCli(); renderGuide();
  }

  RS.views.help = {
    init(root) {
      refs.filter = h('input', { type: 'search', placeholder: 'Filter commands, e.g. index, access, --json', style: { width: '320px', maxWidth: '100%' }, on: { input: () => data && renderCli() } });
      refs.filterBox = h('div', { style: { marginLeft: 'auto' } }, refs.filter);
      refs.nav = h('div', { class: 'subnav', style: { alignItems: 'center' } },
        h('button', { type: 'button', 'data-t': 'cli', class: 'on', on: { click: () => setTab('cli') } }, 'Command line'),
        h('button', { type: 'button', 'data-t': 'guide', on: { click: () => setTab('guide') } }, 'User guide (README)'), refs.filterBox);
      refs.cli = h('div', null, empty('Loading…')); refs.guide = h('div', { class: 'hidden' });
      root.append(h('h2', null, 'Help & command line'), refs.nav, refs.cli, refs.guide);
    },
    show() { load(); setTab(tab); },
    update() { },
  };
})();
