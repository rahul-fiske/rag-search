'use strict';
/* People: name the faces found in photographs, and group the people (family, friends). Names live in one place
   (<home>/people.sqlite), never in the photographs or the index; see core/faces.py and core/people.py. */
(function () {
  let refs = {}, data = null, clusters = null, person = null, skip = 0, ticks = 0;
  const off = new Set();          // faces left out of the next "Name" (clicked in a group of look-alikes)
  const hidden = new Set();       // groups skipped for now (by their first face)

  const faceImg = (f, extra) => h('img', Object.assign({ class: 'face', src: 'api/people/face?id=' + f, loading: 'lazy', alt: '', width: 72, height: 72 }, extra || {}));
  /* a face with a small "photo" button: opens the whole photograph with every face found in it boxed */
  const faceWithPhoto = (f, extra) => h('span', { class: 'facewrap' }, faceImg(f, extra),
    h('button', { class: 'facezoom', type: 'button', title: 'see the whole photograph, with every face found in it', 'aria-label': 'see the whole photograph',
      on: { click: e => { e.stopPropagation(); e.preventDefault(); openPhoto(f); } } }, '\u{1F50D}'));

  /* ---- the whole photograph of a face, with a box on every face found in it ---- */
  let viewer = null;
  async function openPhoto(faceId) {
    if (!viewer) {
      viewer = h('dialog', { class: 'photo-viewer' });
      viewer.addEventListener('click', e => { if (e.target === viewer) viewer.close(); });
      document.body.append(viewer);
    }
    fill(viewer, h('p', { class: 'muted', style: { padding: '24px' } }, 'loading\u2026'));
    if (!viewer.open) viewer.showModal();
    await paintPhoto(faceId, faceId);
  }
  const boxColour = f => f.how === 'you' ? 'var(--ok)' : f.how === 'auto' ? 'var(--accent)' : f.how === 'suggested' ? 'var(--warn)' : 'var(--muted)';
  async function paintPhoto(faceId, selected) {
    const r = await api('people/photo/faces?face=' + faceId);
    if (r.ok === false) { fill(viewer, h('div', { style: { padding: '24px' } }, h('div', { class: 'notice bad' }, r.error || 'cannot show the photograph'),
      h('button', { class: 'btn small', on: { click: () => viewer.close() } }, 'Close'))); return; }
    const faces = r.faces || [], ro = readOnly();
    const sel = faces.find(f => f.id === selected) || faces.find(f => f.id === faceId) || faces[0];
    const pick = id => paintPhoto(faceId, id);
    const boxes = faces.map(f => h('div', { class: 'pv-box' + (f.id === sel.id ? ' sel' : ''), title: (f.name ? f.name : 'unnamed') + ' \u00b7 score ' + f.score,
      style: { left: (100 * f.x) + '%', top: (100 * f.y) + '%', width: (100 * f.w) + '%', height: (100 * f.h) + '%', borderColor: boxColour(f) },
      on: { click: () => pick(f.id) } }, h('span', { class: 'pv-tag', style: { background: boxColour(f) } }, f.name || '?')));
    const input = h('input', { type: 'text', list: 'people-names', placeholder: 'Name (new or existing)', disabled: ro, style: { width: '200px' } });
    const name = async () => {
      const n = input.value.trim(); if (!n) { input.focus(); return; }
      const res = await act('people/name', { name: n, faces: [sel.id] }, `named ${n}`);
      if (res.ok !== false) { await refreshAll(); await paintPhoto(faceId, sel.id); }
    };
    input.addEventListener('keydown', e => { if (e.key === 'Enter') { e.preventDefault(); name(); } });
    const unname = async () => { const res = await act('people/unlabel', { faces: [sel.id] }, 'name taken off'); if (res.ok !== false) { await refreshAll(); await paintPhoto(faceId, sel.id); } };
    const how = { you: 'named by you', auto: 'matched automatically', suggested: 'a weak match: not named', '': 'nobody is matched to it' }[sel.how || ''];
    fill(viewer,
      h('div', { class: 'pv-head' }, h('b', null, r.name || 'photograph'), h('span', { class: 'muted small' }, `${faces.length} face${faces.length === 1 ? '' : 's'} found`),
        h('button', { class: 'btn small', style: { marginLeft: 'auto' }, on: { click: () => viewer.close() } }, 'Close')),
      r.available === false ? h('div', { class: 'notice warn' }, 'The photograph is not in a source folder any more: only its faces can be shown.') : null,
      r.available === false ? null : h('div', { class: 'pv-stage' }, h('img', { src: 'api/people/photo?face=' + faceId, alt: r.name || 'photograph' }), boxes),
      h('div', { class: 'pv-side' },
        h('div', { class: 'row', style: { gap: '8px', alignItems: 'center', flexWrap: 'wrap' } }, faceImg(sel.id, { class: 'face', style: { cursor: 'default' } }),
          h('div', null, h('b', null, sel.name || 'unnamed'), h('div', { class: 'small muted' }, `${how} \u00b7 face score ${sel.score} \u00b7 ${sel.px} px`)),
          input, h('button', { class: 'btn small primary', disabled: ro, on: { click: name } }, sel.how === 'you' ? 'Rename face' : 'Name'),
          sel.how === 'you' ? h('button', { class: 'btn small', disabled: ro, on: { click: unname } }, 'Take the name off') : null),
        h('div', { class: 'small muted', style: { marginTop: '6px' } }, 'Click a box to pick that face. A face the model did not find has no box.',
          '  ', h('span', { class: 'pv-key' }, h('i', { style: { background: 'var(--ok)' } }), 'named by you'), h('span', { class: 'pv-key' }, h('i', { style: { background: 'var(--accent)' } }), 'matched'),
          h('span', { class: 'pv-key' }, h('i', { style: { background: 'var(--warn)' } }), 'weak match'), h('span', { class: 'pv-key' }, h('i', { style: { background: 'var(--muted)' } }), 'nobody'))));
  }

  const taskActive = () => !!(data && data.task && (data.task.status === 'running' || data.task.status === 'queued'));

  async function load() {
    const r = await api('people');
    if (r.ok === false) { fill(refs.setup, h('div', { class: 'notice bad' }, r.error || 'cannot read the people store')); return; }
    data = r; renderSetup(); renderPeople(); renderGroups(); renderList();
  }
  async function loadClusters() {
    const r = await api('people/clusters?limit=20&skip=' + skip);
    clusters = r.ok === false ? { error: r.error } : r; renderClusters();
  }
  async function loadPerson(id) {
    if (!id) { person = null; renderPerson(); return; }
    const r = await api('people/person?id=' + id);
    person = r.ok === false ? null : r; renderPerson();
  }
  async function refreshAll() { await load(); await loadClusters(); if (person) await loadPerson(person.id); }
  async function post(path, body, okMsg) { const r = await act('people/' + path, body, okMsg); if (r.ok !== false) await refreshAll(); return r; }

  function renderList() {
    fill(refs.names, (data.people || []).map(p => h('option', { value: p.name })));
  }

  /* ---- faces: the models, and what the indexing run (step 3.9 Faces) has found ---- */
  function renderSetup() {
    const c = data.counts || {}, t = data.task, ro = readOnly(), busy = taskActive();
    const missing = (data.missing_models || []).length > 0;
    let task = null;
    if (t && (busy || (Date.now() / 1000 - (t.finished_at || t.updated_at || 0)) < 3600)) {
      task = h('div', { style: { margin: '8px 0' } },
        h('div', { class: 'small' }, pill(t.status, t.status === 'failed' ? 'bad' : t.status === 'succeeded' ? 'ok' : 'warn', busy), ' ', busy ? (t.phase || '') : ''),
        t.error ? h('div', { class: 'notice bad' }, t.error) : null);
    }
    fill(refs.setup,
      h('div', { class: 'card-head' }, h('h2', null, 'Faces'),
        h('div', { class: 'spacer' },
          missing ? h('button', { class: 'btn small primary', disabled: ro || busy, on: { click: setup } }, 'Download the face models (38 MB)')
            : h('button', { class: 'btn small primary', disabled: ro, on: { click: indexNow } }, 'Index now to find faces'))),
      c.photos !== undefined ? h('div', { class: 'cd-tiles' }, statCard(num(c.photos), 'photographs looked at'), statCard(num(c.faces), 'faces found'),
        statCard(num(c.people), 'people named'), statCard(num(c.labelled), 'faces you named'), statCard(num(c.named_auto), 'faces matched to them'),
        statCard(num(c.unknown), 'faces nobody is matched to')) : null,
      !missing && data.available ? h('div', { class: 'notice warn' }, data.available) : null,
      task,
      h('p', { class: 'small muted', style: { margin: '8px 0 0' } },
        missing ? 'The face models are downloaded once and run on this computer. ' : '',
        'Faces are found by the indexing run, as its step 3.9 Faces: in each photograph while it is converted, and in photographs indexed before without converting them again, so the first run after the download looks at all of them (about 0.1 s each). Follow it on the ', h('a', { href: '#/indexing' }, 'Indexing'), ' tab. ',
        'Name a few faces of each person below; every other face that is clearly theirs is matched automatically, and a weaker match is asked about under the person. Then search with the names: \u201cme and Samidha\u201d, \u201cAvni eating ice cream\u201d, \u201cmy family at the beach\u201d. The names stay in this store, never in the photographs or the index, so renaming or regrouping changes one place.'));
  }
  async function setup() {
    const r = await api('people/setup', {});
    if (r.ok === false) { toast(r.error || 'failed', 'bad'); return; }
    toast('Downloading the face models.', 'ok'); load();
  }
  async function indexNow() {
    const r = await act('index/start', { mode: 'new' });
    if (r.ok !== false) toast(r.already_running ? 'A run is already active: it finds the faces as it goes' : 'Indexing started', r.already_running ? '' : 'ok');
  }

  /* ---- who is this: faces nobody is matched to, in groups of look-alikes ---- */
  // a group shows its first faces; "Show more" adds the rest, and Name applies only to the faces on show
  const expanded = new Set();
  const shownFaces = c => expanded.has(c.faces[0]) ? c.faces.concat(c.more || []) : c.faces;
  function renderClusters() {
    const ro = readOnly();
    if (!clusters) { fill(refs.clusters, h('p', { class: 'muted' }, 'loading…')); return; }
    if (clusters.error) { fill(refs.clusters, h('div', { class: 'notice bad' }, clusters.error)); return; }
    const shown = (clusters.clusters || []).filter(c => !hidden.has(c.faces[0]));
    const body = shown.length ? shown.map(c => {
      const input = h('input', { type: 'text', list: 'people-names', placeholder: 'Name (new or existing)', disabled: ro, style: { width: '220px' } });
      const name = async () => {
        const n = input.value.trim(); if (!n) { input.focus(); return; }
        const ids = shownFaces(c).filter(f => !off.has(f));
        if (!ids.length) { toast('Every face of this group is left out', 'bad'); return; }
        await post('name', { name: n, faces: ids }, `${ids.length} face(s) named ${n}`);
      };
      input.addEventListener('keydown', e => { if (e.key === 'Enter') { e.preventDefault(); name(); } });
      return h('div', { class: 'cluster' },
        h('div', { class: 'faces' }, shownFaces(c).map(f => faceWithPhoto(f, { class: 'face' + (off.has(f) ? ' off' : ''), title: off.has(f) ? 'left out: click to put back' : 'click to leave this face out',
          on: { click: e => { off.has(f) ? off.delete(f) : off.add(f); e.target.classList.toggle('off', off.has(f)); } } })),
          c.more && c.more.length && !expanded.has(c.faces[0])
            ? h('button', { class: 'btn small', style: { alignSelf: 'center' }, title: 'show them all, to leave out the ones that are someone else', on: { click: () => { expanded.add(c.faces[0]); renderClusters(); } } }, `Show ${num(c.more.length)} more like these`)
            : null),
        h('div', { class: 'row', style: { marginTop: '6px', gap: '8px' } }, input,
          h('button', { class: 'btn small primary', disabled: ro, on: { click: name } }, 'Name'),
          h('button', { class: 'btn small', on: { click: () => { hidden.add(c.faces[0]); renderClusters(); } } }, 'Skip')));
    }) : h('p', { class: 'muted' }, clusters.total ? 'Nothing more to ask in this batch.' : 'No unnamed faces to ask about.');
    // the same Previous / Next 20 above and below the groups; at the bottom they also bring the top of the list back into view
    const nav = (down, below) => {
      const go = to => { skip = to; below && refs.clusters.scrollIntoView({ block: 'start' }); loadClusters(); };
      const prev = skip ? h('button', { class: 'btn small', on: { click: () => go(Math.max(0, skip - 20)) } }, 'Previous 20') : null;
      const next = clusters.total > skip + 20 ? h('button', { class: 'btn small' + (below ? ' primary' : ''), on: { click: () => go(skip + 20) } }, 'Next 20') : null;
      return h(down, { class: below ? 'row' : 'spacer', style: below ? { gap: '8px', marginTop: '10px', alignItems: 'center' } : {} },
        prev, next, below && (prev || next) ? h('span', { class: 'muted small' }, `groups ${num(skip + 1)}\u2013${num(Math.min(clusters.total, skip + 20))} of ${num(clusters.total)}`) : null);
    };
    fill(refs.clusters,
      h('div', { class: 'card-head' }, h('h2', null, 'Who is this?'),
        h('span', { class: 'muted small' }, clusters.total ? `${num(clusters.total)} group(s) of look-alike faces, ${num(clusters.faces)} faces` : ''),
        nav('div', false)),
      h('p', { class: 'small muted', style: { margin: '0 0 6px' } }, 'Faces that look alike are shown together. Click a face that is someone else to leave it out (“Show more” lists the rest of a group), then give the name: it applies to the faces you see. Naming one group of a person is enough to start: the rest of their faces are matched from it.'),
      body, nav('div', true));
  }

  /* ---- people ---- */
  function renderPeople() {
    const ps = data.people || [];
    fill(refs.people, h('div', { class: 'card-head' }, h('h2', null, 'People')),
      ps.length ? h('div', { class: 'table-wrap' }, h('table', { class: 'plain' },
        h('thead', null, h('tr', null, ['Name', 'Also called', 'Photographs', 'Faces named / matched', 'To check', 'Groups'].map((t, n) => h('th', { class: n >= 2 && n <= 4 ? 'num' : '' }, t)))),
        h('tbody', null, ps.map(p => h('tr', { class: person && person.id === p.id ? 'sel' : '' },
          h('td', null, h('a', { href: '#/people', on: { click: e => { e.preventDefault(); loadPerson(p.id); } } }, p.name)),
          h('td', { class: 'muted' }, (p.aliases || []).join(', ')),
          h('td', { class: 'num' }, num(p.photos)), h('td', { class: 'num' }, `${num(p.labelled)} / ${num(p.auto)}`),
          h('td', { class: 'num' }, p.suggested ? num(p.suggested) : ''), h('td', { class: 'muted' }, (p.groups || []).join(', '))))))) :
        h('p', { class: 'muted' }, 'Nobody is named yet: give a name under “Who is this?”.'));
  }

  function renderPerson() {
    if (!person) { fill(refs.person); refs.person.classList.add('hidden'); return; }
    refs.person.classList.remove('hidden');
    const ro = readOnly(), p = person, info = (data.people || []).find(x => x.id === p.id) || {};
    const nameIn = h('input', { type: 'text', value: p.name, disabled: ro, style: { width: '200px' } });
    const aliasIn = h('input', { type: 'text', value: (info.aliases || []).join(', '), placeholder: 'me, papa', disabled: ro, style: { width: '200px' } });
    const rename = async () => {
      const n = nameIn.value.trim(); if (!n || n === p.name) return;
      let r = await api('people/rename', { person: p.id, name: n });
      if (r.merge_needed) {
        if (!await confirmDialog('Merge two people?', `${n} is already a person. ${p.name} and ${n} become one person called ${n}; the faces named ${p.name} are then ${n}'s.`, 'Merge')) return;
        r = await api('people/rename', { person: p.id, name: n, merge: true });
      }
      if (r.ok === false) { toast(r.error || 'failed', 'bad'); return; }
      toast('Saved', 'ok'); await refreshAll(); if (r.merged_into) loadPerson(r.merged_into);
    };
    const del = async () => {
      if (!await confirmDialog(`Forget ${p.name}?`, 'The name and the faces you named go; the faces themselves stay, unnamed.', 'Forget', true)) return;
      await post('delete', { person: p.id }, 'Forgotten'); loadPerson(null);
    };
    const section = (title, note, items, actions) => items.length ? h('div', { style: { marginTop: '12px' } },
      h('h3', { style: { margin: '0 0 4px' } }, title), note ? h('p', { class: 'small muted', style: { margin: '0 0 6px' } }, note) : null,
      h('div', { class: 'faces' }, items.map(it => h('div', { class: 'facebox' }, faceWithPhoto(it.face, { title: it.score !== undefined ? `similarity ${it.score}` : '' }),
        h('div', { class: 'face-actions' }, actions(it)))))) : null;
    const btn = (label, title, fn) => h('button', { class: 'btn tiny', title, disabled: ro, on: { click: fn } }, label);
    fill(refs.person,
      h('div', { class: 'card-head' }, h('h2', null, p.name), h('span', { class: 'muted small' }, `${num(p.photos)} photograph(s)`),
        h('div', { class: 'spacer' }, h('button', { class: 'btn small', on: { click: () => loadPerson(null) } }, 'Close'))),
      h('div', { class: 'row', style: { gap: '8px', flexWrap: 'wrap' } },
        h('label', null, 'Name ', nameIn), h('button', { class: 'btn small', disabled: ro, on: { click: rename } }, 'Rename'),
        h('label', null, 'Also called ', aliasIn),
        h('button', { class: 'btn small', disabled: ro, on: { click: () => post('aliases', { person: p.id, aliases: aliasIn.value }, 'Saved') } }, 'Save'),
        h('button', { class: 'btn small danger', disabled: ro, on: { click: del } }, 'Forget')),
      h('p', { class: 'small muted', style: { margin: '6px 0 0' } }, 'A search can use any of these names; give yourself “me” to search “me and Samidha”. Giving the name of another person merges the two.'),
      section('Is this ' + p.name + '?', p.suggested_total ? `${num(p.suggested_total)} face(s) look somewhat like ${p.name}, the closest first.` : '', p.suggested,
        it => [btn('✓', `yes, ${p.name}`, () => post('confirm', { person: p.id, faces: [it.face] })), btn('✕', `not ${p.name}`, () => post('reject', { person: p.id, faces: [it.face] }))]),
      p.suggested.length > 1 ? h('button', { class: 'btn small', disabled: ro, style: { marginTop: '6px' }, on: { click: () => post('confirm', { person: p.id, faces: p.suggested.map(s => s.face) }, 'Confirmed') } }, `All ${p.suggested.length} shown are ${p.name}`) : null,
      section('Matched automatically', `${num(p.auto_total)} face(s), the least sure first: ✕ the ones that are someone else.`, p.auto,
        it => [btn('✓', `yes, ${p.name} (named by you from now on)`, () => post('confirm', { person: p.id, faces: [it.face] })), btn('✕', `not ${p.name}`, () => post('reject', { person: p.id, faces: [it.face] }))]),
      section('Named by you', `${num(p.labelled_total)} face(s).`, p.labelled.map(f => ({ face: f })),
        it => [btn('✕', 'take the name off this face', () => post('unlabel', { faces: [it.face] }))]));
  }

  /* ---- groups ---- */
  function renderGroups() {
    const ro = readOnly(), gs = data.groups || [], ps = data.people || [];
    const nameOf = id => (ps.find(p => p.id === id) || {}).name || '?';
    const newIn = h('input', { type: 'text', placeholder: 'family', disabled: ro, style: { width: '160px' } });
    const create = async () => { const n = newIn.value.trim(); if (n) await post('group/create', { name: n }, `Group ${n} made`); };
    const edit = async g => {
      const boxes = ps.map(p => h('label', { class: 'check', style: { display: 'flex' } }, h('input', { type: 'checkbox', value: String(p.id), checked: g.members.includes(p.id) }), p.name));
      const nm = h('input', { type: 'text', value: g.name, style: { width: '200px' } });
      if (!await dialog(`Group ${g.name}`, h('div', null, h('label', null, 'Name ', nm), h('p', { class: 'small muted' }, 'Members:'),
        boxes.length ? h('div', { class: 'grid g2' }, boxes) : h('p', { class: 'muted' }, 'Name some people first.')), { okText: 'Save' })) return;
      if (nm.value.trim() && nm.value.trim() !== g.name) { const r = await act('people/group/rename', { group: g.id, name: nm.value.trim() }); if (r.ok === false) return; }
      await post('group/members', { group: g.id, people: boxes.map(b => b.querySelector('input')).filter(i => i.checked).map(i => Number(i.value)) }, 'Saved');
    };
    const del = async g => { if (await confirmDialog(`Delete group ${g.name}?`, 'The people stay; only the group goes.', 'Delete', true)) post('group/delete', { group: g.id }, 'Deleted'); };
    fill(refs.groups, h('div', { class: 'card-head' }, h('h2', null, 'Groups')),
      gs.length ? gs.map(g => h('div', { class: 'row', style: { gap: '8px', padding: '6px 0', borderTop: '1px solid var(--line)', flexWrap: 'wrap' } },
        h('b', null, g.name), h('span', { class: 'muted' }, g.members.length ? g.members.map(nameOf).join(', ') : 'no members yet'),
        h('div', { class: 'spacer row', style: { marginLeft: 'auto', gap: '6px' } },
          h('button', { class: 'btn small', disabled: ro, on: { click: () => edit(g) } }, 'Edit'),
          h('button', { class: 'btn small danger', disabled: ro, on: { click: () => del(g) } }, 'Delete')))) :
        h('p', { class: 'muted' }, 'A group lets a search say “my family”: the photographs with at least one of its members (besides anyone named).'),
      h('div', { class: 'row', style: { gap: '8px', marginTop: '8px' } }, newIn, h('button', { class: 'btn small', disabled: ro, on: { click: create } }, 'New group')));
  }

  RS.views.people = {
    init(root) {
      refs.setup = h('div', { class: 'card' }); refs.clusters = h('div', { class: 'card', style: { marginTop: '16px' } });
      refs.person = h('div', { class: 'card hidden', style: { marginTop: '16px' } });
      refs.people = h('div', { class: 'card' }); refs.groups = h('div', { class: 'card' });
      refs.names = h('datalist', { id: 'people-names' });
      root.append(h('h2', null, 'People'), refs.names, refs.setup, refs.person, h('div', { class: 'grid g2', style: { marginTop: '16px' } }, refs.people, refs.groups), refs.clusters);
    },
    show() { refreshAll(); },
    tick() { if (taskActive() && ++ticks % 2 === 0) load().then(() => { if (!taskActive()) loadClusters(); }); },
  };
})();
