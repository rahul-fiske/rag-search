"""Who is in each photograph, worked out from your labels (``faces.py`` keeps the faces, the names and the labels).

A face you labelled is that person.  Every other face is compared with the labelled ones by its fingerprint (cosine
similarity): it takes the person whose closest labelled face is closest, when that is at least ``MATCH`` and clearly
ahead of the next person (``MARGIN``), the face is big and sharp enough to trust (``MIN_PX``, ``MIN_SCORE``), and you
have not said it is not that person.  A weaker best match (from ``SUGGEST``) is a suggestion the People tab asks about.
Each person keeps every labelled face as its own reference, not an average, so faces of the same person years apart
(a child growing up) are each matched by the labels nearest to them.  Faces nobody is matched to are grouped by how
close they are (``CLUSTER``) so that the People tab can ask "who is this?" once for a group of faces.

None of this is stored: it is worked out again when the store changes, in well under a second for tens of thousands of
faces, so labelling one more face improves every guess at once.

A search names people by their names, aliases ("me") or groups ("family"): ``resolve`` takes those out of the query and
returns the photographs that have all the people named and, for each group, at least one more of its members.  The rest of
the query ("eating ice cream in the jungle") is searched among those photographs as usual.
"""

from __future__ import annotations

import re
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..paths import Paths
from . import faces

MATCH = 0.40                # cosine: a face is this person (OpenCV puts SFace's line for "same person" at 0.363)
MARGIN = 0.06               # ... and this much closer to them than to anyone else (relatives look alike)
SUGGEST = 0.363             # a weaker match is only a suggestion; OpenCV's own line for "the same person" (it was 0.30: 3,400 of 27,000
                            # suggestions sat below it, and most suggestions were faces under 40 px, which SFace cannot tell apart)
MAX_GROUP = 150             # a group of look-alikes the People tab shows is never larger than this
CLUSTER = 0.45              # two unnamed faces this close are grouped together
MIN_PX = 40                 # smaller faces are never named automatically
MIN_SCORE = 0.85            # nor ones the face finder is unsure of (it scores clear faces 0.85 to 0.95)


@dataclass
class Who:
    """Who each face is, worked out from the store once (see the module docstring)."""
    ids: np.ndarray
    vecs: np.ndarray
    shas: list[str]
    names: dict[int, str]                                 # person id -> name
    person: dict[int, tuple[int, str, float]] = field(default_factory=dict)      # face -> (person, "you"/"auto", score)
    suggest: dict[int, tuple[int, float]] = field(default_factory=dict)         # face -> (person, score)
    rejected: set[tuple[int, int]] = field(default_factory=set)
    unknown: list[int] = field(default_factory=list)      # faces worth asking about, nobody matched
    _clusters: list[list[int]] | None = None
    _photos: dict[int, set[str]] | None = None
    _in: dict[str, list[str]] | None = None

    def _index(self) -> None:
        sha_of = dict(zip((int(f) for f in self.ids), self.shas))
        self._photos, self._in = defaultdict(set), defaultdict(list)
        for f in sorted(self.person):
            p = self.person[f][0]
            sha = sha_of.get(f)
            if sha is None or p not in self.names:
                continue
            self._photos[p].add(sha)
            if self.names[p] not in self._in[sha]:
                self._in[sha].append(self.names[p])

    def photos_of(self, person: int) -> set[str]:
        if self._photos is None:
            self._index()
        return set(self._photos.get(person, ()))            # type: ignore[union-attr]

    def people_in(self, sha: str) -> list[str]:
        if self._in is None:
            self._index()
        return list(self._in.get(sha, ()))                   # type: ignore[union-attr]

    def clusters(self) -> list[list[int]]:
        """The unnamed faces in groups of faces close to each other, the largest first (each group by face id)."""
        if self._clusters is None:
            self._clusters = _cluster(self)
        return self._clusters


def work_out(store: faces.Store) -> Who:
    ids, vecs, shas, px, score = store.vectors()
    yes, no = store.labels()
    names = {p["id"]: p["name"] for p in store.people()}
    who = Who(ids=ids, vecs=vecs, shas=shas, names=names, rejected=no)
    if not len(ids):
        return who
    row = {int(f): i for i, f in enumerate(ids)}
    refs = [(f, p) for f, p in yes.items() if f in row and p in names]
    for f, p in refs:
        who.person[f] = (p, "you", 1.0)
    people = sorted({p for _f, p in refs})
    good = (px >= MIN_PX) & (score >= MIN_SCORE)
    if people:
        ref_rows = np.array([row[f] for f, _p in refs])
        ref_person = np.array([people.index(p) for _f, p in refs])
        best = np.full((len(ids), len(people)), -1.0, dtype=np.float32)
        for start in range(0, len(ids), 4096):                   # face x labelled face, a block at a time
            sims = vecs[start:start + 4096] @ vecs[ref_rows].T
            for k in range(len(people)):
                cols = ref_person == k
                best[start:start + 4096, k] = sims[:, cols].max(axis=1)
        for f, p in no:
            if f in row and p in people:
                best[row[f], people.index(p)] = -1.0
        order = np.argsort(-best, axis=1)
        for i, f in enumerate(ids):
            f = int(f)
            if f in who.person:
                continue
            k1 = int(order[i, 0])
            s1 = float(best[i, k1])
            s2 = float(best[i, order[i, 1]]) if len(people) > 1 else -1.0
            if s1 >= MATCH and s1 - s2 >= MARGIN and good[i]:
                who.person[f] = (people[k1], "auto", round(s1, 3))
            elif s1 >= SUGGEST and px[i] >= MIN_PX and score[i] >= faces.SURE_SCORE:    # a tiny face or an unsure box is never guessed at
                who.suggest[f] = (people[k1], round(s1, 3))
    who.unknown = [int(f) for i, f in enumerate(ids) if good[i] and int(f) not in who.person]
    return who


def _groups(v: np.ndarray, idx: list[int], threshold: float) -> list[list[int]]:
    """*idx* (rows of *v*) joined into groups: two faces closer than *threshold* are in one group, directly or through others."""
    parent = {i: i for i in idx}

    def root(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    sub = v[idx]
    for start in range(0, len(idx), 2048):
        sims = sub[start:start + 2048] @ sub.T
        ii, jj = np.nonzero(sims >= threshold)
        for a, b in zip(ii + start, jj):
            if a < b:
                ra, rb = root(idx[int(a)]), root(idx[int(b)])
                if ra != rb:
                    parent[rb] = ra
    out: dict[int, list[int]] = defaultdict(list)
    for i in idx:
        out[root(i)].append(i)
    return list(out.values())


def _cluster(who: Who) -> list[list[int]]:
    if not who.unknown:
        return []
    row = {int(f): i for i, f in enumerate(who.ids)}
    rows = [row[f] for f in who.unknown]
    face_of = {r: f for r, f in zip(rows, who.unknown)}
    done: list[list[int]] = []
    todo = [(rows, CLUSTER)]
    while todo:
        members, threshold = todo.pop()
        for g in _groups(who.vecs, members, threshold):
            # chains of look-alikes join different people into one huge group: split it at a stricter closeness
            if len(g) > MAX_GROUP and threshold < 0.95:
                todo.append((g, threshold + 0.05))
            else:
                done.append([face_of[r] for r in g])
    return sorted(done, key=lambda g: (-len(g), g[0]))


# ── cached per process: the search daemon and the dashboard ask often ─────────

_CACHE: dict[str, tuple[tuple[int, ...], Who]] = {}
_LOCK = threading.Lock()


def who(paths: Paths) -> Who | None:
    """The current ``Who`` (None when there is no store yet), worked out again only when the store has changed."""
    if not faces.store_file(paths).exists():
        return None
    key = str(faces.store_file(paths))
    with _LOCK, faces.Store(paths) as st:
        ver = st.version()
        hit = _CACHE.get(key)
        if hit and hit[0] == ver:
            return hit[1]
        w = work_out(st)
        _CACHE[key] = (st.version(), w)
        return w


# ── names in a search ─────────────────────────────────────────────────────────

_FILLER = {"a", "an", "the", "and", "with", "of", "my", "our", "photo", "photos", "photograph", "photographs", "picture",
           "pictures", "pic", "pics", "image", "images", "show", "find", "give", "get", "me", "all", "where", "in", "together",
           "&", ",", "both", "is", "are", "some", "any", "family's"}
_ASKING = re.compile(r"\b(show|give|find|get|send|tell|let)\s+$", re.I)


def resolve(paths: Paths, query: str) -> dict[str, Any] | None:
    """The people and groups a query names, the photographs (their source paths) that have them, and the rest of the
    query; None when it names nobody (or there are no people)."""
    w = who(paths)
    if w is None or not w.names:
        return None
    with faces.Store(paths) as st:
        people, groups = st.people(), st.groups()
        terms: list[tuple[str, str, int]] = []
        for p in people:
            terms += [(t, "person", p["id"]) for t in [p["name"], *p["aliases"]]]
        terms += [(g["name"], "group", g["id"]) for g in groups]
        terms.sort(key=lambda t: -len(t[0]))
        rest, found = query, []
        for term, kind, ident in terms:
            pat = re.compile(r"(?<![\w])" + re.escape(term) + r"(?:'s)?(?![\w])", re.I)
            for m in list(pat.finditer(rest)):
                if term.casefold() == "me" and _ASKING.search(rest[:m.start()]):
                    continue                                  # "show me ..." asks; it does not name anyone
                rest = rest[:m.start()] + " " * (m.end() - m.start()) + rest[m.end():]
                found.append((m.start(), kind, ident))
                break
        if not found:
            return None
        found.sort()                                          # in the order the query names them
        named = list(dict.fromkeys(i for _at, k, i in found if k == "person"))
        named_groups = [i for _at, k, i in found if k == "group"]
        shas: set[str] | None = None
        for p in named:
            got = w.photos_of(p)
            shas = got if shas is None else shas & got
        for g in dict.fromkeys(named_groups):
            members = next(x["members"] for x in groups if x["id"] == g)
            others = [m for m in members if m not in named] or members
            got: set[str] = set()
            for m in others:
                got |= w.photos_of(m)
            shas = got if shas is None else shas & got
        srcs = set(st.sources(shas or set()).values())
    words = [t for t in re.split(r"\s+", rest.strip()) if t]
    while words and words[0].casefold().strip(",.?!") in _FILLER:
        words.pop(0)
    while words and words[-1].casefold().strip(",.?!") in _FILLER:
        words.pop()
    pname = {p["id"]: p["name"] for p in people}
    gname = {g["id"]: g["name"] for g in groups}
    return {"people": [pname[p] for p in named], "groups": [gname[g] for g in dict.fromkeys(named_groups)],
            "photos": srcs, "rest": " ".join(words)}


def people_of_src(paths: Paths) -> "Any":
    """A function from a photograph's source path to the names of the people in it (for search results)."""
    w = who(paths)
    if w is None or not w.person:
        return None
    with faces.Store(paths) as st:
        rows = st.con.execute("SELECT src, sha FROM photos WHERE faces > 0").fetchall()
    by_src = dict(rows)
    return lambda src: w.people_in(by_src[src]) if src in by_src else []


def people_of_markdown(paths: Paths, md: Path) -> list[str]:
    """The people in the photograph a Markdown file was made from (its checksum is beside it in the indexer's workspace;
    a published copy has none, so it is looked up there)."""
    w = who(paths)
    if w is None or not w.person:
        return []
    beside = [md]
    live = paths.live_markup()
    if live is not None:
        try:
            beside.append(paths.markup / md.relative_to(live))
        except ValueError:
            pass
    for m in beside:
        try:
            sha = m.with_name(m.name + ".sha256").read_text(encoding="utf-8").split("\n")[0].strip()
        except OSError:
            continue
        return w.people_in(sha) if sha else []
    return []


# ── what the People tab shows ─────────────────────────────────────────────────

def overview(paths: Paths) -> dict[str, Any]:
    out: dict[str, Any] = {"available": faces.available(paths), "missing_models": faces.missing_models(paths)}
    if not faces.store_file(paths).exists():
        return {**out, "counts": {}, "people": [], "groups": []}
    w = who(paths)
    with faces.Store(paths) as st:
        counts, people, groups = st.counts(), st.people(), st.groups()
    by_person: dict[int, dict[str, int]] = defaultdict(lambda: {"you": 0, "auto": 0})
    for _f, (p, how, _s) in (w.person.items() if w else []):
        by_person[p][how] += 1
    sugg: dict[int, int] = defaultdict(int)
    for _f, (p, _s) in (w.suggest.items() if w else []):
        sugg[p] += 1
    for p in people:
        p.update(labelled=by_person[p["id"]]["you"], auto=by_person[p["id"]]["auto"], suggested=sugg[p["id"]],
                 photos=len(w.photos_of(p["id"])) if w else 0,
                 groups=[g["name"] for g in groups if p["id"] in g["members"]])
    counts["named_auto"] = sum(v["auto"] for v in by_person.values())
    counts["unknown"] = len(w.unknown) if w else 0
    return {**out, "counts": counts, "people": people, "groups": groups}


def cluster_view(paths: Paths, limit: int = 20, per: int = 24, skip: int = 0) -> dict[str, Any]:
    w = who(paths)
    if w is None:
        return {"clusters": [], "total": 0}
    cl = w.clusters()
    shown = cl[skip:skip + limit]
    return {"total": len(cl), "faces": sum(len(c) for c in cl),
            "clusters": [{"size": len(c), "faces": c[:per], "more": c[per:]} for c in shown]}


def person_view(paths: Paths, person: int, limit: int = 120) -> dict[str, Any]:
    w = who(paths)
    if w is None or person not in w.names:
        return {"error": "no such person"}
    you = [f for f, (p, h, _s) in w.person.items() if p == person and h == "you"]
    auto = sorted(((f, s) for f, (p, h, s) in w.person.items() if p == person and h == "auto"), key=lambda x: x[1])
    sugg = sorted(((f, s) for f, (p, s) in w.suggest.items() if p == person), key=lambda x: -x[1])
    return {"id": person, "name": w.names[person], "photos": len(w.photos_of(person)),
            "labelled": you[:limit], "labelled_total": len(you),
            "auto": [{"face": f, "score": s} for f, s in auto[:limit]], "auto_total": len(auto),
            "suggested": [{"face": f, "score": s} for f, s in sugg[:limit]], "suggested_total": len(sugg)}


def photo_of_face(paths: Paths, face: int) -> dict[str, Any] | None:
    """The photograph a face was found in and every face found in it, for the People tab's photo viewer: ``{"face", "sha",
    "src", "faces": [{"id", "x", "y", "w", "h", "score", "px", "person", "name", "how", "similarity"}]}`` (boxes are fractions of the upright
    picture; ``how`` is "you" for a face you named, "auto" for one matched to a person, "suggested" for a weaker match, "" for
    nobody).  None when there is no such face."""
    if not faces.store_file(paths).exists():
        return None
    with faces.Store(paths) as st:
        row = st.con.execute("SELECT f.sha, p.src FROM faces f JOIN photos p ON p.sha = f.sha WHERE f.id = ?",
                             (int(face),)).fetchone()
        if row is None:
            return None
        sha, src = row
        rows = st.con.execute("SELECT id, x, y, w, h, score, px FROM faces WHERE sha = ? ORDER BY x", (sha,)).fetchall()
    w = who(paths)
    out = []
    for fid, x, y, fw, fh, score, px in rows:
        person, name, how = None, "", ""
        sim = None
        if w is not None and fid in w.person:
            person, how, sim = w.person[fid]
            name = w.names.get(person, "")
        elif w is not None and fid in w.suggest:
            person, sim = w.suggest[fid]
            name, how = w.names.get(person, ""), "suggested"             # a guess: the viewer shows it as one, never as the name
        out.append({"id": fid, "x": x, "y": y, "w": fw, "h": fh, "score": round(score or 0.0, 2), "px": px,
                    "person": person, "name": name, "how": how, "similarity": sim})
    return {"face": int(face), "sha": sha, "src": src, "faces": out}


def source_file(paths: Paths, src: str) -> Path | None:
    """*src* as a file inside a registered source folder, or None (the store remembers where a photograph was; only a file
    that is still in a folder indexing reads is ever shown)."""
    from .. import locations

    p = Path(src)
    try:
        real = p.resolve()
    except OSError:
        return None
    for _name, root in locations.source_roots(paths).locations:
        try:
            if real.is_relative_to(Path(root).resolve()) and real.is_file():
                return real
        except OSError:
            continue
    return None
