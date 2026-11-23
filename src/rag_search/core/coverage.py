"""Coverage: what each indexed file has, step by step, against what it should have -- the check that nothing was left behind.

Every step that works on a photograph after it was converted (3.9 Faces, 3.10 Describe, 5.2 Picture vectors, and the places 3.7
gave its position) keeps its own record, written when the step ran: the People store (``photos``), the describe cache, the
picture vector's ``pictures.json``, the map stamp beside the Markdown.  Each step also decides for itself whether a photograph
still needs it, so a step added later, or one whose check is wrong, leaves photographs behind without anything saying so.
This puts the records side by side: for every collection, how many photographs there are and, per step, how many have it
(``done``), how many should (``expected``), and how many have it from an older producer (``stale``: another description model
or prompt, another picture model, a map that has changed).  A step that is switched off or has no models expects nothing.

Which files are photographs comes from what their conversion recorded (``photo.recorded_photo``), the same answer the steps use.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from .. import maps, models
from ..paths import META_FILE, Paths
from . import describe, faces, pictures

STEPS = ("described", "faces", "pictures", "places")
EXAMPLES = 5


def _blank() -> dict[str, Any]:
    return {"files": 0, "photographs": 0, "videos": 0, "timelines": 0, "documents": 0,
            "steps": {k: {"done": 0, "expected": 0, "stale": 0, "examples": []} for k in STEPS}}


def _described(paths: Paths) -> tuple[set[str], set[str], str]:
    """(photographs described by the current model and prompt, ones the model could not describe, the id that says which)."""
    made_by = describe.current_id(models.describe_selection()[0])
    con = describe._cache(paths)
    if con is None:
        return set(), set(), made_by
    try:
        ok = {r[0] for r in con.execute("SELECT sha FROM d WHERE id = ?", (made_by,))}
        bad = {r[0] for r in con.execute("SELECT sha FROM failed WHERE id = ?", (made_by,))}
    finally:
        con.close()
    return ok, bad, made_by


def _faces(paths: Paths) -> tuple[dict[str, str], str]:
    """(photograph checksum -> the finder that looked at it, why faces cannot be looked for now -- "" when they can)."""
    if not faces.store_file(paths).exists():
        return {}, faces.available(paths)
    with faces.Store(paths) as st:
        return dict(st.con.execute("SELECT sha, detector FROM photos").fetchall()), faces.available(paths)


def report(paths: Paths, collection: str = "") -> dict[str, Any]:
    """``{"collections": {name: {...}}, "total": {...}, "notes": {step: why it expects nothing}}``; each has ``files``,
    ``photographs``, ``videos``, ``timelines``, ``documents`` (of pages) and per step ``done`` / ``expected`` / ``stale`` /
    ``examples`` (a few files that lack it); ``described`` also has ``gave_up`` (the model could not describe it)."""
    ok_desc, bad_desc, made_by = _described(paths)
    looked, no_faces = _faces(paths)
    pic_model = pictures.selected()
    state = maps.map_state(paths)
    notes: dict[str, str] = {}
    if not models.describe_auto() and not ok_desc:
        notes["described"] = "automatic descriptions are off (Models tab)"
    if no_faces:
        notes["faces"] = no_faces
    if not pic_model:
        notes["pictures"] = "picture vectors are turned off (Models tab)"
    out: dict[str, Any] = {}
    root = paths.index
    names = [collection] if collection else sorted(d.name for d in root.iterdir() if d.is_dir()) if root.is_dir() else []
    for name in names:
        c = out[name] = _blank()
        for dp, _dn, fn in os.walk(root / name):
            if META_FILE not in fn:
                continue
            doc = Path(dp)
            try:
                meta = json.loads((doc / META_FILE).read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            c["files"] += 1
            branches = set(((meta.get("conversion") or {}).get("branches")) or {})
            if branches == {"video"}:
                c["videos"] += 1
                continue
            if branches == {"timeline"}:
                c["timelines"] += 1
                continue
            if branches != {"photo"}:
                c["documents"] += 1
                continue
            c["photographs"] += 1
            sha = str(meta.get("src_sha256") or "")
            rel = doc.relative_to(root)
            md = paths.markup / rel.parent / (rel.name + ".md")
            shown = rel.as_posix()

            def step(key: str, have: bool, *, expected: bool = True, stale: bool = False) -> None:
                s = c["steps"][key]
                if not expected:
                    return
                s["expected"] += 1
                if have:
                    s["done"] += 1
                    s["stale"] += stale
                elif len(s["examples"]) < EXAMPLES:
                    s["examples"].append(shown)
            step("described", sha in ok_desc, expected="described" not in notes)
            if sha in bad_desc and sha not in ok_desc:              # the model could not: tried, and not tried again
                c["steps"]["described"]["gave_up"] = c["steps"]["described"].get("gave_up", 0) + 1
            step("faces", sha in looked, expected="faces" not in notes, stale=looked.get(sha) == faces.DETECTOR)
            have_pic = False
            try:
                pj = json.loads((doc / pictures.META).read_text(encoding="utf-8"))
                have_pic = pj.get("src_sha256") == sha and (doc / pictures.FILE).exists()
                step("pictures", have_pic, expected="pictures" not in notes, stale=have_pic and pj.get("model") != pic_model)
            except (OSError, ValueError):
                step("pictures", False, expected="pictures" not in notes)
            try:
                text = md.read_text(encoding="utf-8", errors="replace")
            except OSError:
                text = ""
            if "geo:" in text:                                      # a position: the places of 3.7 are expected
                stamped = maps.stamp_path(md).exists()
                step("places", stamped, stale=stamped and not maps.is_current(md, text, state))
    total = _blank()
    for c in out.values():
        for k in ("files", "photographs", "videos", "timelines", "documents"):
            total[k] += c[k]
        for k, s in c["steps"].items():
            t = total["steps"][k]
            for f in ("done", "expected", "stale", "gave_up"):
                if f in s or f != "gave_up":
                    t[f] = t.get(f, 0) + s.get(f, 0)
            t["examples"] = (t["examples"] + s["examples"])[:EXAMPLES]
    return {"collections": out, "total": total, "notes": notes, "made_by": made_by, "picture_model": pic_model}


def missing(rep: dict[str, Any]) -> dict[str, int]:
    """How many files lack each step in a report: ``{step: expected - done}`` for the steps that lack any."""
    return {k: s["expected"] - s["done"] for k, s in rep["total"]["steps"].items() if s["expected"] > s["done"]}
