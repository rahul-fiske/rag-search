"""A short description of each photograph, written by a vision model and kept apart from the facts.

What the file itself says (when, where, with what) is in the "Photo details" section of a photograph's Markdown
(``photo.py``).  This module adds what a vision model sees, as lines marked as machine-written::

    - Looks like (model-written): A tiger walks through a dry, wooded area with bare trees.
    - Things seen (model-written): tiger, tree, forest, ground, brown, dry
    - People in the picture (model count): 0
    - Faces detected: 0
    - Described by: Qwen3-VL 8B (4-bit), prompt d1

The model is asked to write only what it can clearly see, and a word filter drops the guesses it still makes (seasons,
weather, indoors or outdoors).  It was tried on 38 of the Ranthambore photographs: the main subject was right every time and
the 8B model made no false statement (the 4B about one photograph in twelve), but a model can still be wrong, which is why
the lines say so.

Describing is a step of its own (``rag-search photos describe``, the Models tab), not part of indexing: it loads a model of
several GB, so it runs in one process, can be stopped and picked up again, and a map refresh never repeats it.  Each result
is also kept in ``<workspace>/describe_cache.sqlite`` under the photograph's checksum, the model and the prompt version,
so converting the photograph again (a full rebuild) puts the description back without running the model.  A photograph
counts as one when its Markdown holds little text beyond the details section (a photographed page is read, not described).
The face count is Apple Vision's (a count, never who it is).
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import sqlite3
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .. import machine, models
from ..paths import NODES_FILE, Paths
from . import photo

PROMPT_VERSION = "d1"
PROMPT = (
    "Describe this photograph for a search index. Reply with exactly three lines, each starting with its label.\n"
    "Description: one factual sentence about what is visible: the scene, the main subject, what people or animals are doing.\n"
    "Tags: 6 to 10 short lowercase tags, only for things you can clearly see: objects, animals, the kind of place "
    "(for example street, lake, building, restaurant), activities, main colours.\n"
    "People: the number of people clearly visible (0 if none).\n"
    "Rules: write only what you can clearly see; leave out anything you would have to guess. "
    "Do not state the season or the weather, and do not say whether the place is indoors or outdoors. "
    "You may say it is night when the sky is dark. "
    "Say 'person' or 'people'; use 'child' or 'woman' or 'man' only when it is obvious. "
    "If an object is hard to identify, use a general word (food, device, building) instead of a specific one. "
    "Do not name places, monuments, brands or people. Do not read text unless it is clearly legible.")
MAX_TOKENS = 260
LONG_SIDE = 1000                     # pixels: the model needs far less than 12 megapixels
DROP_TAGS = {"indoor", "indoors", "outdoor", "outdoors", "inside", "winter", "summer", "spring", "autumn", "fall", "monsoon",
             "sunny", "cloudy", "rainy", "overcast", "dusk", "daytime", "day", "morning", "evening", "afternoon", "daylight"}
MAX_TAGS = 10

LOOKS = "Looks like (model-written):"
THINGS = "Things seen (model-written):"
PEOPLE = "People in the picture (model count):"
FACES = "Faces detected:"
BY = "Described by:"
KEYS = (LOOKS, THINGS, PEOPLE, BY)           # the lines a description replaces (the face count is separate)


# ── what is wanted ───────────────────────────────────────────────────────────

def model_label(model_id: str) -> str:
    spec = models.vlm_find(models.READER, model_id)
    return spec.label if spec else model_id.rsplit("/", 1)[-1]


def current_id(model_id: str = "") -> str:
    """What a description says it was made by when it is current: the model and the prompt version."""
    return f"{model_label(model_id or models.describe_selection()[0])}, prompt {PROMPT_VERSION}"


def available() -> str:
    """"" when this machine can describe photographs, else why not."""
    if not machine.mlx_possible():
        return "describing photographs needs the MLX reader of an Apple Silicon Mac"
    return ""


# ── the model's reply ────────────────────────────────────────────────────────

def parse(text: str) -> dict[str, Any]:
    """{"description", "tags", "people"} from the three labelled lines (labels optional: first line, second, third)."""
    lines = [ln.strip() for ln in re.split(r"\n+", text or "") if ln.strip()]
    d = g = ""
    p: int | None = None
    plain: list[str] = []
    for ln in lines:
        m = re.match(r"(description|tags|people)\s*:\s*(.*)", ln, re.I)
        if m:
            key, val = m.group(1).lower(), m.group(2).strip()
            if key == "description":
                d = val
            elif key == "tags":
                g = val
            else:
                n = re.search(r"\d+", val)
                p = int(n.group()) if n else None
        else:
            plain.append(ln)
    if not d and plain:
        d = plain.pop(0)
    if not g and plain:
        g = plain.pop(0)
    if p is None and plain:
        n = re.search(r"\d+", plain[0])
        p = int(n.group()) if n else None
    return {"description": " ".join(d.split()), "tags": clean_tags(g), "people": p}


def clean_tags(raw: str) -> list[str]:
    out: list[str] = []
    for t in re.split(r"[,;\n]", re.sub(r"^\s*tags\s*:", "", raw or "", flags=re.I)):
        t = " ".join(t.lower().strip(" .-*•").split())
        if t and len(t) <= 30 and t not in DROP_TAGS and t not in out:
            out.append(t)
    return out[:MAX_TAGS]


def lines(rec: dict[str, Any], made_by: str) -> list[str]:
    """The Markdown lines (without the leading "- ") of a description."""
    out = []
    if rec.get("description"):
        out.append(f"{LOOKS} {rec['description']}")
    if rec.get("tags"):
        out.append(f"{THINGS} {', '.join(rec['tags'])}")
    if rec.get("people") is not None:
        out.append(f"{PEOPLE} {rec['people']}")
    if out:
        out.append(f"{BY} {made_by}")
    return out


# ── the cache ────────────────────────────────────────────────────────────────

def cache_path(paths: Paths) -> Path:
    return paths.workspace / "describe_cache.sqlite"


def _cache(paths: Paths, create: bool = False) -> sqlite3.Connection | None:
    p = cache_path(paths)
    if not p.exists() and not create:
        return None
    p.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(p, timeout=30)
    con.execute("create table if not exists d(sha text, id text, description text, tags text, people integer, at text, "
                "primary key (sha, id))")
    return con


def cached(paths: Paths, sha: str, made_by: str) -> dict[str, Any] | None:
    con = _cache(paths)
    if con is None or not sha:
        return None
    with contextlib.closing(con):
        try:
            row = con.execute("select description, tags, people from d where sha = ? and id = ?", (sha, made_by)).fetchone()
        except sqlite3.Error:
            return None
    return {"description": row[0], "tags": json.loads(row[1]), "people": row[2]} if row else None


def remember(paths: Paths, sha: str, made_by: str, rec: dict[str, Any]) -> None:
    con = _cache(paths, create=True)
    assert con is not None
    with contextlib.closing(con), con:
        con.execute("insert or replace into d values (?,?,?,?,?,?)",
                    (sha, made_by, rec["description"], json.dumps(rec["tags"]), rec["people"], time.strftime("%Y-%m-%d %H:%M")))


def restore_lines(paths: Paths, sha: str) -> list[str]:
    """The description lines of a photograph that was described before with the current model and prompt, from the cache
    (so converting it again does not run the model); [] when there is none."""
    made_by = current_id()
    rec = cached(paths, sha, made_by)
    return lines(rec, made_by) if rec else []


# ── the photographs ──────────────────────────────────────────────────────────

_SECTION_RE = re.compile(r"^## Photo details\n\n(?:- .*\n?)+\n?", re.M)


def section_of(text: str) -> tuple[str, str]:
    """(the Photo details section, the Markdown after it) of a document; ("", text) without one."""
    m = _SECTION_RE.search(text)
    if not m:
        return "", text
    return m.group(0), text[m.end():]


def described_by(section: str) -> str:
    m = re.search(rf"^- {re.escape(BY)} (.+)$", section, re.M)
    return m.group(1).strip() if m else ""


def is_photograph(text: str) -> bool:
    """A document with a Photo details section and little else (a photographed page has text beyond it)."""
    section, rest = section_of(text)
    return bool(section) and photo.is_photograph(rest)


def photographs(paths: Paths, collection: str = "") -> list[Path]:
    root = paths.markup / collection if collection else paths.markup
    out = []
    if root.is_dir():
        for md in sorted(root.rglob("*.md")):
            with contextlib.suppress(OSError):
                if md.stat().st_size < 200_000:
                    text = md.read_text(encoding="utf-8", errors="replace")
                    if "## Photo details" in text and is_photograph(text):
                        out.append(md)
    return out


def source_of(paths: Paths, md: Path) -> Path | None:
    """The photograph a Markdown file was made from (its index records the path)."""
    rel = md.relative_to(paths.markup).with_suffix("")
    try:
        nodes = json.loads((paths.index / rel / NODES_FILE).read_text(encoding="utf-8"))["nodes"]
        src = Path(nodes[0]["metadata"]["src_path"])
    except (OSError, ValueError, KeyError, IndexError):
        return None
    return src if src.is_file() else None


def source_sha(md: Path) -> str:
    try:
        return md.with_name(md.name + ".sha256").read_text(encoding="utf-8").split("\n")[0].strip()
    except OSError:
        return ""


def status(paths: Paths, collection: str = "") -> dict[str, Any]:
    """What the Models tab and `rag-search photos` show."""
    model, source = models.describe_selection()
    made_by = current_id(model)
    total = current = old = 0
    for md in photographs(paths, collection):
        total += 1
        by = described_by(section_of(md.read_text(encoding="utf-8", errors="replace"))[0])
        if by == made_by:
            current += 1
        elif by:
            old += 1
    spec = models.vlm_find(models.READER, model)
    return {"model": model, "model_label": model_label(model), "source": source, "made_by": made_by,
            "choices": [{"id": m, "label": model_label(m), "need_gb": (models.vlm_find(models.READER, m).mem_gb if models.vlm_find(models.READER, m) else 4.0) + models.VLM_OVERHEAD_GB,
                         "downloaded": models.cache_state(m)["cached"]} for m in models.DESCRIBE_MODELS],
            "downloaded": models.cache_state(model)["cached"], "need_gb": (spec.mem_gb if spec else 4.0) + models.VLM_OVERHEAD_GB,
            "photographs": total, "described": current, "described_by_another_model": old, "pending": total - current,
            "unavailable": available()}


def indexing_lines(paths: Paths, src: Path, sha: str) -> list[str]:
    """What is added to a photograph's section while it is converted: a description from the cache (never a model run) and
    Apple Vision's face count."""
    out = restore_lines(paths, sha)
    tmp = Path(tempfile.mkdtemp(prefix="rag-faces-"))
    try:
        n = photo.faces_in(render(src, tmp / "p.png"))
    except Exception:  # noqa: BLE001 - a picture that cannot be rendered has no count
        n = None
    finally:
        import shutil

        shutil.rmtree(tmp, ignore_errors=True)
    if n is not None:
        out.append(f"{FACES} {n}")
    return out


# ── the work ─────────────────────────────────────────────────────────────────

def render(src: Path, out: Path) -> Path:
    from .conversion import vlm

    return vlm.render_image_frame(src, 1, out, long_side=LONG_SIDE)


def describe_image(reader: Any, image: Path) -> dict[str, Any]:
    """Ask the model about one image (*reader* is a started ``vlm.VlmReader``)."""
    from .conversion import vlm

    reader.ensure_started()
    res = reader.worker.read(image, PROMPT, timeout=vlm.page_timeout(), max_tokens=MAX_TOKENS)
    return parse(res.get("md", ""))


def describe_all(paths: Paths, *, collection: str = "", limit: int = 0, redo: bool = False,
                 progress: Callable[[int, int, str], None] | None = None, log: Callable[[str], None] | None = None,
                 reader: Any = None) -> dict[str, Any]:
    """Describe the photographs that have no description by the current model and prompt (all of them with *redo*; at
    most *limit*).  A photograph described before (same checksum, model and prompt) is taken from the cache.  Each Markdown
    file gets its lines and its index entry is marked out of date, so the next indexing run embeds it again; the picture
    is not read again there.  *reader* is for tests."""
    say = log or (lambda _m: None)
    if not reader:
        why = available()
        if why:
            raise RuntimeError(why)
    model, need = models.describe_choice()
    made_by = current_id(model)
    todo = []
    for md in photographs(paths, collection):
        text = md.read_text(encoding="utf-8", errors="replace")
        if redo or described_by(section_of(text)[0]) != made_by:
            todo.append(md)
    if limit:
        todo = todo[:limit]
    res = {"model": made_by, "candidates": len(todo), "described": 0, "from_cache": 0, "skipped": 0, "faces_only": 0}
    if not todo:
        return res
    own = None
    if reader is None:
        from .conversion import vlm

        own = reader = vlm.VlmReader(model, style="instruct", need_gb=need)
    tmp = Path(tempfile.mkdtemp(prefix="rag-describe-"))
    touched: list[Path] = []
    t0 = time.monotonic()
    try:
        for i, md in enumerate(todo, 1):
            if progress:
                progress(i - 1, len(todo), md.name)
            src = source_of(paths, md)
            if src is None:
                res["skipped"] += 1
                say(f"{md.name}: the source photograph is not where the index says; skipped")
                continue
            sha = source_sha(md)
            rec = cached(paths, sha, made_by)
            image = render(src, tmp / "p.png")
            if rec is None:
                rec = describe_image(reader, image)
                if not rec["description"]:
                    res["skipped"] += 1
                    say(f"{md.name}: the model wrote no description; skipped")
                    continue
                remember(paths, sha, made_by, rec)
                res["described"] += 1
            else:
                res["from_cache"] += 1
            extra = lines(rec, made_by)
            n = photo.faces_in(image)
            if n is not None:
                extra.append(f"{FACES} {n}")
            if photo.set_extra(md, extra):
                touched.append(md)
            if i % 25 == 0:
                say(f"{i}/{len(todo)} photographs, {(time.monotonic() - t0) / i:.1f} s each")
    finally:
        if own is not None:
            own.close()
        import shutil

        shutil.rmtree(tmp, ignore_errors=True)
    from ..paths import META_FILE

    for md in touched:
        rel = md.relative_to(paths.markup).with_suffix("")
        with contextlib.suppress(OSError):
            os.unlink(paths.index / rel / META_FILE)
    res["updated"] = len(touched)
    if progress:
        progress(len(todo), len(todo), "done")
    return res
