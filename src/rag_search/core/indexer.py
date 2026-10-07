"""Index building: source -> Markdown -> chunks -> embeddings -> per-collection merge.

On-disk format (per document, and identically for the merged ``_all/`` index):

    nodes.json        {"format": 1, "nodes": [{"id", "text", "metadata": {...}}]}
    embeddings.npy    float32 (n_nodes, dim), L2-normalised
    index.meta.json   freshness data (source SHA-256, chunk params, model, ...)

Merging a collection concatenates existing per-document embeddings; nothing is
ever re-embedded, so adding one document costs one document.
"""

from __future__ import annotations

import concurrent.futures as cf
import contextlib
import hashlib
import json
import logging
import multiprocessing as mp
import os
import shutil
import signal
import subprocess
import time
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import numpy as np

from .. import stages
from ..locations import ScanPlan, index_is_imported, scan_tree, workspace_excludes
from ..paths import (
    ALL_DIR,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    EMB_FILE,
    INDEX_FORMAT,
    MERGE_MANIFEST,
    META_FILE,
    NODES_FILE,
    PASSTHROUGH_EXTENSIONS,
    IndexBusyError,  # noqa: F401 - re-exported: callers catch indexer.IndexBusyError
    Paths,
    is_within,
    SourceRoots,
    allow_cloud_files,
    ensure_dirs,
    env_int,
    index_dir_for,
    index_lock,
    markup_path_for,
    mirror_rel,
    model_name,
    read_json,
    sha256_file,
    write_json_atomic,
)
from . import chunker, stallwatch
from .bm25 import TOKENIZER_VERSION
from .conversion import (
    applevision,
    costs,
    pagecache,
    pagemd,
    profiler,
    records,
    repair,
    router,
    routed,
    trace,
    vlm,
)
from .docling_convert import (
    EXIT_NO_TEXT,
    ConversionTimeout,
    NoTextError,
    ProtectedPdfError,
    convert_profile,
    damaged_pdf_reason,
    convert_settings,
    describe_error,
    has_real_text,
)

log = logging.getLogger("rag_search.indexer")

ProgressFn = Callable[[dict[str, Any]], None]
_USER_THREADS = os.environ.get("RAG_SEARCH_THREADS")      # an explicit setting always wins


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


# ── discovery ────────────────────────────────────────────────────────────────

def _scan(scan_root: Path, exclude: Iterable[Path] = ()) -> tuple[list[Path], list[dict[str, str]]]:
    """One walk of *scan_root*; see ``locations.scan_tree``."""
    return scan_tree(scan_root, exclude)


def scan_sources(scan_root: Path, exclude: Iterable[Path] = ()) -> list[Path]:
    """All supported files under *scan_root* (hidden files/dirs and *exclude* skipped)."""
    return _scan(scan_root, exclude)[0]


def scan_sources_with_skips(scan_root: Path,
                            exclude: Iterable[Path] = ()) -> tuple[list[Path], list[dict[str, str]]]:
    """Like :func:`scan_sources`, but also returns the files it passed over for having an
    extension indexing doesn't read -- so a caller can report them instead of a document just
    silently never appearing in any status (see ``run_index``'s *unsupported* parameter)."""
    return _scan(scan_root, exclude)


def exclude_dirs(paths: Paths) -> list[Path]:
    return workspace_excludes(paths)


# ── freshness ────────────────────────────────────────────────────────────────

def _params(chunk_size: int, chunk_overlap: int, model: str) -> dict[str, Any]:
    return {
        "format": INDEX_FORMAT,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "chunker": chunker.CHUNKER_VERSION,
        "tokenizer": TOKENIZER_VERSION,
        "model": model,
        "convert": convert_profile(),   # OCR / table / pipeline settings of the conversion step
    }


def is_fresh(idx_dir: Path, src_sha: str, chunk_size: int, chunk_overlap: int,
             model: str) -> bool:
    if not (idx_dir / NODES_FILE).exists() or not (idx_dir / EMB_FILE).exists():
        return False
    meta = read_json(idx_dir / META_FILE)
    if not meta or meta.get("src_sha256") != src_sha:
        return False
    want = _params(chunk_size, chunk_overlap, model)
    return all(meta.get(k) == v for k, v in want.items())


# ── documents that cannot be indexed, and will not be until they change ──────
# A document that works is skipped next time by its checksum (index.meta.json).  One that cannot
# be read for a reason of its own -- a password-protected PDF, a file without any text -- gets the
# same treatment: ``outcome.json`` in its index folder records the source's checksum, the
# conversion settings and the reason, and the next run reports that reason again without opening
# the file in a converter.  A changed file, changed conversion settings, a rebuild or
# "re-convert" (force_md) tries again.  Errors that may pass by themselves (a file the cloud app
# could not fetch, a timeout, a crash, a reader that was not available) are never remembered.
# A run that tries again is a *complete* run; an update run lists a remembered document as ``known``
# ("not tried again"), outside its errors, so that it ends ``succeeded`` when nothing new failed.

OUTCOME_FILE = "outcome.json"
OUTCOME_VERSION = 1
# wordings of a "no text" result that say a reader did not get its turn (routed.no_text_message,
# docling_convert, prepare_document)
NOT_EVERY_READER = ("did not read any page", "document reader not used", "Apple Vision not used",
                    "install the document reader", "unavailable", "crashed", "timeout", "page routing failed")


def lasting_reason(status: str, exc: BaseException | None, message: str) -> str:
    """``protected`` / ``no_text`` when this result will be the same next time, else ""."""
    if isinstance(exc, ProtectedPdfError):
        return "protected"
    # "no text" is final only when every reader had its turn: when the document reader was not
    # used (not installed, short of memory, failed) a later run may well find text
    if status == "no_text" and not any(x in message for x in NOT_EVERY_READER):
        return "no_text"
    return ""


def remember_outcome(idx_dir: Path, sha: str, status: str, reason: str, message: str,
                     ocr: bool | None) -> None:
    with contextlib.suppress(OSError):
        idx_dir.mkdir(parents=True, exist_ok=True)
        write_json_atomic(idx_dir / OUTCOME_FILE, {
            "version": OUTCOME_VERSION, "src_sha256": sha, "convert": convert_profile(ocr),
            "status": status, "reason": reason, "message": message, "at": _now()})


def known_outcome(idx_dir: Path, sha: str, ocr: bool | None) -> dict[str, Any] | None:
    """The remembered result of this exact file with these conversion settings, or None."""
    rec = read_json(idx_dir / OUTCOME_FILE) if (idx_dir / OUTCOME_FILE).exists() else None
    if (not rec or rec.get("version") != OUTCOME_VERSION or rec.get("src_sha256") != sha
            or rec.get("convert") != convert_profile(ocr) or rec.get("status") not in ("error", "no_text")):
        return None
    return rec


def forget_outcome(idx_dir: Path) -> None:
    with contextlib.suppress(OSError):
        (idx_dir / OUTCOME_FILE).unlink()


# ── step 1: source -> Markdown ───────────────────────────────────────────────

def md_is_current(md_path: Path, src_sha: str, *, force: bool = False,
                  ocr: bool | None = None) -> bool:
    """Is *md_path* the Markdown of the source with this SHA, made with the current settings?"""
    sidecar = md_path.with_name(md_path.name + ".sha256")
    if force or not md_path.exists() or not sidecar.exists():
        return False
    # sidecar = source SHA, then the conversion settings the Markdown was made with
    lines = sidecar.read_text(encoding="utf-8").split("\n")
    return lines[0].strip() == src_sha and len(lines) > 1 and lines[1].strip() == convert_profile(ocr)


def convert_source(src: Path, md_path: Path, *, force: bool, src_sha: str,
                   ocr: bool | None = None, info: dict[str, Any] | None = None) -> str:
    """Ensure *md_path* holds Markdown for the *current* content of *src*.

    *info*, when given, receives what the conversion learned (``page_stats``: per-page facts,
    ``pages``, ``ocr``) -- nothing when the Markdown was reused."""
    sidecar = md_path.with_name(md_path.name + ".sha256")
    profile = convert_profile(ocr)
    if md_is_current(md_path, src_sha, force=force, ocr=ocr):
        return "reused"
    md_path.parent.mkdir(parents=True, exist_ok=True)
    with contextlib.suppress(OSError):
        sidecar.unlink()

    docling_py = os.environ.get("RAG_SEARCH_DOCLING_PYTHON", "")
    if src.suffix.lower() not in PASSTHROUGH_EXTENSIONS and docling_py:
        script = Path(__file__).with_name("docling_convert.py")
        cmd = [docling_py, str(script), str(src), str(md_path)]
        if ocr:
            cmd.append("--ocr")
        info_file = md_path.with_name(md_path.name + ".info.json")
        if info is not None:
            cmd += ["--info", str(info_file)]
        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True,
                timeout=env_int("RAG_SEARCH_CONVERT_TIMEOUT", 6000),
            )
        except FileNotFoundError as exc:
            raise RuntimeError(f"RAG_SEARCH_DOCLING_PYTHON not found: {docling_py}") from exc
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(f"conversion timed out for {src.name}") from exc
        if proc.returncode == EXIT_NO_TEXT:
            raise NoTextError((proc.stderr or proc.stdout).strip().removeprefix("ERROR: ")[-400:])
        if proc.returncode != 0:
            tail = (proc.stderr or proc.stdout).strip()[-400:]
            raise RuntimeError(f"docling subprocess failed (rc={proc.returncode}): {tail}")
        if info is not None:
            info.update(read_json(info_file) or {})
            with contextlib.suppress(OSError):
                info_file.unlink()
    else:
        from .docling_convert import convert_file

        res = convert_file(src, md_path, ocr=ocr)
        if info is not None and isinstance(res, dict):
            info.update(res)
    sidecar.write_text(f"{src_sha}\n{profile}\n", encoding="utf-8")
    return "converted"


# ── step 2 (worker): Markdown -> chunks on disk ──────────────────────────────

def _node_id(rel: str, i: int, sha: str) -> str:
    return hashlib.sha1(f"{rel}:{i}:{sha[:16]}".encode()).hexdigest()[:20]


def stage_event(path: str | os.PathLike | None, rel: str, stage: str, status: str,
                **fields: Any) -> None:
    """Append one "this document is in this stage" line to the job's event log.  Called from the
    conversion processes too, which is why it appends to the file itself (one short line per
    write) instead of using the progress callback.  *stage* is a key of ``stages.py`` ("convert", "chunk",
    "embed", ...); the event carries its number as ``id`` ("3", "4", "5").  Never raises."""
    if not path:
        return
    try:
        line = json.dumps({"ts": round(time.time(), 3), "event": "stage", "file": rel,
                           "stage": stage, "id": stages.id_of(stage), "status": status, "pid": os.getpid(), **fields},
                          ensure_ascii=False)
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    except (OSError, ValueError, TypeError):
        pass


def work_event(path: str | os.PathLike | None, rel: str, phase: str, status: str, **fields: Any) -> None:
    """Append one "this process starts / ends a unit of work" line (``status`` start | done) to the job's
    event log.  *phase* is the run phase the work belongs to (``convert``, ``embed``, ``merge``), *rel* the
    document's path inside its collection (or the collection's name).  Written by the conversion processes
    too, hence the append to the file itself; never raises."""
    if not path:
        return
    try:
        pid = int(fields.pop("of_pid", 0) or os.getpid())          # of_pid: written for a process that cannot any more
        line = json.dumps({"ts": round(time.time(), 3), "event": "work", "pid": pid, "phase": phase,
                           "file": rel, "status": status, **fields}, ensure_ascii=False)
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    except (OSError, ValueError, TypeError):
        pass


def plan_event(path: str | os.PathLike | None, rel: str, branches: dict[str, int]) -> None:
    """Append what kind of pages this document has (``{"digital": 40, "raster": 2}``), known as soon as its
    pages are profiled: the dashboard's "pages in active files" is made of these.  Never raises."""
    if not path or not branches:
        return
    try:
        line = json.dumps({"ts": round(time.time(), 3), "event": "plan", "file": rel, "pid": os.getpid(),
                           "pages": sum(branches.values()), "branches": branches}, ensure_ascii=False)
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    except (OSError, ValueError, TypeError):
        pass


def step_event(path: str | os.PathLike | None, rel: str, page: int, what: str) -> None:
    """Append "this process is about to read page N with this reader" (a reader call can take minutes).  The
    dashboard shows it next to the worker, and the stall watch (``stallwatch.py``) takes it as a sign of life.
    Never raises."""
    if not path:
        return
    try:
        line = json.dumps({"ts": round(time.time(), 3), "event": "step", "file": rel, "pid": os.getpid(),
                           "page": page, "what": what}, ensure_ascii=False)
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    except (OSError, ValueError, TypeError):
        pass


def page_event(path: str | os.PathLike | None, rel: str, rec: dict[str, Any], total: int) -> None:
    """Append one finished page to the job's event log (a compact line: the dashboard's live pipeline
    and per-document progress are built from these).  Never raises."""
    if not path:
        return
    try:
        line = json.dumps({
            "ts": round(time.time(), 3), "event": "page", "file": rel, "pid": os.getpid(),
            "page": rec.get("page"), "of": total, "branch": rec.get("branch"), "kind": trace.page_kind(rec),
            "outcome": rec.get("outcome"), "cache": rec.get("cache", ""),
            "read_s": (rec.get("time_s") or {}).get("read"),
            "chars": (rec.get("out") or {}).get("chars"),
            "tokens": rec.get("tokens"), "gpu_s": rec.get("gpu_s"),
            "model": (rec.get("reader") or {}).get("model") if (rec.get("reader") or {}).get("tool") == "vlm" else None,
            "gate": [c.get("name") for c in (rec.get("gate") or {}).get("checks", [])] or None,
            "runway": (rec.get("route") or {}).get("final"),
            "moved": ((rec.get("route") or {}).get("escalated_from") or {}).get("runway"),
            "engine": (rec.get("route") or {}).get("engine"),
        }, ensure_ascii=False)
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    except (OSError, ValueError, TypeError):
        pass


def convert_source_routed(src: Path, md_path: Path, profile: dict[str, Any], *, src_sha: str,
                          ocr: bool | None, cache_root: Path, info: dict[str, Any],
                          stage_log: Any = None, rel_name: str = "") -> str:
    """Per-page conversion of a PDF or an image file (``conversion/routed.py``): writes *md_path* and
    the sidecar like ``convert_source``, and puts one record per page into ``info["page_records"]``.
    The document VLM (``conversion/vlm.py``) reads scanned pages, pictures and image files; docling
    reads the rest and whatever the VLM cannot.  Any error other than "no text" / "timed out"
    propagates, and the caller converts the document whole."""
    sidecar = md_path.with_name(md_path.name + ".sha256")
    md_path.parent.mkdir(parents=True, exist_ok=True)
    with contextlib.suppress(OSError):
        sidecar.unlink()
    cache = pagecache.PageCache(cache_root)
    on_page = lambda rec, total: page_event(stage_log, rel_name, rec, total)       # noqa: E731
    on_step = lambda page, what: step_event(stage_log, rel_name, page, what)       # noqa: E731
    if profile.get("kind") == "image":
        res = routed.convert_image(src, md_path, profile, cache=cache, scan_reader=vlm.shared(),
                                   ocr=ocr, on_page=on_page, repairer=repair.build(), on_step=on_step)
    else:
        res = routed.convert_pdf(src, md_path, profile, cache=cache, scan_reader=vlm.shared(), ocr=ocr,
                                 on_page=on_page, repairer=repair.build(), on_step=on_step)
    info.update({"page_records": res["records"], "pages": res["pages"], "routed": {
        "cache": res["cache"], "runs": res["runs"]}})
    sidecar.write_text(f"{src_sha}\n{convert_profile(ocr)}\n", encoding="utf-8")
    return "converted"


def _wants_routing(kind: str, profile: dict[str, Any] | None, ocr: bool | None) -> bool:
    """Is this document converted page by page?  PDFs with a usable profile, when routing is on, the
    standard pipeline is used and docling runs in this interpreter; image files too, when the
    document reader is on (docling alone reads them whole)."""
    if kind not in ("pdf", "image") or not profile or profile.get("error") or not profile.get("pages"):
        return False
    if os.environ.get("RAG_SEARCH_DOCLING_PYTHON"):
        return False
    try:
        cfg = convert_settings(ocr)
    except ValueError:
        return False
    if kind == "image" and vlm.mode() == "off":
        return False
    return cfg["routing"] == "pages" and cfg["pipeline"] == "standard"


def _why_not_routed(kind: str, profile: dict[str, Any] | None, ocr: bool | None) -> str:
    """Why a PDF or image file is converted as a whole instead of page by page ("" when it is not).  Shown in the
    conversion note, so "no per-page detail" always comes with its reason."""
    if kind not in ("pdf", "image"):
        return ""
    if not profile:
        return "page profile not available"
    if profile.get("error"):
        return f"the page profile failed ({str(profile['error'])[:120]})"
    if not profile.get("pages"):
        return "the page profile found no pages"
    if os.environ.get("RAG_SEARCH_DOCLING_PYTHON"):
        return "RAG_SEARCH_DOCLING_PYTHON is set: docling runs in another Python, whole documents only"
    try:
        cfg = convert_settings(ocr)
    except ValueError as exc:
        return f"conversion settings invalid ({exc})"
    if kind == "image" and vlm.mode() == "off":
        return "RAG_SEARCH_VLM=off: image files are converted whole"
    if cfg["routing"] != "pages":
        return f"RAG_SEARCH_ROUTING={cfg['routing']}: one docling call for the whole file"
    if cfg["pipeline"] != "standard":
        return (f"RAG_SEARCH_PIPELINE={cfg['pipeline']}: docling's own pipeline reads whole pages, so the document "
                f"reader, gate and repair are not used (set the conversion pipeline back to standard)")
    return ""


def _rescue_whole(src: Path, md_path: Path, profile: dict[str, Any] | None, kind: str, *, src_sha: str,
                  ocr: bool | None, info: dict[str, Any], failure: str) -> str:
    """A document docling read whole and found no text in: read its pages with Apple Vision when this Mac can
    (a photographed page is one picture to docling's layout model, and text inside pictures is dropped).
    Returns "converted" after writing the Markdown, or "" when nothing was read."""
    if kind not in ("pdf", "image") or not profile or not profile.get("pages") or applevision.why_not():
        return ""
    is_image = kind == "image"
    texts: dict[int, str] = {}
    recs: list[dict[str, Any]] = []
    for e in routed.plan_pages(profile):
        n = e["page"]
        if e["blank"]:
            recs.append({"page": n, "branch": "fallback", "outcome": "no_text", "why": e["why"], "reader": {"tool": "none"}})
            continue
        t0 = time.perf_counter()
        try:
            text = applevision.read_page(src, n, is_image=is_image, languages=convert_settings(ocr).get("lang") or None)
        except Exception as exc:  # noqa: BLE001 - a rescue never fails the document
            log.info("Apple Vision could not read page %s of %s: %s", n, src.name, exc)
            continue
        ok = has_real_text(text)
        if ok:
            texts[n] = text
        rec = trace.page_record(n, "fallback", e["why"], profile=e["profile"] or None,
                                reader={"tool": "apple-vision", "mode": "page"},
                                outcome="pass" if ok else "no_text",
                                out={"chars": len("".join(text.split()))},
                                note="docling found no text in this document; read by Apple Vision (plain text, no tables)")
        rec["time_s"] = {"read": round(time.perf_counter() - t0, 3)}
        recs.append(rec)
    md = pagemd.join_pages(texts)
    if not has_real_text(md):
        return ""
    sidecar = md_path.with_name(md_path.name + ".sha256")
    tmp = md_path.with_name(md_path.name + ".partial")
    md_path.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_text(md, encoding="utf-8")
    os.replace(tmp, md_path)
    sidecar.write_text(f"{src_sha}\n{convert_profile(ocr)}\n", encoding="utf-8")
    info.update({"page_records": recs, "pages": len(recs)})
    return "converted"


def _readers_of(pages: list[dict[str, Any]], default: str) -> list[str]:
    """Which tools read this document's pages (``docling``, ``vlm``, ``copy``), in order of first use."""
    out: list[str] = []
    for p in pages:
        t = (p.get("reader") or {}).get("tool")
        if t and t != "none" and t not in out:
            out.append(t)
    return out or [default]


def _conv_kw(summary: dict[str, Any] | None) -> dict[str, Any]:
    """``doc`` event field for a conversion summary: only when there is one, so a later event of
    the same document (merged by ``jobs.documents``) never blanks it."""
    return {"conversion": summary} if summary else {}


def _roots(value: Any) -> SourceRoots:
    """A task's source roots (the registered locations, as the plain dict that travels to a worker)."""
    return SourceRoots.of(value)


def _repoint(idx_dir: Path, src: Path) -> None:
    """An unchanged document found at a new place (a location was re-registered with another folder, or
    re-mounted elsewhere): record where it is now, so it is not taken for a deleted one."""
    meta_file = idx_dir / META_FILE
    meta = read_json(meta_file)
    if meta and meta.get("src_path") != str(src):
        meta["src_path"] = str(src)
        write_json_atomic(meta_file, meta)


def _reader_settings(kind: str, ocr: bool | None, used_ocr: str = "") -> dict[str, Any]:
    """How docling was run for this document (stored with every page record)."""
    if kind == "text":
        return {"tool": "copy"}
    try:
        cfg = convert_settings(ocr)
    except ValueError:
        return {"tool": "docling"}
    return {"tool": "docling", "ocr": used_ocr or cfg["ocr"], "engine": cfg["engine"],
            "table": cfg["table"], "backend": cfg["pdf_backend"]}


def _failed_conversion(kind: str, profile: dict[str, Any] | None, outcome: str, prof_s: float,
                       convert_s: float, cpu_s: float) -> dict[str, Any] | None:
    """Summary for a document that was profiled but not converted (no text, or an error): its pages
    count in the run totals with that outcome.  None when nothing is known about its pages."""
    if not profile or not profile.get("pages"):
        return None
    pages = records.build_pages(kind, profile, None, reader=_reader_settings(kind, None),
                                outcome_if_empty=outcome)
    for p in pages:
        p["outcome"] = outcome
    return trace.summarize(pages, time_s={"profile": prof_s, "convert": convert_s},
                           cost={"cpu_s": cpu_s, "peak_mb": costs.peak_rss_mb()},
                           readers=["docling"])


def prepare_document(task: dict[str, Any]) -> dict[str, Any]:
    """Phase-1 worker (runs in a pool process; takes/returns plain dicts): ``_prepare_document`` framed by
    a ``work`` start / done event, so the dashboard knows which process holds which document whatever the
    outcome (a failure closes the work too)."""
    slog, src = task.get("stage_log"), Path(task["src"])
    try:
        rel = mirror_rel(src, _roots(task["roots"])).as_posix()
    except (ValueError, TypeError, KeyError):
        rel = src.name
    work_event(slog, rel, "convert", "start")
    outcome = "error"
    try:
        res = _prepare_document(task)
        outcome = str(res.get("status", "error"))
        return res
    finally:
        work_event(slog, rel, "convert", "done", outcome=outcome)


def _prepare_document(task: dict[str, Any]) -> dict[str, Any]:
    """Profiles the pages, converts the source, chunks it and writes ``nodes.json`` -- see ``prepare_document``.

    Profiles the pages, converts the source, chunks it and writes ``nodes.json`` and the
    conversion trace.  No embedding model is loaded here.  Returns status
    prepared | skipped | no_text | error, plus ``conversion`` (the document's summary, see
    ``core/conversion/trace.py``) whenever the pages were profiled.
    """
    t0 = time.perf_counter()
    allow_cloud_files()                    # sources in a cloud-storage folder may be online-only
    src = Path(task["src"])
    kind = profiler.kind_of(src)
    profile: dict[str, Any] | None = None
    prof_s = convert_s = 0.0
    meter = costs.Meter().start()
    idx_dir: Path | None = None
    sha = ""

    def lasting(status: str, exc: BaseException | None, message: str) -> None:
        """Remember a result that will not change until the file does (see ``lasting_reason``)."""
        reason = lasting_reason(status, exc, message)
        if not reason and status == "error" and src.suffix.lower() == ".pdf" and damaged_pdf_reason(src):
            reason = "damaged"             # these bytes cannot be opened as a PDF: the same next time (a synced copy has another checksum)
        if reason and idx_dir is not None and sha:
            remember_outcome(idx_dir, sha, status, reason, message, task.get("ocr"))

    try:
        roots = _roots(task["roots"])
        markup_root, index_root = Path(task["markup_root"]), Path(task["index_root"])
        cs, co, model = task["chunk_size"], task["chunk_overlap"], task["model"]
        t_fp = time.perf_counter()
        sha = sha256_file(src)
        idx_dir = index_dir_for(src, roots, index_root)
        rel_name = mirror_rel(src, roots).as_posix()
        slog = task.get("stage_log")
        if (not task["rebuild"] and not task["force_md"]
                and is_fresh(idx_dir, sha, cs, co, model)):
            _repoint(idx_dir, src)
            stage_event(slog, rel_name, "fingerprint", "done", seconds=round(time.perf_counter() - t_fp, 3),
                        unchanged=True)
            return {"status": "skipped", "src": str(src), "idx_dir": str(idx_dir)}
        known = None if task["rebuild"] or task["force_md"] else known_outcome(idx_dir, sha, task.get("ocr"))
        if known:
            stage_event(slog, rel_name, "fingerprint", "done", seconds=round(time.perf_counter() - t_fp, 3),
                        unchanged=True)
            since = str(known.get("at") or "")[:10]
            return {"status": known["status"], "src": str(src), "known": True, "reason": known.get("reason", ""),
                    "message": f"{known.get('message', '')} [unchanged since {since}: not tried again]",
                    "elapsed_s": round(time.perf_counter() - t0, 2)}
        stage_event(slog, rel_name, "fingerprint", "done", seconds=round(time.perf_counter() - t_fp, 3),
                    unchanged=False)

        md_path = markup_path_for(src, roots, markup_root)
        trace_file = trace.trace_path_for(md_path)
        stage_event(slog, rel_name, "convert", "start")

        # Page profile (step 1): skipped when this document's Markdown *and* trace are current
        reused_md = md_is_current(md_path, sha, force=task["force_md"], ocr=task.get("ocr"))
        old = trace.read_trace(trace_file) if reused_md else {}
        reuse_trace = bool(old and old.get("src_sha256") == sha
                           and old.get("convert") == convert_profile(task.get("ocr")))
        if not reuse_trace:
            t_prof = time.perf_counter()
            profile = profiler.profile_file(src)
            prof_s = round(time.perf_counter() - t_prof, 2)
            stage_event(slog, rel_name, "profile", "done", seconds=prof_s, pages=len(profile.get("pages", [])))
            planned: dict[str, int] = {}
            for _n, branch, _why in profiler.route_pages(profile) or [(1, router.decide(kind)[0], "")]:
                planned[branch] = planned.get(branch, 0) + 1
            plan_event(slog, rel_name, planned)
        else:                                                    # the stored trace says what the pages are
            planned = {}
            for rec in old.get("pages", []):
                planned[trace.page_kind(rec)] = planned.get(trace.page_kind(rec), 0) + 1
            plan_event(slog, rel_name, planned)

        t_conv = time.perf_counter()
        info: dict[str, Any] = {}
        route_note = ""
        how = ""
        if not reused_md and profile is not None and _wants_routing(kind, profile, task.get("ocr")):
            try:
                how = convert_source_routed(
                    src, md_path, profile, src_sha=sha, ocr=task.get("ocr"),
                    cache_root=Path(task["markup_root"]).parent, info=info, stage_log=slog,
                    rel_name=rel_name)
            except (NoTextError, ConversionTimeout):
                raise
            except vlm.ReaderError as exc:     # an image file the document reader cannot read: docling does
                log.info("document reader not used for %s: %s", src.name, exc)
                route_note = f"document reader not used ({str(exc)[:160]}); converted by docling"
                info = {}
                how = ""
            except Exception as exc:  # noqa: BLE001 - routing must never fail a document
                log.warning("page routing failed for %s (%s); converting the whole document",
                            src.name, describe_error(exc))
                route_note = f"page routing failed ({describe_error(exc)[:160]}); converted as a whole document"
                info = {}
                how = ""
        if not how:
            if not route_note and not reused_md:
                route_note = _why_not_routed(kind, profile, task.get("ocr"))
                route_note = f"not read page by page: {route_note}" if route_note else ""
            if not reused_md:
                step_event(slog, rel_name, 0, "docling, whole document" if kind != "text" else "copy")
            try:
                how = convert_source(src, md_path, force=task["force_md"], src_sha=sha,
                                     ocr=task.get("ocr"), info=info)
            except NoTextError as exc:
                how = _rescue_whole(src, md_path, profile, kind, src_sha=sha, ocr=task.get("ocr"),
                                    info=info, failure=str(exc))
                if not how:
                    why_av = applevision.why_not() if kind in ("pdf", "image") else ""
                    extra = "; ".join(x for x in (route_note, f"Apple Vision not used ({why_av})" if why_av else
                                                  "Apple Vision found no text either" if kind in ("pdf", "image") else "") if x)
                    raise NoTextError(f"{exc} [{extra}]" if extra else str(exc)) from exc
                route_note = "; ".join(x for x in (route_note, "docling found no text: pages read by Apple Vision") if x)
        convert_s = round(time.perf_counter() - t_conv, 2)
        stage_event(slog, rel_name, "convert", "done", seconds=convert_s, how=how,
                    profile_s=prof_s)
        text = md_path.read_text(encoding="utf-8", errors="replace")
        t_chunk = time.perf_counter()
        raw_nodes = chunker.markdown_to_nodes(text, cs, co)
        chunk_s = round(time.perf_counter() - t_chunk, 2)
        stage_event(slog, rel_name, "chunk", "done", seconds=chunk_s, chunks=len(raw_nodes))
        if not raw_nodes:
            lasting("no_text", None, "no text content found in document")
            return {"status": "no_text", "src": str(src),
                    "message": "no text content found in document",
                    "conversion": _failed_conversion(kind, profile, "no_text", prof_s, convert_s,
                                                     meter.cpu_now()),
                    "elapsed_s": round(time.perf_counter() - t0, 2)}

        rel = mirror_rel(src, roots)
        coll = rel.parts[0]
        doc_path = str(Path(*rel.parts[1:]).with_suffix("")) if len(rel.parts) > 1 else rel.stem
        nodes = [{
            "id": _node_id(str(rel), i, sha),
            "text": n["text"],
            "metadata": {
                "page_label": n["page"],
                "heading": n["heading"],
                "file_name": rel.stem,
                "source_name": src.name,
                "collection": coll,
                "doc_path": doc_path,
                "src_path": str(src),
            },
        } for i, n in enumerate(raw_nodes)]

        # the conversion record: page branches + docling's per-page facts, times and cost
        conversion = None
        pages: list[dict[str, Any]] = []
        try:
            trace_rel = trace_file.relative_to(markup_root).as_posix()
            cost = {"cpu_s": meter.cpu_now(), "peak_mb": costs.peak_rss_mb()}
            times = {"profile": prof_s, "convert": convert_s, "chunk": chunk_s}
            if reuse_trace:
                pages = old["pages"]
                # nothing was read in this run: the stored read times and cache hits are not its
                fresh_view = [{k: v for k, v in p.items() if k not in ("time_s", "cache")} for p in pages]
                conversion = trace.summarize(
                    fresh_view, time_s=times, cost=cost, trace=trace_rel,
                    readers=old.get("summary", {}).get("readers") or ["docling"],
                    note="Markdown and trace reused from an earlier run")
            else:
                note = ("" if how == "converted" else
                        "Markdown made by an earlier run: no per-page read facts")
                note = route_note or note
                if info.get("page_records"):
                    pages = info["page_records"]
                else:
                    pages = records.build_pages(
                        kind, profile, info.get("page_stats"),
                        reader=_reader_settings(kind, task.get("ocr"), str(info.get("ocr") or "")),
                        note=note)
                conversion = trace.summarize(
                    pages, time_s=times, cost=cost, trace=trace_rel,
                    readers=_readers_of(pages, "copy" if kind == "text" else "docling"), note=note)
                trace.write_trace(trace_file, source=src.name, src_sha=sha, pages=pages,
                                  summary=conversion, profile=_trace_profile(profile),
                                  settings=convert_profile(task.get("ocr")))
        except Exception as exc:  # noqa: BLE001 - tracing must never fail a document
            log.warning("conversion trace for %s failed: %s", src.name, exc)

        low_pages = {str(p.get("page")) for p in pages if p.get("outcome") == "low"}
        if low_pages:                         # a hit on such a page says so (search, MCP): see ARCHITECTURE 5.3
            for node in nodes:
                if node["metadata"]["page_label"] in low_pages:
                    node["metadata"]["confidence"] = "low"

        idx_dir.mkdir(parents=True, exist_ok=True)
        for stale in (META_FILE, EMB_FILE):  # invalidate before touching nodes
            with contextlib.suppress(OSError):
                (idx_dir / stale).unlink()
        t_write = time.perf_counter()
        write_json_atomic(idx_dir / NODES_FILE, {"format": INDEX_FORMAT, "nodes": nodes},
                          indent=None)
        stage_event(slog, rel_name, "write", "done", seconds=round(time.perf_counter() - t_write, 3),
                    part="nodes.json")
        forget_outcome(idx_dir)
        return {"status": "prepared", "src": str(src), "idx_dir": str(idx_dir), "sha": sha,
                "nodes": len(nodes), "markdown": how, "convert_s": convert_s, "chunk_s": chunk_s,
                "profile_s": prof_s, "conversion": conversion,
                "elapsed_s": round(time.perf_counter() - t0, 2)}
    except NoTextError as exc:     # a photo, a blank document: nothing to index, not a failure
        lasting("no_text", exc, str(exc))
        return {"status": "no_text", "src": str(src), "message": str(exc),
                "conversion": _failed_conversion(kind, profile, "no_text", prof_s, convert_s,
                                                 meter.cpu_now()),
                "elapsed_s": round(time.perf_counter() - t0, 2)}
    except Exception as exc:  # noqa: BLE001 - reported per document
        lasting("error", exc, describe_error(exc))
        return {"status": "error", "src": str(src), "message": describe_error(exc),
                "conversion": _failed_conversion(kind, profile, "error", prof_s, convert_s,
                                                 meter.cpu_now()),
                "elapsed_s": round(time.perf_counter() - t0, 2)}


def _trace_profile(profile: dict[str, Any] | None) -> dict[str, Any]:
    """The part of the profile worth keeping in the trace (the per-page numbers live in the
    page records already)."""
    if not profile:
        return {}
    return {k: profile.get(k) for k in ("kind", "page_count", "truncated", "seconds", "error")}


# ── step 3: embeddings ───────────────────────────────────────────────────────

def _save_npy(path: Path, arr: np.ndarray) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        np.save(f, arr)
    os.replace(tmp, path)


def embed_document(idx_dir: Path, src: Path, sha: str, embedder: Any, *, chunk_size: int,
                   chunk_overlap: int, model: str,
                   progress: Callable[[int, int], None] | None = None,
                   timing: dict[str, float] | None = None,
                   conversion: dict[str, Any] | None = None) -> dict[str, Any]:
    """Embed a prepared document and write its index.meta.json (the completion marker).

    *timing* carries the earlier steps (convert_s, chunk_s); the meta records how long each
    step took (build_s = their sum) and the sizes on disk, for `rag-search list`.  *conversion* is
    the document's conversion summary (``core/conversion/trace.py``), kept in the meta with the
    embedding time added.
    """
    payload = read_json(idx_dir / NODES_FILE)
    nodes = payload.get("nodes", [])
    if not nodes:
        raise RuntimeError("nodes.json is missing or empty")
    t_embed = time.perf_counter()
    emb = embedder.encode([n["text"] for n in nodes], progress)
    embed_s = round(time.perf_counter() - t_embed, 2)
    if emb.shape[0] != len(nodes):
        raise RuntimeError(f"embedder returned {emb.shape[0]} vectors for {len(nodes)} chunks")
    _save_npy(idx_dir / EMB_FILE, np.asarray(emb, dtype=np.float32))
    meta = _params(chunk_size, chunk_overlap, model)
    steps = {"convert_s": 0.0, "chunk_s": 0.0, **(timing or {}), "embed_s": embed_s}
    size = 0
    for f in (NODES_FILE, EMB_FILE):
        with contextlib.suppress(OSError):
            size += (idx_dir / f).stat().st_size
    src_bytes = 0
    with contextlib.suppress(OSError):
        src_bytes = src.stat().st_size
    meta.update({"src_sha256": sha, "src_path": str(src), "nodes": len(nodes),
                 # the exact weights (not part of the freshness key: advisory, and checked by
                 # collection import) -- "" for custom backends and local model folders
                 "model_revision": str(getattr(embedder, "revision", "") or ""),
                 "dim": int(emb.shape[1]), "built_at": _now(), **steps,
                 "build_s": round(sum(steps.values()), 2),
                 "index_bytes": size, "src_bytes": src_bytes})
    conv = dict(conversion or {})
    if conv:
        conv["time_s"] = {**(conv.get("time_s") or {}), "embed": embed_s}
    meta["conversion"] = conv
    write_json_atomic(idx_dir / META_FILE, meta)  # written last: marks the index complete
    return meta


# ── step 4: merge a collection into _all/ ────────────────────────────────────

def per_doc_dirs(coll_dir: Path) -> list[Path]:
    """Complete per-document index dirs under *coll_dir* (``_all`` excluded)."""
    out = []
    for meta in sorted(coll_dir.rglob(META_FILE)):
        d = meta.parent
        if ALL_DIR in d.relative_to(coll_dir).parts:
            continue
        if (d / NODES_FILE).exists() and (d / EMB_FILE).exists():
            out.append(d)
    return out


def _live_docs(coll_dir: Path) -> list[Path]:
    """Per-doc dirs whose source file still exists."""
    live = []
    for d in per_doc_dirs(coll_dir):
        src = read_json(d / META_FILE).get("src_path", "")
        if src and Path(src).exists():
            live.append(d)
    return live


def _manifest_sha(coll_dir: Path, dirs: list[Path]) -> str:
    h = hashlib.sha256()
    for d in sorted(dirs):
        h.update(str(d.relative_to(coll_dir)).encode())
        h.update((d / META_FILE).read_bytes())
    return h.hexdigest()


def merge_collection(coll_dir: Path, *, force: bool = False) -> dict[str, Any]:
    t0 = time.perf_counter()
    name = coll_dir.name
    all_dir = coll_dir / ALL_DIR
    docs = _live_docs(coll_dir)
    if not docs:
        if all_dir.exists():
            shutil.rmtree(all_dir, ignore_errors=True)
        return {"collection": name, "docs": 0, "nodes": 0, "skipped": True}
    sha = _manifest_sha(coll_dir, docs)
    manifest = read_json(all_dir / MERGE_MANIFEST)
    if (not force and manifest.get("sha256") == sha
            and (all_dir / NODES_FILE).exists() and (all_dir / EMB_FILE).exists()):
        return {"collection": name, "docs": len(docs), "nodes": manifest.get("nodes", 0),
                "skipped": True}

    nodes: list[dict[str, Any]] = []
    mats: list[np.ndarray] = []
    for d in docs:
        n = read_json(d / NODES_FILE).get("nodes", [])
        m = np.load(d / EMB_FILE)
        if m.shape[0] != len(n):
            raise RuntimeError(f"{d}: {len(n)} chunks but {m.shape[0]} embeddings; re-index it")
        nodes.extend(n)
        mats.append(m)
    dims = {m.shape[1] for m in mats}
    if len(dims) != 1:
        raise RuntimeError(f"collection {name}: mixed embedding sizes {sorted(dims)}; "
                           "re-run index-all so every document uses the same model")
    all_dir.mkdir(parents=True, exist_ok=True)
    _save_npy(all_dir / EMB_FILE, np.concatenate(mats, axis=0))
    write_json_atomic(all_dir / NODES_FILE, {"format": INDEX_FORMAT, "nodes": nodes}, indent=None)
    write_json_atomic(all_dir / MERGE_MANIFEST, {
        "sha256": sha, "nodes": len(nodes), "built_at": _now(),
        "docs": [str(d.relative_to(coll_dir)) for d in docs],
    })
    return {"collection": name, "docs": len(docs), "nodes": len(nodes), "skipped": False,
            "elapsed_s": round(time.perf_counter() - t0, 2)}


def merge_all(index_root: Path, *, force: bool = False,
              frozen: Iterable[str] = (), on_collection: Any = None) -> list[dict[str, Any]]:
    """Merge every generated collection.  Imported collections (no per-document sources, their
    merged index came with the import) and *frozen* ones (source folder unreachable this run)
    are left exactly as they are."""
    out = []
    if not index_root.is_dir():
        return out
    keep = {f.casefold() for f in frozen}
    note = on_collection or (lambda _name, _status, **_kw: None)      # (collection, "start" | "done", **fields)
    for coll in sorted(index_root.iterdir()):
        if coll.is_dir() and not coll.name.startswith("."):
            if index_is_imported(index_root, coll.name):
                out.append({"collection": coll.name, "skipped": True, "imported": True,
                            "docs": len(read_json(coll / ALL_DIR / MERGE_MANIFEST).get("docs", [])),
                            "nodes": read_json(coll / ALL_DIR / MERGE_MANIFEST).get("nodes", 0)})
                continue
            if coll.name.casefold() in keep:
                out.append({"collection": coll.name, "skipped": True, "unreachable": True})
                continue
            note(coll.name, "start")
            try:
                out.append(merge_collection(coll, force=force))
                note(coll.name, "done", outcome="merged", docs=out[-1].get("docs"), nodes=out[-1].get("nodes"))
            except Exception as exc:  # noqa: BLE001
                out.append({"collection": coll.name, "error": str(exc)})
                note(coll.name, "done", outcome="error")
    return out


# ── step 5: forget documents whose source is gone ────────────────────────────

def _doc_dirs_any(coll_dir: Path) -> list[Path]:
    """Every per-document index dir under *coll_dir*, complete or not (``_all`` excluded)."""
    out = []
    for dp, dirnames, filenames in os.walk(coll_dir):
        dirnames[:] = [d for d in dirnames if d != ALL_DIR and not d.startswith(".")]
        if (META_FILE in filenames or NODES_FILE in filenames or EMB_FILE in filenames
                or OUTCOME_FILE in filenames):
            out.append(Path(dp))
    return out


def _remove_doc_index(d: Path) -> None:
    """Delete one document's index files.  Not ``rmtree``: a folder next to a document with the
    same name (``a.pdf`` and ``a/b.pdf``) nests ``b``'s index inside ``a``'s folder."""
    for f in (META_FILE, EMB_FILE, NODES_FILE, EMB_FILE + ".tmp", OUTCOME_FILE):   # completion marker first
        with contextlib.suppress(OSError):
            (d / f).unlink()
    with contextlib.suppress(OSError):
        for f in d.iterdir():
            if f.is_file() and f.name.startswith(".") and f.name.endswith(".tmp"):
                f.unlink()
    with contextlib.suppress(OSError):
        d.rmdir()


def _remove_empty_dirs(root: Path, keep_root: bool = True) -> None:
    if not root.is_dir():
        return
    for dp, _dirs, _files in sorted(os.walk(root), key=lambda t: -len(t[0])):
        d = Path(dp)
        if keep_root and d == root:
            continue
        with contextlib.suppress(OSError):
            d.rmdir()                          # only succeeds when empty


def prune_orphans(paths: Paths, roots: SourceRoots, sources: list[Path],
                  collections: Iterable[str]) -> list[str]:
    """Delete the converted Markdown and per-document index of every document in *collections*
    whose source no longer exists.

    *sources* must be the complete list of documents found in those collections' source folders
    (a ``ScanPlan`` only lists a collection as covered when its whole folder was read), so
    "not found" really means "deleted".  Only rag-search's own derived files are removed --
    never anything in a source folder.  Imported collections are never touched.  Returns the
    removed documents as ``collection/path`` names."""
    wanted = set(collections)
    if not wanted:
        return []
    expect_idx: set[Path] = set()
    expect_md: set[Path] = set()
    for src in sources:
        try:
            expect_idx.add(index_dir_for(src, roots, paths.index))
            expect_md.add(markup_path_for(src, roots, paths.markup))
        except ValueError:
            continue
    # Compared by identity as well as by path: on a case-insensitive disk (macOS) a folder
    # renamed from "security" to "Security" still maps to the existing index/security/...
    # folder, which must not be taken for a different (orphaned) one.
    keep_idx, keep_md = _identities(expect_idx), _identities(expect_md)
    removed: list[str] = []
    for coll in sorted(wanted):
        if index_is_imported(paths.index, coll):
            continue
        cdir = paths.index / coll
        for d in (_doc_dirs_any(cdir) if cdir.is_dir() else []):
            if d not in expect_idx and _identity(d) not in keep_idx:
                _remove_doc_index(d)
                removed.append(d.relative_to(paths.index).as_posix())
        mdir = paths.markup / coll
        if mdir.is_dir():
            for f in list(mdir.rglob("*")):
                if not f.is_file():
                    continue
                if f.name.endswith(trace.TRACE_SUFFIX):       # <doc>.trace.json belongs to <doc>.md
                    md = f.with_name(f.name[:-len(trace.TRACE_SUFFIX)] + ".md")
                elif f.name.endswith(".md.sha256"):
                    md = f.with_name(f.name[:-len(".sha256")])
                else:
                    md = f
                if md.suffix != ".md" or md in expect_md or _identity(md) in keep_md:
                    continue
                with contextlib.suppress(OSError):
                    f.unlink()
                rel = md.relative_to(paths.markup).with_suffix("").as_posix()
                if f is md and rel not in removed:
                    removed.append(rel)
            _remove_empty_dirs(mdir, keep_root=False)
        _remove_empty_dirs(cdir, keep_root=True)
    return sorted(set(removed))


def _identity(p: Path) -> tuple[int, int] | None:
    try:
        st = p.stat()
    except OSError:
        return None
    return (st.st_dev, st.st_ino)


def _identities(ps: Iterable[Path]) -> set[tuple[int, int]]:
    return {i for i in (_identity(p) for p in ps) if i is not None}


def _drop_empty_collections(paths: Paths) -> None:
    """A generated collection with no documents left (and so no merged index) disappears."""
    if not paths.index.is_dir():
        return
    for cdir in paths.index.iterdir():
        if cdir.is_dir() and not cdir.name.startswith(".") \
                and not index_is_imported(paths.index, cdir.name):
            _remove_empty_dirs(cdir, keep_root=False)


# ── orchestration ────────────────────────────────────────────────────────────

def _wipe(paths: Paths, roots: SourceRoots, sources: list[Path]) -> int:
    n = 0
    colls: set[str] = set()
    for src in sources:
        idx = index_dir_for(src, roots, paths.index)
        colls.add(mirror_rel(src, roots).parts[0])
        if idx.exists():
            _remove_doc_index(idx)
            n += 1
        md = markup_path_for(src, roots, paths.markup)
        for f in (md, md.with_name(md.name + ".sha256"), trace.trace_path_for(md)):
            with contextlib.suppress(OSError):
                f.unlink()
    for c in colls:
        shutil.rmtree(paths.index / c / ALL_DIR, ignore_errors=True)
    return n


LEFT_OUT_FILE = "left_out.json"            # in the workspace: the files a run left out for their name, and said so


def _left_out(paths: Paths, sources: list[Path], errors: list[dict[str, str]], known: list[dict[str, str]], *,
              complete: bool) -> None:
    """A file left out because another one has its document name stays left out until one of them is renamed.  The
    run that finds it reports it as an error; an update run after that lists it as known (moved from *errors* to
    *known*), and a complete run reports it again.  What is remembered about files outside this run is kept."""
    store = paths.workspace / LEFT_OUT_FILE
    old = (read_json(store) if store.exists() else {}) or {}
    seen = old.get("sources") if isinstance(old.get("sources"), dict) else {}
    now_out = {e["src"]: e["message"] for e in errors if e["message"].startswith("same document name")}
    if not complete:
        for e in [e for e in errors if e["src"] in now_out and seen.get(e["src"]) == e["message"]]:
            errors.remove(e)
            known.append({"src": e["src"], "message": e["message"] + " [reported before: not an error of this run]",
                          "was": "error", "reason": "name"})
    mine = {str(x) for x in sources}
    new = {k: v for k, v in seen.items() if k not in mine and Path(k).exists()} | now_out
    if new != seen:
        with contextlib.suppress(OSError):
            if new:
                store.parent.mkdir(parents=True, exist_ok=True)
                write_json_atomic(store, {"version": 1, "sources": new})
            else:
                store.unlink(missing_ok=True)


def _assign_names(paths: Paths, sources: list[Path], roots: SourceRoots,
                  errors: list[dict[str, str]]) -> list[Path]:
    """The sources this run may index.  Index folders are named after the file name without its
    extension, so ``report.pdf`` and ``report.docx`` side by side would share one: only one of
    them is indexed (the one already indexed there, else the first), the other is reported --
    including when the one already indexed is not part of this run (a single-file run), which
    would otherwise silently replace its index."""
    groups: dict[Path, list[Path]] = {}
    for src in sources:
        try:
            idx = index_dir_for(src, roots, paths.index)
        except ValueError:
            errors.append({"src": str(src), "message": "not inside a registered location"})
            continue
        groups.setdefault(idx, []).append(src)
    todo: list[Path] = []
    for idx, group in groups.items():
        owner = read_json(idx / META_FILE).get("src_path", "") if (idx / META_FILE).exists() else ""
        owner_path = Path(owner) if owner else None
        winner = next((s for s in group if owner_path is not None and _same(s, owner_path)), group[0])
        if (owner_path is not None and not _same(winner, owner_path) and owner_path.exists()
                and _index_rel(owner_path, roots) == _index_rel(winner, roots)):
            for s in group:
                errors.append({"src": str(s), "message": f"same document name as {owner} "
                               "(already indexed; rename one of them)"})
            continue
        todo.append(winner)
        for s in group:
            if s is not winner:
                errors.append({"src": str(s), "message": f"same document name as {winner} "
                               "(rename one of them)"})
    return todo


def _same(a: Path, b: Path) -> bool:
    try:
        return a.resolve() == b.resolve()
    except OSError:
        return str(a) == str(b)


def _index_rel(src: Path, roots: SourceRoots) -> str:
    try:
        rel = mirror_rel(src, roots)
        return (rel.parent / rel.stem).as_posix()
    except ValueError:
        return ""


def run_plan(paths: Paths, plan: ScanPlan, **kw: Any) -> dict[str, Any]:
    """``run_index`` over a ``locations.ScanPlan``: its sources, pruning the collections it
    fully covers and leaving unreachable ones untouched."""
    return run_index(paths, plan.sources, plan.roots, unsupported=plan.unsupported,
                     prune=plan.covered, frozen=plan.frozen, orphans=plan.orphans,
                     plan_info=plan.to_dict(), **kw)


POOL_EXIT_GRACE_S = 20.0             # how long a conversion process may take to exit once its work is done


def _shutdown_pool(pool: cf.ProcessPoolExecutor, grace: float = POOL_EXIT_GRACE_S) -> int:
    """Close the conversion pool without ever waiting for a process that cannot exit; returns how
    many had to be killed.

    A conversion process can be unable to exit although all its documents are done: when docling
    gives up on a document (``RAG_SEARCH_DOC_TIMEOUT``) it abandons its OCR / layout threads, and
    one that is stuck inside a native call (seen: an Apple Vision text request that never returns)
    is joined forever by Python's interpreter shutdown.  ``ProcessPoolExecutor.shutdown(wait=True)``
    -- what ``with pool:`` does -- then waits forever too, and the run never reaches embedding.
    Every result is already collected when this is called, so a process still alive after *grace*
    seconds has nothing left to lose and is terminated."""
    from multiprocessing.connection import wait as wait_exit

    procs = list((getattr(pool, "_processes", None) or {}).values())
    pool.shutdown(wait=False, cancel_futures=True)
    left = {p.sentinel: p for p in procs}
    deadline = time.monotonic() + max(0.0, grace)
    try:
        while left and (remaining := deadline - time.monotonic()) > 0:
            for sentinel in wait_exit(list(left), remaining):
                left.pop(sentinel, None)
    except (OSError, ValueError):          # a sentinel already closed: fall through and terminate
        pass
    for p in left.values():
        log.warning("conversion process %s did not exit %.0f s after its last document (a thread "
                    "abandoned after a document timeout is stuck); terminating it", p.pid, grace)
        for stop in (p.terminate, p.kill):
            try:
                stop()
                if wait_exit([p.sentinel], 5):
                    break
            except (OSError, ValueError, AttributeError):
                break
    return len(left)


MAX_POOL_RESTARTS = 8                # a pool whose processes keep ending abruptly is given up after this many new pools
STRIKES = 2                          # a document that was open during this many abrupt ends is not tried again in the run


def _convert_in_pool(tasks: list[dict[str, Any]], workers: int, collect: Callable[[dict[str, Any]], None], *,
                     watch: Any = None, stalled: dict[str, str] | None = None, rel_src: dict[str, str] | None = None,
                     stage_log: Any = None) -> None:
    """Phase 1 in a pool of conversion processes that survives the end of one of them.

    When a pool process ends abruptly -- stopped by the stall watch, killed by the system for memory, a crash in native
    code -- Python marks the whole pool broken and fails every document still waiting.  Here those documents are not
    lost: the documents that finished are kept, the one the stall watch stopped is reported as stalled, a document that
    was open during two abrupt ends is reported as the likely cause, and the rest go to a new pool."""
    from concurrent.futures.process import BrokenProcessPool

    stalled = stalled if stalled is not None else {}
    rel_src = rel_src or {}
    src_rel = {v: k for k, v in rel_src.items()}
    remaining = list(tasks)
    strikes: dict[str, int] = {}
    restarts = 0
    while remaining:
        finished: set[str] = set()
        broke = False
        dead: set[int] = set()
        try:
            pool = cf.ProcessPoolExecutor(max_workers=max(1, min(workers, len(remaining))), mp_context=mp.get_context("spawn"))
            futs = {pool.submit(prepare_document, t): t for t in remaining}
        except Exception as exc:  # noqa: BLE001 - no pool on this machine: one document at a time, in this process
            log.warning("process pool could not be started (%s); continuing serially", exc)
            for t in remaining:
                collect(prepare_document(t))
            return
        try:
            for fut in cf.as_completed(futs):
                t = futs[fut]
                try:
                    res = fut.result()
                except BrokenProcessPool:
                    broke = True                       # every waiting document ends this way: sorted out below
                    continue
                except Exception as exc:  # noqa: BLE001 - this document's worker failed
                    res = {"status": "error", "src": t["src"], "message": f"worker failed: {exc}"}
                finished.add(t["src"])
                collect(res)
        except Exception as exc:  # noqa: BLE001 - the pool itself failed
            log.warning("process pool failed (%s)", exc)
            broke = True
        finally:
            if broke:                                  # which process ended by itself?  The pool then ends the others (SIGTERM)
                time.sleep(0.3)
                for pid, proc in dict(getattr(pool, "_processes", None) or {}).items():
                    if proc.exitcode not in (None, 0, -signal.SIGTERM):
                        dead.add(int(pid))
            _shutdown_pool(pool)
        remaining = [t for t in remaining if t["src"] not in finished]
        if not remaining:
            return
        if not broke:                                  # (not reached: every submitted document ends one way or the other)
            for t in remaining:
                collect({"status": "error", "src": t["src"], "message": "worker failed: no result"})
            return
        restarts += 1
        flying = watch.in_flight() if watch is not None else {}
        # the documents that may have ended the pool: those open in a process that ended by itself; when that cannot
        # be told, every document that was open (or, without an event log, every document left)
        culprits = {pid: f for pid, f in flying.items() if pid in dead} or flying
        open_srcs = {rel_src[f] for f in culprits.values() if f in rel_src} if watch is not None else {t["src"] for t in remaining}
        for pid, f in flying.items():                  # their processes are gone: close their work in the log
            work_event(stage_log, f, "convert", "done", outcome="interrupted", of_pid=pid)
        if watch is not None:
            watch.forget_all()
        keep = []
        for t in remaining:
            src = t["src"]
            if src_rel.get(src, "") in stalled:
                collect({"status": "error", "src": src, "message": stalled.pop(src_rel[src])})
                continue
            if src in open_srcs:
                strikes[src] = strikes.get(src, 0) + 1
                if strikes[src] >= STRIKES:
                    collect({"status": "error", "src": src, "message":
                             "the conversion process ended abruptly twice while it had this document open (out of memory, "
                             "or a crash in a library): the document was left out of this run and the run went on"})
                    continue
            keep.append(t)
        remaining = keep
        if remaining and restarts > MAX_POOL_RESTARTS:
            for t in remaining:
                collect({"status": "error", "src": t["src"], "message":
                         "worker failed: the conversion processes kept ending abruptly; not converted in this run"})
            return
        if remaining:
            log.warning("a conversion process ended abruptly; %d document(s) go to a new pool (restart %d)",
                        len(remaining), restarts)


def run_index(
    paths: Paths,
    sources: list[Path],
    roots: SourceRoots,
    *,
    jobs: int = 2,
    rebuild: bool = False,
    wipe: bool = False,
    force_md: bool = False,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    embedder: Any = None,
    progress: ProgressFn | None = None,
    stage_log: Path | None = None,
    model: str | None = None,
    unsupported: list[dict[str, str]] | None = None,
    prune: Iterable[str] | None = None,
    frozen: Iterable[str] = (),
    orphans: Iterable[str] = (),
    plan_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Index *sources* (all inside the registered locations of *roots*), then merge every
    collection.

    *prune* names collections whose complete document list *sources* is (see
    ``locations.ScanPlan.covered``): documents of theirs that were not found are forgotten
    (``prune_orphans``).  *frozen* collections are not re-merged (their source folder could not
    be read this time; they keep serving their last index).  Both default to "nothing", the
    behaviour of a plain call.

    *model* names the embedding model actually used, for the freshness key and
    ``index.meta.json`` (it must match whatever *embedder* really is -- callers that inject a
    custom *embedder* should pass this too). Defaults to ``model_name()`` (env / this process's
    ``config.json``), which is also what happens when *model* is left as ``None``: production
    callers are unaffected.

    *unsupported* is files the caller already found but left out of *sources* because their
    extension isn't one indexing reads (see ``scan_sources_with_skips``) -- purely for reporting:
    each is recorded in the summary and given a ``doc`` event of its own, so it shows up as
    "unsupported" rather than just never appearing in any status.
    """
    t_start = time.perf_counter()
    ensure_dirs(paths)
    _sink: ProgressFn = progress or (lambda _e: None)
    phase_t0: dict[str, float] = {}      # wall-clock start of each phase, for "running for ..."
    phase_s: dict[str, float] = {}       # seconds spent in each phase

    conv = trace.RunTotals()             # what this run has converted: pages per branch, time, cost

    def emit(ev: dict[str, Any]) -> None:
        if ev.get("phase") in ("convert", "embed", "merge"):
            ev["phase_started_at"] = phase_t0.setdefault(ev["phase"], round(time.time(), 3))
        if "phase" in ev and (conv.docs or conv.files):
            ev["conversion"] = conv.snapshot()
        _sink(ev)

    def doc_event(src: Path, status: str, **fields: Any) -> None:
        """One event per finished document (also written to the job's event log)."""
        try:
            rel = mirror_rel(src, roots)
            coll, inside = rel.parts[0], "/".join(rel.parts[1:])
        except ValueError:
            coll, inside = "", ""
        # path: where the document is inside its collection.  The file name alone does not identify it: two folders
        # of one collection may each hold a "statement.pdf", and a list keyed by name would show (and count) one
        _sink({"doc": {"collection": coll, "source": src.name, "path": inside or src.name, "status": status, **fields}})

    model = model if model is not None else model_name()
    ocr = None    # OCR, table and pipeline settings come from RAG_SEARCH_* (see docling_convert)

    unsupported = list(unsupported or [])
    for u in unsupported:
        doc_event(Path(u["src"]), "unsupported", extension=u["extension"] or "(none)")

    orphans_removed: list[str] = []
    errors: list[dict[str, str]] = []
    no_text: list[dict[str, str]] = []      # converted fine but hold no text: skipped, not failed
    skipped = 0
    known: list[dict[str, str]] = []        # failed, empty or left out before and unchanged: listed, not tried again
    todo = _assign_names(paths, sources, roots, errors)

    with index_lock(paths):
        wiped = _wipe(paths, roots, todo) if wipe else 0
        _left_out(paths, sources, errors, known, complete=bool(wipe or rebuild or force_md))
        for e in errors:                    # left out for its name: listed with the run's documents, as every other failure
            doc_event(Path(e["src"]), "error", message=e["message"])
        for e in known:
            doc_event(Path(e["src"]), "known", message=e["message"], was=e["was"])
        removed = prune_orphans(paths, roots, sources, prune or ())
        for name in orphans:                 # an index no location or import owns: derived data, nothing can update it
            for root in (paths.index, paths.markup):
                if (root / name).is_dir() and is_within(root / name, paths.workspace):
                    shutil.rmtree(root / name, ignore_errors=True)
            orphans_removed.append(name)
        for r in removed:
            parts = r.split("/", 1)
            _sink({"doc": {"collection": parts[0], "source": parts[-1].rsplit("/", 1)[-1], "path": parts[-1], "status": "removed"}})
        rebuild = rebuild or wipe
        force_md = force_md or wipe

        for s_ in todo:
            ext = s_.suffix.lower().lstrip(".") or "(none)"
            conv.files[ext] = conv.files.get(ext, 0) + 1
        tasks = [{
            "src": str(s),
            "roots": roots.to_dict(),
            "markup_root": str(paths.markup),
            "index_root": str(paths.index), "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap, "rebuild": rebuild, "force_md": force_md,
            "model": model, "ocr": ocr,
            "stage_log": str(stage_log) if stage_log else None,
        } for s in todo]

        # Phase 1 — convert + chunk (parallel, no model)
        prepared: list[dict[str, Any]] = []
        total = len(tasks)
        done = 0

        def _collect(res: dict[str, Any]) -> None:
            nonlocal done, skipped
            done += 1
            name = Path(res["src"]).name
            if res.get("known"):           # failed or empty before, the file and the settings unchanged: not this run's failure
                known.append({"src": res["src"], "message": res.get("message", ""), "was": res["status"],
                              "reason": res.get("reason", "")})
                doc_event(Path(res["src"]), "known", message=res.get("message", ""), was=res["status"],
                          total_s=res.get("elapsed_s"))
                emit({"phase": "convert", "done": done, "total": total, "current": name, "message": "not tried again"})
                return
            if res["status"] != "skipped":
                conv.add(res.get("conversion"), ok=res["status"] == "prepared")
            if res["status"] == "prepared":
                prepared.append(res)
                # converted and chunked, waiting for the embed phase: shown in the run's
                # document list now (the "indexed" event replaces it when embedding is done)
                doc_event(Path(res["src"]), "converted", chunks=res.get("nodes"),
                          convert_s=res.get("convert_s"), chunk_s=res.get("chunk_s"),
                          total_s=res.get("elapsed_s"), markdown=res.get("markdown", ""),
                          profile_s=res.get("profile_s"), **_conv_kw(res.get("conversion")))
            elif res["status"] == "skipped":
                skipped += 1
                doc_event(Path(res["src"]), "skipped")
            elif res["status"] == "no_text":
                no_text.append({"src": res["src"], "message": res.get("message", "no text")})
                doc_event(Path(res["src"]), "no_text", message=res.get("message", "no text"),
                          total_s=res.get("elapsed_s"), **_conv_kw(res.get("conversion")))
            else:
                errors.append({"src": res["src"], "message": res.get("message", "error")})
                doc_event(Path(res["src"]), "error", message=res.get("message", "error"),
                          total_s=res.get("elapsed_s"), **_conv_kw(res.get("conversion")))
            note = res["status"]
            if note == "no_text":
                note = "skipped: no text"
            elif note == "error":                  # show the cause, not just the word "error"
                note = "error: " + " ".join(str(res.get("message", "")).split())[:200]
            emit({"phase": "convert", "done": done, "total": total, "current": name,
                  "message": note})

        t_phase = time.perf_counter()
        # docling uses 4 threads unless told otherwise: share the cores between the parallel
        # conversions instead (an explicit RAG_SEARCH_THREADS wins).  Children inherit this.
        if _USER_THREADS is None:          # recomputed per run (a long-lived process runs many)
            os.environ["RAG_SEARCH_THREADS"] = str(
                max(2, min(os.cpu_count() or 4, 8) // max(1, min(jobs, total))))
        emit({"phase": "convert", "done": 0, "total": total, "message": f"{total} file(s)"})
        _sink({"phase_event": {"phase": "convert", "status": "start", "total": total,
                               "workers": min(jobs, total) if total else 0}})
        # the stall watch: a conversion process that writes nothing to the event log for too long is stopped
        watch: stallwatch.StallWatch | None = None
        stalled: dict[str, str] = {}                  # document (path inside its collection) -> what happened
        rel_src: dict[str, str] = {}
        for t in tasks:
            with contextlib.suppress(ValueError):
                rel_src[mirror_rel(Path(t["src"]), roots).as_posix()] = t["src"]
        limit = stallwatch.limit_s() if stage_log else 0.0

        def _on_stall(st: dict[str, Any]) -> None:
            msg = stallwatch.describe(st)
            log.error("%s: %s", st.get("file"), msg)
            if st["pid"] == os.getpid():              # documents are converted in this process: only ending it helps
                with contextlib.suppress(OSError), open(str(stage_log), "a", encoding="utf-8") as fh:
                    fh.write(json.dumps({"ts": round(time.time(), 3), "event": "error",
                                         "error": f"{st.get('file')}: {msg.replace('and the run went on', 'and the run ended')}"},
                                        ensure_ascii=False) + "\n")
                logging.shutdown()
                os._exit(1)
            stalled[str(st.get("file", ""))] = msg
            work_event(stage_log, str(st.get("file", "")), "convert", "done", outcome="stalled", of_pid=st["pid"])
            with contextlib.suppress(OSError):
                os.kill(int(st["pid"]), signal.SIGKILL)

        if stage_log:                                 # with a limit of 0 it stops nothing, and still knows what is open
            watch = stallwatch.StallWatch(Path(str(stage_log)), limit).start(_on_stall)
        try:
            if jobs > 1 and total > 1:
                _convert_in_pool(tasks, min(jobs, total), _collect, watch=watch, stalled=stalled, rel_src=rel_src,
                                 stage_log=stage_log)
            else:
                for t in tasks:
                    _collect(prepare_document(t))
        finally:
            if watch is not None:
                watch.stop()

        phase_s["convert"] = round(time.perf_counter() - t_phase, 1)
        _sink({"phase_event": {"phase": "convert", "status": "done", "total": total, "seconds": phase_s["convert"]}})
        vlm.close_shared()                 # conversion is over: free the reader's memory before the embedder loads

        # Phase 2 — embeddings (one model, this process)
        indexed = 0
        t_phase = time.perf_counter()
        if prepared:
            if embedder is None:
                from .embedding import make_embedder

                embedder = make_embedder()
            _sink({"phase_event": {"phase": "embed", "status": "start", "total": len(prepared),
                                   "chunks": sum(int(r.get("nodes") or 0) for r in prepared), "model": model}})
            for i, res in enumerate(prepared, 1):
                src = Path(res["src"])
                since = round(time.time(), 3)      # when work on this document began
                emit({"phase": "embed", "done": i - 1, "total": len(prepared),
                      "current": src.name, "current_since": since,
                      "message": f"{res['nodes']} chunks"})
                t_doc = time.perf_counter()
                rel_name = mirror_rel(src, roots).as_posix()
                _sink({"stage": {"file": rel_name, "stage": "embed", "id": stages.id_of("embed"),
                                 "status": "start", "chunks": res["nodes"]}})
                _sink({"work": {"phase": "embed", "file": rel_name, "status": "start", "chunks": res["nodes"]}})
                embed_outcome = "error"
                try:
                    def _within(done_chunks: int, all_chunks: int, _n=src.name, _i=i,
                                _t=len(prepared), _s=since) -> None:
                        emit({"phase": "embed", "done": _i - 1, "total": _t, "current": _n,
                              "current_since": _s, "message": f"{done_chunks}/{all_chunks} chunks"})

                    meta = embed_document(
                        Path(res["idx_dir"]), src, res["sha"], embedder,
                        chunk_size=chunk_size, chunk_overlap=chunk_overlap, model=model,
                        progress=_within,
                        timing={"convert_s": res.get("convert_s", 0.0),
                                "chunk_s": res.get("chunk_s", 0.0)},
                        conversion=res.get("conversion"))
                    indexed += 1
                    embed_outcome = "indexed"
                    conv.time_s["embed"] = conv.time_s.get("embed", 0.0) + meta["embed_s"]
                    _sink({"stage": {"file": rel_name, "stage": "embed", "id": stages.id_of("embed"),
                                     "status": "done", "seconds": meta["embed_s"], "chunks": meta["nodes"]}})
                    _sink({"stage": {"file": rel_name, "stage": "write", "id": stages.id_of("write"),
                                     "status": "done", "part": "embeddings.npy, index.meta.json"}})
                    doc_event(src, "indexed", chunks=meta["nodes"], convert_s=meta["convert_s"],
                              chunk_s=meta["chunk_s"], embed_s=meta["embed_s"],
                              total_s=meta["build_s"], markdown=res.get("markdown", ""),
                              profile_s=res.get("profile_s"),
                              **_conv_kw(meta.get("conversion")))
                except Exception as exc:  # noqa: BLE001
                    errors.append({"src": str(src), "message": f"embedding failed: {exc}"})
                    doc_event(src, "error", message=f"embedding failed: {exc}",
                              total_s=round(res.get("elapsed_s", 0.0) + time.perf_counter() - t_doc, 2))
                finally:
                    _sink({"work": {"phase": "embed", "file": rel_name, "status": "done", "outcome": embed_outcome}})
            emit({"phase": "embed", "done": len(prepared), "total": len(prepared)})
            _sink({"phase_event": {"phase": "embed", "status": "done", "total": len(prepared), "indexed": indexed}})

        phase_s["embed"] = round(time.perf_counter() - t_phase, 1)

        # Phase 3 — merge (cheap: concatenation)
        t_phase = time.perf_counter()
        emit({"phase": "merge", "done": 0, "total": 0})
        _sink({"phase_event": {"phase": "merge", "status": "start"}})

        def _merge_note(name: str, status: str, **fields: Any) -> None:
            _sink({"work": {"phase": "merge", "file": name, "status": status, **fields}})

        merged = merge_all(paths.index, force=wipe, frozen=frozen, on_collection=_merge_note)
        _sink({"phase_event": {"phase": "merge", "status": "done",
                               "total": len([m for m in merged if not m.get("skipped")])}})
        _drop_empty_collections(paths)
        phase_s["merge"] = round(time.perf_counter() - t_phase, 1)
        for m in merged:
            if m.get("error"):
                errors.append({"src": f"collection:{m['collection']}", "message": m["error"]})

    vlm.close_shared()                     # the document reader's process (only when this process had one)
    cache_info: dict[str, Any] = {}
    try:                                   # forget cached pages no stored trace refers to any more
        pc = pagecache.PageCache(paths.workspace)
        if pc.root.is_dir():
            cache_info = {**pc.gc(pagecache.referenced_keys(paths.markup)), **pc.stats()}
    except Exception as exc:  # noqa: BLE001 - housekeeping only
        log.warning("page cache cleanup failed: %s", exc)
    summary = {
        "indexed": indexed, "skipped_fresh": skipped, "errors": errors, "wiped": wiped,
        "no_text": no_text,
        "unsupported_extension": unsupported,
        "collections": [m for m in merged if not m.get("error")],
        "elapsed_s": round(time.perf_counter() - t_start, 1),
        "phase_s": phase_s,
        "conversion": conv.snapshot(),
        "page_cache": cache_info,
        "scanned": len(sources),
        "removed": removed,
        "orphans_removed": orphans_removed,
        "known": known, "not_retried": len(known),
        **(plan_info or {}),
    }
    emit({"phase": "done", "done": 1, "total": 1})
    return summary


def dump_summary(summary: dict[str, Any]) -> str:
    return json.dumps(summary, indent=2, ensure_ascii=False)
