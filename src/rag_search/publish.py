"""Publish the indexer workspace as an immutable, atomically-switched generation.

    indexer_workspace/index/<coll>/_all/{nodes.json, embeddings.npy, merge.manifest.json}
    indexer_workspace/markup/<coll>/<doc>.md
            │  hard-link (copy across filesystems)
            ▼
    serving/gen-000007/{index/<coll>/_all/..., markup/<coll>/..., catalog.json}
    serving/current -> gen-000007        (symlink swapped with an atomic rename)

Workspace files are write-once (always replaced by atomic rename), so hard links can
never observe a partial write, and a later rebuild in the workspace never alters an
already-published generation.  Stdlib only: the indexer supervisor imports this.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import re
import shutil
import time
from pathlib import Path
from typing import Any

from .locations import index_is_imported
from .paths import (
    ALL_DIR,
    EMB_FILE,
    MERGE_MANIFEST,
    META_FILE,
    NODES_FILE,
    SUPPORTED_EXTENSIONS,
    Paths,
    read_json,
    write_json_atomic,
)

CATALOG = "catalog.json"
KEEP_GENERATIONS = 3
_GEN_RE = re.compile(r"gen-(\d+)$")


class PublishError(RuntimeError):
    pass


def _mixed_models_error(paths: Paths, where: str, models: set[str], dims: set[int]) -> PublishError:
    """Publishing is refused while documents are embedded with different models: the usual cause
    is a model switch whose re-embedding has not finished (search keeps serving the old index)."""
    from .models import workspace_models

    try:
        by_model = workspace_models(paths, max_age=0)["by_model"]
    except Exception:  # noqa: BLE001 - only used for the message
        by_model = {}
    counts = ", ".join(f"{n} document(s) with {m or 'unknown'}" for m, n in sorted(by_model.items()))
    imported = []
    try:
        from .locations import imported_names, origin

        imported = [f"{c} ({origin(paths, c).get('model', '?')})" for c in imported_names(paths)]
    except Exception:  # noqa: BLE001 - only used for the message
        pass
    return PublishError(
        f"{where} mixes embedding models ({', '.join(sorted(models)) or '?'}; sizes "
        f"{sorted(dims)}): documents are still embedded with different models"
        + (f" - {counts}" if counts else "")
        + ". Finish the switch with `rag-search index new` (search keeps using the previous "
        "index until every document is on one model), or switch back with `rag-search models set`."
        + (f" Imported collections cannot be re-embedded here: {', '.join(imported)} -- delete "
           "any built with another model (`rag-search collection delete NAME`) and import an "
           "export made with the new one." if imported else ""))


def _link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def list_generations(paths: Paths) -> list[int]:
    if not paths.serving.is_dir():
        return []
    out = []
    for p in paths.serving.iterdir():
        m = _GEN_RE.fullmatch(p.name)
        if m and p.is_dir():
            out.append(int(m.group(1)))
    return sorted(out)


def read_catalog(gen_dir: Path | None) -> dict[str, Any]:
    if gen_dir is None:
        return {}
    return read_json(gen_dir / CATALOG) or {}


def _collect(paths: Paths) -> list[dict[str, Any]]:
    """Describe every workspace collection that has a complete merged index."""
    out = []
    if not paths.index.is_dir():
        return out
    for cdir in sorted(paths.index.iterdir()):
        if not cdir.is_dir() or cdir.name.startswith("."):
            continue
        all_dir = cdir / ALL_DIR
        manifest = read_json(all_dir / MERGE_MANIFEST)
        if not manifest or not (all_dir / NODES_FILE).exists() or not (all_dir / EMB_FILE).exists():
            continue
        docs = []
        models: set[str] = set()
        revisions: set[str] = set()
        dims: set[int] = set()
        build_s = 0.0
        timed = 0                      # documents that recorded how long they took
        md_bytes = src_bytes = 0
        for rel in manifest.get("docs", []):
            meta = read_json(cdir / rel / META_FILE)
            if meta.get("model"):
                models.add(meta["model"])
            revisions.add(str(meta.get("model_revision", "") or ""))
            if meta.get("dim"):
                dims.add(int(meta["dim"]))
            if "build_s" in meta:
                build_s += float(meta["build_s"])
                timed += 1
            src_bytes += int(meta.get("src_bytes", 0) or 0)
            with contextlib.suppress(OSError):
                md_bytes += (paths.markup / cdir.name / (rel + ".md")).stat().st_size
            docs.append({
                "name": rel,
                "source": Path(meta.get("src_path", "")).name,
                "chunks": meta.get("nodes", 0),
                "indexed_at": meta.get("built_at", ""),
                "build_s": meta.get("build_s"),           # None for documents indexed by 0.2.0
                "convert_s": meta.get("convert_s"),
                "embed_s": meta.get("embed_s"),
                "index_bytes": meta.get("index_bytes"),
                "source_bytes": meta.get("src_bytes"),
            })
        if len(models) > 1 or len(dims) > 1:
            raise _mixed_models_error(paths, f"collection {cdir.name!r}", models, dims)
        index_bytes = 0
        for fname in (NODES_FILE, EMB_FILE, MERGE_MANIFEST):
            with contextlib.suppress(OSError):
                index_bytes += (all_dir / fname).stat().st_size
        out.append({
            "collection": cdir.name, "manifest_sha": manifest.get("sha256", ""),
            "chunks": manifest.get("nodes", 0), "documents": docs,
            "model": next(iter(models), ""), "dim": next(iter(dims), 0),
            # one commit when every document agrees, "" when unknown or mixed
            "model_revision": next(iter(revisions)) if len(revisions) == 1 else "",
            "origin": "imported" if index_is_imported(paths.index, cdir.name) else "generated",
            # sizes and timings, for `rag-search list`
            "index_bytes": index_bytes,          # the searchable index (chunks + vectors)
            "markdown_bytes": md_bytes,          # converted Markdown, used by grep
            "source_bytes": src_bytes,           # the original files
            "built_at": max((d["indexed_at"] for d in docs if d["indexed_at"]), default=""),
            "build_seconds": round(build_s, 1) if timed else None,
            "build_seconds_documents": timed,    # how many documents the figure covers
        })
    return out


def _has_sources(folder: Path) -> bool:
    if not folder.is_dir():
        return False
    for _dp, dirnames, filenames in os.walk(folder):
        dirnames[:] = [d for d in dirnames if not d.startswith(".")]
        if any(Path(f).suffix.lower() in SUPPORTED_EXTENSIONS and not f.startswith((".", "~$"))
               for f in filenames):
            return True
    return False


def incomplete_documents(paths: Paths, limit: int = 20) -> list[str]:
    """Workspace documents that were started but not finished (e.g. a cancelled run).

    They are missing from the merged index until a later run completes them.
    """
    out: list[str] = []
    if not paths.index.is_dir():
        return out
    for dp, dirnames, filenames in os.walk(paths.index):
        dirnames[:] = [d for d in dirnames if d != ALL_DIR]
        if (NODES_FILE in filenames or EMB_FILE in filenames) and META_FILE not in filenames:
            out.append(str(Path(dp).relative_to(paths.index)))
            if len(out) >= limit:
                break
    return sorted(out)


def _content_sha(collections: list[dict[str, Any]]) -> str:
    h = hashlib.sha256()
    for c in collections:
        h.update(f"{c['collection']}:{c['manifest_sha']}\n".encode())
    return h.hexdigest()


def current_generation(paths: Paths) -> int | None:
    g = paths.current_gen()
    if g is None:
        return None
    m = _GEN_RE.fullmatch(g.name)
    return int(m.group(1)) if m else None


def publish(paths: Paths, *, force: bool = False, allow_drop: bool = False) -> dict[str, Any]:
    """Build the next generation from the workspace and make it current.

    Returns {"changed": bool, "generation": int|None, ...}.  A no-op when the
    workspace content equals what is already live (unless *force*).  A collection that
    is live but has no complete index in the workspace while its source folder still
    holds documents (e.g. after a cancelled full rebuild) is not silently dropped:
    that raises PublishError unless *allow_drop*.
    """
    out = _publish(paths, force, allow_drop)
    partial = incomplete_documents(paths)
    if partial:
        out["incomplete"] = partial
    return out


def _publish(paths: Paths, force: bool, allow_drop: bool) -> dict[str, Any]:
    collections = _collect(paths)
    models = {c["model"] for c in collections if c["model"]}
    if len(models) > 1:
        raise _mixed_models_error(paths, "the workspace", models, set())
    sha = _content_sha(collections)

    cur_gen = paths.current_gen()
    cur = read_catalog(cur_gen)
    if cur and not allow_drop:
        gone = sorted({c["collection"] for c in cur.get("collections", [])}
                      - {c["collection"] for c in collections})
        from .locations import source_root

        gone = [c for c in gone if _has_sources(source_root(paths, c))]
        if gone:
            raise PublishError(
                f"collection(s) {gone} have no complete index in the workspace (interrupted or "
                "cancelled run?); publishing now would remove them from search. Run "
                "`rag-search index new` to finish indexing them.")
    if not force:
        if cur and cur.get("content_sha") == sha:
            return {"changed": False, "generation": current_generation(paths),
                    "note": "workspace unchanged since the live generation"}
        if not cur and not collections:
            return {"changed": False, "generation": None, "note": "nothing to publish yet"}

    paths.serving.mkdir(parents=True, exist_ok=True)
    gens = list_generations(paths)
    n = (gens[-1] + 1) if gens else 1
    tmp = paths.serving / f".tmp-gen-{n:06d}-{os.getpid()}"
    shutil.rmtree(tmp, ignore_errors=True)
    try:
        for c in collections:
            coll = c["collection"]
            for fname in (NODES_FILE, EMB_FILE, MERGE_MANIFEST):
                _link_or_copy(paths.index / coll / ALL_DIR / fname,
                              tmp / "index" / coll / ALL_DIR / fname)
            for d in c["documents"]:
                md = paths.markup / coll / (d["name"] + ".md")
                if md.is_file():
                    _link_or_copy(md, tmp / "markup" / coll / (d["name"] + ".md"))
        (tmp / "index").mkdir(parents=True, exist_ok=True)
        (tmp / "markup").mkdir(parents=True, exist_ok=True)
        write_json_atomic(tmp / CATALOG, {
            "generation": n, "published_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "content_sha": sha, "model": next(iter(models), ""),
            "collections": collections,
        })
        final = paths.gen_dir(n)
        os.rename(tmp, final)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise

    link_tmp = paths.serving / ".current.tmp"
    with contextlib.suppress(FileNotFoundError):
        link_tmp.unlink()
    os.symlink(final.name, link_tmp)
    os.replace(link_tmp, paths.current_link)  # atomic switch
    gc(paths)
    return {"changed": True, "generation": n, "collections": len(collections),
            "chunks": sum(c["chunks"] for c in collections),
            "documents": sum(len(c["documents"]) for c in collections)}


def rollback(paths: Paths) -> dict[str, Any]:
    """Point `current` at the previous generation (still on disk)."""
    cur = current_generation(paths)
    older = [g for g in list_generations(paths) if cur is not None and g < cur]
    if not older:
        raise PublishError("no earlier generation to roll back to")
    target = paths.gen_dir(older[-1])
    link_tmp = paths.serving / ".current.tmp"
    with contextlib.suppress(FileNotFoundError):
        link_tmp.unlink()
    os.symlink(target.name, link_tmp)
    os.replace(link_tmp, paths.current_link)
    return {"changed": True, "generation": older[-1]}


def gc(paths: Paths, keep: int = KEEP_GENERATIONS) -> list[int]:
    """Delete old generations (never the live one) and stale temp dirs."""
    live = current_generation(paths)
    gens = list_generations(paths)
    removed = []
    for g in gens[:-keep] if keep > 0 else gens:
        if g != live:
            shutil.rmtree(paths.gen_dir(g), ignore_errors=True)
            removed.append(g)
    now = time.time()
    if paths.serving.is_dir():
        for p in paths.serving.glob(".tmp-gen-*"):
            with contextlib.suppress(OSError):
                if now - p.stat().st_mtime > 3600:
                    shutil.rmtree(p, ignore_errors=True)
    return removed


def catalog_json(paths: Paths) -> str:
    return json.dumps(read_catalog(paths.current_gen()), indent=2)
