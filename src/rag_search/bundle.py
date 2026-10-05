"""Export a collection's index to one file, and import such a file (stdlib only).

An export is a single gzip-compressed tar file (``NAME.rag.tgz``)::

    manifest.json                       what is inside and how it was built (see below)
    index/_all/nodes.json               the merged chunks of the collection
    index/_all/embeddings.npy           their vectors
    index/_all/merge.manifest.json      the merge record (document list, content hash)
    index/docs/<doc>/index.meta.json    per-document metadata (sizes, timings, model)
    markup/<doc>.md                     the converted Markdown (what rag_grep searches)

``manifest.json`` records the embedding model **and the exact commit of its weights**, the
vector size, the chunking and tokenizer versions, the conversion settings, the collection's
description and a SHA-256 for every file.  Nothing machine-specific travels: absolute source
paths are reduced to file names, and access rules (``access.json``) are never exported --
whoever imports decides who may use the collection.

Import refuses anything it cannot use as-is:

* the embedding model must be the one this installation is configured with (and, when both
  sides know it, the same weights commit): vectors from another model are not comparable, so
  there is no partial or keyword-only import;
* every file must match the manifest's hash, and the archive may contain nothing else (no
  links, no absolute or ``..`` paths);
* the name must be free -- an existing collection is never overwritten, except an earlier
  import of it with ``replace=True``.

An imported collection has no source documents.  A ``collection.origin.json`` file in its
workspace folder marks it, so indexing never scans, prunes, re-merges or rebuilds it; it is
published and searched like any other, and removed with ``rag-search collection delete``.
"""

from __future__ import annotations

import ast
import contextlib
import hashlib
import io
import json
import os
import shutil
import tarfile
import tempfile
import time
from pathlib import Path, PurePosixPath
from typing import Any, Iterator

from . import __version__, descriptions, locations, policy
from .paths import (
    ALL_DIR,
    DEFAULT_COLLECTION,
    EMB_FILE,
    MERGE_MANIFEST,
    META_FILE,
    NODES_FILE,
    IndexBusyError,
    Paths,
    ensure_dirs,
    index_lock,
    is_plain_name,
    model_name,
    read_json,
    write_json_atomic,
    pasted_path,
)

BUNDLE_FORMAT = "rag-search-collection"
BUNDLE_VERSION = 1
SUFFIX = ".rag.tgz"
MAX_MEMBER_BYTES = 8 << 30            # no single file in an export is anywhere near this


class BundleError(ValueError):
    """An export/import the user can fix (unknown collection, model mismatch, name taken...)."""


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def npy_shape(data: bytes | Path) -> tuple[int, ...]:
    """Shape recorded in a ``.npy`` header (read without numpy)."""
    if isinstance(data, Path):
        with open(data, "rb") as f:
            head = f.read(4096)
    else:
        head = data[:4096]
    if head[:6] != b"\x93NUMPY":
        raise BundleError("embeddings.npy is not a NumPy array file")
    major = head[6]
    if major == 1:
        n, start = int.from_bytes(head[8:10], "little"), 10
    else:
        n, start = int.from_bytes(head[8:12], "little"), 12
    try:
        header = ast.literal_eval(head[start:start + n].decode("latin1"))
        return tuple(int(x) for x in header["shape"])
    except (ValueError, SyntaxError, KeyError, TypeError) as exc:
        raise BundleError(f"embeddings.npy has an unreadable header ({exc})") from None


@contextlib.contextmanager
def _index_lock(paths: Paths) -> Iterator[None]:
    """The indexer's lock (``paths.index_lock``): an export reads a consistent workspace, and an
    import never races an indexing run."""
    try:
        with index_lock(paths):
            yield
    except IndexBusyError:
        raise BundleError("an indexing run is in progress; try again when it has finished "
                          "(rag-search index status)") from None


def _workspace_name(paths: Paths, name: str) -> str:
    name = (name or "").strip()
    if not is_plain_name(name):
        raise BundleError(f"{name!r} is not a collection name")
    have = locations.workspace_collections(paths)
    hit = name if name in have else next((c for c in have if c.casefold() == name.casefold()), "")
    if not hit:
        raise BundleError(f"no indexed collection named {name!r} (have: "
                          f"{', '.join(have) or 'none'})")
    return hit


# ── export ──────────────────────────────────────────────────────────────────

def export_collection(paths: Paths, name: str, out: str | Path | None = None) -> dict[str, Any]:
    """Write collection *name*'s index, Markdown and metadata to one ``.rag.tgz`` file."""
    coll = _workspace_name(paths, name)
    cdir = paths.index / coll
    all_dir = cdir / ALL_DIR
    target = Path(pasted_path(out)).expanduser() if out else Path.cwd()
    # a folder (existing, or any name that does not look like an archive) gets NAME.rag.tgz
    if target.is_dir() or not target.name.endswith((".tgz", ".tar.gz")):
        target = target / f"{coll}{SUFFIX}"
    target = target.resolve()
    with _index_lock(paths):
        merge = read_json(all_dir / MERGE_MANIFEST)
        if not merge or not (all_dir / NODES_FILE).is_file() or not (all_dir / EMB_FILE).is_file():
            raise BundleError(f"{coll!r} has no complete index to export (run `rag-search index "
                              "new` first)")
        doc_rels: list[str] = [str(d) for d in merge.get("docs", [])]
        metas: dict[str, dict[str, Any]] = {}
        for rel in doc_rels:
            meta = read_json(cdir / rel / META_FILE)
            if not meta:
                raise BundleError(f"{coll}/{rel}: index.meta.json is missing; re-index first")
            meta = dict(meta)
            meta["src_path"] = Path(str(meta.get("src_path", "")) or rel).name
            metas[rel] = meta
        built = _uniform(coll, metas)
        nodes = read_json(all_dir / NODES_FILE)
        for n in nodes.get("nodes", []):
            md = n.get("metadata") or {}
            md["src_path"] = Path(str(md.get("src_path", "")) or md.get("source_name", "")).name
        files: dict[str, bytes] = {
            "index/_all/nodes.json": json.dumps(nodes, ensure_ascii=False).encode(),
            "index/_all/embeddings.npy": (all_dir / EMB_FILE).read_bytes(),
            "index/_all/merge.manifest.json": json.dumps(merge, indent=2).encode(),
        }
        for rel, meta in metas.items():
            files[f"index/docs/{rel}/{META_FILE}"] = json.dumps(meta, indent=2).encode()
            md_file = paths.markup / coll / (rel + ".md")
            if md_file.is_file():
                files[f"markup/{rel}.md"] = md_file.read_bytes()
    shape = npy_shape(files["index/_all/embeddings.npy"])
    if len(shape) != 2 or shape[0] != len(nodes.get("nodes", [])):
        raise BundleError(f"{coll!r}: {len(nodes.get('nodes', []))} chunks but embeddings of "
                          f"shape {shape}; re-index it before exporting")
    manifest = {
        "format": BUNDLE_FORMAT, "bundle_version": BUNDLE_VERSION,
        "rag_search_version": __version__,
        "exported_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "collection": coll,
        "description": descriptions.get_description(paths, coll),
        **built,
        "dim": shape[1],
        "documents": len(doc_rels),
        "chunks": shape[0],
        "files": {k: _sha(v) for k, v in sorted(files.items())},
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix="." + target.name + ".", suffix=".tmp", dir=target.parent)
    os.close(fd)
    try:
        with tarfile.open(tmp, "w:gz", compresslevel=6) as tf:
            _add(tf, "manifest.json", json.dumps(manifest, indent=2, ensure_ascii=False).encode())
            for k in sorted(files):
                _add(tf, k, files[k])
        os.chmod(tmp, 0o644)               # a file meant to be handed to someone else
        os.replace(tmp, target)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise
    return {"collection": coll, "file": str(target), "bytes": target.stat().st_size,
            "documents": manifest["documents"], "chunks": manifest["chunks"],
            "model": manifest["model"], "model_revision": manifest["model_revision"]}


def _uniform(coll: str, metas: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """The build settings every document of *coll* shares (an export must be one model)."""
    keys = ("model", "model_revision", "chunk_size", "chunk_overlap", "chunker", "tokenizer",
            "format", "convert")
    out: dict[str, Any] = {}
    for k in keys:
        vals = {json.dumps(m.get(k, "")) for m in metas.values()}
        if len(vals) > 1:
            if k in ("model", "format", "chunk_size", "chunker", "tokenizer"):
                raise BundleError(f"{coll!r} mixes {k} values across its documents ({sorted(vals)}); "
                                  "finish re-indexing it (`rag-search index new`) before exporting")
            out[k] = ""                     # e.g. revision unknown for some documents
        else:
            out[k] = json.loads(next(iter(vals))) if vals else ""
    out["index_format"] = out.pop("format")       # "format" names the bundle itself
    if not out.get("model"):
        raise BundleError(f"{coll!r} does not record its embedding model; re-index it first")
    return out


def _add(tf: tarfile.TarFile, name: str, data: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(data)
    info.mtime = int(time.time())
    info.mode = 0o644
    tf.addfile(info, io.BytesIO(data))


# ── import ──────────────────────────────────────────────────────────────────

def read_manifest(archive: str | Path) -> dict[str, Any]:
    """The manifest of an export (checked for format), without unpacking anything else."""
    p = Path(pasted_path(archive)).expanduser()
    try:
        with tarfile.open(p, "r:gz") as tf:
            m = tf.getmember("manifest.json")
            f = tf.extractfile(m) if m.isfile() else None
            data = json.loads(f.read().decode()) if f is not None else None
    except FileNotFoundError:
        raise BundleError(f"no such file: {p}") from None
    except KeyError:
        raise BundleError(f"{p.name} is not a rag-search collection export (no manifest.json)") \
            from None
    except (tarfile.TarError, OSError, ValueError, UnicodeDecodeError) as exc:
        raise BundleError(f"{p.name} is not a readable collection export ({exc})") from None
    if not isinstance(data, dict) or data.get("format") != BUNDLE_FORMAT:
        raise BundleError(f"{p.name} is not a rag-search collection export")
    if int(data.get("bundle_version", 0) or 0) > BUNDLE_VERSION:
        raise BundleError(f"{p.name} was made by a newer rag-search "
                          f"({data.get('rag_search_version', '?')}); upgrade to import it")
    for k in ("collection", "model", "files"):
        if not data.get(k):
            raise BundleError(f"{p.name}: manifest.json has no {k!r}")
    files = data["files"]
    if not isinstance(files, dict) or not all(isinstance(k, str) and isinstance(v, str)
                                              for k, v in files.items()):
        raise BundleError(f"{p.name}: manifest.json lists its files in an unexpected form")
    if not isinstance(data["collection"], str) or not isinstance(data["model"], str):
        raise BundleError(f"{p.name}: manifest.json is damaged")
    for k in ("dim", "documents", "chunks"):
        try:
            data[k] = int(data.get(k) or 0)
        except (TypeError, ValueError):
            raise BundleError(f"{p.name}: manifest.json has a non-numeric {k!r}") from None
    return data


def check_model(manifest: dict[str, Any], paths: Paths | None = None) -> dict[str, str]:
    """Refuse an export built with another embedding model than this installation's.

    "This installation's" model is the configured one *and* -- when something is published --
    the one the published index was built with (what search actually serves).  While the two
    differ a model switch is in progress, and an import is refused until it has finished."""
    from .models import cached_revision

    want, theirs = model_name(), str(manifest.get("model", ""))
    if paths is not None:
        from .catalog import live_catalog

        serving = str(live_catalog(paths).get("model", "") or "")
        if serving and serving != want:
            raise BundleError(
                f"a switch of the embedding model is in progress (search serves {serving!r}, "
                f"{want!r} is configured); finish it (`rag-search index new`) or switch back "
                "before importing")
    if theirs != want:
        raise BundleError(
            f"this export was embedded with {theirs!r}, but this installation uses {want!r}. "
            "Vectors from different models cannot be searched together, so it cannot be "
            f"imported. Either import an export made with {want!r}, or switch this installation "
            f"to {theirs!r} first (`rag-search models set embedding {theirs}` -- note that this "
            "re-embeds your own documents too).")
    mine, rev = cached_revision(want), str(manifest.get("model_revision", "") or "")
    if mine and rev and mine != rev:
        raise BundleError(
            f"this export was embedded with {theirs!r} at weights commit {rev[:12]}, but the "
            f"copy of that model here is commit {mine[:12]} -- different weights give different "
            "vectors. Update the model on one side (`rag-search models download` fetches the "
            "current one) so both match, then export/import again.")
    note = "" if (mine and rev) else ("the model's exact weights could not be compared (one side "
                                      "does not record them); the model name matches")
    return {"model": want, "revision": mine or rev, "note": note}


def _safe_member(m: tarfile.TarInfo) -> PurePosixPath:
    name = PurePosixPath(m.name)
    if not (m.isfile() or m.isdir()):
        raise BundleError(f"refusing {m.name!r}: only plain files may be in an export")
    if name.is_absolute() or ".." in name.parts or not name.parts:
        raise BundleError(f"refusing {m.name!r}: unsafe path")
    if name.parts[0] not in ("manifest.json", "index", "markup"):
        raise BundleError(f"refusing {m.name!r}: not part of a collection export")
    if any(p.startswith(".") for p in name.parts) and name.name != "manifest.json":
        raise BundleError(f"refusing {m.name!r}: hidden files are not part of an export")
    if m.size > MAX_MEMBER_BYTES:
        raise BundleError(f"refusing {m.name!r}: implausibly large")
    return name


def _taken(paths: Paths, name: str) -> tuple[str, str]:
    """(existing spelling, what it is) when *name* is already a collection here, else ("", "")."""
    from .catalog import collection_names, docs_folder_names, live_catalog

    checks = (
        (locations.imported_names(paths), "imported"),
        (locations.names(paths), "a registered location"),
        (docs_folder_names(paths), "a folder in the docs folder"),
        (locations.workspace_collections(paths), "an indexed collection"),
        (collection_names(live_catalog(paths)), "a published collection"),
    )
    for names, what in checks:
        for n in names:
            if n.casefold() == name.casefold():
                return n, what
    return "", ""


def import_collection(paths: Paths, archive: str | Path, *, as_name: str | None = None,
                      replace: bool = False) -> dict[str, Any]:
    """Unpack an export into the workspace as an imported collection (then publish it)."""
    src = Path(pasted_path(archive)).expanduser().resolve()
    manifest = read_manifest(src)
    model = check_model(manifest, paths)
    name = (as_name or manifest["collection"]).strip()
    if not is_plain_name(name) or name.casefold() == DEFAULT_COLLECTION:
        raise BundleError(f"{name!r} cannot be a collection name here; choose one with --as NAME")
    ensure_dirs(paths)
    with _index_lock(paths):
        existing, what = _taken(paths, name)
        if existing:
            if what != "imported":
                raise BundleError(f"{existing!r} is already {what} here; import this export "
                                  "under another name (--as NAME)")
            if not replace:
                raise BundleError(f"{existing!r} was imported before; pass --replace to replace "
                                  "it with this export")
        stage = Path(tempfile.mkdtemp(prefix=".import-", dir=paths.workspace))
        try:
            got = _unpack(src, stage, manifest)
            _check_contents(stage, manifest, got)
            coll = _assemble(stage)
            _rename_collection(coll, manifest["collection"], name)
            locations.write_origin(coll, {
                "source_collection": manifest["collection"], "file": src.name,
                "model": manifest["model"], "model_revision": manifest.get("model_revision", ""),
                "exported_at": manifest.get("exported_at", ""),
                "exported_by_version": manifest.get("rag_search_version", ""),
                "documents": manifest.get("documents", 0), "chunks": manifest.get("chunks", 0)})
            _install(paths, coll, stage / "markup", existing or name, name)
        finally:
            shutil.rmtree(stage, ignore_errors=True)
    desc = str(manifest.get("description", "") or "")
    if desc and not descriptions.get_description(paths, name):
        with contextlib.suppress(descriptions.DescriptionError):
            descriptions.set_description(paths, name, desc)
    rules = policy.current_rules(paths)
    return {"collection": name, "from": manifest["collection"], "file": str(src),
            "documents": manifest.get("documents", 0), "chunks": manifest.get("chunks", 0),
            "model": model["model"], "replaced": bool(existing),
            "access": "everyone" if rules.clients_for(name) is None else "restricted",
            "note": model["note"]}


def _unpack(src: Path, stage: Path, manifest: dict[str, Any]) -> dict[str, str]:
    """Extract every file into *stage* (validated), returning {member: sha256}."""
    got: dict[str, str] = {}
    try:
        with tarfile.open(src, "r:gz") as tf:
            for m in tf:
                rel = _safe_member(m)
                if m.isdir() or str(rel) == "manifest.json":
                    continue
                if str(rel) not in manifest["files"]:
                    raise BundleError(f"refusing {m.name!r}: not listed in the manifest")
                dest = stage.joinpath(*rel.parts)
                dest.parent.mkdir(parents=True, exist_ok=True)
                f = tf.extractfile(m)
                if f is None:
                    raise BundleError(f"cannot read {m.name!r}")
                h = hashlib.sha256()
                with open(dest, "wb") as out:
                    for chunk in iter(lambda: f.read(1 << 20), b""):
                        h.update(chunk)
                        out.write(chunk)
                got[str(rel)] = h.hexdigest()
    except (tarfile.TarError, OSError) as exc:
        raise BundleError(f"{src.name}: cannot unpack ({exc})") from None
    return got


def _check_contents(stage: Path, manifest: dict[str, Any], got: dict[str, str]) -> None:
    want: dict[str, str] = manifest["files"]
    missing = sorted(set(want) - set(got))
    if missing:
        raise BundleError(f"the export is incomplete: missing {', '.join(missing[:5])}")
    bad = sorted(k for k, v in want.items() if got.get(k) != v)
    if bad:
        raise BundleError(f"the export is damaged: {', '.join(bad[:5])} do not match their "
                          "checksums")
    all_dir = stage / "index" / "_all"
    nodes = read_json(all_dir / NODES_FILE).get("nodes")
    if not isinstance(nodes, list):
        raise BundleError("the export has no readable nodes.json")
    shape = npy_shape(all_dir / EMB_FILE)
    if len(shape) != 2 or shape[0] != len(nodes):
        raise BundleError(f"the export holds {len(nodes)} chunks but embeddings of shape {shape}")
    if int(manifest.get("dim", shape[1]) or shape[1]) != shape[1]:
        raise BundleError(f"the manifest says {manifest.get('dim')} dimensions, the vectors have "
                          f"{shape[1]}")


def _assemble(stage: Path) -> Path:
    """Lay the unpacked index out as a workspace collection folder, ``stage/collection``:
    ``index/_all`` -> ``_all``, ``index/docs/<doc>/index.meta.json`` -> ``<doc>/index.meta.json``
    (built in a fresh folder: a document may itself be called "docs" or "_all"-like names)."""
    coll = stage / "collection"
    coll.mkdir()
    os.replace(stage / "index" / "_all", coll / ALL_DIR)
    docs = stage / "index" / "docs"
    if docs.is_dir():
        for meta in sorted(docs.rglob(META_FILE)):
            rel = meta.parent.relative_to(docs)
            if not rel.parts or rel.parts[0] == ALL_DIR:
                raise BundleError(f"refusing document name {rel.as_posix()!r} in the export")
            dest = coll / rel / META_FILE
            dest.parent.mkdir(parents=True, exist_ok=True)
            os.replace(meta, dest)
    (stage / "markup").mkdir(exist_ok=True)
    return coll


def _rename_collection(coll: Path, old: str, new: str) -> None:
    """Imported under another name: chunk metadata names the collection it belongs to."""
    if old == new:
        return
    f = coll / ALL_DIR / NODES_FILE
    data = read_json(f)
    for n in data.get("nodes", []):
        md = n.get("metadata") or {}
        if md.get("collection") == old:
            md["collection"] = new
    write_json_atomic(f, data, indent=None)


def _install(paths: Paths, coll: Path, markup: Path, old: str, name: str) -> None:
    """Move the assembled collection into the workspace, replacing an earlier import of it.

    Both moves are renames inside the workspace.  If either fails, whatever was moved is put
    back, so the workspace never ends up with a half-replaced collection."""
    moves = ((paths.index, coll), (paths.markup, markup))
    parked: list[tuple[Path, Path]] = []        # (where it was, where it is parked)
    placed: list[Path] = []
    try:
        for root, _new in moves:
            cur = root / old
            if cur.exists():
                spot = root / f".old-{old}-{os.getpid()}"
                shutil.rmtree(spot, ignore_errors=True)
                os.replace(cur, spot)
                parked.append((cur, spot))
        for root, new in moves:
            os.replace(new, root / name)
            placed.append(root / name)
    except BaseException:
        for p in placed:
            shutil.rmtree(p, ignore_errors=True)
        for cur, spot in parked:
            with contextlib.suppress(OSError):
                os.replace(spot, cur)
        raise
    for _cur, spot in parked:
        shutil.rmtree(spot, ignore_errors=True)
