"""Static facts for the dashboard's Architecture and Help tabs, taken from the code itself.

Numbers (chunk size, BM25 k1/b, RRF k, pool sizes, models ...) come from the same constants
the indexer and search engine use, so the page cannot drift from the implementation.
Stdlib only.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from pathlib import Path
from typing import Any

from .. import __version__, api, effective, locations, machine, models, register, spec
from ..core.chunker import CHUNKER_VERSION
from ..paths import DEFAULT_TOP_K, INDEX_FORMAT, Paths, env_flag
from ..publish import KEEP_GENERATIONS

APPLIES_LABEL = {
    spec.IMMEDIATE: "applies immediately, to the very next search",
    spec.NEXT_RUN: "applies to the next indexing run (no daemon restart)",
    spec.RESTART: "needs `rag-search daemon restart` to take effect",
}


def tunable_row(t: spec.Tunable) -> dict[str, Any]:
    return {"section": t.section, "key": t.key, "kind": t.kind, "default": t.default,
            "label": t.label, "what": t.what, "impact": t.impact, "applies": t.applies,
            "applies_label": APPLIES_LABEL[t.applies], "cli_flag": t.cli_flag,
            "choices": list(t.choices), "env": t.env, "default_label": t.default_label,
            "choice_help": dict(t.choice_help)}


def tunables() -> list[dict[str, Any]]:
    """Every production tunable, as data -- the dashboard's Settings/Models tabs render their
    controls from this (label/what always shown, impact behind an info icon) instead of
    hardcoding descriptions in JavaScript; `rag-search config set --help` and this list come
    from the exact same `spec.TUNABLES` registry, so they can never drift apart."""
    return [tunable_row(t) for t in spec.TUNABLES]


def config_storage(paths: Paths) -> dict[str, Any]:
    """Where a tunable's value actually lives, for the Architecture tab: one JSON file, one
    section per daemon, environment variables as the final override layer above it."""
    return {
        "file": str(paths.config_file),
        "sections": {
            section: [{"key": t.key, "label": t.label, "env": t.env} for t in ts]
            for section, ts in spec.TUNABLES_BY_SECTION.items()
        },
        "precedence": ["built-in default (hardcoded in spec.py)",
                       "config.json (rag-search config set / the dashboard's Settings and "
                       "Models tabs)",
                       "an actual environment variable of the same name, if the process was "
                       "started with one (RAG_SEARCH_OCR and family) -- see each tunable's "
                       "\"env\""],
        "who_reads_it": {
            "search": "search.* -- read fresh from config.json on every search "
                      "(core/search_daemon.py); no restart needed",
            "indexer": "indexer.* -- read fresh from config.json at the start of every indexing "
                      "run (core/indexer_daemon.py); chunk_size/chunk_overlap go into that run's "
                      "job spec, the rest become environment variables for the worker "
                      "subprocess it spawns",
            "models": "models.* -- read once, into environment variables, when a daemon "
                     "(search or indexer) starts or restarts, before the embedder/reranker are "
                     "constructed",
        },
    }

STATIC = Path(__file__).parent / "static"
DOCS = STATIC / "docs"

OPTIONAL_NODE_FIELDS = frozenset({"metadata.confidence"})     # present only on chunks of flagged pages

# Documentation of the on-disk formats.  tests/test_ui.py checks these field lists against the
# files the real pipeline writes, so they stay correct.
NODE_FIELDS = {
    "id": "stable chunk id: sha-based, from the file path, chunk number and source hash",
    "text": "the chunk's Markdown text (what is embedded, BM25-indexed and shown in results)",
    "metadata.page_label": "page number the chunk starts on, taken from the <!-- page N --> markers",
    "metadata.confidence": "\"low\" when the chunk's page was flagged low-confidence by the quality gate (left out otherwise)",
    "metadata.heading": "nearest Markdown heading in effect where the chunk begins",
    "metadata.file_name": "document name without extension",
    "metadata.source_name": "original file name, with extension",
    "metadata.collection": "the registered location's name",
    "metadata.doc_path": "path of the document inside its collection, without extension",
    "metadata.src_path": "absolute path of the source file (just the file name in an imported "
                         "collection)",
}
META_FIELDS = {
    "format": "index format number (currently 1)",
    "chunk_size": "target chunk size in estimated tokens",
    "chunk_overlap": "overlap carried between chunks, in estimated tokens",
    "chunker": "chunker algorithm version; a change re-chunks every document",
    "tokenizer": "BM25 tokenizer version",
    "model": "embedding model name",
    "model_revision": "commit of the model's weights in the local Hugging Face cache (\"\" when "
                      "unknown); advisory, compared when a collection export is imported",
    "convert": "conversion profile (OCR / table / pipeline settings); a change re-converts",
    "conversion": "the document's conversion summary: pages per branch (a run-length strip) and "
                  "outcome, scripts, docling's quality grades, time per step and CPU cost; the "
                  "per-page trace is markup/<collection>/<doc>.trace.json",
    "src_sha256": "SHA-256 of the source file: unchanged sources are skipped",
    "src_path": "absolute path of the source file",
    "nodes": "number of chunks",
    "dim": "embedding dimension",
    "built_at": "UTC time the document finished indexing",
    "convert_s": "seconds spent converting to Markdown",
    "chunk_s": "seconds spent chunking",
    "embed_s": "seconds spent embedding",
    "build_s": "convert_s + chunk_s + embed_s",
    "index_bytes": "size of nodes.json + embeddings.npy",
    "src_bytes": "size of the source file",
}
CATALOG_FIELDS = {
    "generation": "generation number (also the folder name gen-NNNNNN)",
    "published_at": "when this generation was published",
    "model": "embedding model of every collection in it",
    "content_sha": "hash of all collections' manifests: identical content publishes nothing",
    "collections[].collection": "collection name",
    "collections[].manifest_sha": "identity of the merged index; the search daemon reuses a loaded "
                                  "collection whose sha and model are unchanged",
    "collections[].chunks / dim / model / model_revision": "size and embedding details",
    "collections[].origin": "generated (indexed here) or imported (from a collection export)",
    "collections[].documents[]": "per-document name, chunks, indexed_at and timings",
    "collections[].index_bytes / markdown_bytes / source_bytes": "sizes",
    "collections[].built_at / build_seconds": "when and how long",
}


def _first_meta(paths: Paths) -> dict[str, Any]:
    """One real index.meta.json (for the live embedding dimension); {} when nothing is indexed."""
    for root in (paths.live_index(), paths.index):
        if root and root.is_dir():
            for f in root.rglob("index.meta.json"):
                try:
                    return json.loads(f.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    return {}
    return {}


def architecture(paths: Paths) -> dict[str, Any]:
    """Everything the Architecture tab needs to fill in its diagrams."""
    from ..config import ConfigStore

    cfg = ConfigStore(paths).get()
    pipe = api.pipeline(paths)                    # what each stage really runs with (values + where they come from)
    val = {r["id"]: r["value"] for st in pipe["stages"] for r in st["settings"]}
    ind_env = pipe["daemons"]["indexer"]
    conv, conv_err = effective._conversion(effective.settings_env(
        cfg, ind_env["env_overrides"] if ind_env["running"] else os.environ, ("indexer", "models")))
    conv = {"error": conv_err} if conv_err else conv
    meta = _first_meta(paths)
    rspec = models.find(str(val["models.reranker"]))
    rkind = ("LLM reranker (a language model judges 'yes' or 'no' for each passage)"
             if rspec and rspec.backend == models.QWEN3_RERANKER
             else "cross-encoder (query + passage scored together)")
    # every possible top_k (1..MAX_TOP_K), not just a few samples: the Search tab's debug
    # panel looks up the exact default pool sizes for whatever top_k is set, rather than
    # re-implementing the retrieval_pool()/rerank_pool() formulas in JavaScript.
    examples = {str(k): {"per_retriever": spec.retrieval_pool(k), "to_reranker": spec.rerank_pool(k)}
                for k in range(1, spec.MAX_TOP_K + 1)}
    return {
        "version": __version__,
        "stages": pipe["stages"],            # the numbered pipeline with the values in effect (see stages.py)
        "hosts": [{"name": h.NAME, "label": h.LABEL} for h in register.shown_hosts()],
        "python": platform.python_version(),
        "platform": f"{machine.system()} {machine.arch()}",
        "machine": machine.describe(),
        "models": {
            "embedding": {
                "name": str(val["models.embedding"]), "serving": models.serving_model(paths),
                "kind": "bi-encoder (dense vectors)",
                "dim": meta.get("dim"), "max_seq": int(val["models.max_seq"]),
                "batch": int(val["models.embed_batch"]), "normalized": True,
            },
            "reranker": {
                "name": str(val["models.reranker"]), "kind": rkind,
                "enabled": env_flag("RAG_SEARCH_RERANK", True) and "rerank" in str(val["search.stages"]),
                "batch": int(val["models.rerank_batch"]),
                "max_len": int(val["models.rerank_max_len"]),
                "cap": spec.RERANK_CAP,
            },
            "conversion": {"tool": "docling", **conv},
        },
        # what the next indexing run uses: config.json's indexer section, else the defaults
        "chunking": {"size": int(val["indexer.chunk_size"]),
                     "overlap": int(val["indexer.chunk_overlap"]),
                     "version": CHUNKER_VERSION,
                     "estimator": "max(1.3 x words + 1, characters / 4) tokens"},
        "keyword": {"algorithm": "Okapi BM25", "k1": spec.BM25_K1, "b": spec.BM25_B,
                    "tokenizer": spec.TOKENIZER_VERSION},
        "fusion": {"method": "Reciprocal Rank Fusion", "k": spec.RRF_K,
                  "k_min": spec.RRF_K_MIN, "k_max": spec.RRF_K_MAX},
        "pools": examples,
        "pool_overrides": {"retrieval_pool_max": spec.RETRIEVAL_POOL_MAX,
                           "rerank_pool_max": spec.RERANK_POOL_MAX},
        "limits": {"top_k_default": DEFAULT_TOP_K, "top_k_max": spec.MAX_TOP_K,
                   "snippet_chars": spec.SNIPPET_CHARS},
        "index": {"format": INDEX_FORMAT, "generations_kept": KEEP_GENERATIONS,
                  "node_fields": NODE_FIELDS, "meta_fields": META_FIELDS,
                  "catalog_fields": CATALOG_FIELDS},
        "paths": {"home": str(paths.home)},
        "sources": locations.sources(paths),
        "config_storage": config_storage(paths),
    }


# ── CLI reference, generated from the real argparse tree ────────────────────

def _arg_row(a: argparse.Action) -> dict[str, Any] | None:
    if isinstance(a, (argparse._HelpAction, argparse._SubParsersAction)):
        return None
    if a.help == argparse.SUPPRESS:
        return None
    flags = "/".join(a.option_strings) if a.option_strings else (a.metavar or a.dest)
    if a.option_strings and a.nargs != 0 and not isinstance(a, argparse._VersionAction):
        flags += f" {a.metavar or a.dest.upper()}"
    default = a.default
    return {"flags": flags, "help": a.help or "",
            "default": None if default in (None, argparse.SUPPRESS, False, "", 0) else str(default),
            "positional": not a.option_strings}


def _walk(parser: argparse.ArgumentParser, prefix: str, out: list[dict[str, Any]]) -> None:
    subs = [a for a in parser._actions if isinstance(a, argparse._SubParsersAction)]
    for sub in subs:
        helps = {c.dest: c.help for c in sub._get_subactions()}
        for name, p in sub.choices.items():
            full = f"{prefix} {name}".strip()
            nested = [a for a in p._actions if isinstance(a, argparse._SubParsersAction)]
            if not nested or p.get_default("fn"):
                out.append({"command": full, "help": helps.get(name) or p.description or "",
                            "description": p.description or "",
                            "usage": " ".join(p.format_usage().replace("usage:", "").split()),
                            "args": [r for a in p._actions if (r := _arg_row(a))
                                     and r["flags"].split(" ")[0] not in ("--home", "--json", "--client")]})
            _walk(p, full, out)


def cli_reference() -> dict[str, Any]:
    """{commands: [{command, help, usage, args}], global: [...]} for every `rag-search` command."""
    from ..cli import build_parser

    ap = build_parser()
    cmds: list[dict[str, Any]] = []
    _walk(ap, "", cmds)
    seen, unique = set(), []
    for c in cmds:                       # a group parser and its default action may repeat
        if c["command"] not in seen:
            seen.add(c["command"])
            unique.append(c)
    glob = [r for a in ap._actions if (r := _arg_row(a))]
    return {"commands": unique, "global": glob, "python": sys.version.split()[0]}


def read_doc(name: str) -> str:
    """README.md or ARCHITECTURE.md as packaged with the dashboard ('' when missing)."""
    try:
        return (DOCS / name).read_text(encoding="utf-8")
    except OSError:
        return ""
