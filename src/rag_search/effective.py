"""What each stage of the pipeline will really use (stdlib only).

One place answers "which value does the next run get, and where did it come from?":

    built-in default  <  ``config.json``  <  an environment variable the daemon was started with

``settings_env`` is the single function that turns ``config.json``'s ``indexer`` / ``models`` tunables
into the environment a worker (or the search daemon, or a playground build) reads -- the indexer
daemon, the search daemon and the playground all call it, so they cannot disagree.  ``resolve`` runs the
very parsers the pipeline runs (``docling_convert.convert_settings``, ``vlm.mode``, ``repair.mode``...)
on that environment, so a value shown in the dashboard is a value the code will see.

``READ_BY`` names, for every setting, the function that reads it and when; ``tests/test_effective.py``
checks that function exists and that the setting reaches it.
"""

from __future__ import annotations

import os
from typing import Any, Mapping

from . import spec, stages
from .config import effective_jobs
from .paths import DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE, DEFAULT_MODEL, DEFAULT_RERANK_MODEL, DEFAULT_TOP_K

TUNABLE_ENVS: dict[str, spec.Tunable] = {t.env: t for t in spec.TUNABLES if t.env}
_T: dict[str, spec.Tunable] = {f"{t.section}.{t.key}": t for t in spec.TUNABLES}

# environment variables that choose a model or tune one stage and have no config.json key of the same
# meaning (they are read directly by the code); reported with the overrides so a stage can say "set by ..."
MODEL_ENVS = ("RAG_SEARCH_MODEL", "RAG_SEARCH_RERANK_MODEL", "RAG_SEARCH_VLM_MODEL", "RAG_SEARCH_REPAIR_MODEL")
EXTRA_ENVS: tuple[str, ...] = MODEL_ENVS + tuple(dict.fromkeys(v for s in stages.ALL for v in s.env_only))

# setting -> (module, function that reads it, when it matters)
READ_BY: dict[str, tuple[str, str, str]] = {
    "indexer.routing": ("rag_search.core.indexer", "_wants_routing",
                        "pages: each PDF page takes the path it needs; document: one docling call for the file"),
    "indexer.ocr": ("rag_search.core.docling_convert", "convert_settings",
                    "whether docling OCRs a page (text pages: only pictures; scanned pages: the whole page)"),
    "indexer.ocr_engine": ("rag_search.core.docling_convert", "_build_converter",
                           "whenever docling reads a page: text pages, Office files, and scans the document reader "
                           "did not read"),
    "indexer.ocr_lang": ("rag_search.core.docling_convert", "_build_converter", "docling OCR only"),
    "indexer.table_mode": ("rag_search.core.docling_convert", "_build_converter", "docling's table structure model"),
    "indexer.pdf_backend": ("rag_search.core.docling_convert", "pdf_backend_class", "docling's PDF parser"),
    "indexer.pipeline": ("rag_search.core.indexer", "_wants_routing",
                         "vlm hands whole pages to docling's own vision pipeline and turns page routing off"),
    "indexer.vlm": ("rag_search.core.conversion.vlm", "mode",
                    "auto: scans, large pictures and image files are read by the document reader when it can run"),
    "models.reader": ("rag_search.models", "reader_choice", "the vision model behind lanes 3.2c and 3.2d"),
    "models.memory_limit_gb": ("rag_search.models", "memory_limit_gb",
                               "how much memory the models may count on when the Models tab judges what fits"),
    "indexer.ocr_first": ("rag_search.core.conversion.routed", "ocr_first_mode", "lane b: a clean scan is read by OCR first"),
    "indexer.residue": ("rag_search.core.conversion.routed", "residue_mode",
                        "lane c: regions of ink outside the text layer are read by the document reader"),
    "indexer.escalate_digital": ("rag_search.core.conversion.routed", "escalate_digital_mode",
                                 "a text page that lost text goes on to the document reader (lane d)"),
    "indexer.layer_fill": ("rag_search.core.conversion.routed", "layer_fill_mode",
                           "lane a: a text page is compared with the PDF's own text layer and filled from it"),
    "indexer.stall_timeout": ("rag_search.core.stallwatch", "limit_s",
                              "how long a conversion process may be silent before it is stopped"),
    "indexer.docling_batch": ("rag_search.core.docling_convert", "_apply_batch_sizes", "docling's page batch"),
    "indexer.doc_timeout": ("rag_search.core.docling_convert", "_run", "one document's conversion time limit"),
    "indexer.jobs": ("rag_search.config", "effective_jobs", "documents converted side by side"),
    "indexer.repair": ("rag_search.core.conversion.repair", "mode", "cell repair on scanned pages"),
    "models.repair": ("rag_search.models", "repair_choice", "the model that re-reads suspect cells"),
    "indexer.chunk_size": ("rag_search.core.worker", "run_spec", "passage size, in estimated tokens"),
    "indexer.chunk_overlap": ("rag_search.core.worker", "run_spec", "overlap between neighbouring passages, in "
                                                                     "estimated tokens"),
    "models.embedding": ("rag_search.paths", "model_name", "the embedding model of every run and every query"),
    "models.embed_batch": ("rag_search.core.embedding", "Embedder", "passages encoded per forward pass"),
    "models.max_seq": ("rag_search.core.embedding", "Embedder", "longest passage the embedder reads, in tokens"),
    "models.dtype": ("rag_search.core.embedding", "torch_dtype", "weight precision of the embedder and the reranker"),
    "models.device": ("rag_search.core.embedding", "pick_device", "where the embedder and the reranker run"),
    "indexer.auto_publish": ("rag_search.core.indexer_daemon", "IndexerDaemon", "publish after a successful run"),
    "search.retrieval_pool": ("rag_search.core.search", "SearchEngine", "candidates per retriever per collection"),
    "search.rrf_k": ("rag_search.core.search", "SearchEngine", "reciprocal rank fusion constant"),
    "search.stages": ("rag_search.core.search", "SearchEngine", "which of keyword, vectors, rerank run"),
    "models.reranker": ("rag_search.paths", "rerank_model_name", "the cross-encoder"),
    "search.rerank_pool": ("rag_search.core.search", "SearchEngine", "candidates the cross-encoder reads"),
    "models.rerank_batch": ("rag_search.core.embedding", "Reranker", "pairs scored per forward pass"),
    "models.rerank_max_len": ("rag_search.core.embedding", "Reranker", "longest query + passage pair"),
    "search.top_k": ("rag_search.core.search", "SearchEngine", "results when the caller does not say"),
    "search.prewarm": ("rag_search.core.search_daemon", "SearchDaemon", "load the models when the daemon starts"),
}


def ambient(env: Mapping[str, str] | None = None) -> dict[str, str]:
    """The environment variables set (and non-empty) in *env* that change what a stage does -- the tunables'
    and the model choices -- i.e. the overrides that win over ``config.json``."""
    src = os.environ if env is None else env
    return {k: src[k] for k in (*TUNABLE_ENVS, *EXTRA_ENVS) if src.get(k)}


def settings_env(cfg: Mapping[str, Any], base: Mapping[str, str],
                 sections: tuple[str, ...] = ("indexer", "models")) -> dict[str, str]:
    """*base* plus, for each tunable of *sections* that has an environment variable and a value in *cfg*,
    that variable -- ``setdefault``: an actual environment variable always wins; 0 / "" in the config means
    "no override" and adds nothing."""
    env = dict(base)
    for section in sections:
        scfg = cfg.get(section) or {}
        for t in spec.TUNABLES_BY_SECTION[section]:
            if t.env and scfg.get(t.key):
                env.setdefault(t.env, str(scfg[t.key]))
    return env


def _int(env: Mapping[str, str], name: str, default: int) -> int:
    try:
        return int(env.get(name, "") or default)
    except ValueError:
        return default


def _conversion(env: Mapping[str, str]) -> tuple[dict[str, Any], str]:
    from .core.docling_convert import convert_settings

    try:
        return convert_settings(env=env), ""
    except ValueError as exc:
        return {}, str(exc)


def _value(setting: str, t: spec.Tunable | None, cfg: Mapping[str, Any], env: Mapping[str, str],
           conv: dict[str, Any], conv_err: str) -> Any:
    """What the code reads for *setting* given *env* (the merged environment) and *cfg*."""
    section, key = setting.split(".", 1)
    scfg = cfg.get(section) or {}
    if section == "indexer":
        if key in ("ocr", "ocr_engine", "ocr_lang", "table_mode", "pdf_backend", "pipeline", "routing",
                   "doc_timeout", "docling_batch"):
            if conv_err:
                return f"invalid: {conv_err}"
            m = {"ocr": "ocr", "ocr_engine": "engine", "table_mode": "table", "pdf_backend": "pdf_backend",
                 "pipeline": "pipeline", "routing": "routing"}
            if key in m:
                return conv[m[key]]
            if key == "ocr_lang":
                return ", ".join(conv["lang"]) or "the engine's own default"
            return int(conv["timeout"]) if key == "doc_timeout" else int(conv["batch"])
        if key == "vlm":
            from .core.conversion.vlm import mode
            return mode(env)
        if key == "repair":
            from .core.conversion.repair import mode as repair_mode
            return repair_mode(env)
        if key in ("ocr_first", "residue", "escalate_digital", "layer_fill"):
            from .core.conversion import routed
            return {"ocr_first": routed.ocr_first_mode, "residue": routed.residue_mode,
                    "escalate_digital": routed.escalate_digital_mode, "layer_fill": routed.layer_fill_mode}[key](env)
        if key == "stall_timeout":
            from .core import stallwatch
            return int(stallwatch.limit_s(env))
        if key == "chunk_size":
            return int(scfg.get("chunk_size") or DEFAULT_CHUNK_SIZE)
        if key == "chunk_overlap":
            return int(scfg.get("chunk_overlap") or DEFAULT_CHUNK_OVERLAP)
        if key == "jobs":
            return effective_jobs({"indexer": scfg})
        if key == "auto_publish":
            return bool(scfg.get("auto_publish", True))
    if section == "models":
        defaults = {"embed_batch": ("RAG_SEARCH_EMBED_BATCH", spec.EMBED_BATCH),
                    "max_seq": ("RAG_SEARCH_MAX_SEQ", spec.EMBED_MAX_SEQ),
                    "rerank_batch": ("RAG_SEARCH_RERANK_BATCH", spec.RERANK_BATCH),
                    "rerank_max_len": ("RAG_SEARCH_RERANK_MAX_LEN", spec.RERANK_MAX_LEN)}
        if key in defaults:
            name, dflt = defaults[key]
            return _int(env, name, dflt)
        if key in ("dtype", "device"):
            return (env.get(t.env, "").strip().lower() if t else "") or "chosen for this computer"
        if key == "embedding":
            return env.get("RAG_SEARCH_MODEL") or scfg.get("embedding") or DEFAULT_MODEL
        if key == "reranker":
            return env.get("RAG_SEARCH_RERANK_MODEL") or scfg.get("reranker") or DEFAULT_RERANK_MODEL
        if key in ("reader", "repair"):
            from . import models
            return env.get(models.VLM_ENV[key]) or scfg.get(key) or models.VLM_DEFAULTS[key]
        if key == "memory_limit_gb":
            return scfg.get("memory_limit_gb") or "none (the installed memory)"
    if section == "search":
        if key == "prewarm":
            return bool(scfg.get("prewarm", True))
        if key == "top_k":
            return int(scfg.get("top_k") or DEFAULT_TOP_K)
        if key == "stages":
            return scfg.get("stages") or "keyword, vectors, rerank"
        v = scfg.get(key)
        if v:
            return int(v)
        return t.default_label if t else ""
    return scfg.get(key)


def _source(setting: str, t: spec.Tunable | None, cfg: Mapping[str, Any], amb: Mapping[str, str]) -> str:
    section, key = setting.split(".", 1)
    if t is not None and t.env and amb.get(t.env):
        return "environment"
    envs = {"models.embedding": "RAG_SEARCH_MODEL", "models.reranker": "RAG_SEARCH_RERANK_MODEL",
            "models.reader": "RAG_SEARCH_VLM_MODEL", "models.repair": "RAG_SEARCH_REPAIR_MODEL"}
    if setting in envs and amb.get(envs[setting]):
        return "environment"
    v = (cfg.get(section) or {}).get(key)
    if setting == "indexer.auto_publish":
        return "config" if v is False else "default"
    if setting == "search.prewarm":
        return "config" if v is False else "default"
    return "config" if v not in (None, "", 0, False) else "default"


def resolve(cfg: Mapping[str, Any], base_env: Mapping[str, str] | None = None) -> dict[str, Any]:
    """Every stage with the settings it owns, as the next run / the daemon will see them.

    *cfg* is the loaded ``config.json`` (``config.load_config``), *base_env* the environment the process
    that does the work runs with (the indexer daemon's, reported by its ping); the dashboard's own
    environment when it is not known."""
    base = dict(os.environ if base_env is None else base_env)
    amb = ambient(base)
    env = settings_env(cfg, base, ("indexer", "models"))
    conv, conv_err = _conversion(env)
    rows: dict[str, Any] = {}
    out_stages = []
    for s in stages.ALL:
        items = []
        for setting in s.settings:
            t = _T.get(setting)
            value = _value(setting, t, cfg, env, conv, conv_err)
            mod, fn, when = READ_BY.get(setting, ("", "", ""))
            row = {"id": setting, "label": t.label if t else _LABELS.get(setting, setting),
                   "value": value, "source": _source(setting, t, cfg, amb),
                   "configured": (cfg.get(setting.split(".")[0]) or {}).get(setting.split(".", 1)[1]),
                   "env": t.env if t else "", "applies": t.applies if t else _APPLIES.get(setting, "next-run"),
                   "default": t.default_label if t else "", "read_by": f"{mod}.{fn}" if mod else "", "when": when,
                   "also_used_by": list(s.also_used_by.get(setting, ())), "editable": t is not None,
                   "kind": t.kind if t else "config"}
            items.append(row)
            rows[setting] = row
        envs = [{"name": v, "value": base.get(v, ""), "set": bool(base.get(v))} for v in s.env_only]
        out_stages.append({"id": s.id, "key": s.key, "name": s.name, "scope": s.scope, "where": s.where,
                           "what": s.what, "optional": s.optional, "parent": s.parent,
                           "constants": [{"label": a, "value": b} for a, b in s.constants],
                           "settings": items, "env_only": envs})
    return {"stages": out_stages, "overrides": amb, "errors": [conv_err] if conv_err else []}


_LABELS = {
    "models.embedding": "Embedding model", "models.reranker": "Reranker", "models.reader": "Document reader model",
    "models.repair": "Repair model", "models.memory_limit_gb": "Memory limit for models (GB)",
    "indexer.jobs": "Documents converted side by side", "indexer.auto_publish": "Publish after a run",
    "search.prewarm": "Load the models when the daemon starts",
}
_APPLIES = {"models.embedding": "re-embed", "models.reranker": "restart", "models.reader": "next-run",
            "models.repair": "next-run", "models.memory_limit_gb": "immediate", "indexer.jobs": "next-run",
            "indexer.auto_publish": "next-run", "search.prewarm": "restart"}
