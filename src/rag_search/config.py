"""Settings: built-in defaults, overridden by ``<home>/config.json``, then by environment.

Example ``config.json`` (every key is optional):

    {
      "search":  {"idle_exit_seconds": 0, "prewarm": true,
                   "retrieval_pool": 0, "rerank_pool": 0, "rrf_k": 0, "stages": "", "top_k": 0},
      "indexer": {"idle_exit_seconds": 0, "jobs": 0, "auto_publish": true,
                   "chunk_size": 0, "chunk_overlap": 0, "ocr": "", "ocr_engine": "", "ocr_lang": "",
                   "table_mode": "", "pdf_backend": "", "pipeline": "", "routing": "", "vlm": "", "repair": "",
                   "ocr_first": "", "residue": "", "escalate_digital": "", "layer_fill": "",
                   "stall_timeout": 0, "doc_timeout": 0, "docling_batch": 0},
      "models":  {"embedding": "", "reranker": "", "reader": "", "repair": "", "memory_limit_gb": 0,
                   "embed_batch": 0, "max_seq": 0, "dtype": "", "rerank_batch": 0,
                   "rerank_max_len": 0, "device": ""}
    }

Which clients may use which collections is not configured here: see ``policy.py`` and
``rag-search access``.

``idle_exit_seconds: 0`` means the daemon never exits by itself.  ``jobs: 0`` means
"choose from installed RAM" (1 worker on <= 17 GB machines, otherwise 2).

The pipeline tunables (everything besides ``idle_exit_seconds``/``prewarm``/``jobs``/
``auto_publish``/``embedding``/``reranker``/``memory_limit_gb``) are documented in one place --
``spec.TUNABLES`` -- which also validates them; see that module for what each one means and when a
change takes effect.  ``0``/``""`` there always means "use the built-in default", matching the
existing convention above.  ``rag-search config set`` and the dashboard's Settings/Models tabs are
the normal way to change them; an environment variable of the same family (e.g.
``RAG_SEARCH_OCR``) still wins over whatever is stored here -- see ``spec.Tunable.env`` and
`core/indexer_daemon.py`/`core/embedding.py` for where that final override is applied.
"""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Any

from .paths import Paths, env_flag, write_json_atomic

DEFAULTS: dict[str, Any] = {
    "search": {
        "idle_exit_seconds": 0, "prewarm": True,
        "retrieval_pool": 0, "rerank_pool": 0, "rrf_k": 0, "stages": "", "top_k": 0,
    },
    "indexer": {
        "idle_exit_seconds": 0, "jobs": 0, "auto_publish": True,
        "chunk_size": 0, "chunk_overlap": 0,
        "ocr": "", "ocr_engine": "", "ocr_lang": "", "table_mode": "", "pdf_backend": "",
        "pipeline": "", "routing": "", "vlm": "", "repair": "",
        "ocr_first": "", "residue": "", "escalate_digital": "", "layer_fill": "", "stall_timeout": 0,
        "doc_timeout": 0, "docling_batch": 0,
    },
    # "" = the built-in default model; change with `rag-search models set` or the dashboard.
    # memory_limit_gb: 0 = no limit of your own (models are then judged against the RAM installed)
    "models": {
        "embedding": "", "reranker": "", "reader": "", "repair": "", "memory_limit_gb": 0,
        "embed_batch": 0, "max_seq": 0, "dtype": "", "rerank_batch": 0, "rerank_max_len": 0,
        "device": "",
    },
}


def _merge(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def _env_overrides(cfg: dict[str, Any]) -> None:
    v = os.environ.get("RAG_SEARCH_IDLE_SECONDS")
    if v not in (None, ""):
        try:
            cfg["search"]["idle_exit_seconds"] = int(v)
        except ValueError:
            pass
    if os.environ.get("RAG_SEARCH_PREWARM") not in (None, ""):
        cfg["search"]["prewarm"] = env_flag("RAG_SEARCH_PREWARM", True)
    v = os.environ.get("RAG_SEARCH_JOBS")
    if v not in (None, ""):
        try:
            cfg["indexer"]["jobs"] = int(v)
        except ValueError:
            pass


def load_config(paths: Paths) -> tuple[dict[str, Any], str]:
    """Return (config, error).  A broken file falls back to defaults and reports why."""
    cfg = copy.deepcopy(DEFAULTS)
    err = ""
    f = paths.config_file
    if f.exists():
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                raise ValueError("top level must be an object")
            cfg = _merge(cfg, data)
        except (OSError, ValueError) as exc:
            err = f"{f}: {exc} (using defaults)"
    _env_overrides(cfg)
    return cfg, err


class ConfigStore:
    """Cached config that re-reads the file when its mtime changes."""

    def __init__(self, paths: Paths):
        self.paths = paths
        self._sig: tuple | None = None
        self._cfg: dict[str, Any] = copy.deepcopy(DEFAULTS)
        self.error = ""

    def get(self) -> dict[str, Any]:
        try:
            st = self.paths.config_file.stat()
            sig: tuple | None = (st.st_mtime_ns, st.st_size)
        except OSError:
            sig = None
        if sig != self._sig or self._sig is None:
            self._cfg, self.error = load_config(self.paths)
            self._sig = sig
        return self._cfg


def effective_jobs(cfg: dict[str, Any]) -> int:
    n = int(cfg.get("indexer", {}).get("jobs", 0) or 0)
    if n > 0:
        return n
    try:
        ram = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError, AttributeError):
        return 1
    return 1 if ram <= 17 * 1024 ** 3 else 2


def update_config(paths: Paths, section: str, values: dict[str, Any]) -> Path:
    """Set keys of one section of ``config.json``, keeping everything else the file holds.

    A file that cannot be parsed is not overwritten (the error is raised instead)."""
    f = paths.config_file
    data: dict[str, Any] = {}
    if f.exists():
        data = json.loads(f.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"{f}: top level must be an object")
    current = data.get(section)
    data[section] = {**(current if isinstance(current, dict) else {}), **values}
    write_json_atomic(f, data)
    return f


def write_default_config(paths: Paths, overwrite: bool = False) -> Path:
    f = paths.config_file
    if f.exists() and not overwrite:
        return f
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(DEFAULTS, indent=2) + "\n", encoding="utf-8")
    return f
