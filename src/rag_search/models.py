"""Which embedding model and reranker rag-search uses: catalogue, selection, fit, download state.

Stdlib only (the dashboard server and the CLI import it without loading torch).

* The **catalogue** is a short list of models this build knows how to run, with size, licence and
  memory figures.  Any other Hugging Face id can be chosen too (it is loaded as a plain
  sentence-transformers bi-encoder / cross-encoder and checked by a smoke test first).
* The **selection** is stored in ``config.json`` (``"models": {"embedding": ..., "reranker": ...}``)
  so every process (daemons, indexer workers, the CLI) agrees; ``$RAG_SEARCH_MODEL`` and
  ``$RAG_SEARCH_RERANK_MODEL`` still override it.  See ``paths.model_name()``.
* **Fit** compares the memory a pair of models needs with this machine's RAM, or with
  ``models.memory_limit_gb`` when you set one.  These are estimates, not measurements.
* **Cache state** says which models are already on disk (Hugging Face cache), without importing
  huggingface_hub.

Changing the *embedding* model changes every stored vector, so all documents are embedded again
(conversion is reused); changing the *reranker* only changes how the top hits are ordered.
"""

from __future__ import annotations

import importlib.metadata as md
import os
import platform
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .paths import (DEFAULT_MODEL, DEFAULT_RERANK_MODEL, META_FILE, ALL_DIR, Paths, configured_model,
                    env_flag, read_json)

EMBEDDING, RERANKER = "embedding", "reranker"
KINDS = (EMBEDDING, RERANKER)
ENV_VARS = {EMBEDDING: "RAG_SEARCH_MODEL", RERANKER: "RAG_SEARCH_RERANK_MODEL"}
DEFAULTS = {EMBEDDING: DEFAULT_MODEL, RERANKER: DEFAULT_RERANK_MODEL}

# Repo files we never need (saves ~2-4 GB of downloads).
HF_IGNORE = ["*.onnx", "*.onnx_data", "onnx/*", "openvino/*", "*.msgpack", "*.h5", "*.ot",
             "*.gguf", "imgs/*", "long.jpg"]

# backends (see core/embedding.py)
ST_EMBEDDER = "sentence-transformers"     # bi-encoder loaded with SentenceTransformer
CROSS_ENCODER = "cross-encoder"           # reranker loaded with sentence_transformers.CrossEncoder
QWEN3_RERANKER = "qwen3-reranker"         # causal LM that answers "yes"/"no" (Qwen3-Reranker)

QWEN3_QUERY_PREFIX = ("Instruct: Given a web search query, retrieve relevant passages that "
                      "answer the query\nQuery:")

OVERHEAD_GB = 1.5          # python + torch + the daemon itself, on top of the weights
RAM_SHARE = 0.6            # the part of the installed RAM the models may use when no limit is set


@dataclass(frozen=True)
class ModelSpec:
    id: str                       # Hugging Face repository id
    kind: str                     # embedding | reranker
    label: str
    params_m: int                 # parameters, millions
    mem_gb: float                 # weights in half precision (fp16); twice that on the CPU (fp32)
    license: str
    languages: str
    note: str
    dim: int = 0                  # embedding size (embedding models)
    backend: str = ""
    query_prefix: str = ""        # put in front of the search query (not of stored passages)
    requires: tuple = ()          # (("transformers", "4.51"), ...): minimum library versions
    tier: str = ""                # default | light | quality
    custom: bool = False          # not in the catalogue: chosen by id
    style: str = ""               # reader / repair models: how they are prompted (instruct | paddleocr)


CATALOG: tuple[ModelSpec, ...] = (
    ModelSpec("BAAI/bge-m3", EMBEDDING, "bge-m3", 568, 1.14, "MIT", "100+ languages",
              "The default. 1024 dimensions, 8k-token inputs; strong all-rounder for its size.",
              dim=1024, backend=ST_EMBEDDER, tier="default"),
    ModelSpec("Qwen/Qwen3-Embedding-0.6B", EMBEDDING, "Qwen3-Embedding 0.6B", 596, 1.19,
              "Apache-2.0", "100+ languages",
              "Same size class as bge-m3; instruction-aware queries. Worth trying on English "
              "technical text.", dim=1024, backend=ST_EMBEDDER, query_prefix=QWEN3_QUERY_PREFIX,
              requires=(("transformers", "4.51"),)),
    ModelSpec("Qwen/Qwen3-Embedding-4B", EMBEDDING, "Qwen3-Embedding 4B", 4020, 8.05,
              "Apache-2.0", "100+ languages",
              "Higher quality, about 7x slower to embed than 0.6B; 2560 dimensions (larger index).",
              dim=2560, backend=ST_EMBEDDER, query_prefix=QWEN3_QUERY_PREFIX, tier="quality",
              requires=(("transformers", "4.51"),)),
    ModelSpec("Qwen/Qwen3-Embedding-8B", EMBEDDING, "Qwen3-Embedding 8B", 7570, 15.1,
              "Apache-2.0", "100+ languages",
              "Top of the family; needs a large machine. 4096 dimensions.",
              dim=4096, backend=ST_EMBEDDER, query_prefix=QWEN3_QUERY_PREFIX, tier="quality",
              requires=(("transformers", "4.51"),)),

    ModelSpec("BAAI/bge-reranker-v2-m3", RERANKER, "bge-reranker-v2-m3", 568, 1.14, "MIT",
              "100+ languages", "The default cross-encoder.", backend=CROSS_ENCODER,
              tier="default"),
    ModelSpec("BAAI/bge-reranker-base", RERANKER, "bge-reranker-base", 278, 0.56, "MIT",
              "English, Chinese", "Half the size of v2-m3; a little weaker.",
              backend=CROSS_ENCODER, tier="light"),
    ModelSpec("cross-encoder/ms-marco-MiniLM-L-12-v2", RERANKER, "ms-marco MiniLM-L12", 33, 0.07,
              "Apache-2.0", "English", "Tiny and very fast; clearly weaker than the others.",
              backend=CROSS_ENCODER, tier="light"),
    ModelSpec("mixedbread-ai/mxbai-rerank-base-v2", RERANKER, "mxbai-rerank base v2", 500, 1.0,
              "Apache-2.0", "multilingual",
              "Reinforcement-learning trained reranker; the vendor reports it ahead of "
              "bge-reranker-v2-m3. Needs a recent sentence-transformers (the test says if not).",
              backend=CROSS_ENCODER),
    ModelSpec("mixedbread-ai/mxbai-rerank-large-v2", RERANKER, "mxbai-rerank large v2", 1500, 3.1,
              "Apache-2.0", "multilingual",
              "Larger sibling of the base model; about 3 GB of weights. Needs a recent sentence-transformers.", backend=CROSS_ENCODER,
              tier="quality"),
    ModelSpec("Qwen/Qwen3-Reranker-0.6B", RERANKER, "Qwen3-Reranker 0.6B", 596, 1.19,
              "Apache-2.0", "100+ languages", "Judges relevance with a yes/no answer.",
              backend=QWEN3_RERANKER, requires=(("transformers", "4.51"),)),
    ModelSpec("Qwen/Qwen3-Reranker-4B", RERANKER, "Qwen3-Reranker 4B", 4020, 8.05, "Apache-2.0",
              "100+ languages", "Higher quality; roughly ten times slower per query than 0.6B.",
              backend=QWEN3_RERANKER, tier="quality", requires=(("transformers", "4.51"),)),
)

# name -> (embedding, reranker)
PRESETS: dict[str, tuple[str, str]] = {
    "default": ("BAAI/bge-m3", "BAAI/bge-reranker-v2-m3"),
    "qwen3-small": ("Qwen/Qwen3-Embedding-0.6B", "Qwen/Qwen3-Reranker-0.6B"),
    "qwen3-large": ("Qwen/Qwen3-Embedding-4B", "Qwen/Qwen3-Reranker-4B"),
}

# ── document readers (vision-language models; run in a child process, see core/conversion/vlm.py) ──
# Not part of KINDS: choosing one never re-embeds anything (conversion is redone only for pages
# that go through the reader, and the page cache keys on the model id).
READER, REPAIR = "reader", "repair"
VLM_KINDS = (READER, REPAIR)
VLM_ENV = {READER: "RAG_SEARCH_VLM_MODEL", REPAIR: "RAG_SEARCH_REPAIR_MODEL"}
MLX_VLM = "mlx-vlm"
VLM_OVERHEAD_GB = 2.0      # image tokens, KV cache and the child process, on top of the weights

VLM_CATALOG: tuple[ModelSpec, ...] = (
    ModelSpec("mlx-community/Qwen3-VL-4B-Instruct-4bit", READER, "Qwen3-VL 4B (4-bit)", 4000, 3.1,
              "Apache-2.0", "multilingual",
              "General vision-language model that follows the page prompt (Markdown, tables as HTML). "
              "The default reader and the model used to re-read suspect table cells. Apple Silicon only.",
              backend=MLX_VLM, tier="default", style="instruct"),
    ModelSpec("mlx-community/Qwen3-VL-8B-Instruct-4bit", READER, "Qwen3-VL 8B (4-bit)", 8000, 5.8,
              "Apache-2.0", "multilingual",
              "The larger sibling of the default: slower (about twice as long per page) but better at small print "
              "and non-Latin scripts such as Devanagari (Hindi, Marathi). Same prompt and output as the default. "
              "Apple Silicon only.",
              backend=MLX_VLM, tier="quality", style="instruct"),
    ModelSpec("mlx-community/Qwen3-VL-8B-Instruct-8bit", READER, "Qwen3-VL 8B (8-bit)", 8000, 9.8,
              "Apache-2.0", "multilingual",
              "The 8B model with less quantisation loss: the most accurate reader offered here, and the slowest "
              "and largest. Apple Silicon only.",
              backend=MLX_VLM, tier="quality", style="instruct"),
    ModelSpec("mlx-community/PaddleOCR-VL-1.5-8bit", READER, "PaddleOCR-VL 1.5 (8-bit)", 900, 1.1,
              "Apache-2.0", "multilingual",
              "Small OCR-specialised model (0.9B). Whole pages come back as plain text (it is built to "
              "read one layout region at a time), so try it against the default on your own pages "
              "with `rag-search bench`. Apple Silicon only.",
              backend=MLX_VLM, tier="light", style="paddleocr"),
    ModelSpec("mlx-community/PaddleOCR-VL-1.5-bf16", READER, "PaddleOCR-VL 1.5 (bf16)", 900, 1.9,
              "Apache-2.0", "multilingual",
              "The same model without quantisation: a little more accurate, about twice the size.",
              backend=MLX_VLM, tier="quality", style="paddleocr"),
    ModelSpec("mlx-community/Qwen3-VL-4B-Instruct-4bit", REPAIR, "Qwen3-VL 4B (4-bit)", 4000, 3.1,
              "Apache-2.0", "multilingual",
              "Reads a cropped table cell or a row; a result is used only when the table's arithmetic "
              "agrees and a second, independent read says the same.",
              backend=MLX_VLM, tier="default", style="instruct"),
)
VLM_DEFAULTS = {READER: "mlx-community/Qwen3-VL-4B-Instruct-4bit",
                REPAIR: "mlx-community/Qwen3-VL-4B-Instruct-4bit"}

_ID_RE = re.compile(r"^[A-Za-z0-9][\w.\-]*/[A-Za-z0-9][\w.\-]*$")


class ModelError(ValueError):
    """A model id, kind or preset that cannot be used (the message says why)."""


def by_kind(kind: str) -> list[ModelSpec]:
    return [m for m in CATALOG if m.kind == kind]


def find(model_id: str) -> ModelSpec | None:
    for m in CATALOG:
        if m.id == model_id:
            return m
    return None


def check_kind(kind: str) -> str:
    if kind not in KINDS:
        raise ModelError(f"unknown kind {kind!r}; choose embedding or reranker")
    return kind


def spec_for(kind: str, model_id: str) -> ModelSpec:
    """The catalogue entry, or a generic entry for any other Hugging Face id.

    A custom embedding model is loaded as a plain sentence-transformers model (no query prefix)
    and a custom reranker as a cross-encoder; the smoke test says whether that works."""
    check_kind(kind)
    known = find(model_id)
    if known is not None:
        if known.kind != kind:
            raise ModelError(f"{model_id} is a {known.kind} model, not a {kind} model")
        return known
    if not _ID_RE.match(model_id or ""):
        raise ModelError(f"{model_id!r} is not a Hugging Face model id (expected ORG/NAME)")
    return _generic(kind, model_id)


def _generic(kind: str, model_id: str) -> ModelSpec:
    return ModelSpec(model_id, kind, model_id, 0, 0.0, "see the model card", "?",
                     "Not in the catalogue: loaded as a standard "
                     + ("sentence-transformers model." if kind == EMBEDDING else "cross-encoder."),
                     dim=0, backend=ST_EMBEDDER if kind == EMBEDDING else CROSS_ENCODER,
                     custom=True)


def _safe_spec(kind: str, model_id: str) -> ModelSpec:
    """Like spec_for, but never raises (a bad id in config.json must not break the display)."""
    try:
        return spec_for(kind, model_id)
    except ModelError:
        return find(model_id) or _generic(kind, model_id)


def resolve_preset(name: str) -> tuple[str, str]:
    if name not in PRESETS:
        raise ModelError(f"unknown preset {name!r}; choose from {', '.join(PRESETS)}")
    return PRESETS[name]


# ── selection ────────────────────────────────────────────────────────────────

def selection(kind: str) -> tuple[str, str]:
    """(model id, where it comes from: environment | config | default)."""
    check_kind(kind)
    env = os.environ.get(ENV_VARS[kind])
    if env:
        return env, "environment"
    cfg = configured_model(kind)
    if cfg:
        return cfg, "config"
    return DEFAULTS[kind], "default"


def set_selection(paths: Paths, kind: str, model_id: str) -> None:
    """Write the choice to config.json (other settings there are kept)."""
    from .config import update_config

    spec_for(kind, model_id)                 # validates
    update_config(paths, "models", {kind: model_id})


def vlm_find(kind: str, model_id: str) -> ModelSpec | None:
    return next((m for m in VLM_CATALOG if m.kind == kind and m.id == model_id), None)


def vlm_selection(kind: str) -> tuple[str, str]:
    """(model id, environment | config | default) of the document reader or the repair model."""
    if kind not in VLM_KINDS:
        raise ModelError(f"unknown kind {kind!r}; choose reader or repair")
    env = os.environ.get(VLM_ENV[kind])
    if env:
        return env, "environment"
    cfg = configured_model(kind)
    if cfg:
        return cfg, "config"
    return VLM_DEFAULTS[kind], "default"


def set_vlm_selection(paths: Paths, kind: str, model_id: str) -> None:
    """Choose the reader / repair model (any Hugging Face id of an MLX vision model is accepted;
    one outside the catalogue is prompted like a general instruct model)."""
    from .config import update_config

    if kind not in VLM_KINDS:
        raise ModelError(f"unknown kind {kind!r}; choose reader or repair")
    if not vlm_find(kind, model_id) and not _ID_RE.match(model_id or ""):
        raise ModelError(f"{model_id!r} is not a Hugging Face model id (expected ORG/NAME)")
    update_config(paths, "models", {kind: model_id})


def reader_choice() -> tuple[str, str, float]:
    """(model id, prompt style, GB the reader process needs) for the document reader."""
    mid = vlm_selection(READER)[0]
    spec = vlm_find(READER, mid)
    if spec:
        return mid, spec.style, spec.mem_gb + VLM_OVERHEAD_GB
    return mid, "instruct", 4.0 + VLM_OVERHEAD_GB            # unknown size: assume a 4B 4-bit model


def repair_choice() -> tuple[str, str, float]:
    """(model id, prompt style, GB needed) of the model that re-reads suspect cells and pages."""
    mid = vlm_selection(REPAIR)[0]
    spec = vlm_find(REPAIR, mid) or vlm_find(READER, mid)
    if spec:
        return mid, spec.style, spec.mem_gb + VLM_OVERHEAD_GB
    return mid, "instruct", 4.0 + VLM_OVERHEAD_GB


# The optional Apple-only runtime of the document reader: why it is not part of the base install.
RUNTIME_EXTRA = "mac-vlm"
RUNTIME_PACKAGES = (
    # module, pip name, what it is for, whether the document reader cannot work without it
    ("mlx_vlm", "mlx-vlm", "runs the document reader and the repair model on the Apple GPU (MLX)", True),
    ("ocrmac", "ocrmac", "Apple Vision: the independent second reader that confirms a repaired table cell", False),
    ("pillow_heif", "pillow-heif", "opens iPhone photos (.heic) as image files", False),
)
_RUNTIME_FALLBACK = ("mlx-vlm>=0.3.4", "pillow-heif>=0.18", "ocrmac>=1.0")


def runtime_requirements() -> list[str]:
    """The pip requirements of the ``mac-vlm`` extra, read from this package's own metadata (so the
    list cannot drift from pyproject.toml); a fixed list when the metadata is not available."""
    try:
        reqs = md.requires("rag-search") or []
    except md.PackageNotFoundError:
        reqs = []
    out = []
    for r in reqs:
        if re.search(r"extra\s*==\s*['\"]" + re.escape(RUNTIME_EXTRA) + r"['\"]", r):
            out.append(r.split(";")[0].strip())
    return out or list(_RUNTIME_FALLBACK)


def find_uv() -> str:
    """The ``uv`` executable, or "".  rag-search is installed with ``uv tool install``, whose environment
    has no pip -- but a daemon started by launchd has a short PATH that does not contain uv's folder
    (``~/.local/bin`` for the official installer, ``~/.cargo/bin``), so ``which`` alone is not enough."""
    import shutil

    found = shutil.which("uv")
    if found:
        return found
    home = Path.home()
    for cand in (home / ".local" / "bin" / "uv", home / ".cargo" / "bin" / "uv",
                 Path("/opt/homebrew/bin/uv"), Path("/usr/local/bin/uv")):
        if cand.is_file() and os.access(cand, os.X_OK):
            return str(cand)
    return ""


def runtime_state() -> dict[str, Any]:
    """Which parts of the document-reader runtime are installed in the environment rag-search runs in."""
    import sys

    apple = _apple_silicon()
    pk = [{"module": m, "package": pkg, "what": what, "required": req, "installed": _installed(m)}
          for m, pkg, what, req in RUNTIME_PACKAGES]
    return {"apple_silicon": apple, "packages": pk,
            "complete": all(x["installed"] for x in pk),
            "ready": next(x["installed"] for x in pk if x["module"] == "mlx_vlm"),
            "installer": "uv" if find_uv() else "pip", "python": sys.executable,
            "command": "rag-search models runtime install", "extra": RUNTIME_EXTRA,
            "requirements": runtime_requirements()}


def vlm_state(paths: Paths) -> dict[str, Any]:
    """The Models tab's *document readers* section: per kind the chosen model and the catalogue
    with download state and whether this machine can run it; a readiness checklist; and what
    will actually read scanned pages right now."""
    from .core.conversion import vlm

    free = vlm.available_memory_gb()
    rt = runtime_state()
    mode = vlm.mode()
    out: dict[str, Any] = {"mode": mode, "backend": vlm.backend_spec(), "free_gb": free,
                           "mlx_vlm": rt["ready"], "apple_silicon": rt["apple_silicon"], "runtime": rt}
    blocker = ("off" if mode == "off" else "platform" if not rt["apple_silicon"]
               else "runtime" if not rt["ready"] else "")
    for kind in VLM_KINDS:
        sel, source = vlm_selection(kind)
        listed = [m for m in VLM_CATALOG if m.kind == kind]
        if not any(m.id == sel for m in listed):
            listed.append(ModelSpec(sel, kind, sel, 0, 4.0, "see the model card", "?",
                                    "Not in the catalogue: prompted like a general instruct model.",
                                    backend=MLX_VLM, custom=True, style="instruct"))
        rows = []
        for m in listed:
            cs = cache_state(m.id)
            need = m.mem_gb + VLM_OVERHEAD_GB
            row = {k: getattr(m, k) for k in _PUBLIC}
            row.update(active=m.id == sel, cached=cs["cached"], partial=cs["partial"],
                       downloaded_bytes=cs["bytes"], why_not=cs.get("why", ""), need_gb=round(need, 1),
                       fit=("unknown" if free is None else "ok" if free >= need else "tight"))
            row["state"] = _selected_state(row["active"], cs["cached"], cs["partial"], blocker)
            rows.append(row)
        out[kind] = {"active": sel, "source": source, "models": rows,
                     "state": next(r["state"] for r in rows if r["active"])}
    out["checks"] = _vlm_checks(out, rt, mode, free)
    out["reading"] = _reading(out)
    return out


def _vlm_checks(v: dict[str, Any], rt: dict[str, Any], mode: str, free: float | None) -> list[dict[str, Any]]:
    """The checklist: what the document reader needs, whether each part is there, and the fix."""
    def row(kind: str) -> dict[str, Any]:
        return next(r for r in v[kind]["models"] if r["active"])

    rd, rp = row(READER), row(REPAIR)
    pk = {x["module"]: x for x in rt["packages"]}
    checks = [
        {"id": "mode", "label": "Switched on", "ok": mode != "off", "required": True,
         "detail": "RAG_SEARCH_VLM=off: the reader is never used" if mode == "off" else "automatic: used for scanned pages, pictures and image files",
         "fix": "unset RAG_SEARCH_VLM" if mode == "off" else ""},
        {"id": "platform", "label": "Apple Silicon Mac", "ok": rt["apple_silicon"], "required": True,
         "detail": "MLX runs the model on the Apple GPU" if rt["apple_silicon"] else "this computer cannot run the reader; docling OCR is used",
         "fix": ""},
        {"id": "runtime", "label": "Runtime: mlx-vlm", "ok": pk["mlx_vlm"]["installed"], "required": True,
         "detail": pk["mlx_vlm"]["what"], "fix": "install_runtime" if rt["apple_silicon"] and not pk["mlx_vlm"]["installed"] else ""},
        {"id": "weights_reader", "label": "Reader model downloaded", "ok": rd["cached"], "required": True,
         "detail": rd["id"] + (" (partly downloaded)" if rd["partial"] else
                                f" (files are on disk but the copy is not usable: {rd['why_not']})" if rd["downloaded_bytes"] and not rd["cached"] else ""),
         "fix": "" if rd["cached"] else "download:" + rd["id"]},
        {"id": "weights_repair", "label": "Repair model downloaded", "ok": rp["cached"], "required": False,
         "detail": rp["id"] + ("" if rp["id"] != rd["id"] else " (the same model as the reader)"),
         "fix": "" if rp["cached"] else "download:" + rp["id"]},
        {"id": "ocrmac", "label": "Second reader: ocrmac", "ok": pk["ocrmac"]["installed"], "required": False,
         "detail": pk["ocrmac"]["what"] + (" (without it a repaired cell cannot be confirmed, so the cell is only flagged)" if not pk["ocrmac"]["installed"] else ""),
         "fix": "install_runtime" if rt["apple_silicon"] and not pk["ocrmac"]["installed"] else ""},
        {"id": "heif", "label": "iPhone photos (.heic)", "ok": pk["pillow_heif"]["installed"], "required": False,
         "detail": pk["pillow_heif"]["what"], "fix": "install_runtime" if rt["apple_silicon"] and not pk["pillow_heif"]["installed"] else ""},
        {"id": "memory", "label": "Memory free now", "ok": free is None or free >= rd["need_gb"], "required": False,
         "detail": (f"{free:.1f} GB free, the reader needs about {rd['need_gb']} GB while it runs" if free is not None
                    else "unknown"), "fix": ""},
    ]
    return checks


def _reading(v: dict[str, Any]) -> dict[str, Any]:
    """What reads scanned pages right now: the document reader, or docling OCR and why."""
    blocking = [c for c in v["checks"] if c["required"] and not c["ok"]]
    if not blocking:
        return {"by": "reader", "model": v[READER]["active"],
                "text": f"Scanned pages, pictures and photos are read by the document reader ({v[READER]['active']})."}
    why = "; ".join(c["label"].lower() + ": " + (c["detail"] if c["id"] != "weights_reader" else "not downloaded")
                    for c in blocking)
    return {"by": "docling", "model": "", "blocked_by": [c["id"] for c in blocking],
            "text": "Scanned pages are read by docling OCR only, because the document reader cannot run yet (" + why + "). "
                    "Photographed pages and pictures usually need the document reader."}


def _installed(module: str) -> bool:
    import importlib
    import importlib.util

    importlib.invalidate_caches()               # something may have been installed since the last look
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def _apple_silicon() -> bool:
    return platform.system() == "Darwin" and platform.machine() == "arm64"


def set_memory_limit(paths: Paths, gb: float) -> None:
    from .config import update_config

    if gb < 0:
        raise ModelError("the memory limit cannot be negative (0 = no limit of your own)")
    update_config(paths, "models", {"memory_limit_gb": gb})


def memory_limit_gb(paths: Paths) -> float:
    from .config import load_config

    try:
        return float(load_config(paths)[0].get("models", {}).get("memory_limit_gb") or 0)
    except (TypeError, ValueError):
        return 0.0


def serving_model(paths: Paths) -> str:
    """The embedding model the live (published) index was built with, "" if nothing is published."""
    from .catalog import live_catalog

    return str(live_catalog(paths).get("model", "") or "")


def brief(paths: Paths) -> dict[str, Any]:
    """The few facts the dashboard header and pipeline diagrams need (cheap)."""
    return {"embedding": selection(EMBEDDING)[0], "reranker": selection(RERANKER)[0],
            "serving": serving_model(paths)}


# ── this machine ─────────────────────────────────────────────────────────────

def machine_info() -> dict[str, Any]:
    try:
        ram = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError, AttributeError):
        ram = 0
    system, arch = platform.system(), platform.machine()
    forced = os.environ.get("RAG_SEARCH_DEVICE", "")
    if forced:
        device = forced
    elif system == "Darwin" and arch == "arm64":
        device = "mps"                     # Apple GPU
    else:
        device = "cpu"                     # Intel Macs, plain Linux (a CUDA card is not assumed)
    return {"ram_gb": round(ram / 1024 ** 3, 1), "system": system, "arch": arch, "device": device,
            "weight_bytes": 2 if device in ("mps", "cuda") else 4}


def budget_gb(machine: dict[str, Any], limit_gb: float = 0.0) -> float:
    """Memory the models may use: your own limit, else 60% of the installed RAM."""
    if limit_gb and limit_gb > 0:
        return float(limit_gb)
    return round(machine.get("ram_gb", 0) * RAM_SHARE, 1)


def estimate_gb(emb: ModelSpec, rer: ModelSpec, machine: dict[str, Any], chunks: int = 0) -> float:
    """Rough resident memory of the search daemon for this pair: weights + overhead + index."""
    scale = machine.get("weight_bytes", 2) / 2          # 1 = half precision, 2 = full precision
    weights = emb.mem_gb * scale
    if env_flag("RAG_SEARCH_RERANK", True):
        # a cross-encoder is loaded in full precision on every device
        weights += rer.mem_gb * (2.0 if rer.backend == CROSS_ENCODER else scale)
    index = chunks * ((emb.dim or 1024) * 4 + 9000) / 1e9
    return round(weights + OVERHEAD_GB + index, 1)


def fit_level(estimate: float, budget: float) -> str:
    """ok | tight | too_large  (unknown budget counts as ok)."""
    if budget <= 0:
        return "ok"
    if estimate <= 0.8 * budget:
        return "ok"
    if estimate <= budget:
        return "tight"
    return "too_large"


# ── libraries ────────────────────────────────────────────────────────────────

def parse_version(text: str) -> tuple[int, ...]:
    return tuple(int(n) for n in re.findall(r"\d+", text.split("+")[0])[:3])


def missing_requirements(spec: ModelSpec) -> list[str]:
    """Libraries too old for this model (empty = fine, or not checkable)."""
    out = []
    for pkg, minimum in spec.requires:
        try:
            have = md.version(pkg)
        except md.PackageNotFoundError:
            out.append(f"{pkg} {minimum} or newer is needed (not installed)")
            continue
        if parse_version(have) < parse_version(minimum):
            out.append(f"{pkg} {minimum} or newer is needed ({have} is installed)")
    return out


# ── what is on disk ──────────────────────────────────────────────────────────

def hub_cache() -> Path:
    env = os.environ
    for var in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE"):
        if env.get(var):
            return Path(env[var]).expanduser()
    if env.get("HF_HOME"):
        return Path(env["HF_HOME"]).expanduser() / "hub"
    base = Path(env["XDG_CACHE_HOME"]).expanduser() if env.get("XDG_CACHE_HOME") \
        else Path.home() / ".cache"
    return base / "huggingface" / "hub"


def repo_dir(model_id: str) -> Path:
    return hub_cache() / ("models--" + model_id.replace("/", "--"))


def _snapshot_problem(snap: Path) -> str:
    """Why *snap* is not a complete copy of a model, or "" when it is.

    A sharded model lists its shards in ``*.safetensors.index.json``; every listed shard must be on
    disk.  An index that names files the snapshot does not hold *while the snapshot holds weight
    files the index does not name* is a stale leftover in the repository (mlx-community's
    Qwen3-VL-4B-Instruct-4bit ships a single ``model.safetensors`` next to the index of the
    unquantised two-shard model): the loaders ignore it, so it is ignored here as well.  A
    genuinely interrupted sharded download has the first shard but not the second, and no
    unlisted weight file, so it is still reported."""
    weights = [p for pat in ("*.safetensors", "*.bin") for p in snap.rglob(pat)
               if "training_args" not in p.name and p.exists()]
    if not weights:
        return "no weight file"
    if not ((snap / "config.json").exists() or (snap / "modules.json").exists()):
        return "no config.json"
    have = {p.relative_to(snap).as_posix() for p in weights}
    for index_name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        idx = snap / index_name
        if not idx.exists():
            continue
        listed = set((read_json(idx).get("weight_map") or {}).values())
        missing = sorted(x for x in listed if not (snap / x).exists())
        if missing and not (have - listed):
            return f"shard {missing[0]} is missing" + (f" (and {len(missing) - 1} more)" if len(missing) > 1 else "")
    return ""


def _snapshot_complete(snap: Path) -> bool:
    return not _snapshot_problem(snap)


def cached_revision(model_id: str) -> str:
    """The commit the local cache's ``refs/main`` points at for *model_id* (the weights a load
    actually uses), or "" when unknown (not cached, a local folder, a custom backend).

    A Hugging Face model id names a repository, not fixed weights: the same id can hold
    different weights after an upstream update.  Recording this next to the id (index metadata,
    collection exports) is what makes "built with the same model" checkable."""
    try:
        ref = (repo_dir(model_id) / "refs" / "main").read_text(encoding="utf-8").strip()
    except OSError:
        return ""
    return ref if re.fullmatch(r"[0-9a-f]{7,64}", ref) else ""


def cache_state(model_id: str) -> dict[str, Any]:
    """{"cached": complete copy on disk, "partial": an interrupted download, "bytes": on disk,
    "why": what is missing when it is not complete}.

    A repo directory can hold more than one snapshot (an older complete download plus, say, a
    stray single-file fetch left by a narrower request) -- the newest by mtime is not necessarily
    the complete one, so every snapshot is checked, preferring whichever one ``refs/main`` points
    to when it is present."""
    d = repo_dir(model_id)
    size, partial = 0, False
    try:
        for f in (d / "blobs").iterdir():
            if f.is_file():
                size += f.stat().st_size
                partial = partial or f.name.endswith(".incomplete")
    except OSError:
        return {"cached": False, "partial": False, "bytes": 0, "why": "nothing downloaded yet"}
    snaps = []
    try:
        snaps = sorted((p for p in (d / "snapshots").iterdir() if p.is_dir()),
                       key=lambda p: p.stat().st_mtime, reverse=True)
    except OSError:
        pass
    try:
        main_ref = (d / "refs" / "main").read_text(encoding="utf-8").strip()
    except OSError:
        main_ref = ""
    if main_ref:
        snaps.sort(key=lambda p: p.name != main_ref)  # the main-ref snapshot first, order kept otherwise
    problems = [_snapshot_problem(snap) for snap in snaps]
    complete = not partial and any(not x for x in problems)
    why = ("" if complete else "a download was interrupted (an .incomplete file is left)" if partial
           else (problems[0] if problems else "no snapshot in the cache"))
    return {"cached": complete, "partial": partial and not complete, "bytes": size, "why": why}


# ── the documents in the workspace (for "how many must be embedded again") ───────

_WS: dict[str, Any] = {"at": -1e9, "home": "", "data": {}}


def workspace_models(paths: Paths, max_age: float = 15.0) -> dict[str, Any]:
    """Documents in the workspace per embedding model: {"documents", "chunks", "by_model",
    "embed_s"}.  Cached for a few seconds (it reads one small file per document)."""
    now = time.monotonic()
    if _WS["home"] == str(paths.home) and now - _WS["at"] < max_age:
        return _WS["data"]
    by_model: dict[str, int] = {}
    embed_s: dict[str, float] = {}
    chunks: dict[str, int] = {}
    docs = 0
    if paths.index.is_dir():
        imported: dict[str, bool] = {}
        for meta_file in paths.index.rglob(META_FILE):
            parts = meta_file.relative_to(paths.index).parts
            if ALL_DIR in parts:
                continue
            # an imported collection cannot be re-embedded here: not part of any switch estimate
            if parts[0] not in imported:
                imported[parts[0]] = (paths.index / parts[0] / "collection.origin.json").exists()
            if imported[parts[0]]:
                continue
            meta = read_json(meta_file)
            if not meta:
                continue
            m = str(meta.get("model", ""))
            docs += 1
            by_model[m] = by_model.get(m, 0) + 1
            embed_s[m] = embed_s.get(m, 0.0) + float(meta.get("embed_s", 0) or 0)
            chunks[m] = chunks.get(m, 0) + int(meta.get("nodes", 0) or 0)
    data = {"documents": docs, "chunks": sum(chunks.values()), "by_model": by_model,
            "embed_s": {k: round(v, 1) for k, v in embed_s.items()},
            "chunks_by_model": chunks}
    _WS.update(at=now, home=str(paths.home), data=data)
    return data


def reindex_estimate(paths: Paths, new_id: str) -> dict[str, Any]:
    """What switching the embedding model to *new_id* costs: documents and a time estimate.

    The estimate scales the time the documents took to embed last time by the size ratio of the
    two models; it is only a guide (None when there is nothing to base it on)."""
    ws = workspace_models(paths, max_age=0)
    todo = {m: n for m, n in ws["by_model"].items() if m != new_id}
    docs = sum(todo.values())
    chunks = sum(ws["chunks_by_model"].get(m, 0) for m in todo)
    est = None
    new_spec = find(new_id)
    for m in todo:
        old = find(m)
        secs = ws["embed_s"].get(m, 0.0)
        if secs <= 0:
            continue
        ratio = (new_spec.params_m / old.params_m) if (new_spec and old and old.params_m) else 1.0
        est = (est or 0.0) + secs * ratio
    return {"documents": docs, "chunks": chunks, "total_documents": ws["documents"],
            "estimated_s": round(est) if est is not None else None}


# ── the whole picture, for the CLI and the dashboard ────────────────────────────

_PUBLIC = ("id", "kind", "label", "params_m", "mem_gb", "license", "languages", "note", "dim",
           "backend", "tier", "custom")


def _selected_state(selected: bool, cached: bool, partial: bool, blocker: str = "") -> str:
    """One word for a model row: ``available`` (not chosen), ``in_use`` (chosen and everything it needs is
    there), or ``selected_<why>`` (chosen but something is missing).  "In use" is never shown for a
    model that is not on disk."""
    if not selected:
        return "available"
    if blocker:
        return "selected_" + blocker                    # off | platform | runtime
    if not cached:
        return "selected_partial" if partial else "selected_download"
    return "in_use"


def _row(spec: ModelSpec, sel_id: str, serving: str, other: ModelSpec, machine: dict[str, Any],
         budget: float, chunks: int) -> dict[str, Any]:
    emb, rer = (spec, other) if spec.kind == EMBEDDING else (other, spec)
    est = estimate_gb(emb, rer, machine, chunks)
    cs = cache_state(spec.id)
    row = {k: getattr(spec, k) for k in _PUBLIC}
    row.update(active=spec.id == sel_id, cached=cs["cached"], partial=cs["partial"],
               downloaded_bytes=cs["bytes"], why_not=cs.get("why", ""), estimate_gb=est,
               fit="unknown" if spec.custom else fit_level(est, budget),
               missing=missing_requirements(spec), query_prefix=bool(spec.query_prefix))
    row["state"] = _selected_state(row["active"], cs["cached"], cs["partial"])
    if spec.kind == EMBEDDING:
        row["serving"] = spec.id == serving
    return row


def state(paths: Paths) -> dict[str, Any]:
    """Everything the `models` command and the Models tab show."""
    machine = machine_info()
    limit = memory_limit_gb(paths)
    budget = budget_gb(machine, limit)
    sel = {k: selection(k) for k in KINDS}
    serving = serving_model(paths)
    ws = workspace_models(paths)
    from .catalog import live_catalog

    chunks = sum(c.get("chunks", 0) for c in live_catalog(paths).get("collections", []))
    specs = {k: _safe_spec(k, sel[k][0]) for k in KINDS}
    out: dict[str, Any] = {"machine": machine, "memory_limit_gb": limit, "budget_gb": budget,
                           "presets": {n: {"embedding": e, "reranker": r}
                                       for n, (e, r) in PRESETS.items()},
                           "serving": serving, "workspace": {
                               "documents": ws["documents"], "by_model": ws["by_model"]}}
    for kind in KINDS:
        other = specs[RERANKER if kind == EMBEDDING else EMBEDDING]
        listed = list(by_kind(kind))
        if not any(m.id == sel[kind][0] for m in listed):        # a custom choice stays visible
            listed.append(specs[kind])
        rows = [_row(m, sel[kind][0], serving, other, machine, budget, chunks) for m in listed]
        out[kind] = {"active": sel[kind][0], "source": sel[kind][1], "models": rows}
    out["vlm"] = vlm_state(paths)
    active_emb = sel[EMBEDDING][0]
    out["reindex"] = {"needed": bool(serving and serving != active_emb) or
                      any(m != active_emb for m in ws["by_model"]),
                      "documents_on_other_model": sum(n for m, n in ws["by_model"].items()
                                                      if m != active_emb),
                      "documents": ws["documents"]}
    return out
