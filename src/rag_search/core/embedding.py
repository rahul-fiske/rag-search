"""Embedding and reranking backends (bge-m3 + bge-reranker-v2-m3 by default; see models.py).

Three backends:

* ``Embedder``      any sentence-transformers bi-encoder (bge-m3, Qwen3-Embedding, ...); an
                    optional instruction prefix is put in front of *queries* only.
* ``Reranker``      any sentence-transformers cross-encoder (bge-reranker, mxbai-rerank, ...).
* ``Qwen3Reranker`` the Qwen3-Reranker models: a causal LM asked "does the document answer the
                    query? yes/no", scored as P(yes).

All are lazy: torch / sentence-transformers are imported on first use, so the MCP front-end and
the CLI's cheap commands never pay for them.
"""

from __future__ import annotations

import contextlib
import gc
import importlib
import logging
import os
import re
import sys
import threading
from typing import Any, Callable, Iterator

import numpy as np

from .. import machine, models
from ..paths import env_int, model_name, rerank_model_name
from ..spec import EMBED_BATCH, EMBED_MAX_SEQ, RERANK_BATCH, RERANK_MAX_LEN

log = logging.getLogger("rag_search.embedding")

Progress = Callable[[int, int], None]


def prepare_environment() -> None:
    """Set torch-related env defaults; call before the first torch import."""
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")       # nothing about this installation leaves the computer
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


def _param_bytes(model) -> int | None:
    """Bytes taken by a torch model's weights (None when it cannot be told)."""
    if model is None:
        return None
    target = model if hasattr(model, "parameters") else getattr(model, "model", None)
    try:
        return int(sum(p.numel() * p.element_size() for p in target.parameters()))
    except Exception:  # noqa: BLE001 - never let a status query fail
        return None


def pick_device() -> str:
    prepare_environment()
    if machine.forced_device():
        return machine.forced_device()
    import torch

    return machine.probe_device(torch)


def release_memory() -> None:
    """Give memory of a model that was just dropped back (Python, then the GPU cache)."""
    gc.collect()
    torch = sys.modules.get("torch")
    if torch is None:
        return
    with contextlib.suppress(Exception):
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    with contextlib.suppress(Exception):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def torch_dtype(torch, device: str, default_half: bool = True):
    """Weights precision: $RAG_SEARCH_DTYPE (float16 | bfloat16 | float32), else half precision
    on the GPU when *default_half*, else None (the library's default, full precision)."""
    forced = os.environ.get("RAG_SEARCH_DTYPE", "").strip().lower()
    if forced:
        table = {"float16": torch.float16, "fp16": torch.float16, "half": torch.float16,
                 "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
                 "float32": torch.float32, "fp32": torch.float32, "float": torch.float32}
        if forced not in table:
            raise ValueError(f"RAG_SEARCH_DTYPE={forced!r}: use float16, bfloat16 or float32")
        return table[forced]
    return torch.float16 if default_half and device in ("mps", "cuda") else None


def _finite_or_raise(values: np.ndarray, what: str, name: str) -> None:
    if not np.isfinite(values).all():
        raise RuntimeError(f"{name} produced invalid numbers (NaN/inf) for {what}; the model may "
                           "not work in half precision on this device: try RAG_SEARCH_DTYPE=float32 "
                           "(or bfloat16)")


def _torch_version() -> tuple[int, int]:
    import torch

    nums = re.findall(r"\d+", torch.__version__.split("+")[0])
    return int(nums[0]), int(nums[1]) if len(nums) > 1 else 0


def allow_trusted_legacy_torch_load(model_name: str) -> bool:
    """Let transformers read `pytorch_model.bin` checkpoints with PyTorch < 2.6.

    transformers >= 4.50 refuses `torch.load` on old PyTorch because of CVE-2025-32434 (a crafted
    checkpoint can run code).  Intel Macs cannot go beyond PyTorch 2.2.2, and BAAI/bge-m3 ships
    only a .bin file, so indexing fails there.  The relaxation is applied only to the official
    BAAI/* repositories this package uses, or when RAG_SEARCH_ALLOW_UNSAFE_TORCH_LOAD=1.
    Returns True when the check was disabled.
    """
    try:
        if _torch_version() >= (2, 6):
            return False
    except Exception:  # noqa: BLE001 - no torch: nothing to relax
        return False
    if not (model_name.startswith("BAAI/") or os.environ.get("RAG_SEARCH_ALLOW_UNSAFE_TORCH_LOAD") == "1"):
        return False
    with contextlib.suppress(Exception):
        importlib.import_module("transformers.modeling_utils")
    patched = 0
    for name, mod in list(sys.modules.items()):
        if mod is not None and name.startswith("transformers") and hasattr(mod, "check_torch_load_is_safe"):
            current = mod.check_torch_load_is_safe
            if current is not _relaxed_check:
                _SAVED_CHECKS[name] = (mod, current)
            mod.check_torch_load_is_safe = _relaxed_check
            patched += 1
    if patched:
        log.warning("PyTorch < 2.6: allowing torch.load for trusted model %s (CVE-2025-32434 check "
                    "disabled while it loads)", model_name)
    return bool(patched)


def _relaxed_check(*_a: Any, **_k: Any) -> None:
    return None


_SAVED_CHECKS: dict[str, tuple[Any, Any]] = {}
_LOAD_LOCK = threading.RLock()


def restore_torch_load_check() -> None:
    """Put transformers' own torch.load safety check back (undo the relaxation) -- also in
    modules imported while it was relaxed, which may have copied the relaxed function."""
    originals = {name: orig for name, (_mod, orig) in _SAVED_CHECKS.items()}
    fallback = next(iter(originals.values()), None)
    for name, mod in list(sys.modules.items()):
        if mod is None or not name.startswith("transformers"):
            continue
        with contextlib.suppress(Exception):
            if getattr(mod, "check_torch_load_is_safe", None) is _relaxed_check:
                original = originals.get(name, fallback)
                if original is not None:
                    mod.check_torch_load_is_safe = original
    _SAVED_CHECKS.clear()


@contextlib.contextmanager
def trusted_load(model_name: str) -> Iterator[bool]:
    """The relaxation of ``allow_trusted_legacy_torch_load``, for exactly one model load: the
    check is restored afterwards, so a later load of some other (untrusted) model in the same
    long-lived process is checked again."""
    relaxed = allow_trusted_legacy_torch_load(model_name)
    try:
        yield relaxed
    finally:
        if relaxed:
            restore_torch_load_check()


@contextlib.contextmanager
def offline(on: bool) -> Iterator[None]:
    """While a model that is complete in the local cache is loaded, the Hugging Face libraries are told to stay
    offline.  ``local_files_only`` alone is not enough: transformers starts a background check for a converted copy of
    the weights on the Hub (requests naming the model) unless it is in offline mode.  Search and indexing are meant to
    use no network at all, so the mode is switched on for the load -- in the environment and in the libraries'
    already-imported settings -- and put back afterwards (downloads, asked for on the Models tab, need it off)."""
    if not on:
        yield
        return
    names = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")
    saved_env = {k: os.environ.get(k) for k in names}
    saved_attr: list[tuple[Any, str, Any]] = []
    for k in names:
        os.environ[k] = "1"
    for mod_name, attr in (("huggingface_hub.constants", "HF_HUB_OFFLINE"), ("transformers.utils.hub", "_is_offline_mode")):
        mod = sys.modules.get(mod_name)
        if mod is None and mod_name.startswith("huggingface_hub"):
            with contextlib.suppress(Exception):
                mod = importlib.import_module(mod_name)
        if mod is not None and hasattr(mod, attr):
            saved_attr.append((mod, attr, getattr(mod, attr)))
            with contextlib.suppress(Exception):
                setattr(mod, attr, True)
    try:
        yield
    finally:
        for mod, attr, value in saved_attr:
            with contextlib.suppress(Exception):
                setattr(mod, attr, value)
        for k, v in saved_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _load_model(name: str, factory: Callable[..., Any], **kw: Any) -> Any:
    """Load a Hugging Face model, offline first.

    A model whose complete copy is already in the local cache is loaded with
    ``local_files_only=True``: no round trip to huggingface.co, so a machine without (or with
    blocked) internet starts as fast as one with it.  If that fails (an older library without
    the option, a cache that is not quite complete) it is loaded the normal way; a failure to
    download a model that is not cached is explained in plain words."""
    cached = False
    with contextlib.suppress(Exception):
        cached = bool(models.cache_state(name)["cached"])
    # one load at a time per process: the torch.load relaxation is process-wide while it lasts,
    # so two overlapping loads must not see (or undo) each other's
    with _LOAD_LOCK, trusted_load(name):
        if cached:
            try:
                with offline(True):
                    return factory(local_files_only=True, **kw)
            except TypeError:
                pass                       # this library version has no local_files_only
            except Exception as exc:  # noqa: BLE001 - try the normal way below
                log.info("loading %s from the local cache failed (%s); trying the normal way",
                         name, exc)
        try:
            return factory(**kw)
        except TypeError:
            raise                          # an unsupported option, not a download problem: the caller may have a fallback
        except Exception as exc:
            if cached:
                raise
            from ..model_tasks import explain_download_error

            raise RuntimeError(explain_download_error(name, exc)) from exc


def model_revision(name: str) -> str:
    """The commit of *name* in the local Hugging Face cache (what a load uses), or ""."""
    with contextlib.suppress(Exception):
        return models.cached_revision(name)
    return ""


class Embedder:
    """Dense embeddings via sentence-transformers (bge-m3 ships an ST config)."""

    def __init__(self, name: str | None = None, batch_size: int | None = None,
                 max_seq_length: int | None = None, query_prefix: str | None = None):
        self.name = name or model_name()
        spec = models.find(self.name)
        # instruction models (Qwen3-Embedding) expect a prefix on the query, not on passages
        self.query_prefix = (spec.query_prefix if spec else "") if query_prefix is None \
            else query_prefix
        self.batch_size = batch_size or env_int("RAG_SEARCH_EMBED_BATCH", EMBED_BATCH)
        self.max_seq_length = max_seq_length or env_int("RAG_SEARCH_MAX_SEQ", EMBED_MAX_SEQ)
        self._model = None
        self.device = ""
        self.revision = ""

    def load(self):
        if self._model is None:
            prepare_environment()
            import torch
            from sentence_transformers import SentenceTransformer

            self.device = pick_device()
            kwargs = {}
            dtype = torch_dtype(torch, self.device)
            if dtype is not None:
                kwargs["model_kwargs"] = {"torch_dtype": dtype}
            log.info("Loading embedding model %s (device=%s, batch=%d)",
                     self.name, self.device, self.batch_size)
            model = _load_model(self.name, lambda **kw: SentenceTransformer(
                self.name, device=self.device, **kwargs, **kw))
            self.revision = model_revision(self.name)
            model.max_seq_length = self.max_seq_length
            self._model = model
        return self._model

    def memory_bytes(self) -> int | None:
        return _param_bytes(self._model)

    def encode(self, texts: list[str], progress: Progress | None = None) -> np.ndarray:
        model = self.load()
        out: list[np.ndarray] = []
        step = self.batch_size * 4
        for i in range(0, len(texts), step):
            part = texts[i:i + step]
            vecs = model.encode(part, batch_size=self.batch_size, normalize_embeddings=True,
                                convert_to_numpy=True, show_progress_bar=False)
            vecs = np.asarray(vecs, dtype=np.float32)
            _finite_or_raise(vecs, "some text", self.name)
            out.append(vecs)
            if progress:
                progress(min(i + step, len(texts)), len(texts))
        if not out:
            return np.zeros((0, 0), dtype=np.float32)
        return np.concatenate(out, axis=0)

    def encode_query(self, query: str) -> np.ndarray:
        return self.encode([self.query_prefix + query])[0]


class Reranker:
    """Cross-encoder reranker returning scores in [0, 1]."""

    def __init__(self, name: str | None = None, max_length: int | None = None,
                 batch_size: int | None = None):
        self.name = name or rerank_model_name()
        # the same two settings (spec.py: rerank_max_len, rerank_batch) Qwen3Reranker honours
        self.max_length = max_length or env_int("RAG_SEARCH_RERANK_MAX_LEN", RERANK_MAX_LEN)
        self.batch_size = batch_size or env_int("RAG_SEARCH_RERANK_BATCH", RERANK_BATCH)
        self._model = None
        self._needs_sigmoid = False
        self.device = ""

    def load(self):
        if self._model is None:
            prepare_environment()
            import torch
            from sentence_transformers import CrossEncoder

            self.device = pick_device()
            log.info("Loading reranker %s (device=%s, max_length=%d, batch=%d)", self.name,
                     self.device, self.max_length, self.batch_size)
            extra = {}
            dtype = torch_dtype(torch, self.device, default_half=False)   # full precision unless forced
            if dtype is not None:
                extra["model_kwargs"] = {"torch_dtype": dtype}
            try:
                self._model = _load_model(self.name, lambda **kw: CrossEncoder(
                    self.name, device=self.device, max_length=self.max_length,
                    trust_remote_code=True, activation_fn=torch.nn.Sigmoid(), **extra, **kw))
            except TypeError:  # older/newer sentence-transformers spelling
                self._model = _load_model(self.name, lambda **kw: CrossEncoder(
                    self.name, device=self.device, max_length=self.max_length,
                    trust_remote_code=True, **kw))
                self._needs_sigmoid = True
        return self._model

    def memory_bytes(self) -> int | None:
        return _param_bytes(self._model)

    def score(self, query: str, texts: list[str]) -> list[float]:
        if not texts:
            return []
        model = self.load()
        raw = model.predict([[query, t] for t in texts], batch_size=self.batch_size,
                            show_progress_bar=False)
        vals = np.asarray(raw, dtype=np.float64).reshape(-1)
        _finite_or_raise(vals, "the candidates", self.name)
        if self._needs_sigmoid and vals.size and (vals.min() < 0.0 or vals.max() > 1.0):
            vals = 1.0 / (1.0 + np.exp(-vals))  # raw logits -> [0, 1]
        return [float(v) for v in vals]


QWEN3_RERANK_PREFIX = ("<|im_start|>system\nJudge whether the Document meets the requirements based "
                       "on the Query and the Instruct provided. Note that the answer can only be "
                       "\"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n")
QWEN3_RERANK_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
QWEN3_RERANK_INSTRUCTION = "Given a web search query, retrieve relevant passages that answer the query"


class Qwen3Reranker:
    """Qwen3-Reranker: a causal language model asked whether the document answers the query.

    The prompt and the score follow the model card: the score is P("yes") from a softmax over the
    "no" / "yes" logits at the last position.  Only those two logits are computed (the last hidden
    state times two rows of the output matrix), so the 150k-word vocabulary costs no memory.
    """

    def __init__(self, name: str | None = None, max_length: int | None = None,
                 instruction: str = QWEN3_RERANK_INSTRUCTION, batch_size: int | None = None):
        self.name = name or rerank_model_name()
        self.max_length = max_length or env_int("RAG_SEARCH_RERANK_MAX_LEN", RERANK_MAX_LEN)
        self.instruction = instruction
        self.batch_size = batch_size or env_int("RAG_SEARCH_RERANK_BATCH", RERANK_BATCH)
        self._model = None
        self._tok = None
        self._ids: tuple[int, int] = (0, 0)             # ("no", "yes") token ids
        self._prefix: list[int] = []
        self._suffix: list[int] = []
        self.device = ""

    def load(self):
        if self._model is None:
            prepare_environment()
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self.device = pick_device()
            log.info("Loading reranker %s (device=%s)", self.name, self.device)
            tok = _load_model(self.name, lambda **k: AutoTokenizer.from_pretrained(
                self.name, padding_side="left", **k))
            dtype = torch_dtype(torch, self.device)
            kw = {"torch_dtype": dtype} if dtype is not None else {}
            model = _load_model(self.name, lambda **k: AutoModelForCausalLM.from_pretrained(
                self.name, **kw, **k)).to(self.device).eval()
            no, yes = tok.convert_tokens_to_ids("no"), tok.convert_tokens_to_ids("yes")
            if no is None or yes is None or no == yes or tok.unk_token_id in (no, yes):
                raise RuntimeError(f"{self.name}: the tokenizer has no 'yes'/'no' tokens; this is "
                                   "not a Qwen3-Reranker model")
            self._ids = (int(no), int(yes))
            self._prefix = tok.encode(QWEN3_RERANK_PREFIX, add_special_tokens=False)
            self._suffix = tok.encode(QWEN3_RERANK_SUFFIX, add_special_tokens=False)
            self._tok, self._model = tok, model
        return self._model

    def memory_bytes(self) -> int | None:
        return _param_bytes(self._model)

    def _pair(self, query: str, doc: str) -> str:
        return f"<Instruct>: {self.instruction}\n<Query>: {query}\n<Document>: {doc}"

    def _two_logits(self, inputs):
        """[batch, 2] logits for ("no", "yes") at the last position of each sequence."""
        model, (no, yes) = self._model, self._ids
        base, head = getattr(model, "model", None), getattr(model, "lm_head", None)
        if base is not None and head is not None:
            hidden = base(**inputs).last_hidden_state[:, -1, :]
            return hidden @ head.weight[[no, yes]].T
        return model(**inputs).logits[:, -1, :][:, [no, yes]]

    def score(self, query: str, texts: list[str]) -> list[float]:
        if not texts:
            return []
        self.load()
        import torch

        tok = self._tok
        room = self.max_length - len(self._prefix) - len(self._suffix)
        out: list[float] = []
        for i in range(0, len(texts), self.batch_size):
            pairs = [self._pair(query, t) for t in texts[i:i + self.batch_size]]
            enc = tok(pairs, padding=False, truncation="longest_first",
                      return_attention_mask=False, max_length=room)
            ids = [self._prefix + e + self._suffix for e in enc["input_ids"]]
            inputs = tok.pad({"input_ids": ids}, padding=True, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                two = self._two_logits(inputs).float()
            out.extend(torch.log_softmax(two, dim=1)[:, 1].exp().tolist())
        _finite_or_raise(np.asarray(out), "the candidates", self.name)
        return [float(v) for v in out]


def _from_spec(spec: str):
    mod, _, attr = spec.partition(":")
    if not mod or not attr:
        raise ValueError(f"backend spec must look like 'package.module:ClassName', got {spec!r}")
    return getattr(importlib.import_module(mod), attr)()


def make_embedder(name: str | None = None) -> Embedder:
    """The sentence-transformers embedder for *name* (default: the configured model), or a custom
    backend named by $RAG_SEARCH_EMBEDDER.

    A backend is any class with ``load()`` and ``encode(texts, progress=None) -> ndarray``
    returning L2-normalised float32 vectors (and optionally ``encode_query(query)``).  Also used
    by the tests.
    """
    spec = os.environ.get("RAG_SEARCH_EMBEDDER")
    return _from_spec(spec) if spec else Embedder(name)


def make_reranker(name: str | None = None):
    """The reranker for *name* (default: the configured model): a cross-encoder, or the yes/no
    backend for Qwen3-Reranker; or a custom backend named by $RAG_SEARCH_RERANKER
    (a class with ``load()`` and ``score(query, texts) -> list[float]``)."""
    spec = os.environ.get("RAG_SEARCH_RERANKER")
    if spec:
        return _from_spec(spec)
    name = name or rerank_model_name()
    known = models.find(name)
    if known is not None and known.backend == models.QWEN3_RERANKER:
        return Qwen3Reranker(name)
    return Reranker(name)
