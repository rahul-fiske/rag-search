"""5.2 Picture vectors and S3b: one vector per photograph made from its pixels, searched with the query's text.

A SigLIP 2 model holds an image encoder and a text encoder trained into one space, so a query ("the Golden Gate at
dusk") finds the photographs that look like it even when their description does not say so.  The vectors are a third
list in the search's merge (S4), beside BM25 and the text vectors, and the multimodal reranker (S5) then looks at the
picture of every photograph in its pool.

Storage, per document folder (a photograph only)::

    pictures.npy     float32 [1, dim], L2-normalised
    pictures.json    {"model", "src_sha256", "dim"}: its own freshness key -- a new picture model embeds the photographs
                     again and nothing else; a photograph indexed before picture vectors existed is caught up the same way

Merge (7) puts them side by side in ``_all/pictures.npy`` with ``_all/pictures.rows.json`` (the passage each row belongs
to, the photograph's first passage).  A collection without any is searched as before.

``$RAG_SEARCH_PICTURE_EMBEDDER`` (``package.module:Class``) replaces the model, as the tests do: a class with ``load()``,
``encode_images(paths) -> ndarray`` and ``encode_query(text) -> ndarray``, vectors L2-normalised.
"""

from __future__ import annotations

import contextlib
import importlib
import logging
import os
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .. import models
from ..paths import PICTURES_FILE, PICTURES_ROWS, read_json, write_json_atomic

log = logging.getLogger("rag_search.pictures")

FILE = PICTURES_FILE
META = "pictures.json"
ROWS = PICTURES_ROWS                 # in _all/: {"model", "rows": [passage index per picture row]}
ENV = "RAG_SEARCH_PICTURE_EMBEDDER"
QUERY_PROMPT = "a photo of {}"       # SigLIP was trained on captions: the prompt helps short queries (the trial: +0.01)
QUERY_TOKENS = 64                    # SigLIP 2's text side is trained with max_length 64, padded to it
READ_PX = 768                        # a photograph is opened at most this large (the model resizes it again)
BATCH = 16


def selected() -> str:
    """The picture model chosen now, or "" when picture vectors are turned off."""
    sel = models.picture_selection()[0]
    return "" if sel == "off" else sel


def ready(model: str) -> str:
    """Why picture vectors cannot be made with *model* now ("" when they can): turned off, or not downloaded (nothing is
    ever downloaded by an indexing run; the Models tab does it)."""
    if not model:
        return "picture vectors are turned off (Models tab)"
    if os.environ.get(ENV):
        return ""
    if not models.cache_state(model)["cached"]:
        return f"the picture model {model} is not downloaded (Models tab)"
    return ""


def is_current(idx_dir: Path, sha: str, model: str) -> bool:
    meta = read_json(idx_dir / META)
    return bool(meta) and meta.get("model") == model and meta.get("src_sha256") == sha and (idx_dir / FILE).exists()


def write(idx_dir: Path, vec: np.ndarray, sha: str, model: str) -> None:
    """The vector first, its key last (a half-written pair is never taken for a finished one)."""
    with contextlib.suppress(OSError):
        (idx_dir / META).unlink()
    arr = np.asarray(vec, dtype=np.float32).reshape(1, -1)
    tmp = idx_dir / (FILE + ".tmp")
    with open(tmp, "wb") as f:
        np.save(f, arr)
    os.replace(tmp, idx_dir / FILE)
    write_json_atomic(idx_dir / META, {"model": model, "src_sha256": sha, "dim": int(arr.shape[1])})


def remove(idx_dir: Path) -> None:
    for name in (META, FILE, FILE + ".tmp"):
        with contextlib.suppress(OSError):
            (idx_dir / name).unlink()


def open_image(src: Path, px: int = READ_PX):
    """The photograph as an RGB picture at most *px* on its long side, turned the way the camera held it."""
    from PIL import ImageOps

    from .photo import open_image as open_file

    im = open_file(src)
    with contextlib.suppress(Exception):
        im.draft("RGB", (px, px))        # JPEG: decode at a fraction of the size (much faster)
    im = ImageOps.exif_transpose(im).convert("RGB")
    im.thumbnail((px, px))
    return im


class PictureEmbedder:
    """A SigLIP 2 model through transformers (``AutoModel``): image features for the photographs, text features for the
    query, both L2-normalised so that their product is the cosine."""

    def __init__(self, name: str | None = None, batch_size: int = BATCH):
        self.name = name or selected() or models.PICTURE_DEFAULT
        self.batch_size = batch_size
        self._model = None
        self._proc = None
        self.device = ""

    def load(self):
        if self._model is None:
            from .embedding import _load_model, pick_device, prepare_environment, torch_dtype

            prepare_environment()
            import torch
            from transformers import AutoModel, AutoProcessor

            self.device = pick_device()
            dtype = torch_dtype(torch, self.device)
            kw = {"dtype": dtype} if dtype is not None else {}
            log.info("Loading picture model %s (device=%s)", self.name, self.device)
            model = _load_model(self.name, lambda **k: AutoModel.from_pretrained(self.name, **kw, **k))
            self._proc = _load_model(self.name, lambda **k: AutoProcessor.from_pretrained(self.name, **k))
            self._model = model.to(self.device).eval()
        return self._model

    def memory_bytes(self) -> int | None:
        from .embedding import _param_bytes

        return _param_bytes(self._model)

    @staticmethod
    def _features(out):
        import torch

        if torch.is_tensor(out):
            return out
        pooled = getattr(out, "pooler_output", None)
        return pooled if pooled is not None else out[0]

    def _normalised(self, t) -> np.ndarray:
        import torch

        v = torch.nn.functional.normalize(self._features(t).float(), dim=-1).detach().cpu().numpy()
        from .embedding import _finite_or_raise

        _finite_or_raise(v, "a picture or a query", self.name)
        return v.astype(np.float32)

    def encode_images(self, paths: list[Path], progress: Callable[[int, int], None] | None = None) -> np.ndarray:
        """One vector per photograph; a photograph that cannot be opened gets a row of zeros (it is then never found by
        its picture, and the caller leaves it without picture vectors)."""
        model = self.load()
        import torch

        out: list[np.ndarray] = []
        for i in range(0, len(paths), self.batch_size):
            part = paths[i:i + self.batch_size]
            ims, ok = [], []
            for p in part:
                try:
                    ims.append(open_image(Path(p)))
                    ok.append(True)
                except Exception as exc:  # noqa: BLE001 - one unreadable photograph must not stop the others
                    log.info("picture of %s could not be opened: %s", Path(p).name, exc)
                    ok.append(False)
            vecs = np.zeros((len(part), 0), dtype=np.float32)
            if ims:
                with torch.no_grad():
                    inp = self._proc(images=ims, return_tensors="pt").to(self.device)
                    dt = next(model.parameters()).dtype
                    inp = {k: (v.to(dt) if v.is_floating_point() else v) for k, v in inp.items()}
                    got = self._normalised(model.get_image_features(**inp))
                vecs = np.zeros((len(part), got.shape[1]), dtype=np.float32)
                vecs[np.array(ok)] = got
            out.append(vecs)
            if progress:
                progress(min(i + self.batch_size, len(paths)), len(paths))
        if not out:
            return np.zeros((0, 0), dtype=np.float32)
        width = max(v.shape[1] for v in out)
        return np.concatenate([v if v.shape[1] == width else np.zeros((len(v), width), np.float32) for v in out])

    def encode_query(self, query: str) -> np.ndarray:
        model = self.load()
        import torch

        with torch.no_grad():
            inp = self._proc(text=[QUERY_PROMPT.format(query.strip())], padding="max_length", max_length=QUERY_TOKENS,
                             truncation=True, return_tensors="pt").to(self.device)
            return self._normalised(model.get_text_features(**inp))[0]


def make_picture_embedder(name: str | None = None) -> Any:
    spec = os.environ.get(ENV)
    if spec:
        mod, _, attr = spec.partition(":")
        return getattr(importlib.import_module(mod), attr)()
    return PictureEmbedder(name)


# ── merge (7) ────────────────────────────────────────────────────────────────

def merged(docs: list[tuple[Path, int]]) -> tuple[np.ndarray | None, dict[str, Any]]:
    """The picture vectors of a collection's documents, given as (document folder, index of its first passage in the
    merged list): (matrix, rows record), or (None, {}) when none has any.  Only one model's vectors are kept -- the one most
    documents have (a switch in progress leaves the others out until they are embedded again)."""
    found: list[tuple[str, Path, int]] = []
    for d, first in docs:
        meta = read_json(d / META)
        if meta.get("model") and (d / FILE).exists():
            found.append((str(meta["model"]), d, first))
    if not found:
        return None, {}
    counts: dict[str, int] = {}
    for m, _d, _f in found:
        counts[m] = counts.get(m, 0) + 1
    model = max(counts, key=lambda m: counts[m])
    mats, rows = [], []
    for m, d, first in found:
        if m != model:
            continue
        v = np.load(d / FILE)
        if v.ndim != 2 or v.shape[0] != 1:
            continue
        mats.append(v)
        rows.append(first)
    if not mats or len({v.shape[1] for v in mats}) != 1:
        return None, {}
    return np.concatenate(mats, axis=0).astype(np.float32), {"model": model, "rows": rows}
