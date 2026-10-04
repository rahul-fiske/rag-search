"""Hybrid retrieval over the *published* generation: BM25 + dense vectors -> RRF -> rerank.

`SearchEngine` holds the models and the loaded indexes of one generation.  A new
generation is prepared off to the side (`prepare_generation`) and swapped in with one
pointer assignment (`install`), so searches never wait for a reload and never see a
half-loaded state.

The embedding model always follows the *published* index (its catalog records the model it was
built with): switching the configured model does not disturb searches until the re-embedded
index is published, and the new generation then arrives together with its own embedder.  A
changed reranker setting is picked up by `prepare_reranker` / `install_reranker`.  Engines given
an explicit embedder (tests, custom backends) keep the strict rule instead: a generation built
with another model is refused (`ModelMismatch`).
"""

from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..paths import (ALL_DIR, DEFAULT_TOP_K, EMB_FILE, NODES_FILE, Paths, env_flag, model_name,
                     read_json, rerank_model_name)
from ..publish import read_catalog
from ..spec import (MAX_TOP_K, RRF_K, SNIPPET_CHARS, clamp_rerank_pool, clamp_retrieval_pool,
                    clamp_rrf_k, parse_stages, rerank_pool, retrieval_pool)
from .bm25 import BM25, tokenize

log = logging.getLogger("rag_search.search")



class ModelMismatch(RuntimeError):
    """The generation was built with a different embedding model than this engine uses."""


@dataclass
class IndexData:
    nodes: list[dict[str, Any]]
    emb: np.ndarray
    bm25: BM25


def load_index(d: Path) -> IndexData:
    nodes = read_json(d / NODES_FILE).get("nodes", [])
    emb = np.load(d / EMB_FILE)
    if emb.shape[0] != len(nodes):
        raise RuntimeError(f"{d}: {len(nodes)} chunks but {emb.shape[0]} embeddings")
    return IndexData(nodes, emb, BM25([tokenize(n["text"]) for n in nodes]))


def _top(scores: np.ndarray, n: int, *, positive_only: bool = False) -> list[int]:
    if scores.size == 0:
        return []
    n = min(n, scores.size)
    idx = np.argpartition(-scores, n - 1)[:n]
    idx = idx[np.argsort(-scores[idx], kind="stable")]
    if positive_only:
        idx = idx[scores[idx] > 0]
    return [int(i) for i in idx]


@dataclass
class Generation:
    """An in-memory generation: catalog + lazily/eagerly loaded per-collection indexes."""
    number: int | None = None
    root: Path | None = None
    catalog: dict[str, Any] = field(default_factory=dict)
    indexes: dict[str, IndexData] = field(default_factory=dict)
    reused: list[str] = field(default_factory=list)   # carried over from the previous generation
    loaded: list[str] = field(default_factory=list)   # read from disk for this generation
    embedder: Any = None      # a new embedder that comes with this generation (model changed)
    model: str = ""           # embedding model the generation was built with
    _lock: Any = field(default_factory=threading.Lock, repr=False, compare=False)

    def collections(self) -> list[str]:
        return [c["collection"] for c in self.catalog.get("collections", [])]

    def get(self, coll: str) -> IndexData:
        data = self.indexes.get(coll)
        if data is not None:
            return data
        with self._lock:                       # concurrent searches load a collection once
            if coll not in self.indexes:
                if self.root is None:
                    raise KeyError(coll)
                self.indexes[coll] = load_index(self.root / "index" / coll / ALL_DIR)
                self.loaded.append(coll)
            return self.indexes[coll]


class SearchEngine:
    def __init__(self, paths: Paths, embedder: Any = None, reranker: Any = None,
                 rerank: bool | None = None):
        self.paths = paths
        self.embedder = embedder
        self.reranker = reranker
        self.rerank = env_flag("RAG_SEARCH_RERANK", True) if rerank is None else rerank
        self.gen = Generation()
        self.model = model_name()
        # Searches run concurrently.  `_swap` makes (generation, embedder, reranker) one
        # consistent snapshot per query; `_infer` serialises only the model calls (one model on
        # one device), so BM25, the vector product and fusion of different queries overlap.
        self._swap = threading.Lock()
        self._infer = threading.RLock()
        # default backends follow the published index / the settings; injected ones stay as given
        self._follow_embedder = embedder is None and not os.environ.get("RAG_SEARCH_EMBEDDER")
        self._follow_reranker = reranker is None and not os.environ.get("RAG_SEARCH_RERANKER")

    # models ---------------------------------------------------------------
    def _model_to_serve(self) -> str:
        """The embedding model to load: the one the published index was built with (so search
        keeps working while the configured model is being switched), else the configured one."""
        if self._follow_embedder:
            built_with = read_catalog(self.paths.current_gen()).get("model", "")
            if built_with:
                return str(built_with)
        return model_name()

    def load_models(self) -> None:
        from .embedding import make_embedder, make_reranker

        with self._infer:
            if self.embedder is None:
                model = self._model_to_serve()
                emb = make_embedder(model)
                with self._swap:
                    self.model, self.embedder = model, emb
            if hasattr(self.embedder, "load"):
                self.embedder.load()
            if self.rerank:
                if self.reranker is None:
                    rr = make_reranker()
                    with self._swap:
                        self.reranker = rr
                if hasattr(self.reranker, "load"):
                    self.reranker.load()

    def prepare_reranker(self) -> Any:
        """A reranker for the *configured* model, loaded off to the side; None when the loaded
        one is already right (or was injected, or reranking is off, or nothing is loaded yet)."""
        if not (self._follow_reranker and self.rerank) or self.reranker is None:
            return None
        want = rerank_model_name()
        if want == getattr(self.reranker, "name", want):
            return None
        from .embedding import make_reranker

        new = make_reranker(want)
        if hasattr(new, "load"):
            new.load()
        return new

    def install_reranker(self, new: Any) -> None:
        with self._swap:
            old, self.reranker = self.reranker, new
        del old
        from .embedding import release_memory

        with self._infer:                      # never flush the GPU cache mid-inference
            release_memory()

    def models_info(self) -> dict[str, str]:
        """Names of the models loaded now (what search uses)."""
        return {"embedding": getattr(self.embedder, "name", "") or self.model,
                "reranker": getattr(self.reranker, "name", "") if self.reranker is not None else ""}

    def memory_info(self) -> dict[str, Any]:
        """What the serving generation keeps in memory (embeddings and chunk text exactly;
        the BM25 postings and Python overhead show up in the process's resident size)."""
        colls: dict[str, Any] = {}
        emb = txt = 0
        for name, d in list(self.gen.indexes.items()):
            e = int(d.emb.nbytes)
            t = sum(len(n["text"]) for n in d.nodes)
            colls[name] = {"chunks": len(d.nodes), "embeddings_bytes": e, "text_bytes": t}
            emb += e
            txt += t
        models: dict[str, int] = {}
        for label, m in (("embedder", self.embedder), ("reranker", self.reranker)):
            fn = getattr(m, "memory_bytes", None)
            b = fn() if callable(fn) else None
            if b is not None:
                models[label] = b
        return {"embeddings_bytes": emb, "text_bytes": txt, "models_bytes": models,
                "collections": colls, "models": self.models_info()}

    # generations ----------------------------------------------------------
    def prepare_generation(self, prewarm: bool = True) -> Generation | None:
        """Load the live generation from disk without touching the serving one.

        Returns None when nothing is published.  Raises ModelMismatch if the
        generation's embedding model differs from ours.
        """
        root = self.paths.current_gen()
        if root is None:
            return None
        catalog = read_catalog(root)
        built_with = catalog.get("model", "")
        new_embedder = None
        if built_with and built_with != self.model and catalog.get("collections"):
            if not self._follow_embedder:
                raise ModelMismatch(f"generation {catalog.get('generation')} was built with "
                                    f"{built_with!r} but this daemon uses {self.model!r}")
            if self.embedder is not None:      # a re-embedded index: its model comes with it
                from .embedding import make_embedder

                new_embedder = make_embedder(built_with)
                if hasattr(new_embedder, "load"):
                    new_embedder.load()
        gen = Generation(catalog.get("generation"), root, catalog, embedder=new_embedder,
                         model=built_with)
        self._carry_over(gen)
        if prewarm:
            for coll in gen.collections():
                gen.get(coll)
        return gen

    def _carry_over(self, gen: Generation) -> None:
        """Reuse in-memory indexes of collections that did not change.

        `manifest_sha` (in catalog.json) identifies a collection's merged content, so a
        collection with the same sha and the same embedding model is byte-identical to
        the one already loaded.  IndexData is never mutated, so sharing it between the
        serving generation and the one being prepared is safe.
        """
        old = self.gen
        if not old.indexes or old.catalog.get("model") != gen.catalog.get("model"):
            return
        old_sha = {c["collection"]: c.get("manifest_sha") for c in old.catalog.get("collections", [])}
        for c in gen.catalog.get("collections", []):
            name = c["collection"]
            if name in old.indexes and c.get("manifest_sha") and old_sha.get(name) == c["manifest_sha"]:
                gen.indexes[name] = old.indexes[name]
                gen.reused.append(name)

    def install(self, gen: Generation | None) -> None:
        old = None
        with self._swap:                       # generation and its embedder change together
            if gen is not None and gen.embedder is not None:
                old, self.embedder, self.model = self.embedder, gen.embedder, gen.model
            self.gen = gen or Generation()
        if old is not None:                    # the previous model is no longer needed
            del old
            from .embedding import release_memory

            with self._infer:                  # never flush the GPU cache mid-inference
                release_memory()

    @property
    def generation(self) -> int | None:
        return self.gen.number

    # search ---------------------------------------------------------------
    def search(self, query: str, top_k: int = DEFAULT_TOP_K,
               collections: list[str] | None = None, *,
               stages: "tuple[str, ...] | list[str] | str | None" = None,
               retrieval_pool_n: int | None = None, rerank_pool_n: int | None = None,
               rrf_k: int | None = None) -> dict[str, Any]:
        """Search the given collections (already policy-checked by the caller).

        ``stages`` (default: bm25+dense+rerank, i.e. today's behaviour) lets a caller isolate
        one retriever, or turn reranking off, to troubleshoot the pipeline; at least one of
        bm25/dense must stay on (`spec.parse_stages` normalises/validates it, same as the
        daemon does at the request boundary -- a bad value here raises ValueError rather than
        silently misbehaving).  ``retrieval_pool_n``/``rerank_pool_n``/``rrf_k`` override the
        `spec.py` formulas for the same purpose, clamped to safe ceilings.  With only one
        retrieval stage active there is nothing to fuse, so ranking falls back to that stage's
        own score instead of RRF (RRF over a single list is just its rank order -- it would
        discard the stage's raw score, which is exactly what a single-stage debug search wants
        to see).
        """
        t0 = time.perf_counter()
        query = (query or "").strip()
        if not query:
            return {"results": [], "error": "empty query"}
        top_k = max(1, min(int(top_k), MAX_TOP_K))
        stages = parse_stages(stages)
        use_bm25, use_dense = "bm25" in stages, "dense" in stages
        want_rerank = "rerank" in stages
        fused = use_bm25 and use_dense
        with self._swap:  # one consistent snapshot for the whole query
            gen, embedder, reranker = self.gen, self.embedder, self.reranker
        have = gen.collections()
        # each collection once: a repeated name would add its rank contributions twice
        wanted = list(dict.fromkeys(c for c in (collections if collections is not None else have)
                                    if c in have))
        pool_n = clamp_retrieval_pool(retrieval_pool_n, top_k) if retrieval_pool_n \
            else retrieval_pool(top_k)
        rr_pool_n = clamp_rerank_pool(rerank_pool_n, top_k) if rerank_pool_n else rerank_pool(top_k)
        k = clamp_rrf_k(rrf_k) if rrf_k else RRF_K
        timing: dict[str, Any] = {"generation": gen.number, "collections": len(wanted),
                                  "stages": list(stages), "retrieval_pool": pool_n,
                                  "rerank_pool": rr_pool_n, "rrf_k": k}
        if not wanted:
            note = ("nothing is published yet; index some documents first"
                    if gen.number is None else "no matching collections")
            timing["total_ms"] = int((time.perf_counter() - t0) * 1000)
            return {"results": [], "note": note, "timing": timing}
        if embedder is None:
            self.load_models()
            with self._swap:
                embedder, reranker = self.embedder, self.reranker

        qtok = tokenize(query) if use_bm25 else []
        qvec = None
        t_embed = time.perf_counter()
        if use_dense:
            encode_query = getattr(embedder, "encode_query", None)  # model's query prefix
            with self._infer:
                qvec = np.asarray(encode_query(query) if callable(encode_query)
                                  else embedder.encode([query])[0], dtype=np.float32)
        timing["embed_query_ms"] = int((time.perf_counter() - t_embed) * 1000)
        cand: dict[tuple[str, str], dict[str, Any]] = {}
        keyword_s = dense_s = 0.0
        for coll in wanted:
            data = gen.get(coll)
            bm25_scores = dense_scores = None
            if use_bm25:
                t = time.perf_counter()
                bm25_scores = data.bm25.scores(qtok)
                kw = _top(bm25_scores, pool_n, positive_only=True)
                keyword_s += time.perf_counter() - t
            else:
                kw = []
            if use_dense:
                t = time.perf_counter()
                dense_scores = data.emb @ qvec
                dn = _top(dense_scores, pool_n)
                dense_s += time.perf_counter() - t
            else:
                dn = []
            for r, i in enumerate(kw):
                node = data.nodes[i]
                slot = cand.setdefault((coll, node["id"]), {"rrf": 0.0, "node": node, "coll": coll,
                                       "bm25_score": None, "bm25_rank": None,
                                       "dense_score": None, "dense_rank": None})
                slot["rrf"] += 1.0 / (k + r + 1)
                slot["bm25_score"], slot["bm25_rank"] = float(bm25_scores[i]), r + 1
            for r, i in enumerate(dn):
                node = data.nodes[i]
                slot = cand.setdefault((coll, node["id"]), {"rrf": 0.0, "node": node, "coll": coll,
                                       "bm25_score": None, "bm25_rank": None,
                                       "dense_score": None, "dense_rank": None})
                slot["rrf"] += 1.0 / (k + r + 1)
                slot["dense_score"], slot["dense_rank"] = float(dense_scores[i]), r + 1
        timing["keyword_ms"] = int(keyword_s * 1000)     # BM25 over all collections in scope
        timing["dense_ms"] = int(dense_s * 1000)         # cosine over all collections in scope
        timing["retrieve_ms"] = int((time.perf_counter() - t0) * 1000)   # embed + both + fusion
        timing["candidates"] = len(cand)
        _stage_stats(timing, cand.values())

        sort_key = ((lambda s: s["rrf"]) if fused else
                   (lambda s: s["bm25_score"]) if use_bm25 else (lambda s: s["dense_score"]))
        pool = sorted(cand.values(), key=sort_key, reverse=True)[:rr_pool_n]
        t1 = time.perf_counter()
        scores: list[float] | None = None
        if want_rerank and pool:
            if self.rerank:
                try:
                    if reranker is None:
                        self.load_models()
                        reranker = self.reranker
                    with self._infer:
                        scores = reranker.score(query, [s["node"]["text"] for s in pool])
                except Exception as exc:  # noqa: BLE001
                    log.warning("rerank failed, using prior order: %s", exc)
                    timing["rerank_error"] = str(exc)
            else:
                timing["rerank_error"] = ("the reranker is turned off in this deployment's "
                    "model config; turn it on (rag-search models set reranker ...) to use it")
        timing["rerank_ms"] = int((time.perf_counter() - t1) * 1000)
        timing["reranked"] = scores is not None
        if scores is not None:
            top1, top2 = _top1_top2(scores)
            timing["rerank_top1"], timing["rerank_gap"] = top1, (None if top2 is None else
                                                                  round(top1 - top2, 4))

        order = (sorted(range(len(pool)), key=lambda i: scores[i], reverse=True)
                 if scores is not None else list(range(len(pool))))
        results = []
        for rank, i in enumerate(order[:top_k], 1):
            c = pool[i]
            node, coll = c["node"], c["coll"]
            rrf = c["rrf"] if fused else None
            rerank_score = scores[i] if scores is not None else None
            final = (rerank_score if rerank_score is not None else
                    rrf if rrf is not None else sort_key(c))
            m = node["metadata"]
            text = node["text"]
            results.append({
                "rank": rank,
                "score": round(final, 4),
                "rrf_score": None if rrf is None else round(rrf, 5),
                "bm25_score": None if c["bm25_score"] is None else round(c["bm25_score"], 4),
                "bm25_rank": c["bm25_rank"],
                "dense_score": None if c["dense_score"] is None else round(c["dense_score"], 4),
                "dense_rank": c["dense_rank"],
                "rerank_score": None if rerank_score is None else round(rerank_score, 4),
                "collection": coll,
                "file": m.get("file_name", ""),
                "source": m.get("source_name", ""),
                "page": m.get("page_label", "?"),
                "heading": m.get("heading", ""),
                "text": text if len(text) <= SNIPPET_CHARS else text[:SNIPPET_CHARS] + " …",
                **({"confidence": "low"} if m.get("confidence") == "low" else {}),
            })
        timing["total_ms"] = int((time.perf_counter() - t0) * 1000)
        return {"results": results, "timing": timing}


def _top1_top2(values: "list[float] | np.ndarray") -> tuple[float, float | None]:
    """The two largest values, rounded -- used for the "score cliff" diagnostic: a big gap
    between rank 1 and rank 2 within one stage is a sign RRF/fusion may be burying a strong hit."""
    ordered = sorted((float(v) for v in values), reverse=True)
    if not ordered:
        return 0.0, None
    top1 = round(ordered[0], 4)
    return top1, (round(ordered[1], 4) if len(ordered) > 1 else None)


def _stage_stats(timing: dict[str, Any], candidates: "list[dict[str, Any]] | Any") -> None:
    """Per-stage candidate counts, retrieval overlap, and top1/gap -- computed once over the
    full (pre rerank-pool-truncation) candidate set, so they reflect what each retriever
    actually found rather than only what survived into the reranker's pool."""
    candidates = list(candidates)
    bm25_vals = [c["bm25_score"] for c in candidates if c["bm25_score"] is not None]
    dense_vals = [c["dense_score"] for c in candidates if c["dense_score"] is not None]
    both = sum(1 for c in candidates if c["bm25_score"] is not None and c["dense_score"] is not None)
    timing["bm25_candidates"] = len(bm25_vals)
    timing["dense_candidates"] = len(dense_vals)
    timing["overlap_count"] = both
    timing["bm25_only_count"] = len(bm25_vals) - both
    timing["dense_only_count"] = len(dense_vals) - both
    for label, vals in (("bm25", bm25_vals), ("dense", dense_vals)):
        if not vals:
            continue
        top1, top2 = _top1_top2(vals)
        timing[f"{label}_top1"] = top1
        timing[f"{label}_gap"] = None if top2 is None else round(top1 - top2, 4)
