"""Playground: a structurally separate sandbox for trying models/tunables and benchmarking,
without ever touching a production collection.

Isolation is by construction, not convention: every playground function takes the *production*
``Paths`` only to resolve ``<home>/playground/<name>/`` (`paths.get_playground_paths`), and from
there works exclusively inside that sub-tree -- its own ``locations.json`` (the folders the user
chose as its sources, registered exactly like a production collection's; never copied), its own
``workspace/index/`` and ``config.json``.  Nothing here ever opens ``serving/``, a ``run/*.sock``, or the
production ``config.json``; there is no generation/publish/hot-swap machinery at all (a
playground has exactly one reader), and there is no standing daemon -- ``search``/``bench`` load
the small index and the chosen models in-process and return, the same way `rag-search index
foreground` does today.

Each experiment pins its own embedding model, reranker, chunk params and default search tunables
in its own ``config.json`` (`get_config`/`update_config`).  Because `core.search.SearchEngine`
already accepts pre-built embedder/reranker *objects* and `core.indexer.run_index` accepts an
explicit *model* override (see its docstring), a playground experiment never depends on
`paths.model_name()` / `$RAG_SEARCH_MODEL` / this process's ambient ``config.json`` -- so running
two experiments with two different models, or from a long-lived process such as the dashboard,
never races or leaks between them.
"""

from __future__ import annotations

import contextlib
import json
import math
import os
import shutil
import time
import uuid
from pathlib import Path
from typing import Any

from ..effective import settings_env
from ..paths import (
    ALL_DIR,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_MODEL,
    DEFAULT_RERANK_MODEL,
    DEFAULT_TOP_K,
    Paths,
    ensure_dirs,
    get_playground_paths,
    is_within,
    list_playground_names,
    playground_root,
    read_json,
    write_json_atomic,
)
from ..spec import (
    TUNABLES_BY_KEY,
    TUNABLES_BY_SECTION,
    parse_stages,
    validate_section,
    validate_tunable,
)
from .indexer import run_plan

# Every docling/OCR/table/PDF-backend knob a playground experiment can pin -- everything in
# spec.TUNABLES's "indexer" section that has an environment variable of its own (chunk_size/
# chunk_overlap are "indexer" tunables too, but they have no env var -- they're applied as
# run_index() parameters above, not env vars, and already have their own dedicated fields
# below).  Sharing spec.py's registry, instead of re-describing these knobs here, means the
# choices/validation/help text can never drift from what `rag-search config set` and the
# Settings tab already show for the exact same knobs.
DOCLING_TUNABLES = tuple(t for t in TUNABLES_BY_SECTION["indexer"] if t.env)
DOCLING_TUNABLE_KEYS = frozenset(t.key for t in DOCLING_TUNABLES)
# experiment key -> (stage setting, environment variable the reading code looks at)
MODEL_PINS = {"reader_model": ("models.reader", "RAG_SEARCH_VLM_MODEL"),
              "repair_model": ("models.repair", "RAG_SEARCH_REPAIR_MODEL")}

DEFAULT_CONFIG: dict[str, Any] = {
    "embedding_model": DEFAULT_MODEL,
    "rerank_model": DEFAULT_RERANK_MODEL,
    "chunk_size": DEFAULT_CHUNK_SIZE,
    "chunk_overlap": DEFAULT_CHUNK_OVERLAP,
    "rerank": True,
    "stages": None,          # None = all stages (bm25+dense+rerank), same default as production
    "retrieval_pool": None,
    "rerank_pool": None,
    "rrf_k": None,
    # The document reader and the repair model (stages 3.2 / 3.4).  Blank = production's choice (the Models
    # tab); a model id pins this experiment to that model, whatever production uses.
    "reader_model": "",
    "repair_model": "",
    # Blank/0 (each tunable's own sentinel -- see spec.Tunable) means "no override": this
    # experiment converts documents exactly like production's own built-in defaults, same as
    # if RAG_SEARCH_OCR & co. were never set.
    **{t.key: t.default for t in DOCLING_TUNABLES},
}
_CONFIG_KEYS = frozenset(DEFAULT_CONFIG)


class PlaygroundError(ValueError):
    """A user-facing playground problem (bad name, missing docs, no index yet, ...)."""


# ── experiment lifecycle ──────────────────────────────────────────────────────

def create_experiment(base: Paths, name: str, *, sources: list[str] | None = None,
                      collection: str = "", from_production: bool = False) -> dict[str, Any]:
    """Start an experiment.  *sources* are folders registered as its source locations (never copied);
    *collection* names the collection when exactly one folder is given (default: the folder's own name)."""
    exp = get_playground_paths(base, name)
    if exp.home.exists():
        raise PlaygroundError(f"experiment {name!r} already exists "
                              f"(rag-search playground index {name} to (re)build it)")
    if collection and len(sources or []) != 1:
        raise PlaygroundError("--collection names the collection of exactly one --from folder")
    ensure_dirs(exp)
    (exp.home / "bench" / "runs").mkdir(parents=True, exist_ok=True)
    example = exp.home / "bench" / "queries.jsonl.example"
    example.write_text(
        '{"query": "how is a session token refreshed", '
        '"relevant": [{"file": "example.pdf", "page": 4}]}\n'
        '{"query": "second example question", '
        '"relevant": [{"file": "example.pdf", "page": 1}, {"file": "other.docx", "page": 2}]}\n',
        encoding="utf-8")
    if not (exp.home / "config.json").exists():
        seed = dict(DEFAULT_CONFIG)
        if from_production:
            seed.update(production_snapshot(base))
        write_json_atomic(exp.home / "config.json", seed)
    try:
        added = [add_source(base, name, folder, collection) for folder in (sources or [])]
    except PlaygroundError:
        shutil.rmtree(exp.home, ignore_errors=True)         # an experiment that could not get its sources is not left half made
        raise
    return {"name": name, "home": str(exp.home), "sources": added,
            "from_production": from_production, "config": get_config(base, name)}


def _exp(base: Paths, name: str) -> Paths:
    exp = get_playground_paths(base, name)
    if not exp.home.is_dir():
        raise PlaygroundError(f"no such experiment: {name} (rag-search playground create {name})")
    return exp


def add_source(base: Paths, name: str, folder: str, collection: str = "") -> dict[str, Any]:
    """Register *folder* as a source of the experiment: the same rules and registry as production
    (``locations``), kept in the experiment's own ``locations.json``."""
    from .. import locations

    try:
        return locations.add(_exp(base, name), collection, folder)
    except locations.LocationError as exc:
        raise PlaygroundError(str(exc)) from exc


def remove_source(base: Paths, name: str, collection: str) -> str:
    from .. import locations

    removed = locations.remove_entry(_exp(base, name), collection)
    if not removed:
        raise PlaygroundError(f"experiment {name!r} has no source {collection!r}")
    return removed


def list_sources(base: Paths, name: str) -> dict[str, Any]:
    from .. import locations

    return locations.sources(_exp(base, name)) | {"status": locations.status(_exp(base, name))}


def list_experiments(base: Paths) -> list[dict[str, Any]]:
    from .. import locations

    out = []
    for name in list_playground_names(base):
        exp = get_playground_paths(base, name)
        locs = locations.load(exp)[0]
        collections = sorted(d.name for d in exp.index.iterdir()
                             if d.is_dir() and (d / ALL_DIR).is_dir()) if exp.index.is_dir() else []
        runs = len(list((exp.home / "bench" / "runs").glob("*.json"))) \
            if (exp.home / "bench" / "runs").is_dir() else 0
        out.append({"name": name, "sources": [{"collection": n, "folder": f} for n, f in sorted(locs.items())],
                    "indexed_collections": collections, "bench_runs": runs, "config": get_config(base, name)})
    return out


def remove_experiment(base: Paths, name: str) -> None:
    exp = get_playground_paths(base, name)   # validates *name*
    root = playground_root(base)
    if not is_within(exp.home, root) or exp.home == root:
        raise PlaygroundError("refusing to remove a path outside the playground root")
    if not exp.home.is_dir():
        raise PlaygroundError(f"no such experiment: {name}")
    shutil.rmtree(exp.home)


# ── per-experiment config (model choice, chunk params, default tunables) ──────

def get_config(base: Paths, name: str) -> dict[str, Any]:
    exp = get_playground_paths(base, name)
    stored = read_json(exp.home / "config.json")
    cfg = dict(DEFAULT_CONFIG)
    if isinstance(stored, dict):
        cfg.update({k: v for k, v in stored.items() if k in _CONFIG_KEYS})
    return cfg


def update_config(base: Paths, name: str, **overrides: Any) -> dict[str, Any]:
    exp = get_playground_paths(base, name)
    if not exp.home.is_dir():
        raise PlaygroundError(f"no such experiment: {name} (rag-search playground create {name})")
    cfg = get_config(base, name)
    for key, value in overrides.items():
        if value is None or key not in _CONFIG_KEYS:
            continue
        if key in ("chunk_size", "chunk_overlap") and int(value) <= 0:
            raise PlaygroundError(f"{key} must be a positive integer")
        if key == "stages":
            parse_stages(value)   # validate only; stored as given (string or list)
        if key in MODEL_PINS:
            value = _model_pin(value)
        if key in DOCLING_TUNABLE_KEYS:
            # same validation/normalisation `rag-search config set` and the Settings tab use for
            # this exact knob -- blank/0 always comes back out as "no override" (see spec.py)
            value = validate_tunable(TUNABLES_BY_KEY[key], value)
        cfg[key] = value
    write_json_atomic(exp.home / "config.json", cfg)
    return cfg


def _model_pin(value: Any) -> str:
    """A Hugging Face model id, or "" for "use production's choice" (also spelled "production" / "default")."""
    from .. import models

    v = str(value or "").strip()
    if v.lower() in ("", "production", "default", "none"):
        return ""
    if not models._ID_RE.match(v):
        raise PlaygroundError(f"{v!r} is not a Hugging Face model id (expected ORG/NAME); "
                              "leave it empty to use production's choice")
    return v


# ── the playground <-> production bridge ──────────────────────────────────────
# "copy from production" (create_experiment(..., from_production=True)) seeds a new experiment
# with today's *effective* production settings (env var > config.json > built-in default, the
# same precedence `paths.model_name()` and friends use), so a benchmark starts from what real
# searches actually get instead of this module's own defaults.  "promote" is the reverse: write
# an experiment's combo back into production's config.json.  Both go through `production_snapshot`
# so create and promote agree on what "production's settings" means, and neither one ever touches
# an index -- promoting only changes what the *next* index/search does.

def _prod_vlm(kind: str) -> str:
    from .. import models

    try:
        return models.vlm_selection(kind)[0]
    except Exception:          # noqa: BLE001 - a snapshot never fails over a model choice
        return ""


def production_snapshot(base: Paths) -> dict[str, Any]:
    """Production's current effective settings, shaped like an experiment's config.json."""
    from ..config import load_config

    cfg, _ = load_config(base)
    scfg, icfg, mcfg = cfg["search"], cfg["indexer"], cfg["models"]
    return {
        "embedding_model": (os.environ.get("RAG_SEARCH_MODEL") or mcfg.get("embedding")
                            or DEFAULT_MODEL),
        "rerank_model": (os.environ.get("RAG_SEARCH_RERANK_MODEL") or mcfg.get("reranker")
                         or DEFAULT_RERANK_MODEL),
        "chunk_size": int(icfg.get("chunk_size") or DEFAULT_CHUNK_SIZE),
        "chunk_overlap": int(icfg.get("chunk_overlap") or DEFAULT_CHUNK_OVERLAP),
        "rerank": True,
        "stages": scfg.get("stages") or None,
        "retrieval_pool": scfg.get("retrieval_pool") or None,
        "rerank_pool": scfg.get("rerank_pool") or None,
        "rrf_k": scfg.get("rrf_k") or None,
        "reader_model": _prod_vlm("reader"),
        "repair_model": _prod_vlm("repair"),
        # Same env > config.json > built-in-default precedence _worker_env uses at index time --
        # a docling tunable that's on neither comes back as its own blank/0 "no override"
        # sentinel, which is the faithful copy: production isn't overriding it either.
        **{t.key: (os.environ.get(t.env) or icfg.get(t.key) or t.default) for t in DOCLING_TUNABLES},
    }


# Fields a promotion can change, and the subset of those that make the existing index stale
# (a different embedding model, chunk size or overlap all change what a stored chunk/vector
# means, so it needs re-embedding -- see `models.reindex_estimate`).  A different reranker or
# search tunable only changes the *next* search/index run, nothing to warn about there.
_PROMOTABLE = ("embedding_model", "rerank_model", "chunk_size", "chunk_overlap", "stages",
              "retrieval_pool", "rerank_pool", "rrf_k", "reader_model", "repair_model")
_REINDEX_FIELDS = ("embedding_model", "chunk_size", "chunk_overlap")


def promotion_preview(base: Paths, name: str) -> dict[str, Any]:
    """What promoting experiment *name* would change in production, and whether that change
    needs a reindex (with `models.reindex_estimate`'s cost estimate when it does)."""
    exp_cfg = get_config(base, name)
    prod = production_snapshot(base)
    # a blank reader/repair pin means "production's choice": nothing to promote
    changes = {f: {"from": prod.get(f), "to": exp_cfg.get(f)} for f in _PROMOTABLE
              if prod.get(f) != exp_cfg.get(f) and not (f in MODEL_PINS and not exp_cfg.get(f))}
    needs_reindex = any(f in changes for f in _REINDEX_FIELDS)
    reindex = None
    if needs_reindex:
        from .. import models

        reindex = models.reindex_estimate(base, exp_cfg["embedding_model"])
    return {"experiment": name, "changes": changes, "needs_reindex": needs_reindex,
            "reindex_estimate": reindex}


def promote_to_production(base: Paths, name: str, *, confirm: bool = False) -> dict[str, Any]:
    """Write experiment *name*'s embedding/reranker/chunk/search settings into production's
    config.json.  Never touches an index itself: if this changes the embedding model or chunk
    size/overlap, the existing production index still reflects the *old* settings until it is
    rebuilt (``rag-search index --rebuild``) -- ``confirm=True`` is required in that case, so a
    promotion never silently leaves the live index stale without the caller knowing."""
    preview = promotion_preview(base, name)
    if not preview["changes"]:
        return {"promoted": name, "changes": {}, "needs_reindex": False, "reindex_estimate": None}
    if preview["needs_reindex"] and not confirm:
        est = preview["reindex_estimate"] or {}
        cost = f"~{est.get('documents', '?')} document(s)"
        if est.get("estimated_s") is not None:
            cost += f", about {est['estimated_s']}s"
        raise PlaygroundError(
            "this changes the embedding model and/or chunk size/overlap, which makes the "
            f"existing production index stale until it is rebuilt ({cost}). Re-run with "
            "confirm=true to promote anyway, then `rag-search index --rebuild`.")
    exp_cfg = get_config(base, name)
    changes = preview["changes"]
    from .. import models
    from ..config import update_config as write_prod_section

    if "embedding_model" in changes:
        models.set_selection(base, models.EMBEDDING, exp_cfg["embedding_model"])
    if "rerank_model" in changes:
        models.set_selection(base, models.RERANKER, exp_cfg["rerank_model"])
    for pin in MODEL_PINS:
        if pin in changes:
            models.set_vlm_selection(base, pin.split("_")[0], exp_cfg[pin])
    indexer_raw = {"chunk_size": exp_cfg["chunk_size"], "chunk_overlap": exp_cfg["chunk_overlap"]}
    indexer_over = {k: v for k, v in validate_section("indexer", indexer_raw).items() if k in changes}
    if indexer_over:
        write_prod_section(base, "indexer", indexer_over)
    search_raw = {"stages": exp_cfg.get("stages") or "",
                 "retrieval_pool": exp_cfg.get("retrieval_pool") or 0,
                 "rerank_pool": exp_cfg.get("rerank_pool") or 0, "rrf_k": exp_cfg.get("rrf_k") or 0}
    search_over = {k: v for k, v in validate_section("search", search_raw).items() if k in changes}
    if search_over:
        write_prod_section(base, "search", search_over)
    return {"promoted": name, "changes": changes, "needs_reindex": preview["needs_reindex"],
            "reindex_estimate": preview["reindex_estimate"]}


# ── building the sample index (no publish, no generations) ────────────────────

def _config_like(base: Paths, cfg: dict[str, Any]) -> dict[str, Any]:
    """The experiment as a ``config.json``-shaped dict (what ``effective.resolve`` and ``settings_env`` read).

    The experiment owns the models it pins, chunking, conversion and search settings; the *hardware*
    settings of the Models section (device, dtype, batch sizes, memory limit) and an unpinned reader /
    repair model are production's, because they describe this computer, not the experiment."""
    from ..config import load_config

    prod, _ = load_config(base)
    pmodels = dict(prod.get("models") or {})
    return {
        "indexer": {"chunk_size": cfg["chunk_size"], "chunk_overlap": cfg["chunk_overlap"], "jobs": 1,
                    **{t.key: cfg.get(t.key) for t in DOCLING_TUNABLES}},
        "models": {**pmodels, "embedding": cfg["embedding_model"], "reranker": cfg["rerank_model"],
                   "reader": cfg.get("reader_model") or pmodels.get("reader", ""),
                   "repair": cfg.get("repair_model") or pmodels.get("repair", "")},
        "search": {"stages": cfg.get("stages") or "", "retrieval_pool": cfg.get("retrieval_pool") or 0,
                   "rerank_pool": cfg.get("rerank_pool") or 0, "rrf_k": cfg.get("rrf_k") or 0,
                   "top_k": (prod.get("search") or {}).get("top_k", 0)},
    }


def experiment_env(base: Paths, cfg: dict[str, Any], environ: dict[str, str] | Any = None) -> dict[str, str]:
    """The environment variables this experiment adds for its run: the same ``effective.settings_env`` the
    production indexer daemon uses for its workers (an actual environment variable always wins; blank / 0 adds
    nothing), plus the reader / repair model an experiment pins."""
    environ = os.environ if environ is None else environ
    merged = settings_env(_config_like(base, cfg), dict(environ), ("indexer", "models"))
    for key, (_setting, env_name) in MODEL_PINS.items():
        if cfg.get(key):
            merged.setdefault(env_name, str(cfg[key]))
    return {k: v for k, v in merged.items() if k not in environ}


@contextlib.contextmanager
def _experiment_env(base: Paths, cfg: dict[str, Any]):
    """Apply :func:`experiment_env` for the run that is about to start, then restore exactly what was there --
    scoped to the one ``build_index()`` / search / bench call, never left set afterwards, so back-to-back
    experiments with different settings are safe even in one process (tests, the dashboard)."""
    added = experiment_env(base, cfg)
    try:
        os.environ.update(added)
        yield added
    finally:
        for name in added:
            os.environ.pop(name, None)


def effective_settings(base: Paths, name: str) -> dict[str, Any]:
    """What this experiment's next run will really use, by pipeline stage: ``effective.resolve`` over the
    experiment's settings, with where each value comes from (experiment | production config | environment |
    default).  The stages are the production ones (``stages.py``); 7 Merge and 8 Publish do not exist here."""
    from .. import effective
    from .. import stages as stg

    if not get_playground_paths(base, name).home.is_dir():       # defaults for a name that is nothing are misleading
        raise PlaygroundError(f"no such experiment: {name} (rag-search playground create {name})")
    cfg = get_config(base, name)
    like = _config_like(base, cfg)
    res = effective.resolve(like, os.environ)
    pinned = {"models.embedding", "models.reranker", *(f"indexer.{k}" for k in like["indexer"]),
              *(f"search.{k}" for k in like["search"])}
    pinned |= {setting for key, (setting, _e) in MODEL_PINS.items() if cfg.get(key)}
    out = []
    for st in res["stages"]:
        if st["id"].startswith("S") or st["id"] == "8":
            continue
        rows = []
        for r in st["settings"]:
            if r["id"] in ("indexer.jobs", "indexer.auto_publish"):
                continue
            src = r["source"]
            if src == "config":
                src = "experiment" if r["id"] in pinned else "production config"
            rows.append({**r, "source": src})
        out.append({**st, "settings": rows})
    return {"ok": True, "stages": out, "errors": res["errors"], "overrides": res["overrides"],
            "note": "8 Publish does not exist in the playground: nothing is published, switched or served",
            "stage_ids": [s.id for s in stg.INDEXING if s.id != "8"]}


def build_index(base: Paths, name: str, *, jobs: int = 1, rebuild: bool = False,
                wipe: bool = False, force_md: bool = False,
                progress: Any = None, stage_log: Any = None) -> dict[str, Any]:
    from .. import locations
    from .embedding import make_embedder, prepare_environment

    exp = _exp(base, name)
    cfg = get_config(base, name)
    ensure_dirs(exp)
    try:
        plan = locations.plan_scan(exp)
    except locations.LocationError as exc:
        if str(exc) == locations.NO_LOCATIONS:
            raise PlaygroundError(f"no source folders are registered for experiment {name!r}: "
                                  f"rag-search playground source {name} add FOLDER") from exc
        raise PlaygroundError(str(exc)) from exc
    if not plan.sources:
        raise PlaygroundError("no supported documents in the experiment's source folders: "
                              + ", ".join(f for _n, f in plan.roots.locations))
    with _experiment_env(base, cfg):
        prepare_environment()
        embedder = make_embedder(cfg["embedding_model"])
        return run_plan(exp, plan, jobs=jobs, rebuild=rebuild, wipe=wipe,
                                force_md=force_md, chunk_size=int(cfg["chunk_size"]),
                                chunk_overlap=int(cfg["chunk_overlap"]), embedder=embedder,
                                model=cfg["embedding_model"], progress=progress, stage_log=stage_log)


# ── loading the engine directly from workspace/index (no serving/current) ─────

def _build_engine(exp: Paths, cfg: dict[str, Any], *, rerank: bool | None = None, progress: Any = None):
    from .embedding import make_embedder, make_reranker, prepare_environment
    from .search import SearchEngine

    prepare_environment()
    emb_model = cfg["embedding_model"]
    rr_model = cfg["rerank_model"]
    want_rerank = bool(cfg.get("rerank", True)) if rerank is None else rerank
    if progress:
        progress({"phase": "load", "done": 0, "total": 1,
                  "message": f"loading {emb_model}" + (f" and {rr_model}" if want_rerank else "")})
    embedder = make_embedder(emb_model)
    reranker = make_reranker(rr_model) if want_rerank else None
    engine = SearchEngine(exp, embedder=embedder, reranker=reranker, rerank=want_rerank)
    engine.model = emb_model
    engine.load_models()
    if progress:
        progress({"phase": "load", "done": 1, "total": 1, "message": "models loaded"})
    return engine, emb_model, (rr_model if want_rerank else None)


def _load_collections(engine: Any, exp: Paths) -> list[str]:
    from .search import Generation, load_index

    gen = Generation()
    names: list[str] = []
    if exp.index.is_dir():
        for d in sorted(exp.index.iterdir()):
            if d.is_dir() and (d / ALL_DIR).is_dir():
                gen.indexes[d.name] = load_index(d / ALL_DIR)
                names.append(d.name)
    gen.catalog = {"collections": [{"collection": n} for n in names]}
    engine.gen = gen
    return names


def _load_engine_and_index(base: Paths, name: str, *, rerank: bool | None = None, progress: Any = None):
    exp = get_playground_paths(base, name)
    if not exp.home.is_dir():
        raise PlaygroundError(f"no such experiment: {name} (rag-search playground create {name})")
    cfg = get_config(base, name)
    with _experiment_env(base, cfg):          # hardware settings of the Models section apply to the models loaded here
        engine, emb_model, rr_model = _build_engine(exp, cfg, rerank=rerank, progress=progress)
    names = _load_collections(engine, exp)
    if not names:
        raise PlaygroundError(f"no index built yet for {name!r} "
                              f"(rag-search playground index {name})")
    return exp, cfg, engine, emb_model, rr_model


# ── interactive search (same debug parameters as production `search`) ─────────

def search(base: Paths, name: str, query: str, *, top_k: int | None = None,
          collections: list[str] | None = None,
          stages: str | list[str] | None = None, retrieval_pool_n: int | None = None,
          rerank_pool_n: int | None = None, rrf_k: int | None = None,
          rerank: bool | None = None) -> dict[str, Any]:
    exp, cfg, engine, emb_model, rr_model = _load_engine_and_index(base, name, rerank=rerank)
    res = engine.search(
        query, top_k=top_k if top_k is not None else DEFAULT_TOP_K, collections=collections,
        stages=stages if stages is not None else cfg.get("stages"),
        retrieval_pool_n=retrieval_pool_n if retrieval_pool_n is not None else cfg.get("retrieval_pool"),
        rerank_pool_n=rerank_pool_n if rerank_pool_n is not None else cfg.get("rerank_pool"),
        rrf_k=rrf_k if rrf_k is not None else cfg.get("rrf_k"))
    res["models"] = {"embedding": emb_model, "reranker": rr_model}
    return res


# ── benchmarking ────────────────────────────────────────────────────────────

def _load_queries(path: Path) -> list[dict[str, Any]]:
    out = []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            obj = json.loads(line)
        except ValueError as exc:
            raise PlaygroundError(f"{path}:{i}: invalid JSON ({exc})") from exc
        if not isinstance(obj, dict) or not obj.get("query"):
            raise PlaygroundError(f"{path}:{i}: expected {{\"query\": ..., \"relevant\": [...]}}")
        obj.setdefault("relevant", [])
        out.append(obj)
    return out


def _match_key(d: dict[str, Any]) -> tuple[str, str]:
    """A label {"file": "kb.pdf", "page": 4} and a search hit (whose own "file" is the stem,
    e.g. "kb" -- "source" is the full name) both resolve to the same key here."""
    name = d.get("source") or d.get("file", "")
    return (str(name).strip(), str(d.get("page", "")).strip())


def _score_query(hits: list[dict[str, Any]], relevant: list[dict[str, Any]]) -> tuple[int | None, list[int]]:
    want = {_match_key(r) for r in relevant}
    flags: list[int] = []
    hit_rank = None
    for i, h in enumerate(hits, 1):
        is_rel = _match_key(h) in want
        flags.append(1 if is_rel else 0)
        if is_rel and hit_rank is None:
            hit_rank = i
    return hit_rank, flags


def _ndcg(flags: list[int], n_relevant: int) -> float:
    dcg = sum(r / math.log2(i + 2) for i, r in enumerate(flags))
    ideal_n = min(max(n_relevant, 0), len(flags))
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_n))
    return round(dcg / idcg, 4) if idcg else 0.0


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = min(len(s) - 1, max(0, int(round(p / 100 * (len(s) - 1)))))
    return round(s[idx], 1)


def bench(base: Paths, name: str, *, queries_path: str | None = None, k: int = 5,
         stages: str | list[str] | None = None, retrieval_pool_n: int | None = None,
         rerank_pool_n: int | None = None, rrf_k: int | None = None,
         rerank: bool | None = None, label: str | None = None, progress: Any = None) -> dict[str, Any]:
    exp, cfg, engine, emb_model, rr_model = _load_engine_and_index(base, name, rerank=rerank, progress=progress)
    qpath = Path(queries_path).expanduser() if queries_path else exp.home / "bench" / "queries.jsonl"
    if not qpath.exists():
        raise PlaygroundError(f"no labeled queries at {qpath} -- copy "
                              f"{exp.home / 'bench' / 'queries.jsonl.example'} to queries.jsonl "
                              "and fill in real questions/answers for your sample documents")
    queries = _load_queries(qpath)
    if not queries:
        raise PlaygroundError(f"{qpath} has no queries")

    eff_stages = stages if stages is not None else cfg.get("stages")
    eff_pool = retrieval_pool_n if retrieval_pool_n is not None else cfg.get("retrieval_pool")
    eff_rpool = rerank_pool_n if rerank_pool_n is not None else cfg.get("rerank_pool")
    eff_rrfk = rrf_k if rrf_k is not None else cfg.get("rrf_k")

    per_query = []
    recalls, mrrs, ndcgs, latencies = [], [], [], []
    for qi, q in enumerate(queries, 1):
        if progress:
            progress({"phase": "bench", "done": qi - 1, "total": len(queries), "current": q["query"][:80],
                      "message": f"query {qi}/{len(queries)}"})
        res = engine.search(q["query"], top_k=k, stages=eff_stages, retrieval_pool_n=eff_pool,
                            rerank_pool_n=eff_rpool, rrf_k=eff_rrfk)
        hits = res.get("results", [])
        hit_rank, flags = _score_query(hits, q["relevant"])
        recall = 1.0 if hit_rank else 0.0
        mrr = round(1.0 / hit_rank, 4) if hit_rank else 0.0
        ndcg = _ndcg(flags, len(q["relevant"]))
        total_ms = res.get("timing", {}).get("total_ms", 0)
        recalls.append(recall)
        mrrs.append(mrr)
        ndcgs.append(ndcg)
        latencies.append(total_ms)
        if progress:
            progress({"query": {"i": qi, "of": len(queries), "query": q["query"], "hit_rank": hit_rank,
                                "recall": recall, "ndcg": ndcg, "total_ms": total_ms,
                                "timing": {k2: v for k2, v in res.get("timing", {}).items() if k2.endswith("_ms")}}})
        per_query.append({"query": q["query"], "relevant": q["relevant"], "hit_rank": hit_rank,
                          "recall": recall, "mrr": mrr, "ndcg": ndcg, "total_ms": total_ms,
                          "top_result": {k2: hits[0].get(k2) for k2 in
                                        ("file", "page", "score", "rerank_score")} if hits else None})

    run_id = time.strftime("%Y%m%dT%H%M%S", time.gmtime()) + "-" + uuid.uuid4().hex[:6]
    record = {
        "run_id": run_id, "experiment": name, "label": label or "",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "queries_file": str(qpath), "n_queries": len(queries),
        "combo": {"embedding_model": emb_model, "rerank_model": rr_model,
                  "chunk_size": cfg["chunk_size"], "chunk_overlap": cfg["chunk_overlap"],
                  "stages": list(parse_stages(eff_stages)), "retrieval_pool": eff_pool,
                  "rerank_pool": eff_rpool, "rrf_k": eff_rrfk, "k": k},
        "metrics": {
            "recall_at_k": round(sum(recalls) / len(recalls), 4),
            "mrr": round(sum(mrrs) / len(mrrs), 4),
            "ndcg_at_k": round(sum(ndcgs) / len(ndcgs), 4),
            "latency_ms": {"mean": round(sum(latencies) / len(latencies), 1),
                          "p50": _percentile(latencies, 50), "p95": _percentile(latencies, 95)},
        },
        "per_query": per_query,
    }
    runs_dir = exp.home / "bench" / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(runs_dir / f"{run_id}.json", record)
    if progress:
        progress({"phase": "bench", "done": len(queries), "total": len(queries), "message": "finished"})
    return record


def compare(base: Paths, name: str) -> list[dict[str, Any]]:
    exp = get_playground_paths(base, name)
    runs_dir = exp.home / "bench" / "runs"
    if not runs_dir.is_dir():
        return []
    out = []
    for f in sorted(runs_dir.glob("*.json")):
        rec = read_json(f)
        if isinstance(rec, dict) and rec.get("run_id"):
            out.append({k: v for k, v in rec.items() if k != "per_query"})
    return out
