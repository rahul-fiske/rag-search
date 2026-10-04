"""Search daemon: models + every published index resident in memory, always on.

Actions: ping, search, grep, list, reload, status, shutdown.

Startup order: the socket is bound first, then models load in a background thread, so
`ping` answers immediately (state = loading_models -> loading_index -> ready) and
`list`/`grep` work while the models are still loading.  A new published generation
is loaded off to the side and swapped in without a pause; the daemon also notices a
changed `serving/current` link by itself, so a lost `reload` message is harmless.
"""

from __future__ import annotations

import contextlib
import logging
import os
import socket
import sys
import threading
import time
from typing import Any

from .. import policy, protocol
from ..catalog import collection_names, list_view, live_catalog
from ..config import ConfigStore
from ..grep import grep_isolated
from ..paths import DEFAULT_TOP_K, Paths, get_paths, parse_collections
from ..effective import ambient, settings_env
from ..spec import parse_stages
from .daemon_base import LOG_DATE_FORMAT, DaemonBase, process_memory

log = logging.getLogger("rag_search.search_daemon")
WATCH_SECONDS = 5.0


def apply_model_env(mcfg: dict[str, Any]) -> None:
    """Model-runtime tunables (embed_batch, max_seq, dtype, rerank_batch, rerank_max_len, device)
    become environment variables before the embedder/reranker are constructed -- ``setdefault``,
    so an actual ``RAG_SEARCH_*`` the daemon was started with always wins.  0/"" means "no
    override".  Called once at daemon construction: the embedder/reranker are built once and kept
    warm, so a config change here needs `rag-search daemon restart` to take effect."""
    for name, value in settings_env({"models": mcfg}, {}, ("models",)).items():
        os.environ.setdefault(name, value)


class SearchDaemon(DaemonBase):
    kind = "search"

    def __init__(self, paths: Paths, engine: Any = None, idle_exit_seconds: int | None = None,
                 prewarm: bool | None = None):
        self.cfgs = ConfigStore(paths)
        self.access = policy.AccessStore(paths)
        self.seen: dict[str, float] = {}     # client -> last request (this daemon run)
        cfg = self.cfgs.get()
        scfg = cfg["search"]
        self.env_overrides = ambient()       # before apply_model_env adds config values to the environment
        super().__init__(paths, scfg["idle_exit_seconds"] if idle_exit_seconds is None
                         else idle_exit_seconds)
        if engine is None:
            apply_model_env(cfg["models"])   # before the embedder/reranker are constructed below
            from .search import SearchEngine  # heavy import (numpy) only when actually serving

            engine = SearchEngine(paths)
        self.engine = engine
        self.prewarm = scfg["prewarm"] if prewarm is None else prewarm
        self.ready = threading.Event()
        self.index_error = ""
        self._search_lock = threading.Lock()
        self._reload_lock = threading.Lock()
        self.watch_seconds = WATCH_SECONDS
        self.retry_seconds = 30.0  # first retry delay after a failed model load
        self.warmup: dict[str, Any] = {"started_at": round(time.time(), 3)}
        self.last_reload: dict[str, Any] = {}
        self._mem_cache: tuple[float, tuple, dict[str, Any]] = (0.0, (), {})

    # ── lifecycle ───────────────────────────────────────────────────────────
    def on_start(self) -> None:
        threading.Thread(target=self._boot, name="boot", daemon=True).start()
        threading.Thread(target=self._watch_generation, name="gen-watch", daemon=True).start()

    def _boot(self) -> None:
        t0 = time.monotonic()
        delay = self.retry_seconds
        while not self.stop.is_set():
            self.state = "loading_models"
            try:
                t_models = time.monotonic()
                self.engine.load_models()
                self.warmup["models_s"] = round(time.monotonic() - t_models, 1)
                break
            except Exception as exc:  # noqa: BLE001
                # e.g. the first-run download failed: report it, then keep retrying with
                # backoff so that the daemon heals itself once the network is back
                self.state = "error"
                self.error = f"{type(exc).__name__}: {exc}"
                log.exception("model load failed (retrying in %.0fs)", delay)
                self.ready.set()          # requests get 'model_error' instead of waiting
                if self.stop.wait(delay):
                    return
                delay = min(delay * 2, 600.0)
                self.ready.clear()        # next attempt: callers wait / see warming_up
        if self.stop.is_set():
            return
        self.error = ""
        self.state = "loading_index"
        t_index = time.monotonic()
        try:
            self.reload(prewarm=self.prewarm)
        except Exception as exc:  # noqa: BLE001
            self.index_error = f"{type(exc).__name__}: {exc}"
            log.exception("initial index load failed")
        self.warmup["index_s"] = round(time.monotonic() - t_index, 1)
        self.warmup["total_s"] = round(time.monotonic() - t0, 1)
        self.warmup["ready_at"] = round(time.time(), 3)
        self.state = "ready"
        log.info("ready after %.1fs (generation %s)", time.monotonic() - t0,
                 self.engine.generation)
        self.ready.set()

    def _watch_generation(self) -> None:
        while not self.stop.wait(self.watch_seconds):
            if not self.ready.is_set():
                continue
            live = self.paths.current_gen()
            live_no = None
            if live is not None:
                cat = live_catalog(self.paths)
                live_no = cat.get("generation")
            if live_no != self.engine.generation and (live_no or self.engine.generation):
                try:
                    self.reload(prewarm=self.prewarm)
                except Exception as exc:  # noqa: BLE001
                    self.index_error = f"{type(exc).__name__}: {exc}"
                    log.warning("auto-reload failed: %s", exc)
                    # avoid retrying a broken generation every tick
                    self.stop.wait(30)

    def _swap_reranker(self) -> dict[str, Any]:
        """Pick up a changed reranker setting: load the new one off to the side, then swap.
        A failure keeps the working reranker and is reported (searches are never disturbed)."""
        prepare = getattr(self.engine, "prepare_reranker", None)
        if not callable(prepare):
            return {}
        try:
            new = prepare()
        except Exception as exc:  # noqa: BLE001
            log.warning("reranker switch failed: %s", exc)
            return {"reranker_error": f"{type(exc).__name__}: {exc}"}
        if new is None:
            return {}
        with self._search_lock:
            self.engine.install_reranker(new)
        name = getattr(new, "name", "")
        log.info("reranker switched to %s", name)
        return {"reranker": name}

    def reload(self, prewarm: bool = True) -> dict[str, Any]:
        """Load the live generation (slow part outside the search lock), then swap.
        Also applies a changed reranker setting."""
        with self._reload_lock:
            t0 = time.monotonic()
            extra = self._swap_reranker()
            before = self.engine.generation
            gen = self.engine.prepare_generation(prewarm=prewarm)
            if gen is not None and gen.number == before:
                return {"generation": before, "changed": False, **extra}
            with self._search_lock:
                self.engine.install(gen)
            self.index_error = ""
            log.info("generation %s -> %s", before, self.engine.generation)
            out = {"generation": self.engine.generation, "changed": True,
                   "previous": before, "reused": gen.reused if gen else [],
                   "loaded": gen.loaded if gen else [], **extra}
            self.last_reload = {"at": round(time.time(), 3), "seconds": round(time.monotonic() - t0, 1),
                                "generation": out["generation"], "reused": out["reused"],
                                "loaded": out["loaded"]}
            return out

    def _warmup_view(self) -> dict[str, Any]:
        """warming_up / warm / error, and how long the warm-up took (or has taken so far)."""
        w = dict(self.warmup)
        if self.state == "ready":
            w["status"] = "warm"
        elif self.state == "error":
            w["status"] = "error"
        else:
            w["status"] = "warming_up"
            w["phase"] = self.state                       # loading_models | loading_index
            w["elapsed_s"] = round(time.time() - w["started_at"], 1)
        return w

    def _memory_view(self) -> dict[str, Any]:
        """Resident memory of this daemon; cached for 2 s (a `ps` call on macOS)."""
        stamp, key, cached = self._mem_cache
        now_key = (self.state, self.engine.generation)      # a new state/generation changes it
        if cached and key == now_key and time.monotonic() - stamp < 2.0:
            return cached
        mem = process_memory()
        info = getattr(self.engine, "memory_info", None)
        if callable(info):
            with contextlib.suppress(Exception):
                mem.update(info())
        self._mem_cache = (time.monotonic(), now_key, mem)
        return mem

    def _models_view(self) -> dict[str, str]:
        info = getattr(self.engine, "models_info", None)
        try:
            return info() if callable(info) else {}
        except Exception:  # noqa: BLE001
            return {}

    def ping_info(self) -> dict[str, Any]:
        cat = self.engine.gen.catalog
        return {"generation": self.engine.generation,
                "collections": len(cat.get("collections", [])),
                "chunks": sum(c.get("chunks", 0) for c in cat.get("collections", [])),
                "prewarm": self.prewarm, "index_error": self.index_error,
                "error": self.error if self.state == "error" else "",
                "config_error": self.cfgs.error, "access_error": self.access.error,
                "clients_seen": dict(self.seen),
                "warm": self.state == "ready", "warmup": self._warmup_view(),
                "last_reload": self.last_reload, "memory": self._memory_view(),
                "models": self._models_view(), "env_overrides": self.env_overrides}

    # ── actions ─────────────────────────────────────────────────────────────
    def dispatch(self, req: dict[str, Any], conn: socket.socket) -> dict[str, Any] | None:
        action, client = req["action"], req["client"]
        if action in ("list", "grep", "search") and req.get("origin") != "ui":
            self.seen[client] = time.time()      # (the dashboard's "view as" searches do not count)
        rules = self.access.get()
        if action == "list":
            return {"ok": True, "result": list_view(self.paths, rules, client,
                                                     full=bool(req.get("full", False)))}
        if action == "grep":
            return self._grep(req, rules, client)
        if action == "search":
            return self._search(req, rules, client)
        if action == "reload":
            if not self.ready.wait(max(0.0, min(float(req.get("wait_s", 30)), 300.0))):
                return protocol.error(protocol.WARMING_UP, "daemon is still starting")
            try:
                return {"ok": True, **self.reload(prewarm=self.prewarm)}
            except Exception as exc:  # noqa: BLE001
                from .search import ModelMismatch

                code = protocol.MODEL_MISMATCH if isinstance(exc, ModelMismatch) \
                    else protocol.INTERNAL
                return protocol.error(code, f"{type(exc).__name__}: {exc}")
        return protocol.error(protocol.BAD_REQUEST, f"unknown action: {action!r}")

    def _scope(self, req: dict[str, Any], rules: policy.Rules, client: str,
               existing: list[str]) -> tuple[list[str], dict[str, Any] | None]:
        try:
            requested = parse_collections(req.get("collections"))
        except ValueError as exc:
            return [], protocol.error(protocol.BAD_REQUEST, str(exc))
        scope, err = policy.resolve_scope(rules, client, requested, existing)
        if err:
            return [], protocol.error(protocol.BAD_REQUEST, err)
        return scope, None

    def _grep(self, req: dict[str, Any], rules: policy.Rules, client: str) -> dict[str, Any]:
        cat = live_catalog(self.paths)
        scope, err = self._scope(req, rules, client, collection_names(cat))
        if err:
            return err
        t0 = time.perf_counter()
        res = grep_isolated(self.paths.live_markup(), str(req.get("pattern", "")), scope,
                            int(req.get("context_lines", 2)), int(req.get("max_matches", 20)))
        res.setdefault("timing", {})["server_ms"] = int((time.perf_counter() - t0) * 1000)
        return {"ok": True, "result": res}

    def _search(self, req: dict[str, Any], rules: policy.Rules, client: str) -> dict[str, Any]:
        t_req = time.perf_counter()
        wait = max(0.0, min(float(req.get("wait_s", 30)), 300.0))
        if not self.ready.wait(wait):
            return protocol.error(protocol.WARMING_UP, "search engine is still loading its models",
                                  state=self.state,
                                  elapsed_s=int(time.monotonic() - self.started))
        if self.state == "error":
            return protocol.error(protocol.MODEL_ERROR, self.error)
        gen_collections = self.engine.gen.collections()
        scope, err = self._scope(req, rules, client, gen_collections)
        if err:
            return err
        if not scope:
            return {"ok": True, "result": {"results": [], "note": "no searchable collections "
                    "for this request (nothing published, or none is available to this client)"}}
        # Fresh on every request (ConfigStore only re-reads the file when it changed): a caller
        # that does not ask for a specific value gets whatever an administrator has set as the
        # production default (`rag-search config set` / the dashboard's Settings tab), and from
        # there the built-in formula/constant -- so this takes effect immediately, no restart.
        scfg = self.cfgs.get()["search"]
        req_stages = req.get("stages")
        try:
            stages = parse_stages(req_stages if req_stages is not None else (scfg.get("stages") or None))
        except ValueError as exc:
            return protocol.error(protocol.BAD_REQUEST, str(exc))
        pool_kw: dict[str, Any] = {}
        for key, dest in (("retrieval_pool", "retrieval_pool_n"), ("rerank_pool", "rerank_pool_n"),
                          ("rrf_k", "rrf_k")):
            v = req.get(key)
            if v is None:
                v = scfg.get(key) or None   # 0 in config means "no override -- use the formula"
            if v is None:
                continue
            try:
                pool_kw[dest] = int(v)
            except (TypeError, ValueError):
                return protocol.error(protocol.BAD_REQUEST, f"{key} must be a whole number")
        top_k = req.get("top_k")
        if top_k is None:
            top_k = scfg.get("top_k") or DEFAULT_TOP_K
        try:
            top_k = int(top_k)
        except (TypeError, ValueError):
            return protocol.error(protocol.BAD_REQUEST, "top_k must be a whole number")
        # No daemon-wide lock: the engine takes a consistent snapshot of the generation and its
        # models per query and serialises only the model calls, so concurrent clients are not
        # queued behind one another's BM25/vector/fusion work (or a slow many-collection query).
        queue_ms = int((time.perf_counter() - t_req) * 1000)       # warm-up wait
        result = self.engine.search(str(req.get("query", "")), top_k,
                                    scope, stages=stages, **pool_kw)
        result.setdefault("timing", {}).update(
            queue_ms=queue_ms, server_ms=int((time.perf_counter() - t_req) * 1000))
        return {"ok": True, "result": result}


def main() -> int:
    logging.basicConfig(level=logging.INFO, stream=sys.stderr,
                        format="%(asctime)s [search-daemon] %(levelname)s %(message)s",
                        datefmt=LOG_DATE_FORMAT)
    from .embedding import prepare_environment

    prepare_environment()
    return SearchDaemon(get_paths()).run()


if __name__ == "__main__":
    raise SystemExit(main())
