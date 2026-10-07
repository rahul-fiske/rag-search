"""Indexer daemon: a light supervisor that guarantees ONE indexing run at a time.

Stdlib only (never imports numpy/torch/docling): the heavy work runs in a killable
worker subprocess (`core/worker.py`) that writes only into ``indexer_workspace/``.

Actions
  start    {mode: new|all, path, rebuild, force_md, restart}
           idempotent: while a run is active it just reports it, unless restart=true,
           which kills the running worker (whole process group) and starts over.
           Finished documents are skipped by SHA-256, so a restart is cheap.
  cancel   stop the active run
  status   {job_id?}  summary of the active (or last) run + recent history
  follow   {job_id?}  streamed JSON lines: status, progress..., then {"event":"end"}
  publish  publish the workspace now and tell the search daemon (refused while running)

After a successful run (config ``indexer.auto_publish``) the daemon publishes a new
serving generation and sends ``reload`` to the search daemon.
"""

from __future__ import annotations

import json
import logging
import os
import secrets
import signal
import socket
import subprocess
import sys
import threading
import time
from typing import Any

from .. import api, locations, protocol, stages
from ..config import ConfigStore, effective_jobs
from ..effective import ambient, settings_env
from ..jobs import ACTIVE, all_records, events_file, job_file, now, read_record, view
from ..paths import SUPPORTED_EXTENSIONS, Paths, detached_start, get_paths, write_json_atomic
from . import stallwatch
from .daemon_base import LOG_DATE_FORMAT, DaemonBase
from .worker import EXIT_BUSY

log = logging.getLogger("rag_search.indexer_daemon")

KEEP_JOBS = 30
MODES = ("new", "all")


def new_job_id() -> str:
    return time.strftime("%Y%m%d-%H%M%S") + "-" + secrets.token_hex(2)


def _pid_is_worker(pid: int) -> bool:
    try:
        cmd = subprocess.run(["ps", "-ww", "-p", str(pid), "-o", "command="], capture_output=True,
                             text=True, timeout=5).stdout
    except (OSError, subprocess.SubprocessError):
        return False
    return "rag_search.core.worker" in cmd


def _kill_group(pid: int, grace: float = 5.0) -> None:
    try:
        pgid = os.getpgid(pid)
    except OSError:
        return
    for sig, wait in ((signal.SIGTERM, grace), (signal.SIGKILL, 3.0)):
        try:
            os.killpg(pgid, sig)
        except OSError:
            return
        end = time.monotonic() + wait
        while time.monotonic() < end:
            try:
                os.killpg(pgid, 0)
            except OSError:
                return
            time.sleep(0.05)


def _secs(value: Any) -> str:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "?"
    return f"{v:.1f} s" if v < 120 else f"{int(v // 60)} min {int(v % 60):02d} s"


def _log_event(jid: str, ev: dict[str, Any]) -> None:
    """One readable line in the daemon log for each document entering or leaving a stage (this
    is what the dashboard's System > Indexer log shows while a run is going)."""
    kind = ev.get("event")
    if kind == "start":
        log.info("run %s: looking at %s file(s), mode %s", jid, ev.get("files", "?"), ev.get("mode", "?"))
    elif kind == "stage":
        f, st, status = ev.get("file", "?"), ev.get("stage", "?"), ev.get("status", "")
        tag = f"{stages.named(st):<14}"
        if st == "convert" and status == "start":
            log.info("%s  %s started", f, tag)
        elif st == "convert":
            log.info("%s  %s done in %s (%s)", f, tag, _secs(ev.get("seconds")), ev.get("how", ""))
        elif st == "fingerprint":
            log.info("%s  %s %s", f, tag, "unchanged: skipped" if ev.get("unchanged") else "changed")
        elif st == "profile":
            log.info("%s  %s %s page(s) in %s", f, tag, ev.get("pages", "?"), _secs(ev.get("seconds")))
        elif st == "chunk":
            log.info("%s  %s %s chunks in %s", f, tag, ev.get("chunks", "?"), _secs(ev.get("seconds")))
        elif st == "embed" and status == "start":
            log.info("%s  %s started (%s chunks)", f, tag, ev.get("chunks", "?"))
        elif st == "embed":
            log.info("%s  %s done in %s", f, tag, _secs(ev.get("seconds")))
        elif st == "write":
            log.info("%s  %s %s", f, tag, ev.get("part", ""))
    elif kind == "doc":
        f = f"{ev.get('collection', '')}/{ev.get('path') or ev.get('source', '?')}".lstrip("/")   # with its folder, as the stage lines
        if ev.get("status") == "indexed":
            log.info("%s  INDEXED  %s chunks, total %s (convert %s, embed %s)", f, ev.get("chunks", "?"),
                     _secs(ev.get("total_s")), _secs(ev.get("convert_s")), _secs(ev.get("embed_s")))
        elif ev.get("status") == "no_text":
            log.info("%s  SKIPPED  no text to index", f)
        elif ev.get("status") == "error":
            log.warning("%s  FAILED   %s", f, " ".join(str(ev.get("message", "")).split())[:500])


def _describe_exit(rc: int, job_id: str) -> str:
    """Say what happened to a worker that ended without a result. A negative code is the
    signal that killed it (SIGKILL is what the system sends when it runs out of memory)."""
    if rc < 0:
        try:
            name = signal.Signals(-rc).name
        except ValueError:
            name = f"signal {-rc}"
        hint = (" - usually the system stopping it for lack of memory; try RAG_SEARCH_JOBS=1"
                if -rc == signal.SIGKILL else "")
        return f"worker was killed by {name}{hint} (see {job_id}.log)"
    return f"worker exited with code {rc} (see {job_id}.log)"


class IndexerDaemon(DaemonBase):
    kind = "indexer"

    def __init__(self, paths: Paths, idle_exit_seconds: int | None = None):
        self.cfgs = ConfigStore(paths)
        icfg = self.cfgs.get()["indexer"]
        super().__init__(paths, icfg["idle_exit_seconds"] if idle_exit_seconds is None
                         else idle_exit_seconds)
        self.lock = threading.RLock()
        self.job: dict[str, Any] | None = None       # active job record
        self.thread: threading.Thread | None = None
        self.proc: subprocess.Popen | None = None
        self.cancel_flag = False
        self.last: dict[str, Any] | None = None       # last finished record
        self.worker_module = "rag_search.core.worker"  # tests may swap in a stub

    # ── lifecycle ───────────────────────────────────────────────────────────
    def on_start(self) -> None:
        self._cleanup_orphans()
        self.last = self._load_last()
        self.state = "ready"

    def on_stop(self) -> None:
        with self.lock:
            active = self.job is not None
        if active:
            self._cancel("interrupted", "indexer daemon stopped"
                         + (f": {self.stop_reason}" if self.stop_reason else ""))

    def is_busy(self) -> bool:
        return self.job is not None

    def ping_info(self) -> dict[str, Any]:
        with self.lock:
            j = self.job
            return {"running": j is not None, "job_id": j["id"] if j else None,
                    "last_status": (self.last or {}).get("status"),
                    "config_error": self.cfgs.error,
                    "env_overrides": ambient()}       # what wins over config.json in this daemon's workers

    # ── persistence ─────────────────────────────────────────────────────────
    def _save(self, rec: dict[str, Any]) -> None:
        write_json_atomic(job_file(self.paths, rec["id"]), rec)

    def _all_records(self) -> list[dict[str, Any]]:
        return all_records(self.paths)

    def _load_last(self) -> dict[str, Any] | None:
        for r in self._all_records():
            if r.get("status") not in ACTIVE:
                return r
        return None

    def _cleanup_orphans(self) -> None:
        """A previous daemon may have died while its worker kept running (own session)."""
        for rec in self._all_records():
            if rec.get("status") not in ACTIVE:
                continue
            pid = rec.get("pid")
            if pid and _pid_is_worker(int(pid)):
                log.warning("killing orphaned worker %s of job %s", pid, rec["id"])
                _kill_group(int(pid))
            rec.update(status="interrupted", finished_at=now(),
                       error="indexer daemon restarted while this run was active")
            self._save(rec)
        self._gc_jobs()

    def _gc_jobs(self) -> None:
        recs = self._all_records()
        for rec in recs[KEEP_JOBS:]:
            for f in (job_file(self.paths, rec["id"]), events_file(self.paths, rec["id"]),
                      self.paths.jobs / f"{rec['id']}.log"):
                try:
                    f.unlink()
                except OSError:
                    pass

    # ── job control ─────────────────────────────────────────────────────────
    def _validate_spec(self, req: dict[str, Any]) -> tuple[dict[str, Any] | None, str]:
        mode = str(req.get("mode", "new"))
        if mode not in MODES:
            return None, f"mode must be one of {MODES}"
        raw = str(req.get("path") or "").strip()
        from ..locations import NO_LOCATIONS, LocationError, resolve_target

        if not raw and not locations.names(self.paths):
            return None, NO_LOCATIONS
        if raw:
            try:
                resolve_target(self.paths, raw)    # a registered location, or a path inside one
            except LocationError as exc:
                msg = str(exc)
                return None, ("path " + msg) if msg.startswith("not found") else msg
        cfg = self.cfgs.get()
        icfg = cfg["indexer"]
        return {"mode": mode, "path": raw, "rebuild": bool(req.get("rebuild")),
                "force_md": bool(req.get("force_md")),
                "jobs": int(req.get("jobs") or effective_jobs(cfg)),
                # 0/missing -> worker.py's own DEFAULT_CHUNK_SIZE/DEFAULT_CHUNK_OVERLAP applies.
                # Read fresh here (job start), not at daemon boot: a config change is picked up
                # by the very next run, no daemon restart.
                "chunk_size": int(req.get("chunk_size") or icfg.get("chunk_size") or 0),
                "chunk_overlap": int(req.get("chunk_overlap") or icfg.get("chunk_overlap") or 0)}, ""

    def start(self, req: dict[str, Any]) -> dict[str, Any]:
        spec, err = self._validate_spec(req)
        if err or spec is None:
            return protocol.error(protocol.BAD_REQUEST, err)
        restart = bool(req.get("restart"))
        with self.lock:
            if self.job is not None:
                if not restart:
                    return {"ok": True, "started": False, "already_running": True,
                            "job": view(self.job)}
            restarted = self.job is not None
        if restarted:
            self._cancel("cancelled", "restarted by a new request", wait=True)
        with self.lock:
            if self.job is not None:  # lost a race with a concurrent start
                return {"ok": True, "started": False, "already_running": True,
                        "job": view(self.job)}
            rec = {"id": new_job_id(), "status": "queued", "mode": spec["mode"],
                   "path": spec["path"], "spec": spec, "created_at": now(),
                   "started_at": None, "finished_at": None, "pid": None,
                   "progress": {"phase": "queued"}, "summary": None, "error": "",
                   "publish": None, "search_reload": None}
            self.job = rec
            self.cancel_flag = False
            self._save(rec)
            self.thread = threading.Thread(target=self._supervise, args=(rec,),
                                           name=f"job-{rec['id']}", daemon=True)
            self.thread.start()
            return {"ok": True, "started": True, "restarted": restarted, "job": view(rec)}

    def _cancel(self, status: str, reason: str, wait: bool = True) -> bool:
        with self.lock:
            rec, proc, th = self.job, self.proc, self.thread
            if rec is None:
                return False
            self.cancel_flag = True
            rec["_final_status"], rec["_final_error"] = status, reason
        if proc is not None and proc.poll() is None:
            _kill_group(proc.pid)
        if wait and th is not None and th is not threading.current_thread():
            th.join(30)
        return True

    # ── supervisor thread ───────────────────────────────────────────────────
    def _update(self, rec: dict[str, Any], **kw: Any) -> None:
        with self.lock:
            rec.update(kw)
            self._save({k: v for k, v in rec.items() if not k.startswith("_")})

    def _worker_env(self) -> dict[str, str]:
        """The worker subprocess's environment: the parent's own environment first (so an actual
        ``RAG_SEARCH_OCR=...`` etc. the daemon was started with always wins), then config.json's
        ``indexer``/``models`` tunables that have an environment variable of their own -- read
        fresh here (job start), so a config change reaches the very next run without a daemon
        restart.  0/"" in config means "no override"; docling_convert.py / embedding.py's own
        built-in default then applies exactly as if the variable were never set."""
        return settings_env(self.cfgs.get(), dict(os.environ, RAG_SEARCH_HOME=str(self.paths.home)))

    def _supervise(self, rec: dict[str, Any]) -> None:
        jid = rec["id"]
        ev_path = events_file(self.paths, jid)
        ev_path.write_text("")
        offset = 0
        result: dict[str, Any] | None = None
        err = ""
        try:
            out = open(self.paths.jobs / f"{jid}.log", "ab")
            env = self._worker_env()
            proc = subprocess.Popen([sys.executable, "-m", self.worker_module, jid],
                                    stdin=subprocess.DEVNULL, stdout=out, stderr=out,
                                    start_new_session=True, close_fds=True,
                                    **detached_start(self.paths.home, env))
            out.close()
            with self.lock:
                self.proc = proc
            self._update(rec, status="running", started_at=now(), pid=proc.pid)
            log.info("run %s started (worker pid %d)", jid, proc.pid)
            if self.cancel_flag:  # cancelled in the instant before the pid was known
                _kill_group(proc.pid)

            def drain() -> None:
                nonlocal offset, result, err
                with open(ev_path, "rb") as fh:
                    fh.seek(offset)
                    data = fh.read()
                    end = data.rfind(b"\n")
                    if end < 0:
                        return
                    offset += end + 1
                for line in data[:end].split(b"\n"):
                    try:
                        ev = json.loads(line)
                    except ValueError:
                        continue
                    if ev.get("event") in ("start", "stage", "doc"):
                        _log_event(jid, ev)
                    if ev.get("event") == "progress":
                        self._update(rec, progress={k: v for k, v in ev.items()
                                                    if k not in ("ts", "event")})
                    elif ev.get("event") == "result":
                        result = ev.get("summary")
                    elif ev.get("event") == "error":
                        err = ev.get("error", "")

            # The run's own stall watch stops a conversion process that goes silent (stallwatch.py).  This is the second
            # line: a run whose event log has not grown at all for twice that limit -- its main process hangs, in a
            # phase the first watch does not cover -- is stopped and reported, instead of staying "running" for ever.
            run_limit = stallwatch.RUN_FACTOR * stallwatch.limit_s(env)
            quiet, tick = 0.0, time.monotonic()
            while proc.poll() is None:
                before = offset
                drain()
                now_t = time.monotonic()
                quiet = 0.0 if offset != before else quiet + min(now_t - tick, 5.0)   # capped: a sleeping laptop is not a stall
                tick = now_t
                if run_limit > 0 and quiet >= run_limit and not self.cancel_flag:
                    log.error("run %s wrote nothing for %d min: stopping it", jid, round(quiet / 60))
                    self._cancel("failed", f"stalled: the run reported nothing for {round(quiet / 60)} minutes and was "
                                 "stopped. Start it again: documents and pages already done are kept "
                                 "(RAG_SEARCH_STALL_TIMEOUT sets the limit).", wait=False)
                time.sleep(0.25)
            drain()
            rc = proc.returncode
        except Exception as exc:  # noqa: BLE001
            log.exception("supervisor failed")
            rc, err = -1, f"{type(exc).__name__}: {exc}"
        self._finish(rec, rc, result, err)

    def _finish(self, rec: dict[str, Any], rc: int, result: dict[str, Any] | None,
                err: str) -> None:
        try:
            with self.lock:
                stopped = self.cancel_flag
                why = rec.get("_final_error") or rec.get("_final_status", "cancelled")
            if stopped:
                log.info("run %s stopped: %s", rec["id"], why)
            elif result is not None and rc == 0:
                log.info("run %s finished: %d indexed, %s unchanged, %d without text, %d error(s), %s",
                         rec["id"], result.get("indexed", 0), result.get("skipped_fresh", "?"),
                         len(result.get("no_text") or []), len(result.get("errors") or []),
                         _secs(result.get("elapsed_s")))
            else:
                log.warning("run %s ended without a result (rc %s)%s", rec["id"], rc,
                            f": {err}" if err else "")
        except Exception:  # noqa: BLE001 - logging must never break bookkeeping
            pass
        with self.lock:
            cancelled = self.cancel_flag
            final_status = rec.get("_final_status", "cancelled")
            final_err = rec.get("_final_error", "")
        if cancelled:
            self._update(rec, status=final_status, error=final_err, finished_at=now())
        elif result is not None and rc == 0:
            errors = result.get("errors") or []
            self._update(rec, status="partial" if errors else "succeeded", summary=result,
                         finished_at=now(), progress={"phase": "done"})
            if self.cfgs.get()["indexer"].get("auto_publish", True):
                self._auto_publish(rec)
        else:
            why = err or ("another indexing run holds the index lock (started outside the "
                          "daemon?)" if rc == EXIT_BUSY else
                          _describe_exit(rc, rec["id"]))
            self._update(rec, status="failed", error=why, finished_at=now())
        with self.lock:
            done = {k: v for k, v in rec.items() if not k.startswith("_")}
            self.last = done
            self.job = None
            self.proc = None
            self.touch()
        self._gc_jobs()

    def _auto_publish(self, rec: dict[str, Any]) -> None:
        t0 = time.time()
        self._update(rec, progress={"phase": "publish", "since": round(t0, 3)})    # publishing is a phase of the run, shown like the others
        pub = api.publish_and_reload(self.paths)
        self._update(rec, publish=pub.get("publish"), search_reload=pub.get("search_reload"),
                     publish_s=round(time.time() - t0, 1), progress={"phase": "done"})

    # ── request handling ────────────────────────────────────────────────────
    def dispatch(self, req: dict[str, Any], conn: socket.socket) -> dict[str, Any] | None:
        action = req["action"]
        if action == "start":
            return self.start(req)
        if action == "cancel":
            return self._do_cancel()
        if action == "status":
            return self._status(str(req.get("job_id") or ""), int(req.get("history", 0)))
        if action == "follow":
            self._follow(conn, str(req.get("job_id") or ""))
            return None
        if action == "publish":
            with self.lock:
                if self.job is not None:
                    return protocol.error(protocol.BUSY, "an indexing run is active; publishing "
                                          "is done automatically when it finishes")
            return {"ok": True, **api.publish_and_reload(self.paths)}
        return protocol.error(protocol.BAD_REQUEST, f"unknown action: {action!r}")

    def _do_cancel(self) -> dict[str, Any]:
        with self.lock:
            jid = self.job["id"] if self.job else None
        if not jid or not self._cancel("cancelled", "cancelled by request"):
            return {"ok": True, "cancelled": False, "note": "no indexing run is active"}
        return {"ok": True, "cancelled": True, "job_id": jid}

    def _find(self, job_id: str) -> dict[str, Any] | None:
        with self.lock:
            if self.job is not None and (not job_id or job_id == self.job["id"]):
                return self.job
            if not job_id:
                return self.last
        return read_record(self.paths, job_id)

    def _status(self, job_id: str, history: int) -> dict[str, Any]:
        rec = self._find(job_id)
        if job_id and rec is None:
            return protocol.error(protocol.BAD_REQUEST, f"no such job: {job_id}")
        out: dict[str, Any] = {"ok": True, "running": self.job is not None,
                               "job": view(rec) if rec else None,
                               "sources": locations.sources(self.paths),
                               "supported_extensions": sorted(SUPPORTED_EXTENSIONS)}
        if history:
            out["history"] = [view(r) for r in self._all_records()[:history]]
        return out

    def _follow(self, conn: socket.socket, job_id: str) -> None:
        conn.settimeout(None)
        rec = self._find(job_id)

        def send(obj: dict[str, Any]) -> bool:
            try:
                conn.sendall(protocol.encode(obj))
                return True
            except OSError:
                return False

        if rec is None:
            send(protocol.error(protocol.BAD_REQUEST, "no indexing job yet"))
            return
        jid = rec["id"]
        if not send({"event": "status", "job": view(rec)}):
            return
        if rec.get("status") not in ACTIVE:
            send({"event": "end", "job": view(rec)})
            return
        ev_path = events_file(self.paths, jid)
        try:
            offset = ev_path.stat().st_size  # only what happens from now on
        except OSError:
            offset = 0
        while not self.stop.is_set():
            try:
                with open(ev_path, "rb") as fh:
                    fh.seek(offset)
                    data = fh.read()
            except OSError:
                data = b""
            end = data.rfind(b"\n")
            if end >= 0:
                offset += end + 1
                for line in data[:end].split(b"\n"):
                    try:
                        ev = json.loads(line)
                    except ValueError:
                        continue
                    if ev.get("event") in ("progress", "doc") and not send(ev):
                        return
            with self.lock:
                finished = self.job is None or self.job["id"] != jid
            if finished:
                final = self._find(jid)
                send({"event": "end", "job": view(final) if final else None})
                return
            time.sleep(0.3)


def main() -> int:
    logging.basicConfig(level=logging.INFO, stream=sys.stderr,
                        format="%(asctime)s [indexer-daemon] %(levelname)s %(message)s",
                        datefmt=LOG_DATE_FORMAT)
    return IndexerDaemon(get_paths()).run()


if __name__ == "__main__":
    raise SystemExit(main())
