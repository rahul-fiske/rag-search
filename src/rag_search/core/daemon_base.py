"""Shared plumbing for the two daemons: single instance, socket server, idle exit.

Stdlib only (the indexer supervisor is built on this and must stay light).
Subclasses implement `dispatch(req, conn)`; return a dict to send it as the single
reply, or None after streaming lines to *conn* themselves.
"""

from __future__ import annotations

import fcntl
import json
import logging
import os
import signal
import socket
import subprocess
import sys
import threading
import time
from typing import Any

from .. import __version__, protocol
from ..paths import Paths, allow_cloud_files, ensure_dirs

log = logging.getLogger("rag_search.daemon")
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def process_memory() -> dict[str, int]:
    """Resident memory of this process in bytes: {"rss_bytes", "peak_rss_bytes"} (0 = unknown)."""
    rss = peak = 0
    try:
        with open("/proc/self/statm") as fh:                       # Linux
            rss = int(fh.read().split()[1]) * os.sysconf("SC_PAGE_SIZE")
    except (OSError, ValueError, IndexError):
        try:                                                        # macOS
            out = subprocess.run(["ps", "-o", "rss=", "-p", str(os.getpid())],
                                 capture_output=True, text=True, timeout=3).stdout.strip()
            rss = int(out) * 1024 if out else 0
        except (OSError, ValueError, subprocess.SubprocessError):
            rss = 0
    try:
        import resource

        ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        peak = ru if sys.platform == "darwin" else ru * 1024       # macOS: bytes, Linux: KiB
    except (ImportError, OSError, ValueError):
        peak = 0
    return {"rss_bytes": rss, "peak_rss_bytes": max(peak, rss)}


class DaemonBase:
    kind = ""

    def __init__(self, paths: Paths, idle_exit_seconds: int = 0):
        self.paths = paths
        self.idle_exit_seconds = int(idle_exit_seconds or 0)
        self.state = "starting"
        self.error = ""
        self.started = time.monotonic()
        self.last_activity = time.monotonic()
        self.requests = 0
        self.stop = threading.Event()
        self.stop_reason = ""        # why the daemon is shutting down (logged, and put on a cancelled run)
        # test-friendly knobs
        self.idle_poll = 15.0
        self.accept_timeout = 1.0

    # ── hooks ───────────────────────────────────────────────────────────────
    def on_start(self) -> None:  # called once the socket is bound
        pass

    def on_stop(self) -> None:  # called after the accept loop ends
        pass

    def is_busy(self) -> bool:  # True = never idle-exit right now
        return False

    def ping_info(self) -> dict[str, Any]:
        return {}

    def dispatch(self, req: dict[str, Any], conn: socket.socket) -> dict[str, Any] | None:
        return protocol.error(protocol.BAD_REQUEST, f"unknown action: {req['action']!r}")

    # ── request handling ────────────────────────────────────────────────────
    def touch(self) -> None:
        self.last_activity = time.monotonic()

    def _handle(self, req: dict[str, Any], conn: socket.socket) -> dict[str, Any] | None:
        action = req["action"]
        if action == "ping":
            return {"ok": True, "role": self.kind, "protocol": protocol.PROTOCOL_VERSION,
                    "version": __version__, "pid": os.getpid(), "state": self.state,
                    "error": self.error, "uptime_s": int(time.monotonic() - self.started),
                    "requests": self.requests, "home": str(self.paths.home),
                    "idle_exit_seconds": self.idle_exit_seconds, **self.ping_info()}
        self.touch()
        self.requests += 1
        if action == "shutdown":
            self.stop_reason = "shutdown requested by a client (rag-search daemon stop/restart)"
            log.warning("%s daemon: %s", self.kind, self.stop_reason)
            self.stop.set()
            return {"ok": True}
        return self.dispatch(req, conn)

    def _serve_conn(self, conn: socket.socket) -> None:
        try:
            conn.settimeout(30)
            raw = protocol.read_line(conn)
            try:
                req, err = protocol.validate_request(json.loads(raw.decode("utf-8")))
                resp = err if err else self._handle(req, conn)  # type: ignore[arg-type]
            except ValueError as exc:
                resp = protocol.error(protocol.BAD_REQUEST, f"invalid JSON: {exc}")
            except Exception as exc:  # noqa: BLE001
                log.exception("request failed")
                resp = protocol.error(protocol.INTERNAL, f"{type(exc).__name__}: {exc}")
            if resp is not None:
                conn.settimeout(None)
                conn.sendall(protocol.encode(resp))
        except OSError:
            pass
        finally:
            conn.close()

    # ── lifecycle ───────────────────────────────────────────────────────────
    def _idle_watch(self) -> None:
        while not self.stop.wait(self.idle_poll):
            if (self.idle_exit_seconds and not self.is_busy()
                    and time.monotonic() - self.last_activity > self.idle_exit_seconds):
                log.info("idle for %ds; exiting", self.idle_exit_seconds)
                self.stop_reason = f"idle for {self.idle_exit_seconds}s"
                self.stop.set()

    def run(self) -> int:
        allow_cloud_files()                # launchd starts daemons without it; the worker inherits it
        ensure_dirs(self.paths)
        lock_fd = os.open(self.paths.alive_lock(self.kind), os.O_CREAT | os.O_RDWR, 0o600)
        # a client probing the lock (client.alive_lock_held) holds it for an instant: retry
        # briefly so that a starting daemon does not mistake the probe for a running daemon
        for attempt in range(10):
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError:
                if attempt == 9:
                    log.info("another %s daemon already holds the lock; exiting", self.kind)
                    os.close(lock_fd)
                    return 0
                time.sleep(0.1)
        sock = self.paths.socket(self.kind)
        if sock.parent != self.paths.run:  # long-path fallback location
            sock.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            os.chmod(sock.parent, 0o700)
        sock_path = str(sock)
        try:
            os.unlink(sock_path)  # stale socket from a dead daemon
        except FileNotFoundError:
            pass
        srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        srv.bind(sock_path)
        os.chmod(sock_path, 0o600)
        srv.listen(16)
        srv.settimeout(self.accept_timeout)
        self.paths.pid_file(self.kind).write_text(str(os.getpid()))

        def _sig(signum: int, _frame: Any) -> None:
            try:
                name = signal.Signals(signum).name
            except ValueError:
                name = str(signum)
            self.stop_reason = f"received {name} (daemon stop/restart, install or upgrade, or kill)"
            log.warning("%s daemon: %s", self.kind, self.stop_reason)
            self.stop.set()

        if threading.current_thread() is threading.main_thread():
            signal.signal(signal.SIGTERM, _sig)
            signal.signal(signal.SIGINT, _sig)
        log.info("%s daemon listening on %s (pid %d)", self.kind, sock_path, os.getpid())
        self.on_start()
        threading.Thread(target=self._idle_watch, name="idle-watch", daemon=True).start()
        try:
            while not self.stop.is_set():
                try:
                    conn, _ = srv.accept()
                except socket.timeout:
                    continue
                except OSError:
                    break
                threading.Thread(target=self._serve_conn, args=(conn,), daemon=True).start()
        finally:
            srv.close()
            try:
                self.on_stop()
            except Exception:  # noqa: BLE001
                log.exception("on_stop failed")
            for p in (self.paths.socket(self.kind), self.paths.pid_file(self.kind)):
                try:
                    p.unlink()
                except OSError:
                    pass
            os.close(lock_fd)
            log.info("%s daemon stopped", self.kind)
        return 0
