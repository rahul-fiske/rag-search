"""Socket clients for the two daemons, with on-demand start (stdlib only).

Every call is synchronous and time-bounded.  Async callers wrap `request_sync` in
`asyncio.to_thread`; because the start lock is taken and released inside one thread
function, a cancelled MCP call can never leave the lock held.
"""

from __future__ import annotations

import contextlib
import fcntl
import json
import os
import socket
import subprocess
import sys
import threading
import time
from typing import Any, Iterator

from . import protocol
from .paths import Paths, detached_start, ensure_dirs

LOG_ROTATE_BYTES = 5 * 1024 * 1024
DAEMON_MODULES = {
    "search": "rag_search.core.search_daemon",
    "indexer": "rag_search.core.indexer_daemon",
}


def client_id() -> str:
    return os.environ.get("RAG_SEARCH_CLIENT", "cli")


def _connect(paths: Paths, kind: str, timeout: float) -> socket.socket:
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.settimeout(timeout)
    try:
        s.connect(str(paths.socket(kind)))
    except BaseException:
        s.close()
        raise
    return s


def roundtrip(paths: Paths, kind: str, req: dict[str, Any], timeout: float) -> dict[str, Any]:
    """Send one request, return the decoded single reply.  Raises OSError on connect/IO failure."""
    s = _connect(paths, kind, timeout)
    try:
        s.sendall(protocol.encode(req))
        buf = b""
        while b"\n" not in buf:
            chunk = s.recv(1 << 20)
            if not chunk:
                break
            buf += chunk
    finally:
        s.close()
    if not buf.strip():
        raise ConnectionError("daemon closed the connection without a reply")
    return json.loads(buf.split(b"\n", 1)[0].decode("utf-8"))


def stream(paths: Paths, kind: str, req: dict[str, Any],
           timeout: float | None = None) -> Iterator[dict[str, Any]]:
    """Yield JSON lines from a streaming action until the daemon sends {"event": "end"}."""
    s = _connect(paths, kind, 5.0)
    try:
        s.settimeout(timeout)
        s.sendall(protocol.encode(req))
        buf = b""
        while True:
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                if line.strip():
                    obj = json.loads(line.decode("utf-8"))
                    yield obj
                    if obj.get("event") == "end" or obj.get("ok") is False:
                        return
            chunk = s.recv(65536)
            if not chunk:
                return
            buf += chunk
    finally:
        s.close()


def ping(paths: Paths, kind: str, timeout: float = 2.0) -> dict[str, Any] | None:
    try:
        resp = roundtrip(paths, kind, protocol.make_request("ping", client_id()), timeout)
    except (OSError, ValueError):
        return None
    return resp if resp.get("ok") else None


def alive_lock_held(paths: Paths, kind: str) -> bool:
    """True if a daemon process currently holds the single-instance lock."""
    fd = os.open(paths.alive_lock(kind), os.O_CREAT | os.O_RDWR, 0o600)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            return True
        fcntl.flock(fd, fcntl.LOCK_UN)
        return False
    finally:
        os.close(fd)


def spawn(paths: Paths, kind: str, lock_wait: float = 10.0) -> bool:
    """Start a daemon unless one is running/starting.  Returns True if we spawned one."""
    ensure_dirs(paths)
    fd = os.open(paths.start_lock(kind), os.O_CREAT | os.O_RDWR, 0o600)
    try:
        deadline = time.monotonic() + lock_wait
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError:
                if time.monotonic() > deadline:
                    return False
                time.sleep(0.1)
        if ping(paths, kind, 1.0) or alive_lock_held(paths, kind):
            return False
        log = paths.log_file(kind)
        with contextlib.suppress(OSError):
            if log.exists() and log.stat().st_size > LOG_ROTATE_BYTES:
                os.replace(log, log.with_suffix(".log.1"))
        out = open(log, "ab")
        env = dict(os.environ, RAG_SEARCH_HOME=str(paths.home))
        proc = subprocess.Popen(
            [sys.executable, "-m", DAEMON_MODULES[kind]],
            stdin=subprocess.DEVNULL, stdout=out, stderr=out,
            start_new_session=True, close_fds=True, **detached_start(paths.home, env),
        )
        out.close()
        threading.Thread(target=proc.wait, daemon=True).start()  # reap; avoids a zombie
        # Keep the start lock until the daemon owns its alive-lock (or died): otherwise a
        # second caller would see "no daemon yet" and spawn a duplicate that loses the race.
        end = time.monotonic() + 20.0
        while time.monotonic() < end and proc.poll() is None and not alive_lock_held(paths, kind):
            time.sleep(0.05)
        return True
    finally:
        os.close(fd)


def request_sync(paths: Paths, kind: str, action: str, *, client: str | None = None,
                 wait_s: float = 45.0, request_timeout: float = 180.0,
                 autostart: bool = True, **fields: Any) -> dict[str, Any]:
    """Send *action* to a daemon, starting it if needed; give up after ~wait_s of warm-up.

    Returns the daemon's reply, or {"ok": False, "code": "warming_up"|"unavailable", ...}.
    """
    req = protocol.make_request(action, client or client_id(), **fields)
    deadline = time.monotonic() + wait_s
    spawned = False
    last_err = ""
    while True:
        remaining = max(0.0, deadline - time.monotonic())
        try:
            return roundtrip(paths, kind, dict(req, wait_s=remaining),
                             timeout=remaining + request_timeout)
        except (FileNotFoundError, ConnectionRefusedError) as exc:
            last_err = str(exc)
            if not autostart:
                return protocol.error(protocol.UNAVAILABLE, f"{kind} daemon is not running")
            if not spawned:
                spawn(paths, kind)
                spawned = True
        except OSError as exc:  # timeouts, resets
            return protocol.error(protocol.UNAVAILABLE, f"{type(exc).__name__}: {exc}")
        except ValueError as exc:
            return protocol.error(protocol.UNAVAILABLE, f"bad daemon reply: {exc}")
        if time.monotonic() >= deadline:
            return protocol.error(protocol.WARMING_UP,
                                  f"{kind} daemon not accepting connections yet ({last_err})",
                                  log=str(paths.log_file(kind)))
        time.sleep(0.4)


def stop(paths: Paths, kind: str, wait: float | None = None) -> bool:
    """Ask a daemon to shut down; True only once it has really exited (lock released).

    The indexer needs longer: it first kills its worker's process group.
    """
    wait = wait if wait is not None else (60.0 if kind == "indexer" else 20.0)
    try:
        ok = bool(roundtrip(paths, kind, protocol.make_request("shutdown", client_id()), 5)
                  .get("ok"))
    except (OSError, ValueError):
        return False
    end = time.monotonic() + wait
    while time.monotonic() < end and alive_lock_held(paths, kind):
        time.sleep(0.1)
    return ok and not alive_lock_held(paths, kind)
