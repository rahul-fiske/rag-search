"""The dashboard server: `rag-search ui` (stdlib only, no model is ever loaded here).

* Serves one static page (``static/``) and a small JSON API over ``rag_search.api`` /
  ``rag_search.access``: exactly what the CLI can do, so nothing here is special-purpose.
* Live data reaches the browser over Server-Sent Events: a background thread samples the two
  daemons once a second and pushes only what changed (`live`: daemons + indexing, `catalog`:
  collections + access).
* Safety: it listens on 127.0.0.1 only; every request needs the per-user token (cookie or
  header) and a loopback ``Host``; state-changing calls need a same-origin JSON request.
  ``--read-only`` refuses everything that changes state.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import importlib.metadata
import json
import os
import secrets
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from http.cookies import SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from .. import __version__, access, api, catalog, model_tasks, models, playground_runs, policy, spec
from ..config import ConfigStore, update_config
from ..paths import Paths, allow_cloud_files, detached_start, pasted_path, ensure_dirs, get_paths, parse_collections
from . import info, markdown

_INSTALLED: dict[str, Any] = {"at": -1e9, "version": ""}


def installed_version() -> str:
    """Version of rag-search installed on disk right now (may be newer than the code this
    long-running process loaded).  Read at most every 30 s; "" when it cannot be told."""
    now = time.monotonic()
    if now - _INSTALLED["at"] > 30:
        try:
            version = importlib.metadata.version("rag-search")
        except Exception:  # noqa: BLE001 - not installed as a package (source tree), or unreadable
            version = ""
        _INSTALLED.update(at=now, version=version)
    return str(_INSTALLED["version"])

DEFAULT_PORT = 8765
COOKIE = "rag_search_ui"
MAX_BODY = 64 * 1024
CSP = ("default-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; "
       "connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'")
CONTENT_TYPES = {".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8",
                 ".css": "text/css; charset=utf-8", ".svg": "image/svg+xml",
                 ".png": "image/png", ".ico": "image/x-icon"}


# ── token and files ──────────────────────────────────────────────────────────

def token_file(paths: Paths) -> Path:
    return paths.run / "ui.token"


def load_token(paths: Paths) -> str:
    """The dashboard token: created once (mode 0600) so bookmarks and cookies survive restarts."""
    ensure_dirs(paths)
    f = token_file(paths)
    try:
        tok = f.read_text(encoding="utf-8").strip()
        if len(tok) >= 20:
            return tok
    except OSError:
        pass
    tok = secrets.token_urlsafe(24)
    fd = os.open(f, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(tok + "\n")
    return tok


def home_id(paths: Paths) -> str:
    return hashlib.sha256(str(paths.home).encode()).hexdigest()[:12]


def tail_file(path: Path, lines: int, max_bytes: int = 256 * 1024) -> list[str]:
    try:
        with open(path, "rb") as fh:
            fh.seek(0, os.SEEK_END)
            size = fh.tell()
            fh.seek(max(0, size - max_bytes))
            data = fh.read().decode("utf-8", errors="replace")
    except OSError:
        return []
    return data.splitlines()[-lines:]


# ── live snapshots ───────────────────────────────────────────────────────────


def _scoped_paths(base: Any, q: dict[str, list[str]]) -> Any:
    """Production's paths, or those of the playground experiment named by ``?exp=``: the document list, page
    records, source pages and converted Markdown are read by the same functions for both."""
    exp = (q.get("exp") or [""])[0]
    if not exp:
        return base
    from ..paths import get_playground_paths, list_playground_names

    if exp not in list_playground_names(base):
        raise ValueError(f"no such experiment: {exp}")
    return get_playground_paths(base, exp)


class Live:
    """Samples the daemons in the background; SSE clients wait on `cond` for changes."""

    def __init__(self, paths: Paths, read_only: bool):
        self.paths, self.read_only = paths, read_only
        self.cond = threading.Condition()
        self.seq = {"live": 0, "catalog": 0}
        self.payload: dict[str, str] = {}
        self.clients = 0
        self.stopping = False
        self._catalog_due = 0.0
        self._thread: threading.Thread | None = None
        self._nudge = threading.Event()

    # -- what is sampled --------------------------------------------------------
    def build_live(self) -> dict[str, Any]:
        daemons = api.daemon_status(self.paths)
        index = api.index_status(self.paths, docs=60, history=8, client="cli")
        job = (index.get("job") or {}) if isinstance(index, dict) else {}
        conversion = api.conversion_run(self.paths, job.get("id", ""), client="cli") if job else {}
        return {"ts": round(time.time(), 1), "daemons": daemons, "index": index,
                "conversion": conversion}

    def build_catalog(self, seen: dict[str, float]) -> dict[str, Any]:
        return {"version": __version__, "installed_version": installed_version(),
                "home": str(self.paths.home), "read_only": self.read_only,
                "models": models.brief(self.paths),
                "list": catalog.list_view(self.paths, policy.Rules(), "cli", full=True),
                "access": access.overview(self.paths, seen)}

    def sample(self, force_catalog: bool = False) -> None:
        try:
            live = self.build_live()
        except Exception as exc:  # noqa: BLE001 - one bad sample must not kill the thread
            live = {"ts": round(time.time(), 1), "error": f"{type(exc).__name__}: {exc}"}
        self._store("live", live)
        now = time.monotonic()
        if force_catalog or now >= self._catalog_due:
            seen = ((live.get("daemons") or {}).get("search") or {}).get("clients_seen") or {}
            try:
                cat = self.build_catalog(seen)
            except Exception as exc:  # noqa: BLE001
                cat = {"error": f"{type(exc).__name__}: {exc}", "version": __version__,
                       "read_only": self.read_only}
            self._store("catalog", cat)
            self._catalog_due = now + 2.0

    def _store(self, name: str, data: dict[str, Any]) -> None:
        text = json.dumps(data, separators=(",", ":"), ensure_ascii=False)
        with self.cond:
            if self.payload.get(name) != text:
                self.payload[name] = text
                self.seq[name] += 1
                self.cond.notify_all()

    # -- lifecycle --------------------------------------------------------------
    def start(self) -> None:
        self.sample(force_catalog=True)
        self._thread = threading.Thread(target=self._loop, name="ui-sampler", daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while not self.stopping:
            self._nudge.wait(1.0 if self.clients else 5.0)
            self._nudge.clear()
            if self.stopping:
                return
            self.sample()

    def nudge(self, catalog: bool = False) -> None:
        """Sample soon (after an action) instead of waiting for the next tick."""
        if catalog:
            self._catalog_due = 0.0
        self._nudge.set()

    def stop(self) -> None:
        self.stopping = True
        self._nudge.set()
        with self.cond:
            self.cond.notify_all()

    def snapshot(self) -> dict[str, Any]:
        with self.cond:
            return {k: json.loads(v) for k, v in self.payload.items()}


# ── the HTTP handler ─────────────────────────────────────────────────────────

class Handler(BaseHTTPRequestHandler):
    server_version = f"rag-search-ui/{__version__}"
    protocol_version = "HTTP/1.1"

    # the app object lives on the server: server.app
    @property
    def app(self) -> "UiApp":
        return self.server.app  # type: ignore[attr-defined]

    def log_message(self, fmt: str, *args: Any) -> None:  # noqa: D401 - silence per-request lines
        return

    # -- plumbing ---------------------------------------------------------------
    def _send(self, status: int, body: bytes, ctype: str, extra: dict[str, str] | None = None) -> None:
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        if "Cache-Control" not in (extra or {}):
            self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Content-Security-Policy", CSP)
        self.send_header("X-Frame-Options", "DENY")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _json(self, obj: Any, status: int = 200) -> None:
        self._send(status, json.dumps(obj, ensure_ascii=False).encode(), "application/json; charset=utf-8")

    def _error(self, status: int, message: str) -> None:
        self._json({"ok": False, "error": message}, status)

    def _host_ok(self) -> bool:
        host = (self.headers.get("Host") or "").lower()
        port = self.server.server_address[1]
        return host in {f"127.0.0.1:{port}", f"localhost:{port}", f"[::1]:{port}"}

    def _authorised(self) -> bool:
        tok = self.app.token
        given = self.headers.get("X-RagSearch-Token") or ""
        if not given:
            jar = SimpleCookie(self.headers.get("Cookie") or "")
            given = jar[COOKIE].value if COOKIE in jar else ""
        return bool(given) and hmac.compare_digest(given.encode(), tok.encode())

    def _origin_ok(self) -> bool:
        origin = self.headers.get("Origin")
        return origin is None or origin == f"http://{self.headers.get('Host')}"

    def _body(self) -> dict[str, Any] | None:
        ctype = (self.headers.get("Content-Type") or "").split(";")[0].strip().lower()
        if ctype != "application/json":
            self.close_connection = True         # the unread body must not be parsed as a new request
            self._error(415, "send application/json")
            return None
        try:
            n = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            n = -1
        if not 0 <= n <= MAX_BODY:
            self.close_connection = True
            self._error(413, "request body too large")
            return None
        try:
            data = json.loads(self.rfile.read(n).decode("utf-8") or "{}")
        except (ValueError, UnicodeDecodeError):
            self._error(400, "invalid JSON")
            return None
        if not isinstance(data, dict):
            self._error(400, "expected a JSON object")
            return None
        return data

    # -- routing ----------------------------------------------------------------
    def do_HEAD(self) -> None:  # noqa: N802
        self.do_GET()

    def do_GET(self) -> None:  # noqa: N802
        if not self._host_ok():
            return self._error(403, "unexpected Host header")
        url = urlparse(self.path)
        if url.path == "/api/ping":            # no token: lets `rag-search ui` find a running one
            return self._json({"app": "rag-search-ui", "home": home_id(self.app.paths),
                               "version": __version__})
        q = parse_qs(url.query)
        if q.get("token") and url.path == "/":
            if hmac.compare_digest(q["token"][0].encode(), self.app.token.encode()):
                cookie = f"{COOKIE}={self.app.token}; Path=/; HttpOnly; SameSite=Strict"
                return self._send(302, b"", "text/plain", {"Location": "/", "Set-Cookie": cookie})
            return self._error(401, "wrong token; start the dashboard with `rag-search ui`")
        if not self._authorised():
            return self._send(401, b"Open the dashboard with:  rag-search ui\n", "text/plain; charset=utf-8")
        path = url.path
        try:
            if path in ("/", "/index.html"):
                return self._static("index.html")
            if path.startswith("/static/"):
                return self._static(path[len("/static/"):])
            if path.startswith("/api/"):
                return self._api_get(path[len("/api/"):], q)
        except (BrokenPipeError, ConnectionResetError):
            return
        except Exception as exc:  # noqa: BLE001
            return self._error(500, f"{type(exc).__name__}: {exc}")
        self._error(404, "not found")

    def do_POST(self) -> None:  # noqa: N802
        if not self._host_ok():
            return self._error(403, "unexpected Host header")
        if not self._authorised():
            return self._error(401, "not authorised; open the dashboard with `rag-search ui`")
        if not self._origin_ok():
            return self._error(403, "cross-origin request refused")
        path = urlparse(self.path).path
        if not path.startswith("/api/"):
            return self._error(404, "not found")
        body = self._body()
        if body is None:
            return
        try:
            self._api_post(path[len("/api/"):], body)
        except (BrokenPipeError, ConnectionResetError):
            return
        except Exception as exc:  # noqa: BLE001
            self._error(500, f"{type(exc).__name__}: {exc}")

    def _static(self, name: str) -> None:
        f = (info.STATIC / name).resolve()
        if f.parent != info.STATIC.resolve() or not f.is_file() or f.suffix not in CONTENT_TYPES:
            return self._error(404, "not found")
        self._send(200, f.read_bytes(), CONTENT_TYPES[f.suffix])

    # -- GET API ----------------------------------------------------------------
    def _api_get(self, name: str, q: dict[str, list[str]]) -> None:
        app = self.app
        if name == "state":
            return self._json(app.live.snapshot())
        if name == "index/documents":
            def one(key: str) -> str:
                return (q.get(key) or [""])[0]
            r = api.index_status(app.paths, job_id=one("job_id"), docs=int(one("limit") or 200),
                                 doc_status=one("status"), doc_collection=one("collection"),
                                 doc_q=one("q"), doc_branch=one("branch"),
                                 doc_outcome=one("outcome"), client="cli")
            return self._json(r)
        try:                                     # an experiment's own workspace when ?exp=NAME, else production
            sp = _scoped_paths(app.paths, q)
        except ValueError as exc:
            return self._error(400, str(exc))
        if name == "conversion/run":
            return self._json(api.conversion_run(sp, (q.get("job_id") or [""])[0], client="cli"))
        if name == "conversion/documents":
            def one(key: str) -> str:
                return (q.get(key) or [""])[0]
            r = api.conversion_documents(
                sp, job_id=one("job_id"), status=one("status"), collection=one("collection"),
                q=one("q"), branch=one("branch"), outcome=one("outcome"),
                limit=int(one("limit") or 200), client="cli")
            return self._json(r) if r.get("ok") else self._error(400, r.get("error", "failed"))
        if name == "conversion/trace":
            def one(key: str) -> str:
                return (q.get(key) or [""])[0]
            r = api.conversion_trace(sp, one("collection"), one("doc"),
                                     page=int(one("page") or 0))
            return self._json(r) if r.get("ok") else self._error(400, r.get("error", "failed"))
        if name == "conversion/page-image":
            def one(key: str) -> str:
                return (q.get(key) or [""])[0]
            r = api.conversion_page_image(sp, one("collection"), one("doc"),
                                          int(one("page") or 1), int(one("width") or 900))
            if not r.get("ok"):
                return self._error(400, r.get("error", "failed"))
            return self._send(200, r["png"], "image/png", {"Cache-Control": "private, max-age=300"})
        if name == "conversion/markdown":           # the converted text; raw=1 = plain text for a browser tab
            def one(key: str) -> str:
                return (q.get(key) or [""])[0]
            r = api.conversion_markdown(sp, one("collection"), one("doc"), page=int(one("page") or 0))
            if not r.get("ok"):
                return self._error(400, r.get("error", "failed"))
            if one("raw"):
                return self._send(200, r["result"]["markdown"].encode(), "text/plain; charset=utf-8")
            return self._json(r)
        if name == "conversion/page-cache":
            return self._json(api.page_cache(app.paths))
        if name == "conversion/bench":              # gold sets and stored runs (listing only)
            g, runs = api.bench_gold_list(app.paths), api.bench_list(app.paths, "")
            return self._json({"ok": True, "sets": g["result"]["sets"], "engines": g["result"]["engines"],
                               "runs": runs.get("result", [])})
        if name in ("conversion/bench-run", "conversion/bench-compare"):
            def one(key: str) -> str:
                return (q.get(key) or [""])[0]
            r = (api.bench_show(app.paths, one("set"), one("run")) if name.endswith("run")
                 else api.bench_compare(app.paths, one("set"), one("a"), one("b")))
            return self._json(r) if r.get("ok") else self._error(400, r.get("error", "failed"))
        if name == "events":
            return self._events()
        if name == "download":
            return self._download((q.get("id") or [""])[0])
        if name == "collection/info":
            r = api.collection_info(app.paths, (q.get("name") or [""])[0])
            if not r.get("ok"):
                return self._error(400, r.get("error", "failed"))
            return self._json(r)
        if name == "architecture":
            return self._json(info.architecture(app.paths))
        if name == "pipeline":
            return self._json(api.pipeline(app.paths))
        if name == "config":
            store = ConfigStore(app.paths)
            return self._json({"ok": True, "values": store.get(), "tunables": info.tunables(),
                               "error": store.error})
        if name == "help":
            readme, arch = info.read_doc("README.md"), info.read_doc("ARCHITECTURE.md")
            return self._json({"cli": info.cli_reference(),
                               "readme": {"html": markdown.render(readme), "toc": markdown.headings(readme)},
                               "architecture_doc": {"html": markdown.render(arch),
                                                    "toc": markdown.headings(arch)}})
        if name == "models":
            return self._json({"ok": True, **models.state(app.paths),
                               "task": model_tasks.read_task(app.paths)})
        if name == "models/plan":
            try:
                pl = model_tasks.plan_switch(app.paths, (q.get("kind") or [""])[0],
                                             (q.get("model") or [""])[0].strip())
            except models.ModelError as exc:
                return self._error(400, str(exc))
            return self._json({"ok": True, "plan": pl})
        if name == "logs":
            kind = (q.get("kind") or ["search"])[0]
            if kind not in api.KINDS + ("ui",):
                return self._error(400, "kind must be search, indexer or ui")
            n = max(1, min(int((q.get("lines") or ["200"])[0] or 200), 2000))
            return self._json({"kind": kind, "file": str(app.paths.log_file(kind)),
                               "lines": tail_file(app.paths.log_file(kind), n)})
        self._error(404, "not found")

    def _events(self) -> None:
        live = self.app.live
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "keep-alive")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()
        sent = {"live": -1, "catalog": -1}
        with live.cond:
            live.clients += 1
        live.nudge()
        try:
            while not live.stopping:
                with live.cond:
                    live.cond.wait_for(lambda: live.stopping or any(
                        live.seq[k] != sent[k] and k in live.payload for k in sent), timeout=15)
                    msgs = [(k, live.seq[k], live.payload[k]) for k in sent
                            if k in live.payload and live.seq[k] != sent[k]]
                if not msgs:
                    self.wfile.write(b": keepalive\n\n")
                    self.wfile.flush()
                    continue
                for k, seq, text in msgs:
                    self.wfile.write(f"event: {k}\ndata: {text}\n\n".encode())
                    sent[k] = seq
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass
        finally:
            with live.cond:
                live.clients -= 1
            self.close_connection = True

    # -- POST API ---------------------------------------------------------------
    def _api_post(self, name: str, body: dict[str, Any]) -> None:
        app, paths = self.app, self.app.paths
        readonly_ok = {"search", "grep", "doctor", "playground/search", "playground/compare",
                      "playground/list", "playground/preview", "playground/run", "playground/runs",
                      "playground/settings", "conversion/estimate"}
        if app.read_only and name not in readonly_ok:
            return self._error(403, "the dashboard was started with --read-only")

        def text(key: str, default: str = "") -> str:
            v = body.get(key, default)
            return v if isinstance(v, str) else default

        def flag(key: str) -> bool:
            return body.get(key) is True

        def names(key: str) -> list[str]:
            v = body.get(key)
            if isinstance(v, list):
                v = ",".join(str(x) for x in v)
            return parse_collections(v if isinstance(v, str) else "")

        view_as = policy.normalize_client(text("client", "cli"))
        try:
            if name == "search":
                top_k = body.get("top_k")   # None = the production default (config.json)
                top_k = int(top_k) if top_k is not None else None
                stages = body.get("stages")   # list or comma-string; None = today's default (all)
                pool_kw: dict[str, Any] = {}
                for key in ("retrieval_pool", "rerank_pool", "rrf_k"):
                    v = body.get(key)
                    if v is not None:
                        pool_kw[key] = int(v)
                return self._json(api.search(paths, text("query"), top_k=top_k, collections=names("collections"),
                                             client=view_as, wait_s=45, origin="ui", stages=stages, **pool_kw))
            if name == "grep":
                return self._json(api.grep(paths, text("pattern"), collections=names("collections"),
                                           context_lines=int(body.get("context_lines", 2)),
                                           max_matches=int(body.get("max_matches", 20)),
                                           client=view_as, origin="ui"))
            if name == "conversion/estimate":     # a dry run: reads the sources, converts nothing
                r = api.conversion_estimate(paths, text("target"), budget_s=30.0)
                return self._json(r) if r.get("ok") else self._error(400, r.get("error", "failed"))
            if name == "index/start":
                mode = text("mode", "new")
                if mode not in ("new", "all"):
                    return self._error(400, "mode must be new or all")
                if mode == "all" and not flag("confirm"):
                    return self._error(400, "a full rebuild needs confirm: true")
                r = api.index_start(paths, mode=mode, path=text("path"), rebuild=flag("rebuild"),
                                    force_md=flag("force_md"), restart=flag("restart"), client="cli")
                app.live.nudge()
                return self._json(r)
            if name == "index/cancel":
                r = api.index_cancel(paths, client="cli")
                app.live.nudge()
                return self._json(r)
            if name == "index/publish":
                r = api.index_publish(paths, client="cli")
                app.live.nudge(catalog=True)
                return self._json(r)
            if name == "access":
                return self._access(body)
            if name.startswith("collection/") or name == "location/remove":
                return self._collection_action(name, body)
            if name == "describe":
                coll, desc = body.get("collection"), body.get("description", "")
                if not isinstance(coll, str) or not isinstance(desc, str):
                    return self._error(400, "expected {collection, description} (an empty "
                                       "description clears it)")
                r = api.describe_collection(paths, coll, desc, client="cli")
                if not r.get("ok"):
                    return self._error(400, r.get("error", "failed"))
                app.live.nudge(catalog=True)
                return self._json(r)
            if name == "daemon":
                which, action = text("which", "all"), text("action")
                if action not in ("start", "stop", "restart"):
                    return self._error(400, "action must be start, stop or restart")
                r = getattr(api, f"daemon_{action}")(paths, which)
                app.live.nudge(catalog=True)
                return self._json({"ok": True, "result": r})
            if name == "doctor":
                return self._json(self._doctor())
            if name == "config/set":
                section = text("section")
                values = body.get("values")
                if section not in spec.TUNABLES_BY_SECTION or not isinstance(values, dict):
                    return self._error(400, "section must be search, indexer or models, with a "
                                       "'values' object")
                validated = spec.validate_section(section, values)
                if validated:
                    update_config(paths, section, validated)
                    app.live.nudge(catalog=True)
                store = ConfigStore(paths)
                return self._json({"ok": True, "values": store.get(), "changed": validated})
            if name.startswith("models/"):
                return self._models_post(name[len("models/"):], body)
            if name.startswith("playground/"):
                return self._playground_post(name[len("playground/"):], body)
        except ValueError as exc:
            return self._error(400, str(exc))
        self._error(404, "not found")

    def _playground_cli(self, args: list[str], timeout: float) -> dict[str, Any]:
        """Every playground action runs as `rag-search playground ...` in a child process --
        like `doctor`, this keeps the dashboard's own process from ever loading a model (a
        playground experiment can use a different model than whatever production has loaded)."""
        env = dict(os.environ, RAG_SEARCH_HOME=str(self.app.paths.home))
        try:
            p = subprocess.run([sys.executable, "-m", "rag_search.cli", "playground", *args],
                               capture_output=True, text=True, timeout=timeout, env=env)
        except subprocess.TimeoutExpired:
            return {"ok": False, "error": f"timed out after {int(timeout)}s"}
        try:
            data = json.loads(p.stdout)
        except ValueError:
            return {"ok": False, "error": (p.stderr or p.stdout or "command failed").strip()[-800:]}
        return {"ok": True, "result": data, "failed": p.returncode != 0}

    @staticmethod
    def _retrieval_args(body: dict[str, Any]) -> list[str]:
        """The stage / pool / RRF flags `playground search` and `playground bench` share."""
        args: list[str] = []
        stages = body.get("stages")
        if stages:
            args += ["--stages", stages if isinstance(stages, str) else ",".join(stages)]
        for key, flag in (("retrieval_pool", "--retrieval-pool"), ("rerank_pool", "--rerank-pool"),
                          ("rrf_k", "--rrf-k")):
            if body.get(key) is not None:
                args += [flag, str(int(body[key]))]
        if body.get("no_rerank") is True:
            args.append("--no-rerank")
        return args

    def _playground_run(self, action: str, exp: str, body: dict[str, Any]) -> None:
        """Index and bench runs are started in the background and watched through /api/playground/run, like
        an indexing run; cancel stops the experiment's running one (see playground_runs.py)."""
        try:
            if action == "cancel":
                return self._json({"ok": True, "result": playground_runs.cancel(self.app.paths, exp)})
            args: list[str] = []
            if action == "index":
                for key, flag in (("rebuild", "--rebuild"), ("force_md", "--force-md"), ("wipe", "--wipe")):
                    if body.get(key) is True:
                        args.append(flag)
                opts = {k: body.get(k) is True for k in ("rebuild", "force_md", "wipe")}
            else:
                if body.get("k"):
                    args += ["-k", str(int(body["k"]))]
                if body.get("label"):
                    args += ["--label", str(body["label"])]
                args += self._retrieval_args(body)
                opts = {"k": body.get("k"), "label": body.get("label") or ""}
            return self._json({"ok": True, "result": playground_runs.start(self.app.paths, exp, action, args, opts)})
        except ValueError as exc:
            return self._error(400, str(exc))

    def _playground_view(self, exp: str, body: dict[str, Any]) -> None:
        try:
            run_id = body.get("run") if isinstance(body.get("run"), str) else ""
            return self._json({"ok": True, "result": playground_runs.view(self.app.paths, exp, run_id or "")})
        except ValueError as exc:
            return self._error(400, str(exc))

    def _playground_post(self, action: str, body: dict[str, Any]) -> None:
        if action == "list":
            return self._json(self._playground_cli(["list", "--json"], timeout=30))
        exp = body.get("name")
        if not isinstance(exp, str) or not exp:
            return self._error(400, "expected {name: EXPERIMENT, ...}")
        if action == "create":
            args = ["create", exp, "--json"]
            coll = body.get("collection")
            if isinstance(coll, str) and coll:
                args += ["--collection", coll]
            folders = body.get("folders")
            for f in folders if isinstance(folders, list) else []:
                if isinstance(f, str) and f.strip():
                    args += ["--from", f.strip()]
            if body.get("from_production") is True:
                args.append("--from-production")
            return self._json(self._playground_cli(args, timeout=30))
        if action == "sources":                 # list / add / remove the experiment's source folders
            op = body.get("op") if body.get("op") in ("list", "add", "remove") else "list"
            args = ["source", exp, op, "--json"]
            arg = body.get("folder") if op == "add" else body.get("collection") if op == "remove" else ""
            if isinstance(arg, str) and arg:
                args.insert(3, arg)
            if op == "add" and isinstance(body.get("collection"), str) and body["collection"]:
                args += ["--as", body["collection"]]
            return self._json(self._playground_cli(args, timeout=30))
        if action == "config":
            args = ["config", exp, "--json"]
            # Same (key, flag) pairs `rag-search playground config` itself accepts -- the docling/
            # OCR/table/PDF-backend knobs are read straight from spec.py's shared registry (every
            # "indexer" tunable that has an env var of its own) instead of being listed by hand
            # here too, so a new one added to spec.py needs no matching edit in this file.
            docling_pairs = tuple((t.key, t.cli_flag) for t in spec.TUNABLES_BY_SECTION["indexer"]
                                  if t.env)
            for key, flag in (("embedding_model", "--embedding-model"),
                              ("rerank_model", "--rerank-model"), ("chunk_size", "--chunk-size"),
                              ("reader_model", "--reader-model"), ("repair_model", "--repair-model"),
                              ("chunk_overlap", "--chunk-overlap"), ("stages", "--stages"),
                              ("retrieval_pool", "--retrieval-pool"),
                              ("rerank_pool", "--rerank-pool"), ("rrf_k", "--rrf-k"),
                              *docling_pairs):
                v = body.get(key)
                if v not in (None, "") or (key in ("reader_model", "repair_model") and v == ""):
                    args += [flag, str(v) if v != "" else "production"]      # "" = back to production's choice
            if body.get("rerank") is True:
                args.append("--rerank")
            elif body.get("rerank") is False:
                args.append("--no-rerank")
            return self._json(self._playground_cli(args, timeout=30))
        if action in ("index", "bench") or action == "cancel":
            return self._playground_run(action, exp, body)
        if action == "run":
            return self._playground_view(exp, body)
        if action == "runs":
            try:
                return self._json({"ok": True, "result": playground_runs.history(self.app.paths, exp)})
            except ValueError as exc:
                return self._error(400, str(exc))
        if action == "settings":
            return self._json(self._playground_cli(["settings", exp, "--json"], timeout=30))
        if action == "search":
            args = ["search", exp, str(body.get("query") or "")]
            if body.get("top_k"):
                args += ["--top-k", str(int(body["top_k"]))]
            return self._json(self._playground_cli([*args, *self._retrieval_args(body), "--json"], timeout=180))
        if action == "compare":
            return self._json(self._playground_cli(["compare", exp, "--json"], timeout=30))
        if action == "preview":
            return self._json(self._playground_cli(["promote", exp, "--dry-run", "--json"], timeout=30))
        if action == "promote":
            args = ["promote", exp, "--json"]
            if body.get("confirm") is True:
                args.append("--confirm")
            return self._json(self._playground_cli(args, timeout=30))
        if action == "rm":
            if body.get("confirm") is not True:
                return self._error(400, "deleting an experiment needs confirm: true")
            return self._json(self._playground_cli(["rm", exp, "--yes", "--json"], timeout=30))
        self._error(404, "not found")

    def _models_post(self, action: str, body: dict[str, Any]) -> None:
        """Model downloads and switches run as detached tasks (`model_tasks`); the page polls
        GET /api/models for their progress."""
        paths = self.app.paths
        kind, model = body.get("kind"), body.get("model")
        try:
            if action == "cancel":
                return self._json({"ok": True, "cancelled": model_tasks.cancel(paths)})
            if action == "limit":
                gb = float(body.get("gb") or 0)
                models.set_memory_limit(paths, gb)
                return self._json({"ok": True, "memory_limit_gb": gb})
            if action in ("reader", "repair"):       # the document reader / repair model: a choice, nothing else
                if not isinstance(model, str) or not model.strip():
                    return self._error(400, "expected {model}")
                models.set_vlm_selection(paths, action, model.strip())
                self.app.live.nudge(catalog=True)
                return self._json({"ok": True, "kind": action, "model": model.strip()})
            if action == "switch":
                if not isinstance(kind, str) or not isinstance(model, str):
                    return self._error(400, "expected {kind, model}")
                model = model.strip()
                plan = model_tasks.plan_switch(paths, kind, model, force=body.get("force") is True)
                if plan["blocking"] and body.get("force") is not True:
                    return self._error(400, "; ".join(plan["blocking"]))
                req = {"op": "switch", "kind": kind, "model": model, "force": body.get("force") is True,
                       "no_verify": body.get("no_verify") is True,
                       "reindex": body.get("reindex") is not False}
            elif action == "download":
                ids = body.get("models") if isinstance(body.get("models"), list) else [model]
                for mid in ids:
                    if not isinstance(mid, str) or not (models.find(mid) or models._ID_RE.match(mid)):
                        return self._error(400, f"not a Hugging Face model id: {mid!r}")
                req = {"op": "download", "models": ids, "model": ids[0]}
            elif action == "runtime":
                req = {"op": "runtime", "model": "mlx-vlm"}
            elif action == "verify":
                ks = [kind] if kind in models.KINDS else list(models.KINDS)
                req = {"op": "verify", "targets": [[k, models.selection(k)[0]] for k in ks]}
            else:
                return self._error(404, "not found")
            rec = model_tasks.start_detached(paths, req)
        except model_tasks.TaskBusy as exc:
            return self._error(409, str(exc))
        except (models.ModelError, model_tasks.TaskError) as exc:
            return self._error(400, str(exc))
        self.app.live.nudge(catalog=True)
        self._json({"ok": True, "task": rec})

    def _collection_action(self, name: str, body: dict[str, Any]) -> None:
        """Add (register a folder), import, export, delete; location remove.  The same `api`
        functions as `rag-search location|collection ...`; deletion needs the collection's name
        typed back as `confirm`, here as in the browser."""
        paths, app = self.app.paths, self.app

        def text(key: str) -> str:
            v = body.get(key, "")
            return v.strip() if isinstance(v, str) else ""

        if name == "collection/add-location":
            r = api.location_add(paths, text("name"), text("folder"))
        elif name == "collection/import":
            r = api.collection_import(paths, text("file"), as_name=text("as_name") or None,
                                      replace=body.get("replace") is True)
        elif name == "collection/export":
            folder = pasted_path(text("folder")) or str(paths.home / "exports")
            try:
                Path(folder).expanduser().mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                return self._error(400, f"cannot use {folder}: {exc}")
            r = api.collection_export(paths, text("name"), folder)
            if r.get("ok"):
                r["result"]["download"] = "download?id=" + app.remember_download(
                    Path(r["result"]["file"]))
        elif name in ("collection/delete", "location/remove"):
            coll, confirm = text("name"), text("confirm")
            if not coll or confirm.casefold() != coll.casefold():
                return self._error(400, "type the collection's name to confirm")
            r = (api.collection_delete(paths, coll) if name == "collection/delete"
                 else api.location_remove(paths, coll))
        else:
            return self._error(404, "not found")
        if not r.get("ok"):
            return self._error(400, r.get("error", "failed"))
        app.live.nudge(catalog=True)
        self._json(r)

    def _download(self, did: str) -> None:
        with self.app.downloads_lock:
            path = self.app.downloads.get(did)
        if path is None or not path.is_file():
            return self._error(404, "no such download (export again)")
        size = path.stat().st_size
        self.send_response(200)
        self.send_header("Content-Type", "application/gzip")
        self.send_header("Content-Length", str(size))
        self.send_header("Content-Disposition", f'attachment; filename="{path.name}"')
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                self.wfile.write(chunk)

    def _access(self, body: dict[str, Any]) -> None:
        action, coll = body.get("action"), body.get("collection")
        clients = body.get("clients", [])
        if action not in ("restrict", "grant") or not isinstance(coll, str) \
                or not isinstance(clients, list) or not all(isinstance(c, str) for c in clients):
            return self._error(400, "expected {action: restrict|grant, collection, clients: [..]}")
        try:
            res = getattr(access, action)(self.app.paths, coll, clients)
        except access.AccessError as exc:
            return self._error(400, str(exc))
        self.app.live.nudge(catalog=True)
        self._json({"ok": True, "result": res})

    def _doctor(self) -> dict[str, Any]:
        """Run `rag-search doctor --json` in a child process (keeps this process light)."""
        env = dict(os.environ, RAG_SEARCH_HOME=str(self.app.paths.home))
        try:
            p = subprocess.run([sys.executable, "-m", "rag_search.cli", "doctor", "--json"],
                               capture_output=True, text=True, timeout=120, env=env)
        except subprocess.TimeoutExpired:
            return {"ok": False, "error": "doctor timed out after 120 s"}
        try:
            rows = json.loads(p.stdout)
        except ValueError:
            return {"ok": False, "error": (p.stderr or p.stdout or "doctor failed").strip()[-800:]}
        return {"ok": True, "rows": rows, "failed": p.returncode != 0}


# ── the server object ────────────────────────────────────────────────────────

class UiApp:
    def __init__(self, paths: Paths, token: str, read_only: bool = False):
        self.paths, self.token, self.read_only = paths, token, read_only
        self.live = Live(paths, read_only)
        # exports made from this dashboard, downloadable by id (never an arbitrary path)
        self.downloads: dict[str, Path] = {}
        self.downloads_lock = threading.Lock()

    def remember_download(self, path: Path) -> str:
        import secrets

        did = secrets.token_urlsafe(12)
        with self.downloads_lock:
            self.downloads[did] = path
            while len(self.downloads) > 50:              # keep the newest few
                self.downloads.pop(next(iter(self.downloads)))
        return did


class UiServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True
    request_queue_size = 32

    def __init__(self, addr: tuple[str, int], app: UiApp):
        self.app = app
        super().__init__(addr, Handler)

    def handle_error(self, request: Any, client_address: Any) -> None:
        """A browser that closes a tab or reloads ends its connection in the middle of a request: that is not
        an error of the dashboard and does not belong in its log as a traceback."""
        if isinstance(sys.exc_info()[1], (ConnectionError, TimeoutError)):
            return
        super().handle_error(request, client_address)

    def shutdown_all(self) -> None:
        self.app.live.stop()
        self.shutdown()


def make_server(paths: Paths, port: int = DEFAULT_PORT, read_only: bool = False,
                token: str | None = None) -> UiServer:
    """Bind 127.0.0.1:port (0 = any free port) and start the sampler; caller runs serve_forever()."""
    allow_cloud_files()                    # "Add collection" lists a folder that may be a cloud placeholder
    app = UiApp(paths, token or load_token(paths), read_only)
    server = UiServer(("127.0.0.1", port), app)
    app.live.start()
    return server


def url_for(server: UiServer) -> str:
    return f"http://127.0.0.1:{server.server_address[1]}/?token={server.app.token}"


# ── command line ─────────────────────────────────────────────────────────────

def _ping(port: int) -> dict | None:
    """The answer of whatever listens on *port* to /api/ping, {} if it is not a dashboard,
    None if nothing listens."""
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/ping", timeout=1.5) as r:
            data = json.loads(r.read().decode())
    except urllib.error.URLError:
        return None
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _probe(port: int, paths: Paths) -> str:
    """'ours' if a dashboard for this data folder answers on *port*, 'other' if something else
    does, '' if nothing."""
    data = _ping(port)
    if data is None:
        return ""
    return "ours" if data.get("app") == "rag-search-ui" and data.get("home") == home_id(paths) else "other"


def pid_file(paths: Paths) -> Path:
    return paths.run / "ui.pid"


def port_file(paths: Paths) -> Path:
    return paths.run / "ui.port"


def running_port(paths: Paths) -> int:
    """Port of a dashboard for this data folder that is up right now (0 = none)."""
    try:
        port = int(port_file(paths).read_text().strip())
    except (OSError, ValueError):
        return 0
    return port if port and _probe(port, paths) == "ours" else 0


def _open(url: str) -> None:
    import webbrowser

    try:
        webbrowser.open(url)
    except Exception:  # noqa: BLE001 - printing the URL is the fallback
        pass


def stop_running(paths: Paths) -> str:
    try:
        pid = int(pid_file(paths).read_text().strip())
    except (OSError, ValueError):
        return "no dashboard process id is recorded (started by an older version?)"
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        pid_file(paths).unlink(missing_ok=True)
        return "the dashboard was not running"
    except PermissionError:
        return f"cannot signal process {pid}"
    for _ in range(50):
        time.sleep(0.1)
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            break
    pid_file(paths).unlink(missing_ok=True)
    port_file(paths).unlink(missing_ok=True)
    return f"stopped the dashboard (pid {pid})"


def run(paths: Paths | None = None, port: int | None = None, open_browser: bool = True,
        read_only: bool = False, detach: bool = False, url_only: bool = False,
        stop: bool = False) -> int:
    paths = paths or get_paths()
    if stop:
        print(stop_running(paths))
        return 0
    port = DEFAULT_PORT if port is None else port
    if url_only:                            # only report: the recorded port first, then the default one
        found = running_port(paths) or (port if port and _probe(port, paths) == "ours" else 0)
        if not found:
            print("the dashboard is not running (start it with: rag-search ui)", file=sys.stderr)
            return 1
        print(f"http://127.0.0.1:{found}/?token={load_token(paths)}")
        return 0
    if port:
        at = running_port(paths) or port    # a dashboard of this data folder wherever it listens
        state = _probe(at, paths)
        running = (_ping(at) or {}).get("version") if state == "ours" else None
        if state == "ours" and running and running != __version__:
            # an upgrade leaves the old dashboard running the old code: replace it
            print(f"the running dashboard is version {running}, this is {__version__}: restarting it")
            msg = stop_running(paths)
            print(msg)
            if not msg.startswith("stopped"):
                print("it was started before this version wrote its process id: press Ctrl-C in the "
                      "terminal where it runs (or kill that process), then run this command again",
                      file=sys.stderr)
                return 1
        elif state == "ours":               # already running for this data folder: reuse it
            url = f"http://127.0.0.1:{at}/?token={load_token(paths)}"
            print(f"rag-search dashboard is already running: {url}")
            if open_browser:
                _open(url)
            return 0
        if state == "other":
            print(f"error: port {port} is used by something else; choose another with --port "
                  "(or --port 0 for any free port)", file=sys.stderr)
            return 1
    if detach:
        return _detach(paths, port, read_only, open_browser)
    try:
        server = make_server(paths, port, read_only)
    except OSError as exc:
        print(f"error: cannot listen on 127.0.0.1:{port}: {exc}", file=sys.stderr)
        return 1
    url = url_for(server)
    ensure_dirs(paths)
    port_file(paths).write_text(str(server.server_address[1]))
    pid_file(paths).write_text(str(os.getpid()))         # so `rag-search ui --stop` finds it too
    print(f"rag-search dashboard {__version__}: {url}")
    print("  (only this computer can reach it; Ctrl-C to stop"
          + ("; read-only)" if read_only else ")"), flush=True)
    if open_browser:
        _open(url)
    signal.signal(signal.SIGTERM, lambda *_: threading.Thread(target=server.shutdown_all, daemon=True).start())
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.app.live.stop()
        server.server_close()
        port_file(paths).unlink(missing_ok=True)
        try:
            if pid_file(paths).read_text().strip() == str(os.getpid()):
                pid_file(paths).unlink(missing_ok=True)
        except OSError:
            pass
    return 0


def _detach(paths: Paths, port: int, read_only: bool, open_browser: bool) -> int:
    ensure_dirs(paths)
    args = [sys.executable, "-m", "rag_search.ui.server", "--port", str(port), "--no-browser"]
    if read_only:
        args.append("--read-only")
    log = open(paths.log_file("ui"), "ab")
    proc = subprocess.Popen(args, stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                            start_new_session=True, close_fds=True,
                            **detached_start(paths.home, dict(os.environ, RAG_SEARCH_HOME=str(paths.home))))
    log.close()
    pid_file(paths).write_text(str(proc.pid))
    threading.Thread(target=proc.wait, daemon=True).start()
    for _ in range(100):
        time.sleep(0.1)
        if proc.poll() is not None:
            print(f"error: the dashboard exited at once; see {paths.log_file('ui')}", file=sys.stderr)
            return 1
        if (port and _probe(port, paths) == "ours") or (not port and running_port(paths)):
            break
    else:                                   # ten seconds and it never answered: do not claim it runs
        print(f"error: the dashboard did not come up; see {paths.log_file('ui')}", file=sys.stderr)
        return 1
    port = port or running_port(paths)
    if not port:
        print(f"error: the dashboard did not come up; see {paths.log_file('ui')}", file=sys.stderr)
        return 1
    url = f"http://127.0.0.1:{port}/?token={load_token(paths)}"
    print(f"rag-search dashboard running in the background (pid {proc.pid}): {url}")
    print("  stop it with: rag-search ui --stop")
    if open_browser:
        _open(url)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m rag_search.ui.server")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--read-only", action="store_true")
    a = ap.parse_args(argv)
    return run(port=a.port, open_browser=not a.no_browser, read_only=a.read_only)


if __name__ == "__main__":
    raise SystemExit(main())
