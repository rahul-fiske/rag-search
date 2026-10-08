"""Downloading, testing and switching models (the work behind `rag-search models` and the Models tab).

One task runs at a time.  The task record ``run/models.task.json`` is what the dashboard polls; the
runner (this module) holds ``run/models.task.lock`` for as long as it works, so a record that says
"running" while nobody holds the lock is known to be stale (the runner died).

    download   fetch a model into the Hugging Face cache (progress = bytes on disk / expected)
    verify     load it in a *separate process* and run a tiny relevance check
    switch     check the machine -> download -> verify -> write config.json -> apply:
                 reranker   the search daemon loads the new one and swaps it in; nothing is re-indexed
                 embedding  every document is embedded again; the new index goes live only when
                            all of it is done (the old index and model keep serving until then)

The CLI runs a task in the foreground (`run`); the dashboard starts it detached (`start_detached`)
so that it survives the page being closed.  Only the standard library is imported here; torch and
huggingface_hub are loaded inside the steps that need them.
"""

from __future__ import annotations

import contextlib
import fcntl
import fnmatch
import json
import os
import signal
import subprocess
import sys
import threading
import time
import uuid
from typing import Any, Callable

from . import models
from .paths import Paths, detached_start, ensure_dirs, get_paths, read_json, write_json_atomic

TASK_FILE = "models.task.json"
LOCK_FILE = "models.task.lock"
LOG_FILE = "models.log"
ACTIVE = ("queued", "running")
QUEUED_GRACE_S = 30.0        # a queued task whose runner has not started by then is stale

# the relevance check: each query must rank its own passage first among the three
DOCS = ("Sourdough bread rises using a fermented starter of wild yeast and lactic acid bacteria "
        "instead of commercial yeast.",
        "An aurora is a natural light display in the sky, caused by charged solar particles "
        "colliding with atmospheric gases near the poles.",
        "TCP uses a three-way handshake (SYN, SYN-ACK, ACK) to establish a reliable connection "
        "between two hosts.")
QUERIES = (("how do sourdough loaves rise", 0), ("what causes the northern lights", 1),
           ("how does the SYN-ACK handshake work", 2))


class TaskError(RuntimeError):
    """The task cannot go on; the message is for the user."""


class TaskBusy(TaskError):
    """Another model task is running."""


class Cancelled(BaseException):
    """The task was cancelled (SIGTERM from `models cancel` / the dashboard, or Ctrl-C)."""


# ── the task record ──────────────────────────────────────────────────────────

def task_file(paths: Paths):
    return paths.run / TASK_FILE


def lock_file(paths: Paths):
    return paths.run / LOCK_FILE


def log_file(paths: Paths):
    return paths.run / LOG_FILE


def _lock_held(paths: Paths) -> bool:
    try:
        fd = os.open(lock_file(paths), os.O_RDWR)
    except OSError:
        return False
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            return True
        fcntl.flock(fd, fcntl.LOCK_UN)
        return False
    finally:
        os.close(fd)


def read_task(paths: Paths) -> dict[str, Any] | None:
    """The latest task record; a "running" one whose runner is gone is reported as failed."""
    rec = read_json(task_file(paths))
    if not isinstance(rec, dict) or not rec.get("id"):
        return None
    if rec.get("status") in ACTIVE and not _lock_held(paths):
        fresh = rec.get("status") == "queued" and time.time() - rec.get("updated_at", 0) < QUEUED_GRACE_S
        if not fresh:
            rec = {**rec, "status": "failed", "finished_at": rec.get("updated_at"),
                   "error": rec.get("error") or "the task was interrupted (its process is gone)"}
    return rec


def is_active(paths: Paths) -> bool:
    rec = read_task(paths)
    return bool(rec and rec.get("status") in ACTIVE)


class Task:
    """The running task: holds the lock, keeps the record up to date."""

    def __init__(self, paths: Paths, request: dict[str, Any], echo: Callable[[str], None] | None,
                 progress_echo: Callable[[dict[str, Any]], None] | None = None):
        self.paths, self.request, self.echo = paths, request, echo
        self.progress_echo = progress_echo
        ensure_dirs(paths)
        self._fd = os.open(lock_file(paths), os.O_RDWR | os.O_CREAT, 0o600)
        try:
            fcntl.flock(self._fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            os.close(self._fd)
            raise TaskBusy("another model download or switch is already running "
                           "(see `rag-search models status`)") from None
        self.rec: dict[str, Any] = {
            "id": request.get("task_id") or uuid.uuid4().hex[:10], "op": request["op"],
            "kind": request.get("kind", ""), "model": request.get("model", ""),
            "status": "running", "phase": "starting", "pid": os.getpid(),
            "started_at": round(time.time(), 3), "log": [], "progress": {}, "result": {},
            "request": {k: v for k, v in request.items() if k != "task_id"}}
        self._lock = threading.Lock()
        self.save()

    def save(self) -> None:
        with self._lock:
            self.rec["updated_at"] = round(time.time(), 3)
            write_json_atomic(task_file(self.paths), self.rec, indent=None)

    def update(self, **fields: Any) -> None:
        self.rec.update(fields)
        self.save()

    def log(self, msg: str) -> None:
        self.rec["log"] = (self.rec.get("log", []) + [{"t": round(time.time(), 1), "msg": msg}])[-40:]
        if self.echo:
            self.echo(msg)
        self.save()

    def phase(self, name: str, msg: str = "") -> None:
        self.rec["phase"], self.rec["progress"] = name, {}
        if msg:
            self.log(msg)
        else:
            self.save()

    def progress(self, **fields: Any) -> None:
        self.rec["progress"] = fields
        if self.progress_echo:
            self.progress_echo(fields)
        self.save()

    def finish(self, status: str, error: str = "", **result: Any) -> None:
        self.rec.update(status=status, finished_at=round(time.time(), 3), phase="done")
        if error:
            self.rec["error"] = error
        if result:
            self.rec["result"] = result
        self.save()
        with contextlib.suppress(OSError):
            fcntl.flock(self._fd, fcntl.LOCK_UN)
            os.close(self._fd)


# ── planning: what would this switch do? ─────────────────────────────────────

def plan_switch(paths: Paths, kind: str, model_id: str, *, force: bool = False) -> dict[str, Any]:
    """Everything to show before a switch (the CLI asks, the dashboard's dialog lists it).

    ``blocking`` reasons stop the switch unless *force*; ``warnings`` do not."""
    spec = models.spec_for(kind, model_id)
    machine = models.machine_info()
    budget = models.budget_gb(machine, models.memory_limit_gb(paths))
    other_kind = models.RERANKER if kind == models.EMBEDDING else models.EMBEDDING
    other = models.spec_for(other_kind, models.selection(other_kind)[0])
    emb, rer = (spec, other) if kind == models.EMBEDDING else (other, spec)
    from .catalog import live_catalog

    chunks = sum(c.get("chunks", 0) for c in live_catalog(paths).get("collections", []))
    est = models.estimate_gb(emb, rer, machine, chunks)
    fit = models.fit_level(est, budget) if not spec.custom else "unknown"
    cs = models.cache_state(model_id)
    current, source = models.selection(kind)
    blocking: list[str] = []
    warnings: list[str] = []
    missing = models.missing_requirements(spec)
    blocking.extend(missing)
    if fit == "too_large":
        blocking.append(f"needs about {est:.1f} GB of memory with the other model; "
                        f"this machine's budget is {budget:.1f} GB "
                        "(raise it with `rag-search models limit GB`)")
    elif fit == "tight":
        warnings.append(f"about {est:.1f} GB of memory with the other model, close to the "
                        f"{budget:.1f} GB budget")
    if spec.custom:
        warnings.append("not in the catalogue: it is loaded as a standard "
                        + ("sentence-transformers model" if kind == models.EMBEDDING
                           else "cross-encoder")
                        + ", and a quick test runs first; nothing is changed if that fails")
    if source == "environment":
        warnings.append(f"${models.ENV_VARS[kind]} is set in the environment and overrides the "
                        "saved choice; unset it for the switch to take effect")
    reindex = None
    if kind == models.EMBEDDING:
        reindex = models.reindex_estimate(paths, model_id)
    return {"kind": kind, "model": model_id, "label": spec.label, "custom": spec.custom,
            "current": current, "same": current == model_id and not (reindex and reindex["documents"]),
            "cached": cs["cached"], "partial": cs["partial"], "weights_gb": spec.mem_gb,
            "fit": fit, "estimate_gb": est, "budget_gb": budget, "missing": missing,
            "blocking": blocking, "warnings": warnings, "force": force, "reindex": reindex,
            "license": spec.license}


# ── download ─────────────────────────────────────────────────────────────────

def expected_bytes(model_id: str) -> int | None:
    """Size of what will be downloaded (best effort; None when the hub cannot be asked)."""
    try:
        from huggingface_hub import HfApi

        info = HfApi().model_info(model_id, files_metadata=True)
        total = 0
        for f in info.siblings or []:
            if any(fnmatch.fnmatch(f.rfilename, pat) for pat in models.HF_IGNORE):
                continue
            total += int(getattr(f, "size", 0) or 0)
        return total or None
    except Exception:  # noqa: BLE001
        return None


def explain_download_error(model_id: str, exc: BaseException) -> str:
    names = {c.__name__ for c in type(exc).__mro__}
    text = str(exc).splitlines()[0] if str(exc) else ""
    if "GatedRepoError" in names:
        return (f"{model_id} is gated on Hugging Face: accept its licence on the model page and "
                "log in (`huggingface-cli login`), or choose another model")
    if "RepositoryNotFoundError" in names or "RevisionNotFoundError" in names:
        return f"{model_id} was not found on Hugging Face (check the spelling of the id)"
    if isinstance(exc, OSError) and getattr(exc, "errno", None) == 28:
        return "the disk is full (the models cache needs several GB free)"
    if names & {"LocalEntryNotFoundError", "ConnectionError", "Timeout", "HTTPError",
                "OfflineModeIsEnabled", "ProxyError", "SSLError", "ConnectTimeout"}:
        return f"cannot reach huggingface.co to download {model_id}: {text or type(exc).__name__}"
    return f"download of {model_id} failed: {type(exc).__name__}: {text}"


def download_model(model_id: str, task: Task) -> None:
    """Fetch *model_id* into the cache (no-op when it is complete), reporting progress."""
    if models.cache_state(model_id)["cached"]:
        task.log(f"{model_id}: already downloaded")
        return
    from .core.embedding import prepare_environment

    prepare_environment()
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise TaskError(f"huggingface_hub is not installed ({exc})") from exc
    task.log(f"downloading {model_id} ...")
    total = expected_bytes(model_id)
    outcome: dict[str, Any] = {}

    def work() -> None:
        try:
            outcome["path"] = snapshot_download(repo_id=model_id, ignore_patterns=models.HF_IGNORE)
        except BaseException as exc:  # noqa: BLE001 - re-raised in the caller's thread
            outcome["error"] = exc

    th = threading.Thread(target=work, name="download", daemon=True)
    th.start()
    while th.is_alive():
        th.join(1.0)
        task.progress(model=model_id, done=models.cache_state(model_id)["bytes"], total=total)
    if "error" in outcome:
        raise TaskError(explain_download_error(model_id, outcome["error"]))
    state = models.cache_state(model_id)
    task.progress(model=model_id, done=state["bytes"], total=total)
    if not state["cached"]:    # the hub call returned, but what is on disk is not a usable copy: say so
        raise TaskError(f"{model_id}: the download finished but the copy in the cache is not complete "
                        f"({state.get('why', '')}); the cache folder is {models.repo_dir(model_id)}")
    task.log(f"{model_id}: downloaded")


# ── verify (in its own process) ──────────────────────────────────────────────

def _memory(model: Any) -> int | None:
    fn = getattr(model, "memory_bytes", None)
    return fn() if callable(fn) else None


def _verify_here(kind: str, model_id: str) -> dict[str, Any]:
    """Load the model and check that it ranks a relevant passage above irrelevant ones."""
    import numpy as np

    from .core.embedding import make_embedder, make_reranker, prepare_environment

    prepare_environment()
    spec = models.spec_for(kind, model_id)
    t0 = time.time()
    docs = list(DOCS)
    if kind == models.EMBEDDING:
        emb = make_embedder(model_id)
        emb.load()
        t_load = time.time() - t0
        t1 = time.time()
        vecs = np.asarray(emb.encode(docs), dtype=np.float32)
        encode_query = getattr(emb, "encode_query", None) or (lambda q: emb.encode([q])[0])
        qs = [np.asarray(encode_query(q), dtype=np.float32).reshape(-1) for q, _ in QUERIES]
        t_enc = time.time() - t1
        if vecs.ndim != 2 or vecs.shape[0] != len(docs) or any(q.shape[0] != vecs.shape[1] for q in qs):
            raise TaskError(f"unexpected output shape {vecs.shape}")
        if not (np.isfinite(vecs).all() and all(np.isfinite(q).all() for q in qs)):
            raise TaskError("the model produced NaN/inf values on this machine "
                            "(try RAG_SEARCH_DEVICE=cpu or RAG_SEARCH_DTYPE=float32, or another model)")
        if spec.dim and vecs.shape[1] != spec.dim:
            raise TaskError(f"expected {spec.dim} dimensions but got {vecs.shape[1]}")
        ranked = [int(np.argmax(vecs @ q)) for q in qs]
        detail = {"dim": int(vecs.shape[1]), "load_s": round(t_load, 1),
                  "encode_s": round(t_enc, 2), "memory_bytes": _memory(emb)}
    else:
        rer = make_reranker(model_id)
        rer.load()
        t_load = time.time() - t0
        t1 = time.time()
        rows = [np.asarray(rer.score(q, docs), dtype=np.float32) for q, _ in QUERIES]
        t_enc = time.time() - t1
        if any(r.shape != (len(docs),) or not np.isfinite(r).all() for r in rows):
            raise TaskError("the reranker returned unusable scores on this machine "
                            "(try RAG_SEARCH_DEVICE=cpu or RAG_SEARCH_DTYPE=float32)")
        ranked = [int(np.argmax(r)) for r in rows]
        detail = {"load_s": round(t_load, 1), "score_s": round(t_enc, 2),
                  "memory_bytes": _memory(rer)}
    if ranked != [want for _, want in QUERIES]:
        raise TaskError(f"{model_id} did not put the relevant passage first for every test question; "
                        "it is probably not a "
                        + ("text-embedding" if kind == models.EMBEDDING else "reranking")
                        + " model in a format rag-search can use")
    return {"ok": True, **detail}


def verify_model(kind: str, model_id: str, timeout: float = 900.0) -> dict[str, Any]:
    """Run the smoke test in a subprocess (a crash or a huge model cannot take the caller down).
    -> {"ok": True, ...timings...} or {"ok": False, "error": "..."}."""
    try:
        proc = subprocess.run([sys.executable, "-m", "rag_search.model_tasks", "verify", kind,
                               model_id], capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": f"the test did not finish within {int(timeout)} s"}
    for line in reversed(proc.stdout.splitlines()):
        if line.startswith("RESULT "):
            with contextlib.suppress(ValueError):
                return json.loads(line[7:])
    tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-3:]
    return {"ok": False, "error": f"the test process failed (exit {proc.returncode}): "
            + " | ".join(tail)}


# ── the tasks ────────────────────────────────────────────────────────────────

def _check_switch(plan: dict[str, Any], force: bool) -> None:
    if plan["blocking"] and not force:
        raise TaskError("; ".join(plan["blocking"]) + " (use --force to try anyway)")


def _apply_reranker(paths: Paths, task: Task, model_id: str, previous: str) -> dict[str, Any]:
    from . import api, client

    models.set_selection(paths, models.RERANKER, model_id)
    if client.ping(paths, "search", 1.0) is None:
        task.log("the search daemon is not running; it will use the new reranker when it starts")
        return {"applied": True, "search_daemon": "not running"}
    task.phase("apply", "loading the new reranker in the search daemon ...")
    r = api.reload_search(paths)
    if r.get("reranker_error") or not r.get("ok"):
        err = r.get("reranker_error") or r.get("error") or "reload failed"
        _restore(paths, models.RERANKER, previous)
        raise TaskError(f"the search daemon could not load {model_id}: {err}; the previous "
                        "reranker stays in use")
    return {"applied": True, "search_daemon": "reloaded", "reranker": r.get("reranker", model_id)}


def _restore(paths: Paths, kind: str, previous: str) -> None:
    with contextlib.suppress(Exception):
        from .config import update_config

        update_config(paths, "models", {kind: previous if previous != models.DEFAULTS[kind] else ""})


def _apply_embedding(paths: Paths, task: Task, model_id: str, reindex: bool) -> dict[str, Any]:
    from . import api

    models.set_selection(paths, models.EMBEDDING, model_id)
    est = models.reindex_estimate(paths, model_id)
    if not reindex:
        task.log("saved; documents are embedded with the new model the next time they are "
                 "indexed (`rag-search index new`); search keeps using the current index until "
                 "every document has been re-embedded")
        return {"applied": True, "reindex": "not started", **{"documents": est["documents"]}}
    if not est["documents"]:
        task.log("no document needs re-embedding")
        return {"applied": True, "reindex": "nothing to do"}
    task.phase("apply", f"re-embedding {est['documents']} document(s) with {model_id} ...")
    r = api.index_start(paths, mode="new", restart=True, client="cli")
    if not r.get("ok"):
        raise TaskError(f"the model was saved, but indexing could not be started: "
                        f"{r.get('error', r)}; run `rag-search index new`")
    job = (r.get("job") or {}).get("id", "")
    task.log("re-embedding started" + (f" (job {job})" if job else "")
             + ": search keeps serving the previous index until it is complete")
    return {"applied": True, "reindex": "started", "job": job, "documents": est["documents"]}


def run_switch(paths: Paths, task: Task, req: dict[str, Any]) -> dict[str, Any]:
    kind, model_id = req["kind"], req["model"]
    force = bool(req.get("force"))
    plan = plan_switch(paths, kind, model_id, force=force)
    task.update(plan=plan)
    if plan["same"] and plan["cached"]:
        task.log(f"{model_id} is already the {kind} model; nothing to do")
        return {"applied": True, "unchanged": True}
    _check_switch(plan, force)
    previous = models.selection(kind)[0]
    task.phase("download", f"switching {kind} to {model_id}")
    download_model(model_id, task)
    if not req.get("no_verify"):
        task.phase("verify", f"testing {model_id} (loads it once in a separate process) ...")
        v = verify_model(kind, model_id)
        if not v.get("ok"):
            raise TaskError(f"{model_id} failed its test: {v.get('error')}. Nothing was changed.")
        task.log(f"test passed ({', '.join(f'{k} {v[k]}' for k in ('dim', 'load_s') if k in v)})")
    task.phase("apply")
    if kind == models.RERANKER:
        return _apply_reranker(paths, task, model_id, previous)
    return _apply_embedding(paths, task, model_id, bool(req.get("reindex", True)))


def run_download(paths: Paths, task: Task, req: dict[str, Any]) -> dict[str, Any]:
    ids = list(req.get("models") or [])
    if not ids:
        ids = [req["model"]]
    for i, mid in enumerate(ids, 1):
        task.phase("download", f"[{i}/{len(ids)}] {mid}")
        download_model(mid, task)
    return {"downloaded": ids}


def run_verify(paths: Paths, task: Task, req: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for kind, mid in req["targets"]:
        task.phase("verify", f"testing {mid} ...")
        r = verify_model(kind, mid)
        out[mid] = r
        task.log(f"{mid}: " + ("ok" if r.get("ok") else f"FAILED - {r.get('error')}"))
    if not all(r.get("ok") for r in out.values()):
        raise TaskError("; ".join(f"{m}: {r.get('error')}" for m, r in out.items() if not r.get("ok")))
    return {"verified": out}


def install_command(reqs: list[str]) -> list[str]:
    """How to install *reqs* into the environment rag-search itself runs in: uv when it can be found
    (``uv tool install`` environments have no pip), otherwise this interpreter's own pip.  Raises
    TaskError, with the command to run by hand, when neither is available."""
    uv = models.find_uv()
    if uv:
        return [uv, "pip", "install", "--python", sys.executable, *reqs]
    import importlib.util

    if importlib.util.find_spec("pip") is not None:
        return [sys.executable, "-m", "pip", "install", *reqs]
    quoted = " ".join(f'"{r}"' for r in reqs)
    raise TaskError("neither uv nor pip could be found for rag-search's Python environment (it was installed "
                    "with uv, which does not include pip, and uv is not in the dashboard's PATH). "
                    f"In a terminal run:  uv pip install --python \"{sys.executable}\" {quoted}")


def _stream(cmd: list[str], task: Task) -> int:
    """Run *cmd*, logging each output line to the task; returns the exit code."""
    proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1)
    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            if line.strip():
                task.log(line.rstrip()[:300])
        return proc.wait()
    except BaseException:
        proc.terminate()
        raise


def run_runtime(paths: Paths, task: Task, req: dict[str, Any]) -> dict[str, Any]:
    """Install the runtime of the document reader (mlx-vlm, ocrmac, pillow-heif: dependencies of rag-search, so this
    repairs an environment that lacks them).  Apple Silicon only; nothing else is touched and no model is downloaded."""
    if not models._apple_silicon():
        raise TaskError("the document reader runs on Apple Silicon Macs only; this computer uses docling OCR")
    reqs = models.runtime_requirements()
    cmd = install_command(reqs)
    task.phase("install", "installing " + ", ".join(reqs))
    task.log("$ " + " ".join(cmd))
    code = _stream(cmd, task)
    if code != 0:
        raise TaskError(f"the installer exited with code {code} (see the log; you can also run "
                        f"`uv pip install {' '.join(reqs)}` yourself)")
    task.phase("check", "checking the installation")
    state = models.runtime_state()
    if not state["ready"]:
        raise TaskError("the installer finished but mlx-vlm cannot be found by rag-search; "
                        f"it was installed for {sys.executable}")
    return {"installed": [x["package"] for x in state["packages"] if x["installed"]],
            "missing": [x["package"] for x in state["packages"] if not x["installed"]]}


RUNNERS: dict[str, Callable[[Paths, Task, dict[str, Any]], dict[str, Any]]] = {
    "switch": run_switch, "download": run_download, "verify": run_verify, "runtime": run_runtime}


def run(paths: Paths, request: dict[str, Any], echo: Callable[[str], None] | None = None,
        progress_echo: Callable[[dict[str, Any]], None] | None = None) -> dict[str, Any]:
    """Run one task to the end in this process and return its final record.
    Raises TaskBusy when another one is running."""
    if request.get("op") not in RUNNERS:
        raise TaskError(f"unknown task {request.get('op')!r}")
    task = Task(paths, request, echo, progress_echo)
    old: Any = None
    if threading.current_thread() is threading.main_thread():
        def on_term(_sig: int, _frame: Any) -> None:
            raise Cancelled()
        old = signal.signal(signal.SIGTERM, on_term)
    try:
        result = RUNNERS[request["op"]](paths, task, request)
        task.finish("succeeded", **result)
    except (Cancelled, KeyboardInterrupt):
        task.log("cancelled")
        task.finish("cancelled")
    except TaskError as exc:
        task.finish("failed", error=str(exc))
    except models.ModelError as exc:
        task.finish("failed", error=str(exc))
    except Exception as exc:  # noqa: BLE001
        task.finish("failed", error=f"{type(exc).__name__}: {exc}")
    finally:
        if old is not None:
            signal.signal(signal.SIGTERM, old)
    return task.rec


def start_detached(paths: Paths, request: dict[str, Any]) -> dict[str, Any]:
    """Start a task in its own background process (the dashboard); returns the queued record."""
    ensure_dirs(paths)
    if is_active(paths):
        raise TaskBusy("another model download or switch is already running")
    if request.get("op") not in RUNNERS:
        raise TaskError(f"unknown task {request.get('op')!r}")
    tid = uuid.uuid4().hex[:10]
    request = {**request, "task_id": tid}
    queued = {"id": tid, "op": request["op"], "kind": request.get("kind", ""),
              "model": request.get("model", ""), "status": "queued", "phase": "starting",
              "started_at": round(time.time(), 3), "updated_at": round(time.time(), 3),
              "log": [], "progress": {}, "result": {}}
    write_json_atomic(task_file(paths), queued, indent=None)
    log = open(log_file(paths), "ab")
    try:
        subprocess.Popen([sys.executable, "-m", "rag_search.model_tasks", "run",
                          json.dumps(request)], stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                         start_new_session=True,
                         **detached_start(paths.home, {**os.environ, "RAG_SEARCH_HOME": str(paths.home)}))
    finally:
        log.close()
    return queued


def cancel(paths: Paths) -> bool:
    """Stop the running task.  True if there was one."""
    rec = read_task(paths)
    if not rec or rec.get("status") not in ACTIVE or not rec.get("pid"):
        return False
    with contextlib.suppress(OSError):
        os.kill(int(rec["pid"]), signal.SIGTERM)
        return True
    return False


# ── entry point for the detached runner and the verify subprocess ────────────

def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) == 3 and argv[0] == "verify":
        try:
            res = _verify_here(argv[1], argv[2])
        except BaseException as exc:  # noqa: BLE001 - the parent reads this line
            msg = str(exc) if isinstance(exc, TaskError) else f"{type(exc).__name__}: {exc}"
            res = {"ok": False, "error": msg}
        print("RESULT " + json.dumps(res), flush=True)
        return 0 if res.get("ok") else 1
    if len(argv) == 2 and argv[0] == "run":
        req = json.loads(argv[1])
        paths = get_paths()
        try:
            rec = run(paths, req, echo=lambda m: print(m, flush=True))
        except TaskBusy as exc:
            print(str(exc), flush=True)
            return 2
        return 0 if rec.get("status") == "succeeded" else 1
    print("usage: python -m rag_search.model_tasks (verify KIND MODEL | run REQUEST_JSON)",
          file=sys.stderr)
    return 64


if __name__ == "__main__":
    raise SystemExit(main())
