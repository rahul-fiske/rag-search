"""The document VLM reader, parent side (stdlib only; nothing heavy is imported here).

A vision-language model reads a *page image* and writes Markdown.  It runs in a **child process**
(``vlm_worker.py``) so that

* its memory (MLX / Metal) never sits inside the indexer worker next to docling and torch, and
  goes back to the system the moment the child exits;
* a crash, a hang or an out-of-memory kill costs one page, not the run: the page is read by docling
  instead (branch ``fallback``, and the trace says why).

Protocol: one JSON object per line on the child's stdin / stdout.

    child  -> {"event": "ready", "model": ..., "backend": ..., "load_s": 7.1, "rss_mb": 3100}
    parent -> {"op": "read", "id": 1, "image": "/tmp/p1.png", "prompt": "...", "max_tokens": 4096}
    child  -> {"id": 1, "ok": true, "md": "...", "tokens": 812, "seconds": 9.4, "rss_mb": 3600}
    parent -> {"op": "quit"}

``VlmReader`` has the same ``read(src, first, last, mode)`` as the docling reader (``routed.py``), so
the router can hand it the scanned pages of a PDF or the frames of an image file.  The reader is
started when the first page is queued and kept for the rest of the run (``shared`` / ``close_shared``).

Backends: ``mlx`` (mlx-vlm, Apple Silicon; the extra ``rag-search[mac-vlm]``) or ``module:attr`` of
your own class (the tests use ``tests.helpers:FakeVlmBackend``).  ``$RAG_SEARCH_VLM_BACKEND`` chooses.
The real backend never downloads a model: one that is not on disk is reported, and the page is read
by docling.
"""

from __future__ import annotations

import json
import os
import queue
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

DEFAULT_BACKEND = "mlx"
LOAD_TIMEOUT_S = 900.0               # first start: the weights may still be read from disk
PAGE_TIMEOUT_S = 300.0               # one page; a dense page at 4k tokens takes ~1-2 minutes on a Mac
RENDER_PX = 2000                     # long side of the page image sent to the model
MEMORY_HEADROOM_GB = 2.0             # free memory wanted on top of the model's own size
MAX_RESTARTS = 2                     # a run restarts a dead reader this many times, then gives up
MAX_TOKENS = 4096

PROMPT_VERSION = "p2"                # bump when a prompt changes: pages are then read again
PROMPT_PAGE = (
    "Read this page and write its content as Markdown, exactly as printed. Keep the reading order. "
    "Write every table as an HTML <table> with one <tr> per row, one <td> per cell and no merged "
    "cells unless the page has them. Copy every number exactly, with its commas, decimal point and "
    "Dr/Cr or minus sign. Do not summarise, translate, correct or add anything. If part of the page "
    "is unreadable, write [illegible] there. If the page has no text at all (a photograph, a blank "
    "page), reply with nothing."
)
PROMPT_PICTURE = (
    "This is a picture taken from a document. Write all the text in it, exactly as printed, as "
    "Markdown (tables as HTML <table>). If it contains no text, answer with an empty reply."
)
PROMPT_CELL = (
    "This image is one cell of a table. Reply with only the text of the cell, exactly as printed: "
    "keep every comma, the decimal point and any minus sign or Dr/Cr. Do not add words, units or "
    "explanations."
)
PROMPT_PADDLE = {"page": "OCR:", "picture": "OCR:", "cell": "OCR:", "table": "Table Recognition:"}


class ReaderError(RuntimeError):
    """The VLM reader could not read a page; the page is read by docling instead."""
    reason = "error"


class ReaderUnavailable(ReaderError):
    reason = "unavailable"           # not installed, not enough memory, failed to load


class ReaderTimeout(ReaderError):
    reason = "timeout"


class ReaderCrashed(ReaderError):
    reason = "crashed"


# ── memory ────────────────────────────────────────────────────────────────────────────────

def available_memory_gb() -> float | None:
    """Memory that can be given to a new process without swapping, in GB; None when unknown.
    ``$RAG_SEARCH_VLM_FREE_GB`` overrides (tests, or a machine where the figure is misleading)."""
    env = os.environ.get("RAG_SEARCH_VLM_FREE_GB")
    if env:
        try:
            return float(env)
        except ValueError:
            pass
    try:                                                      # Linux
        with open("/proc/meminfo", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / 1024 / 1024
    except (OSError, ValueError, IndexError):
        pass
    if sys.platform == "darwin":                              # macOS: free + inactive + speculative
        try:
            out = subprocess.run(["vm_stat"], capture_output=True, text=True, timeout=5).stdout
            page = int(out.split("page size of")[1].split()[0])
            n = 0
            for line in out.splitlines():
                key, _, val = line.partition(":")
                if key.strip() in ("Pages free", "Pages inactive", "Pages speculative", "Pages purgeable"):
                    n += int(val.strip().rstrip("."))
            return n * page / 1024 ** 3
        except (OSError, ValueError, IndexError, subprocess.SubprocessError):
            pass
    return None


def memory_guard(need_gb: float) -> None:
    """Raise ReaderUnavailable when the machine does not have *need_gb* (plus headroom) free."""
    free = available_memory_gb()
    want = need_gb + MEMORY_HEADROOM_GB
    if free is not None and free < want:
        raise ReaderUnavailable(
            f"only {free:.1f} GB of memory is free and the document reader needs about {want:.1f} GB "
            "(model plus working memory)")


# ── settings ──────────────────────────────────────────────────────────────────────────────

def mode(env: Any = None) -> str:
    """``auto`` (use the reader when it can run) or ``off`` (``$RAG_SEARCH_VLM``)."""
    v = (os.environ if env is None else env).get("RAG_SEARCH_VLM", "auto").strip().lower() or "auto"
    return "off" if v in ("off", "0", "false", "no", "none") else "auto"


def backend_spec() -> str:
    return os.environ.get("RAG_SEARCH_VLM_BACKEND", "").strip() or DEFAULT_BACKEND


def page_timeout() -> float:
    try:
        return max(1.0, float(os.environ.get("RAG_SEARCH_VLM_PAGE_TIMEOUT") or PAGE_TIMEOUT_S))
    except ValueError:
        return PAGE_TIMEOUT_S


def prompt_for(style: str, what: str = "page") -> str:
    if style == "paddleocr":
        return PROMPT_PADDLE.get(what, "OCR:")
    return {"picture": PROMPT_PICTURE, "cell": PROMPT_CELL}.get(what, PROMPT_PAGE)


# ── the child process ─────────────────────────────────────────────────────────────────────

class Worker:
    """One ``vlm_worker`` child: start it, send it pages, stop it."""

    def __init__(self, backend: str, model: str, *, python: str | None = None) -> None:
        self.backend, self.model = backend, model
        self.python = python or sys.executable
        self.proc: subprocess.Popen | None = None
        self.lines: "queue.Queue[str | None]" = queue.Queue()
        self.ready: dict[str, Any] = {}
        self.seq = 0
        self.stderr_tail: list[str] = []

    # reading the child's stdout on a thread gives every wait a timeout
    def _pump(self, stream: Any, out: "queue.Queue[str | None]") -> None:
        try:
            for line in iter(stream.readline, ""):
                out.put(line)
        except (OSError, ValueError):
            pass
        out.put(None)

    def _pump_err(self, stream: Any) -> None:
        try:
            for line in iter(stream.readline, ""):
                self.stderr_tail = (self.stderr_tail + [line.rstrip()])[-12:]
        except (OSError, ValueError):
            pass

    def start(self, timeout: float = LOAD_TIMEOUT_S) -> dict[str, Any]:
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join(p for p in sys.path if p)
        env["PYTHONUNBUFFERED"] = "1"
        if self.backend == "mlx":
            env["HF_HUB_OFFLINE"] = "1"                   # a reader never downloads by itself
        cmd = [self.python, "-m", "rag_search.core.conversion.vlm_worker",
               "--backend", self.backend, "--model", self.model]
        try:
            self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                         stderr=subprocess.PIPE, text=True, env=env, bufsize=1,
                                         encoding="utf-8")
        except OSError as exc:
            raise ReaderUnavailable(f"cannot start the reader process: {exc}") from exc
        threading.Thread(target=self._pump, args=(self.proc.stdout, self.lines), daemon=True).start()
        threading.Thread(target=self._pump_err, args=(self.proc.stderr,), daemon=True).start()
        try:
            msg = self._next(timeout, "loading the model")
        except ReaderError as exc:                        # a model that does not load is not a page problem
            raise ReaderUnavailable(str(exc)) from exc
        if msg.get("event") != "ready":
            self.close()
            raise ReaderUnavailable(str(msg.get("error") or "the reader did not start"))
        self.ready = msg
        return msg

    def _next(self, timeout: float, what: str) -> dict[str, Any]:
        end = time.monotonic() + timeout
        while True:
            left = end - time.monotonic()
            if left <= 0:
                self.kill()
                raise ReaderTimeout(f"timed out after {round(timeout)} s {what}")
            try:
                line = self.lines.get(timeout=min(left, 1.0))
            except queue.Empty:
                continue
            if line is None:
                rc = self.proc.poll() if self.proc else None
                time.sleep(0.05)                          # let the stderr thread catch up
                tail = " | ".join(self.stderr_tail[-3:])
                self.kill()
                raise ReaderCrashed(f"the reader process ended (exit {rc}) {what}"
                                    + (f": {tail[:300]}" if tail else ""))
            try:
                msg = json.loads(line)
            except ValueError:
                continue                                  # stray output from a library
            if isinstance(msg, dict):
                return msg

    def alive(self) -> bool:
        return bool(self.proc and self.proc.poll() is None)

    def read(self, image: Path, prompt: str, *, timeout: float, max_tokens: int = MAX_TOKENS) -> dict[str, Any]:
        if not self.alive():
            raise ReaderCrashed("the reader process is not running")
        self.seq += 1
        req = {"op": "read", "id": self.seq, "image": str(image), "prompt": prompt, "max_tokens": max_tokens}
        try:
            assert self.proc and self.proc.stdin
            self.proc.stdin.write(json.dumps(req) + "\n")
            self.proc.stdin.flush()
        except (OSError, ValueError) as exc:
            self.kill()
            raise ReaderCrashed(f"the reader process is gone: {exc}") from exc
        while True:
            msg = self._next(timeout, "reading a page")
            if msg.get("id") == self.seq:
                break
        if not msg.get("ok"):
            raise ReaderError(str(msg.get("error") or "the reader failed on this page"))
        return msg

    def kill(self) -> None:
        p = self.proc
        if p and p.poll() is None:
            try:
                p.kill()
                p.wait(timeout=10)
            except (OSError, subprocess.SubprocessError):
                pass

    def close(self) -> None:
        p = self.proc
        if not p:
            return
        if p.poll() is None:
            try:
                assert p.stdin
                p.stdin.write(json.dumps({"op": "quit"}) + "\n")
                p.stdin.flush()
                p.wait(timeout=8)
            except (OSError, ValueError, subprocess.SubprocessError):
                pass
        self.kill()
        for s in (p.stdin, p.stdout, p.stderr):
            try:
                if s:
                    s.close()
            except OSError:
                pass


# ── page images ───────────────────────────────────────────────────────────────────────────

def render_pdf_page(src: Path, page: int, out: Path, *, crop: tuple[float, float, float, float] | None = None,
                    long_side: int = RENDER_PX) -> Path:
    """Write page *page* (1-based) of *src* as a PNG (rendered with pypdfium2; the source is only
    read).  *crop* = (left, top, right, bottom) as fractions of the page cuts a region out."""
    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument(str(src))
    try:
        pg = pdf[page - 1]
        try:
            w, h = pg.get_size()
            if crop:
                w, h = w * (crop[2] - crop[0]), h * (crop[3] - crop[1])
            scale = max(0.5, min(4.0, long_side / max(1.0, max(w, h))))
            im = pg.render(scale=scale).to_pil().convert("RGB")
        finally:
            pg.close()
    finally:
        pdf.close()
    if crop:
        W, H = im.size
        im = im.crop((int(crop[0] * W), int(crop[1] * H), max(int(crop[2] * W), int(crop[0] * W) + 1),
                      max(int(crop[3] * H), int(crop[1] * H) + 1)))
    out.parent.mkdir(parents=True, exist_ok=True)
    im.save(out, "PNG")
    return out


def render_image_frame(src: Path, frame: int, out: Path, *, long_side: int = RENDER_PX,
                       crop: tuple[float, float, float, float] | None = None) -> Path:
    """Write frame *frame* (1-based) of an image file as an upright PNG (EXIF orientation applied,
    shrunk to *long_side*).  *crop* = (left, top, right, bottom) as fractions cuts a region out first.
    HEIC/HEIF need pillow-heif."""
    from PIL import Image, ImageOps

    try:
        import pillow_heif                                    # type: ignore

        pillow_heif.register_heif_opener()
    except ImportError:
        pass
    with Image.open(src) as im:
        if frame > 1:
            im.seek(frame - 1)
        im = ImageOps.exif_transpose(im.convert("RGB"))
    if crop:
        W, H = im.size
        im = im.crop((int(crop[0] * W), int(crop[1] * H), max(int(crop[2] * W), int(crop[0] * W) + 1),
                      max(int(crop[3] * H), int(crop[1] * H) + 1)))
    im.thumbnail((long_side, long_side))
    out.parent.mkdir(parents=True, exist_ok=True)
    im.save(out, "PNG")
    return out


# ── the reader the router uses ────────────────────────────────────────────────────────────

class VlmReader:
    """Reads pages with the document VLM.  ``read`` has the docling reader's shape."""

    def __init__(self, model: str, *, style: str = "", need_gb: float = 0.0, backend: str | None = None,
                 tmp_dir: Path | None = None) -> None:
        self.model = model
        self.style = style
        self.need_gb = need_gb
        self.backend = backend or backend_spec()
        self.tool = "vlm"
        self.id = f"vlm:{model}#{PROMPT_VERSION}"      # part of every page-cache key
        self.worker: Worker | None = None
        self.restarts = 0
        self.dead = ""                       # why the reader was given up for this run
        self.load_s = 0.0
        self.peak_mb = 0.0
        self.tokens = 0
        self.pages_read = 0
        self.gpu_s = 0.0
        self.tmp = tmp_dir

    # -- lifecycle
    def _tmp_dir(self) -> Path:
        if self.tmp is None:
            self.tmp = Path(tempfile.mkdtemp(prefix="rag-vlm-"))
        return self.tmp

    def usable(self) -> bool:
        return not self.dead

    def preflight(self) -> None:
        """The cheap checks before a process is started.  The real backend never downloads: a model
        that is not on disk is reported (models are fetched only when you ask for them)."""
        if self.backend != "mlx":
            return
        from ... import models

        if sys.platform != "darwin" or os.uname().machine != "arm64":
            raise ReaderUnavailable("the MLX document reader needs an Apple Silicon Mac")
        import importlib.util

        if importlib.util.find_spec("mlx_vlm") is None:
            raise ReaderUnavailable("mlx-vlm is not installed (pip install 'rag-search[mac-vlm]')")
        if not models.cache_state(self.model)["cached"]:
            raise ReaderUnavailable(f"the model {self.model} is not downloaded "
                                    f"(rag-search models download {self.model})")

    def check(self) -> None:
        """Rule the reader out for the rest of the run when it cannot be started at all (not
        installed, wrong machine, model not downloaded).  Cheap; nothing is started."""
        if self.dead:
            return
        try:
            self.preflight()
        except ReaderUnavailable as exc:
            self.dead = str(exc)

    def ensure_started(self) -> None:
        if self.dead:
            raise ReaderUnavailable(self.dead)
        if self.worker and self.worker.alive():
            return
        if self.worker is not None:                   # it died: restart a limited number of times
            self.worker.close()                       # release the dead process's pipes
            self.worker = None
            if self.restarts >= MAX_RESTARTS:
                self.dead = "the reader process stopped too often in this run"
                raise ReaderUnavailable(self.dead)
            self.restarts += 1
        self.check()
        if self.dead:
            raise ReaderUnavailable(self.dead)
        memory_guard(self.need_gb)                    # not sticky: memory may be free again for the next page
        try:
            w = Worker(self.backend, self.model)
            t0 = time.perf_counter()
            w.start()
            self.load_s += round(time.perf_counter() - t0, 2)
            self.worker = w
        except ReaderUnavailable as exc:
            w.close()
            self.dead = str(exc)                      # a model that fails to load will not load on the next page
            raise

    def close(self) -> None:
        if self.worker:
            self.worker.close()
            self.worker = None
        if self.tmp:
            shutil.rmtree(self.tmp, ignore_errors=True)
            self.tmp = None

    # -- reading
    def read_image(self, image: Path, what: str = "page") -> dict[str, Any]:
        """``{"md", "tokens", "seconds", "rss_mb"}`` for one image file."""
        self.ensure_started()
        assert self.worker
        res = self.worker.read(image, prompt_for(self.style, what), timeout=page_timeout())
        self.tokens += int(res.get("tokens") or 0)
        self.pages_read += 1
        self.gpu_s += float(res.get("seconds") or 0.0)
        self.peak_mb = max(self.peak_mb, float(res.get("rss_mb") or 0.0))
        return dict(res, md=clean_reply(res.get("md")))

    def read(self, src: Path, first: int, last: int, mode: str) -> dict[str, Any]:
        """Read pages *first*..*last* of the PDF (or frames of the image file) *src*, one model call
        each.  Pages the reader cannot read are left out of ``pages`` and listed in ``failed`` with the
        reason: the router reads those with docling.  Returns ``{"pages", "stats", "seconds",
        "failed"}``."""
        t0 = time.perf_counter()
        pages: dict[int, str] = {}
        stats: dict[int, dict[str, Any]] = {}
        failed: dict[int, str] = {}
        tmp = self._tmp_dir()
        for n in range(first, last + 1):
            t_page = time.perf_counter()
            img = tmp / f"p{n}.png"
            try:
                if src.suffix.lower() == ".pdf":
                    render_pdf_page(src, n, img)
                else:
                    render_image_frame(src, n, img)
                res = self.read_image(img, "page")
            except ReaderError as exc:
                failed[n] = f"{exc.reason}: {exc}"
                if self.dead:                          # nothing more will be read this run
                    for m in range(n + 1, last + 1):
                        failed[m] = f"unavailable: {self.dead}"
                    break
                continue
            except Exception as exc:  # noqa: BLE001 - a page that cannot be drawn
                failed[n] = f"error: cannot render the page: {type(exc).__name__}: {exc}"
                continue
            finally:
                img.unlink(missing_ok=True)
            pages[n] = str(res.get("md") or "")
            stats[n] = {"chars": len("".join(pages[n].split())), "tokens": int(res.get("tokens") or 0),
                        "gpu_s": round(float(res.get("seconds") or 0.0), 3),
                        "read_s": round(time.perf_counter() - t_page, 3),
                        "model": self.model}
        return {"pages": pages, "stats": stats, "failed": failed,
                "seconds": round(time.perf_counter() - t0, 2)}


_FENCE_OPEN = re.compile(r"^\s*```[ \t]*(markdown|md|html)?[ \t]*\n", re.I)
_FENCE_CLOSE = re.compile(r"\n[ \t]*```[ \t]*$")


def clean_reply(text: Any) -> str:
    """The model's answer as Markdown: a vision model often wraps the whole page in a ```markdown fence, which
    would index the page as one code block.  A fence around the *whole* answer is removed (an opening
    ```markdown / ```md fence with a closing one at the end; a bare fence only when it is the answer's one
    pair); fenced code inside the page is left alone."""
    s = str(text or "").strip()
    m = _FENCE_OPEN.match(s)
    if not m:
        return s
    body = s[m.end():]
    fences = len(re.findall(r"(?m)^[ \t]*```", body))
    if _FENCE_CLOSE.search(s):
        if m.group(1) and m.group(1).lower() in ("markdown", "md") or fences == 1:
            return _FENCE_CLOSE.sub("", body).strip()
    elif fences == 0:                               # an opening fence the answer never closed (cut off at the token limit)
        return body.strip()
    return s


# ── one reader per run ────────────────────────────────────────────────────────────────────

_SHARED: dict[str, VlmReader] = {}


def selected_model() -> tuple[str, str, float]:
    """(model id, prompt style, memory it needs in GB) of the chosen document reader."""
    from ... import models

    return models.reader_choice()


def shared() -> VlmReader | None:
    """The reader this process uses, created on first use; None when the reader is switched off."""
    if mode() == "off":
        return None
    model, style, need = selected_model()
    key = f"{backend_spec()}|{model}"
    r = _SHARED.get(key)
    if r is None:
        if not _SHARED:
            import atexit

            atexit.register(close_shared)             # a pool process that exits takes its reader with it
        r = _SHARED[key] = VlmReader(model, style=style, need_gb=need)
    return r


def repair_shared() -> VlmReader | None:
    """The reader that re-reads suspect cells and pages: the document reader itself when the repair
    model is the same (one process, one model in memory), else its own reader.  None when switched off."""
    if mode() == "off":
        return None
    from ... import models

    model, style, need = models.repair_choice()
    rd = shared()
    if rd is not None and rd.model == model:
        return rd
    key = f"repair|{backend_spec()}|{model}"
    r = _SHARED.get(key)
    if r is None:
        r = _SHARED[key] = VlmReader(model, style=style, need_gb=need)
    return r


def close_shared() -> None:
    for r in list(_SHARED.values()):
        r.close()
    _SHARED.clear()
