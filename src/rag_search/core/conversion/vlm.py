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

Backends: ``mlx`` (mlx-vlm, Apple Silicon; a dependency of rag-search there) or ``module:attr`` of
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

from ... import machine
from . import degenerate

DEFAULT_BACKEND = "mlx"
LOAD_TIMEOUT_S = 900.0               # first start: the weights may still be read from disk
PAGE_TIMEOUT_S = 300.0               # one page; a dense page at 4k tokens takes ~1-2 minutes on a Mac
RENDER_PX = 2000                     # long side of the page image sent to the model
MEMORY_HEADROOM_GB = 2.0             # free memory wanted on top of the model's own size
MAX_RESTARTS = 2                     # a run restarts a dead reader this many times, then gives up
MAX_TOKENS = 4096
GUARD_VERSION = 1                    # the loop guard below; a page cached before it, and looping, is read again
RETRY_MAX_TOKENS = 3072              # a page that looped is read again with this limit,
RETRY_PENALTY = 1.2                  # a repetition penalty, and if that fails
STRIPS = 3                           # ... in this many horizontal strips

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
    return machine.available_memory_gb()


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
        self.limit = 0.0                 # the time limit of the request in progress, for the message
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
                raise ReaderTimeout(f"timed out {what} (limit {round(self.limit or timeout)} s)")
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

    def read(self, image: Path, prompt: str, *, timeout: float, max_tokens: int = MAX_TOKENS,
             repetition_penalty: float | None = None) -> dict[str, Any]:
        if not self.alive():
            raise ReaderCrashed("the reader process is not running")
        self.seq += 1
        self.limit = timeout
        req = {"op": "read", "id": self.seq, "image": str(image), "prompt": prompt, "max_tokens": max_tokens}
        if repetition_penalty:
            req["repetition_penalty"] = float(repetition_penalty)
        try:
            assert self.proc and self.proc.stdin
            self.proc.stdin.write(json.dumps(req) + "\n")
            self.proc.stdin.flush()
        except (OSError, ValueError) as exc:
            self.kill()
            raise ReaderCrashed(f"the reader process is gone: {exc}") from exc
        end = time.monotonic() + timeout                   # one limit for the page, whatever else the child prints
        while True:
            msg = self._next(max(0.0, end - time.monotonic()), "reading a page")
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


def split_bands(img: Path, n: int) -> list[Path]:
    """Cut the image file *img* into *n* horizontal strips, each cut at the whitest row near the even split
    (so a line of print is not cut in half); the strips are written next to it."""
    from PIL import Image

    out: list[Path] = []
    with Image.open(img) as im:
        w, h = im.size
        grey = im.convert("L").resize((1, h), Image.BOX)
        rows = list(grey.tobytes())                        # one brightness per row of the page
        cuts = [0]
        for i in range(1, n):
            mid, span = int(h * i / n), max(4, int(h * 0.06))
            lo, hi = max(cuts[-1] + 1, mid - span), min(h - 1, mid + span)
            cuts.append(max(range(lo, hi + 1), key=lambda y: rows[y]) if lo <= hi else mid)
        cuts.append(h)
        for i in range(n):
            p = img.with_name(f"{img.stem}.strip{i + 1}.png")
            im.crop((0, cuts[i], w, cuts[i + 1])).save(p)
            out.append(p)
    return out


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
        from . import profiler

        im = profiler.opaque(ImageOps.exif_transpose(im)).convert("RGB")       # upright, and on paper (a transparent ground is black otherwise)
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

        if not machine.mlx_possible():
            raise ReaderUnavailable("the MLX document reader needs an Apple Silicon Mac")
        import importlib.util

        if importlib.util.find_spec("mlx_vlm") is None:
            raise ReaderUnavailable("mlx-vlm is not installed (rag-search models runtime install)")
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
    def read_image(self, image: Path, what: str = "page", *, max_tokens: int = MAX_TOKENS,
                   repetition_penalty: float | None = None) -> dict[str, Any]:
        """``{"md", "tokens", "seconds", "rss_mb"}`` for one image file."""
        self.ensure_started()
        assert self.worker
        res = self.worker.read(image, prompt_for(self.style, what), timeout=page_timeout(),
                               max_tokens=max_tokens, repetition_penalty=repetition_penalty)
        self.tokens += int(res.get("tokens") or 0)
        self.pages_read += 1
        self.gpu_s += float(res.get("seconds") or 0.0)
        self.peak_mb = max(self.peak_mb, float(res.get("rss_mb") or 0.0))
        return dict(res, md=clean_reply(res.get("md")))

    def read_page_guarded(self, img: Path) -> tuple[dict[str, Any], str]:
        """One page image, read with a guard against a runaway.  A small model decoding greedily can fall
        into a loop on a dense page (the same line hundreds of times, or a stray script); the worker stops
        it after a few hundred tokens (``stopped == "loop"``) and this reads the page again: first with a
        repetition penalty and a lower limit, then in horizontal strips.  Of the readings that are not
        runaways the most complete (most text) is kept -- so a false alarm on a healthy page costs time, not
        text; when none is, the least repetitive one is, and the gate flags the page.  Returns (reading, a
        note saying what happened or "")."""
        first = self.read_image(img, "page")
        first_bad = degenerate.assess(first["md"])["bad"]
        if first.get("stopped") != "loop" and not first_bad:
            return first, ""
        tried = [first]
        good = [] if first_bad else [first]
        extra = {"tokens": int(first.get("tokens") or 0), "seconds": float(first.get("seconds") or 0.0)}
        ladder = (("with a repetition penalty", lambda: self.read_image(
                       img, "page", max_tokens=RETRY_MAX_TOKENS, repetition_penalty=RETRY_PENALTY)),
                  (f"in {STRIPS} strips", lambda: self.read_strips(img)))
        how_kept = ""
        for how, fn in ladder:
            try:
                res = fn()
            except ReaderError:
                if self.dead:
                    raise
                continue
            extra["tokens"] += int(res.get("tokens") or 0)
            extra["seconds"] += float(res.get("seconds") or 0.0)
            tried.append(res)
            if res.get("stopped") != "loop" and res["md"].strip() and not degenerate.assess(res["md"])["bad"]:
                good.append(res)
                how_kept = how
                break

        def runaway(r: dict[str, Any]) -> int:
            a = degenerate.assess(r["md"])
            return a["line_repeats"] + a["phrase_repeats"] + (10 ** 6 if not r["md"].strip() else 0)

        def done(res: dict[str, Any], note: str) -> tuple[dict[str, Any], str]:
            return dict(res, tokens=extra["tokens"], seconds=extra["seconds"]), note

        if good:
            best = max(good, key=lambda r: len(r["md"].strip()))
            if best is first:
                return done(best, "the reader looked as if it was repeating itself, but the first reading was "
                                  "the most complete and was kept")
            return done(best, f"the reader repeated itself on this page; read again {how_kept}")
        return done(min(tried, key=runaway), "the reader repeated itself on this page and two more readings did "
                                             "too; the least repetitive was kept")

    def read_strips(self, img: Path) -> dict[str, Any]:
        """Read *img* as STRIPS horizontal strips (cut where the page is blank) and join the texts."""
        parts, tokens, seconds = [], 0, 0.0
        strips = split_bands(img, STRIPS)
        try:
            for strip in strips:
                res = self.read_image(strip, "page")
                parts.append(res["md"].strip())
                tokens += int(res.get("tokens") or 0)
                seconds += float(res.get("seconds") or 0.0)
                if res.get("stopped") == "loop":
                    return {"md": "\n\n".join(p for p in parts if p), "tokens": tokens, "seconds": seconds,
                            "stopped": "loop"}
        finally:
            for strip in strips:
                strip.unlink(missing_ok=True)
        return {"md": "\n\n".join(p for p in parts if p), "tokens": tokens, "seconds": seconds, "stopped": ""}

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
                res, note = self.read_page_guarded(img)
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
                        "model": self.model, "note": note}
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
