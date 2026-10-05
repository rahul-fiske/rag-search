"""Shared test helpers: a deterministic fake embedder/reranker (no ML deps needed)."""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for _p in (str(SRC), str(ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
SUBPROC_PYTHONPATH = str(SRC) + os.pathsep + str(ROOT)

from rag_search.paths import Paths, ensure_dirs, get_paths  # noqa: E402

DIM = 64


class FakeEmbedder:
    """Hashed bag-of-words vectors: texts that share words are close."""

    def __init__(self):
        self.calls = 0

    def load(self):
        return self

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(DIM, dtype=np.float32)
        for w in re.findall(r"\w+", text.lower()):
            h = int(hashlib.md5(w.encode()).hexdigest(), 16)
            v[h % DIM] += 1.0
        n = np.linalg.norm(v)
        return v / n if n else v

    def encode(self, texts, progress=None):
        self.calls += 1
        arr = np.stack([self._vec(t) for t in texts]) if texts else np.zeros((0, DIM), np.float32)
        if progress:
            progress(len(texts), len(texts))
        return arr


class FakeVlmBackend:
    """A document-reader backend (``vlm_worker``) that needs no model.  It "reads" a page image and
    answers with text derived from the image, so a test can tell pages apart.  Behaviour is steered
    by a JSON file named in $RAG_TEST_VLM_PLAN (read on every call, shared with the child process):

    ``{"load_error": "...", "crash_on": [3], "hang_on": [2], "error_on": [4], "empty": true,
       "text": "custom page text", "log": "/path/calls.log"}``

    A runaway reader: ``"loop_over": 1500`` makes an image taller than that (a whole page; its strips are
    shorter) answer with a loop and ``stopped: "loop"``; ``"penalty_fixes": true`` lets a repetition
    penalty cure it; ``"always_loop": true`` loops on every image; ``"quiet_loop": true`` leaves out the
    ``stopped`` flag (the text alone gives it away).

    ``crash_on`` / ``hang_on`` / ``error_on`` count calls across all child processes (the counter
    lives next to the plan file), 1-based.
    """

    def __init__(self, model_id):
        plan = self._plan()
        if plan.get("load_error"):
            raise RuntimeError(plan["load_error"])
        self.model = model_id

    @staticmethod
    def _plan():
        import json
        f = os.environ.get("RAG_TEST_VLM_PLAN")
        return json.loads(Path(f).read_text()) if f else {}

    def _count(self):
        f = Path(os.environ["RAG_TEST_VLM_PLAN"]).with_suffix(".count")
        n = (int(f.read_text()) if f.exists() else 0) + 1
        f.write_text(str(n))
        return n

    def read(self, image_path, prompt, max_tokens, repetition_penalty=None):
        import hashlib
        import time
        plan = self._plan()
        n = self._count() if os.environ.get("RAG_TEST_VLM_PLAN") else 0
        if plan.get("log"):
            with open(plan["log"], "a", encoding="utf-8") as fh:
                fh.write(f"{Path(image_path).name}\t{prompt[:20]}\t{repetition_penalty or ''}\n")
        if plan.get("native_write"):
            os.write(1, b"native library chatter on fd 1\n")      # must not reach the protocol channel
        if n in plan.get("crash_on", []):
            os._exit(9)
        if n in plan.get("hang_on", []):
            time.sleep(60)
        if n in plan.get("error_on", []):
            raise RuntimeError("the model could not read this page")
        if prompt.startswith("This image is one cell") or (prompt == "OCR:" and plan.get("cell_ocr")):
            by_call = plan.get("cell_by_call") or {}
            text = by_call.get(str(n), plan.get("cell", ""))
            return {"md": text, "tokens": 3}
        if plan.get("empty"):
            return {"md": "", "tokens": 0}
        runaway = plan.get("always_loop")
        if plan.get("loop_over") is not None:
            from PIL import Image
            with Image.open(image_path) as im:
                runaway = runaway or im.size[1] > plan["loop_over"]
        if runaway and not (plan.get("penalty_fixes") and repetition_penalty):
            loop = {"md": "The said property shall be conveyed to the purchaser free of all charges. " * 80,
                    "tokens": 900}
            return loop if plan.get("quiet_loop") else dict(loop, stopped="loop")
        time.sleep(0.02)
        digest = hashlib.sha256(Path(image_path).read_bytes()).hexdigest()[:8]
        text = (plan.get("text_by_model") or {}).get(self.model) or plan.get("text") or (
            f"Scanned statement page {digest}. The account holder deposited the amount on the first of the "
            "month and the bank confirmed the balance carried forward.")
        return {"md": text, "tokens": 40 + len(text) // 8}


class FakeSecondReader:
    """The independent second reader (``repair.second_reader``) without Apple Vision: words and boxes
    from the JSON file named by $RAG_TEST_SECOND_PLAN: ``{"words": [["text", l, t, r, b], ...],
    "error": "..."}`` (fractions of the page, origin top-left)."""

    id = "fake-second"

    def words(self, image):
        import json
        from rag_search.core.conversion.repair import Word
        plan = json.loads(Path(os.environ["RAG_TEST_SECOND_PLAN"]).read_text())
        if plan.get("error"):
            raise RuntimeError(plan["error"])
        return [Word(w[0], *w[1:5]) for w in plan.get("words", [])]


class FakeEngine:
    """A benchmark engine that "reads" pages from a JSON file named by $RAG_TEST_FAKE_PAGES:
    ``{"<file name>": {"<page>": "<markdown>"}}``.  Files not listed raise (a reader failure)."""

    name = "fake"

    def describe(self):
        return {"name": self.name}

    def read_pages(self, src, pages):
        import json
        table = json.loads(Path(os.environ["RAG_TEST_FAKE_PAGES"]).read_text())
        if Path(src).name not in table:
            raise RuntimeError(f"fake engine cannot read {Path(src).name}")
        got = table[Path(src).name]
        return {"pages": {p: got.get(str(p), "") for p in pages}, "seconds": 0.5 * len(got),
                "pages_read": len(got), "page_count": len(got)}


class SlowEmbedder(FakeEmbedder):
    """FakeEmbedder that sleeps $TEST_EMBED_SLEEP seconds per call (to test cancel/restart)."""

    def encode(self, texts, progress=None):
        import time
        time.sleep(float(os.environ.get("TEST_EMBED_SLEEP", "0")))
        return super().encode(texts, progress)


class FakeReranker:
    def load(self):
        return self

    def score(self, query, texts):
        q = set(re.findall(r"\w+", query.lower()))
        out = []
        for t in texts:
            w = set(re.findall(r"\w+", t.lower()))
            out.append(len(q & w) / (len(q) or 1))
        return out


class TempHome(unittest.TestCase):
    """Each test gets an isolated data folder and a clean environment."""

    ENV_KEYS = ("RAG_SEARCH_HOME",
                "RAG_SEARCH_DOCLING_PYTHON", "RAG_SEARCH_OCR", "RAG_SEARCH_MODEL",
                "RAG_SEARCH_OCR_ENGINE", "RAG_SEARCH_OCR_LANG", "RAG_SEARCH_TABLE_MODE", "RAG_SEARCH_PIPELINE", "RAG_SEARCH_ROUTING", "RAG_SEARCH_PDF_BACKEND", "RAG_SEARCH_THREADS", "RAG_SEARCH_DOC_TIMEOUT",
                "RAG_SEARCH_EMBEDDER", "RAG_SEARCH_RERANKER", "RAG_SEARCH_CLIENT",
                "RAG_SEARCH_IDLE_SECONDS", "RAG_SEARCH_PREWARM", "RAG_SEARCH_JOBS",
                "TEST_EMBED_SLEEP", "RAG_SEARCH_RERANK_MODEL", "RAG_SEARCH_RERANK",
                "RAG_SEARCH_DTYPE", "RAG_SEARCH_DEVICE", "HF_HOME", "HF_HUB_CACHE",
                "HUGGINGFACE_HUB_CACHE", "RAG_SEARCH_DOCLING_BATCH", "RAG_TEST_FAKE_PAGES", "RAG_TEST_VLM_PLAN", "RAG_SEARCH_VLM", "RAG_SEARCH_VLM_BACKEND", "RAG_SEARCH_VLM_FREE_GB", "RAG_SEARCH_VLM_PAGE_TIMEOUT", "RAG_SEARCH_VLM_MODEL", "RAG_SEARCH_REPAIR_MODEL", "RAG_SEARCH_REPAIR", "RAG_SEARCH_REPAIR_SECOND", "RAG_TEST_SECOND_PLAN",
                "PYTHONPATH")

    def setUp(self):
        self._saved = {k: os.environ.pop(k, None) for k in self.ENV_KEYS}
        self.tmp = Path(tempfile.mkdtemp(prefix="ragtest-")).resolve()
        os.environ["RAG_SEARCH_HOME"] = str(self.tmp / "home")
        self.paths: Paths = get_paths()
        ensure_dirs(self.paths)
        self.sdir = self.tmp / "sources"      # where the tests' source folders live; each is registered
        self.sdir.mkdir()

    def tearDown(self):
        for k, v in self._saved.items():
            os.environ.pop(k, None)
            if v is not None:
                os.environ[k] = v
        stop_leftover_processes(self.tmp)
        shutil.rmtree(self.tmp, ignore_errors=True)

    def source(self, collection: str) -> Path:
        """The folder of *collection* under ``self.sdir``, created and registered as its location
        (written directly: the reachability and overlap checks of ``locations.add`` are not under test)."""
        from rag_search import locations
        folder = self.sdir / collection
        folder.mkdir(parents=True, exist_ok=True)
        locs = locations.load(self.paths)[0]
        if locs.get(collection) != str(folder):
            locs[collection] = str(folder)
            locations._save(self.paths, locs)
        return folder

    def register_tree(self, root: Path | None = None) -> None:
        """Register every first-level folder of *root* (default ``self.sdir``) as a location."""
        from rag_search import locations
        root = root or self.sdir
        locs = locations.load(self.paths)[0]
        locs.update({d.name: str(d) for d in sorted(root.iterdir()) if d.is_dir() and not d.name.startswith(".")})
        locations._save(self.paths, locs)

    def roots(self):
        """The registered locations as run_index takes them (every folder under ``self.sdir`` first)."""
        from rag_search import locations
        self.register_tree()
        return locations.source_roots(self.paths)

    def write_doc(self, rel: str, text: str) -> Path:
        p = self.source(Path(rel).parts[0]).parent / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
        return p

    # -- convenience for pipeline tests ---------------------------------------
    def index(self, embedder=None, **kw):
        from rag_search.core import indexer
        roots = self.roots()
        srcs = [f for _n, r in roots.locations
                for f in indexer.scan_sources(Path(r), indexer.exclude_dirs(self.paths))]
        return indexer.run_index(self.paths, srcs, roots, jobs=1,
                                 embedder=embedder or FakeEmbedder(), **kw)

    def publish(self, **kw):
        from rag_search import publish
        return publish.publish(self.paths, **kw)

    def engine(self, **kw):
        from rag_search.core.search import SearchEngine
        kw.setdefault("embedder", FakeEmbedder())
        kw.setdefault("reranker", FakeReranker())
        e = SearchEngine(self.paths, **kw)
        e.install(e.prepare_generation())
        return e

    def use_fake_backends(self):
        """Make spawned daemons/workers use the fake embedder/reranker."""
        os.environ["RAG_SEARCH_EMBEDDER"] = "tests.helpers:FakeEmbedder"
        os.environ["RAG_SEARCH_RERANKER"] = "tests.helpers:FakeReranker"
        os.environ["PYTHONPATH"] = SUBPROC_PYTHONPATH


def leave_a_stuck_thread() -> str:
    """Pool task for the pool-shutdown test: returns at once but leaves a non-daemon thread that
    never ends, like a docling stage thread abandoned inside a native call."""
    import threading

    threading.Thread(target=threading.Event().wait).start()
    return "done"


def no_real_reader(case) -> None:
    """Make the MLX document reader unavailable for one test, on any machine: the code under test
    then sees "not an Apple Silicon Mac", which is what a test about the fallback path needs.  Without
    it such a test passes in the cloud and fails on a Mac that has the reader and its model."""
    import types
    from unittest import mock

    p = mock.patch("os.uname", return_value=types.SimpleNamespace(
        sysname="Linux", nodename="test", release="0", version="0", machine="x86_64"))
    p.start()
    case.addCleanup(p.stop)


def stop_leftover_processes(tmp: Path) -> int:
    """Stop the daemons and dashboards a test started in its temporary data folder and did not stop
    (they are always-on by design, so they would outlive the test run; dozens per run).  Only a
    process whose id is in a ``run/*.pid`` file under *tmp* and which is a rag-search process."""
    import signal
    import subprocess

    stopped = 0
    for pid_file in Path(tmp).rglob("run/*.pid"):
        try:
            pid = int(pid_file.read_text().strip())
            cmd = subprocess.run(["ps", "-p", str(pid), "-o", "command="], capture_output=True,
                                 text=True, timeout=10).stdout
            if "rag_search" in cmd:
                os.kill(pid, signal.SIGTERM)
                stopped += 1
        except (OSError, ValueError, subprocess.SubprocessError):
            continue
    return stopped

