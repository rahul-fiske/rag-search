"""Real-tool tests: docling with OCR, real (small) embedding and reranking models, real daemons.

They run wherever those tools are installed -- a Linux cloud session as well as the Mac -- and test the
framework, not the quality of the models: see "Tests" in CONTRIBUTING.md and ``scripts/cloud_setup.sh``.
"""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from tests import guard
from tests.helpers import SUBPROC_PYTHONPATH, stop_leftover_processes

# small models: the framework needs *a* sentence-transformers embedder and *a* cross-encoder, not good ones
EMBED_MODEL = os.environ.get("RAG_TEST_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
RERANK_MODEL = os.environ.get("RAG_TEST_RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-2-v2")
TOOLS = ("docling", "torch", "sentence_transformers", "pypdfium2", "PIL")


def missing_tools() -> str:
    gone = [m for m in TOOLS if importlib.util.find_spec(m) is None]
    if gone:
        return "real-tool tests need " + ", ".join(gone) + " (scripts/cloud_setup.sh, or the uv tool environment)"
    if sys.platform != "darwin" and not shutil.which("tesseract"):
        return "real-tool tests need an OCR engine: tesseract (apt-get install tesseract-ocr)"
    return ""


def real_env(home: Path) -> dict[str, str]:
    """The environment of every process a real test starts (CLI, daemons, workers)."""
    env = {k: v for k, v in os.environ.items() if not k.startswith("RAG_SEARCH_") and not k.startswith("RAG_TEST_")}
    env.update(RAG_SEARCH_HOME=str(home), RAG_SEARCH_MODEL=EMBED_MODEL, RAG_SEARCH_RERANK_MODEL=RERANK_MODEL,
               RAG_SEARCH_VLM="off",                  # the document reader is the Mac's: tests/machine/
               RAG_SEARCH_REPAIR="off", TOKENIZERS_PARALLELISM="false",
               RAG_SEARCH_MAX_SEQ="512",              # the small test models stop at 512 tokens, bge-m3 at 8192
               PYTHONPATH=os.pathsep.join(filter(None, [SUBPROC_PYTHONPATH, os.environ.get("PYTHONPATH")])))
    if shutil.which("tesseract"):                    # Linux has no Apple Vision; docling's default OCR downloads
        env.update(RAG_SEARCH_OCR_ENGINE="tesseract", RAG_SEARCH_OCR_LANG="eng")   # from a host often blocked
    return env


class RealCase(unittest.TestCase):
    """A temporary data folder shared by the tests of one class, and the CLI run as a real process."""

    @classmethod
    def setUpClass(cls):
        guard.unblock()                               # the portable tier blocks the real libraries in this process
        why = missing_tools()
        if why:
            raise unittest.SkipTest(why)
        cls.tmp = Path(tempfile.mkdtemp(prefix="rag-real-")).resolve()
        cls.home = cls.tmp / "home"
        cls.env = real_env(cls.home)
        cls._saved = {k: os.environ.get(k) for k in cls.env}
        os.environ.update(cls.env)                    # in-process api calls see the same data folder
        from rag_search.paths import ensure_dirs, get_paths
        cls.paths = get_paths()
        ensure_dirs(cls.paths)

    @classmethod
    def tearDownClass(cls):
        if not hasattr(cls, "tmp"):
            return
        try:
            cls.cli("daemon", "stop", check=False)
        finally:
            stop_leftover_processes(cls.tmp)
            for k, v in cls._saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
            shutil.rmtree(cls.tmp, ignore_errors=True)

    @classmethod
    def cli(cls, *args: str, check: bool = True, timeout: int = 1800) -> subprocess.CompletedProcess:
        p = subprocess.run([sys.executable, "-m", "rag_search", *args], capture_output=True, text=True,
                           env=cls.env, cwd=str(cls.tmp), timeout=timeout)
        if check and p.returncode != 0:
            raise AssertionError(f"rag-search {' '.join(args)} exited {p.returncode}:\n{p.stdout[-2000:]}\n{p.stderr[-3000:]}")
        return p

    @classmethod
    def cli_json(cls, *args: str, ok_codes: tuple[int, ...] = (0,), **kw):
        """Run a command with ``--json`` and parse its output; an exit code outside *ok_codes* fails
        (``index foreground`` exits 1 when a document failed, which the corpus does on purpose)."""
        p = cls.cli(*args, "--json", check=False, **kw)
        if p.returncode not in ok_codes:
            raise AssertionError(f"rag-search {' '.join(args)} exited {p.returncode}:\n{p.stdout[-2000:]}\n{p.stderr[-3000:]}")
        return json.loads(p.stdout[p.stdout.index("{"):])
