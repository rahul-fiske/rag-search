"""Page readers as the benchmark (and later the router) sees them (stdlib at import).

An *engine* turns some pages of a source file into Markdown::

    engine.read_pages(src, pages) -> {"pages": {n: markdown}, "seconds": float,
                                       "pages_read": int, "page_count": int}

``pages_read`` is how many pages the engine actually worked on to produce the answer (a reader
that must convert the whole document to give one page reports the document's page count), so
seconds per page is honest.  Engines are chosen by name (``current``, ...) or as
``module:attr`` for a class or factory of your own -- the tests use ``tests.helpers:FakeEngine``.
"""

from __future__ import annotations

import importlib
import os
import tempfile
import time
from pathlib import Path
from typing import Any

from . import pagemd


class EngineError(RuntimeError):
    pass


class CurrentEngine:
    """Whole-document docling with the settings in force (``RAG_SEARCH_OCR`` and friends): what
    production did before routing.  The baseline every other engine is compared with."""

    name = "current"

    def __init__(self, **_: Any) -> None:
        self._cache: dict[str, dict[int, str]] = {}

    def describe(self) -> dict[str, Any]:
        from ..docling_convert import convert_profile

        return {"name": self.name, "settings": convert_profile()}

    def read_pages(self, src: Path, pages: list[int]) -> dict[str, Any]:
        from ..docling_convert import convert_file

        t0 = time.perf_counter()
        with tempfile.TemporaryDirectory(prefix="rag-bench-") as tmp:
            out = Path(tmp) / "doc.md"
            info = convert_file(src, out)
            doc = pagemd.split_pages(out.read_text(encoding="utf-8"))
        n = int(info.get("pages") or len(doc) or 1)
        return {"pages": {p: doc.get(p, "") for p in pages}, "seconds": round(time.perf_counter() - t0, 3),
                "pages_read": n, "page_count": n}


class VlmEngine:
    """The document VLM alone: every wanted page is rendered and read by the reader chosen in the
    Models tab (``rag-search models reader``), whatever kind of page it is.  Measures the model, not
    the router.  A page the reader cannot read fails the file (so a benchmark never silently measures
    the docling fallback)."""

    name = "vlm"

    def __init__(self, **_: Any) -> None:
        from . import vlm

        self.reader = vlm.shared()
        if self.reader is None:
            raise EngineError("the document reader is switched off (RAG_SEARCH_VLM=off)")

    def describe(self) -> dict[str, Any]:
        r = self.reader
        return {"name": self.name, "model": r.model, "backend": r.backend, "load_s": r.load_s,
                "tokens": r.tokens, "gpu_s": round(r.gpu_s, 1), "reader_peak_mb": r.peak_mb}

    @property
    def peak_mb(self) -> float:
        return float(self.reader.peak_mb)

    def close(self) -> None:
        from . import vlm

        vlm.close_shared()

    def read_pages(self, src: Path, pages: list[int]) -> dict[str, Any]:
        t0 = time.perf_counter()
        out: dict[int, str] = {}
        page_s: dict[int, float] = {}
        for n in pages:
            res = self.reader.read(src, n, n, "scan")
            if n not in res["pages"]:
                raise EngineError(f"page {n}: {res['failed'].get(n, 'not read')}")
            out[n] = res["pages"][n]
            page_s[n] = res["seconds"]
        return {"pages": out, "page_s": page_s, "seconds": round(time.perf_counter() - t0, 3),
                "pages_read": len(pages), "page_count": len(pages)}


class RoutedEngine:
    """The production pipeline for a whole file (profile, route, docling for text pages, the document
    VLM for scans, the gate), without the page cache so every run reads everything."""

    name = "routed"

    def __init__(self, **_: Any) -> None:
        pass

    def describe(self) -> dict[str, Any]:
        from ..docling_convert import convert_profile
        from . import vlm

        r = vlm.shared()
        return {"name": self.name, "settings": convert_profile(), "reader": r.model if r else "off"}

    def close(self) -> None:
        from . import vlm

        vlm.close_shared()

    def read_pages(self, src: Path, pages: list[int]) -> dict[str, Any]:
        from . import profiler, repair, routed, vlm

        t0 = time.perf_counter()
        prof = profiler.profile_file(src)
        if prof.get("error") or not prof.get("pages"):
            raise EngineError(prof.get("error") or "cannot profile the file")
        with tempfile.TemporaryDirectory(prefix="rag-bench-") as tmp:
            out = Path(tmp) / "doc.md"
            if prof["kind"] == "image":
                routed.convert_image(src, out, prof, cache=None, scan_reader=vlm.shared(), repairer=repair.build())
            else:
                routed.convert_pdf(src, out, prof, cache=None, scan_reader=vlm.shared(), repairer=repair.build())
            doc = pagemd.split_pages(out.read_text(encoding="utf-8"))
        n = int(prof.get("page_count") or len(doc) or 1)
        return {"pages": {p: doc.get(p, "") for p in pages}, "seconds": round(time.perf_counter() - t0, 3),
                "pages_read": n, "page_count": n}


BUILTIN = {"current": CurrentEngine, "routed": RoutedEngine, "vlm": VlmEngine}


def resolve(spec: str = "current", **kw: Any):
    """The engine for *spec*: a built-in name or ``module:attr``."""
    spec = (spec or "current").strip()
    if spec in BUILTIN:
        return BUILTIN[spec](**kw)
    if ":" not in spec:
        raise EngineError(f"unknown engine {spec!r}; use one of {sorted(BUILTIN)} or module:attr")
    mod, _, attr = spec.partition(":")
    try:
        obj = getattr(importlib.import_module(mod), attr)
    except (ImportError, AttributeError) as exc:
        raise EngineError(f"cannot load engine {spec!r}: {exc}") from exc
    eng = obj(**kw) if isinstance(obj, type) or (callable(obj) and not hasattr(obj, "read_pages")) else obj
    if not hasattr(eng, "read_pages"):
        raise EngineError(f"{spec!r} has no read_pages(src, pages)")
    return eng


def available() -> list[str]:
    extra = [s.strip() for s in os.environ.get("RAG_SEARCH_BENCH_ENGINES", "").split(",") if s.strip()]
    return [*BUILTIN, *extra]
