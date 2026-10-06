#!/usr/bin/env python3
"""Convert one document to page-annotated Markdown with docling.

Deliberately standalone (imports nothing from rag_search) so it can also be run
by a *different* Python interpreter that has docling installed:

    python docling_convert.py INPUT OUTPUT.md [--ocr] [--info INFO.json]

``--info`` also writes what the conversion learned about each page (characters, script, tables,
pictures, docling's confidence scores) for the indexer's conversion trace.

Settings (environment variables, read on every call; see ``convert_settings``):

    RAG_SEARCH_OCR         force (default) | smart | auto | off
                           smart = force for PDFs whose own text layer looks unreliable (scans,
                           garbled fonts), auto for the rest: much faster on born-digital PDFs
    RAG_SEARCH_PDF_BACKEND pypdfium2 (default) | docling-parse | default (= whatever docling uses)
    RAG_SEARCH_OCR_ENGINE  auto (default) | ocrmac | rapidocr | easyocr | tesseract | tesserocr
    RAG_SEARCH_OCR_LANG    comma-separated language codes in the engine's own spelling
    RAG_SEARCH_TABLE_MODE  accurate (default) | fast
    RAG_SEARCH_THREADS     threads docling may use for one document (default: the indexer sets
                           it from the CPU count and the number of parallel workers)
    RAG_SEARCH_DOC_TIMEOUT seconds one document may take before it is given up (default 2700 = 45 min, 0 = no limit)
    RAG_SEARCH_DOCLING_BATCH  pages per batch for layout, table and OCR models (default 8; 0 = docling's
                           own default of 4).  Only speed and memory change, never the text
    RAG_SEARCH_PIPELINE    standard (default) | vlm   (experimental: docling's VLM pipeline)
    RAG_SEARCH_ROUTING     pages (default) | document.  pages = every PDF page takes the path that suits
                           it (text-layer pages without forced OCR, scanned pages with full-page OCR);
                           document = the whole file in one docling call, as before routing existed

Output format: one ``<!-- page N -->`` marker before each page's Markdown.
Formats without pages (docx, html, ...) are emitted as a single page 1.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
import time
import traceback
import unicodedata
from pathlib import Path
from typing import Any

PASSTHROUGH = {".md", ".txt"}
IMAGES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}

# Bump when the conversion code changes what it writes, so that documents are converted again.
CONVERT_VERSION = "c5"
POST_VERSION = "3"                  # what is done to a page after it was read (HTML tables written as pipe tables,
                                    # the loop guard and its gate check, the text-layer fill of digital pages): part of a document's conversion profile,
                                    # *not* of a page's cache key, so a change re-converts documents from the page
                                    # cache without reading a page again (only pages that ran away are read again)

OCR_MODES = ("auto", "off", "force", "smart")
ROUTING_MODES = ("pages", "document")
DEFAULT_DOC_TIMEOUT = 2700          # 45 minutes
DEFAULT_BATCH = 8

# The defaults are the combination that reads table contents best in our PDFs; it equals
#   docling convert --force-ocr --pdf-backend pypdfium2
# (every page is rendered and OCRed instead of trusting the PDF's own text layer).
DEFAULT_OCR = "force"
DEFAULT_PDF_BACKEND = "pypdfium2"
PDF_BACKENDS = ("pypdfium2", "docling-parse", "default")
_PDF_BACKEND_ALIASES = {"docling": "docling-parse", "docling_parse": "docling-parse",
                        "dlparse": "docling-parse", "auto": "default", "": DEFAULT_PDF_BACKEND}
OCR_ENGINES = {           # name -> class in docling.datamodel.pipeline_options
    "ocrmac": "OcrMacOptions",           # Apple Vision (macOS only)
    "rapidocr": "RapidOcrOptions",
    "easyocr": "EasyOcrOptions",
    "tesseract": "TesseractCliOcrOptions",
    "tesserocr": "TesseractOcrOptions",
}


def _env_number(name: str, default: float, env: Any = None) -> float:
    raw = (os.environ if env is None else env).get(name, "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        raise ValueError(f"{name} must be a number, got {raw!r}") from None
    if value < 0:
        raise ValueError(f"{name} must not be negative, got {raw!r}")
    return value


def convert_settings(ocr: bool | None = None, env: Any = None) -> dict:
    """Conversion settings from the environment (pure stdlib; safe to call anywhere).

    *ocr* is the legacy per-call switch: True means "OCR PDFs" even if the environment says off.
    *env* is the mapping to read instead of ``os.environ`` (the dashboard asks what a worker with
    that environment would use).
    """
    environ = os.environ if env is None else env
    raw = environ.get("RAG_SEARCH_OCR", DEFAULT_OCR).strip().lower()
    if raw in ("0", "off", "false", "no", "none"):
        mode = "off"
    elif raw in ("force", "full", "always", ""):
        mode = "force"
    elif raw in ("smart", "adaptive"):
        mode = "smart"
    else:                                   # auto, 1, on, true, yes, anything else
        mode = "auto"
    if ocr and mode == "off":
        mode = "auto"
    engine = environ.get("RAG_SEARCH_OCR_ENGINE", "auto").strip().lower() or "auto"
    if engine != "auto" and engine not in OCR_ENGINES:
        raise ValueError(f"RAG_SEARCH_OCR_ENGINE must be auto or one of {sorted(OCR_ENGINES)}, "
                         f"got {engine!r}")
    lang = [x.strip() for x in environ.get("RAG_SEARCH_OCR_LANG", "").split(",") if x.strip()]
    table = environ.get("RAG_SEARCH_TABLE_MODE", "accurate").strip().lower()
    if table not in ("accurate", "fast"):
        raise ValueError(f"RAG_SEARCH_TABLE_MODE must be accurate or fast, got {table!r}")
    pipeline = environ.get("RAG_SEARCH_PIPELINE", "standard").strip().lower()
    if pipeline not in ("standard", "vlm"):
        raise ValueError(f"RAG_SEARCH_PIPELINE must be standard or vlm, got {pipeline!r}")
    routing = environ.get("RAG_SEARCH_ROUTING", "pages").strip().lower() or "pages"
    if routing not in ROUTING_MODES:
        raise ValueError(f"RAG_SEARCH_ROUTING must be one of {list(ROUTING_MODES)}, got {routing!r}")
    backend = environ.get("RAG_SEARCH_PDF_BACKEND", DEFAULT_PDF_BACKEND).strip().lower()
    backend = _PDF_BACKEND_ALIASES.get(backend, backend)
    if backend not in PDF_BACKENDS:
        raise ValueError(f"RAG_SEARCH_PDF_BACKEND must be one of {list(PDF_BACKENDS)}, got {backend!r}")
    threads = _env_number("RAG_SEARCH_THREADS", 0, environ)
    timeout = _env_number("RAG_SEARCH_DOC_TIMEOUT", DEFAULT_DOC_TIMEOUT, environ)
    batch = _env_number("RAG_SEARCH_DOCLING_BATCH", DEFAULT_BATCH, environ)
    return {"ocr": mode, "engine": engine, "lang": lang, "table": table, "pipeline": pipeline,
            "pdf_backend": backend, "threads": int(threads), "timeout": float(timeout),
            "batch": int(batch), "routing": routing}


def _switched_off(var: str) -> bool:
    return os.environ.get(var, "").strip().lower() in ("off", "0", "false", "no", "none")


def convert_profile(ocr: bool | None = None, *, readers: bool = True) -> str:
    """Short string identifying the conversion settings; stored with every converted document
    so that changing a setting converts (and re-indexes) the documents again.

    With *readers*, a document reader or repair step that is switched **off** is part of it
    (``|vlm=off``, ``|repair=off``): the default (``auto``) adds nothing, so a document converted
    with the readers on stays current, and turning one off converts the documents again.  The page
    cache asks without them (*readers* false): a digital page does not depend on the document
    reader, and the reader's identity is in the key of the pages it reads."""
    try:
        s = convert_settings(ocr)
    except ValueError:                       # bad value: convert_file reports it properly
        return f"{CONVERT_VERSION}|invalid"
    out = (f"{CONVERT_VERSION}|ocr={s['ocr']}|engine={s['engine']}|lang={','.join(s['lang'])}"
           f"|table={s['table']}|pipeline={s['pipeline']}|pdf={s['pdf_backend']}|route={s['routing']}")
    if readers:
        out += f"|post={POST_VERSION}"
        if _switched_off("RAG_SEARCH_VLM"):
            out += "|vlm=off"
        if _switched_off("RAG_SEARCH_REPAIR"):
            out += "|repair=off"
        for var, tag in (("RAG_SEARCH_OCR_FIRST", "ocrfirst"), ("RAG_SEARCH_RESIDUE", "residue"),
                         ("RAG_SEARCH_ESCALATE_DIGITAL", "updigital")):     # lanes that are off by default: on is part of it
            if os.environ.get(var, "").strip().lower() == "auto":
                out += f"|{tag}=auto"
    return out


def pdf_backend_class(name: str):
    """docling's backend class for *name* (None = keep docling's own default).

    Imports docling lazily and tolerates different docling versions: the class names of the
    docling-parse backends changed several times, ``PyPdfiumDocumentBackend`` has not.
    """
    if name == "default":
        return None
    import importlib

    if name == "pypdfium2":
        candidates = [("docling.backend.pypdfium2_backend", "PyPdfiumDocumentBackend")]
    else:                                   # docling-parse: newest generation that is installed
        candidates = [("docling.backend.docling_parse_v4_backend", "DoclingParseV4DocumentBackend"),
                      ("docling.backend.docling_parse_v2_backend", "DoclingParseV2DocumentBackend"),
                      ("docling.backend.docling_parse_backend", "DoclingParseDocumentBackend")]
    for module, cls in candidates:
        try:
            return getattr(importlib.import_module(module), cls)
        except (ImportError, AttributeError):
            continue
    raise RuntimeError(f"PDF backend {name!r} is not available in this docling version "
                       "(set RAG_SEARCH_PDF_BACKEND=default)")


def _force_full_page_ocr(ocr_options, po) -> None:
    """OCR every page, whichever way this docling version spells it (mode, or the older flag)."""
    import warnings

    mode = getattr(po, "OcrMode", None)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")      # newer docling deprecates force_full_page_ocr
        if mode is not None and hasattr(mode, "FULL_PAGE"):
            try:
                ocr_options.mode = mode.FULL_PAGE
                return
            except (AttributeError, ValueError, TypeError):
                pass
        ocr_options.force_full_page_ocr = True


# ── is the PDF's own text layer trustworthy? (RAG_SEARCH_OCR=smart) ────────────────────────
_SAMPLE_PAGES = 12
_MIN_TEXT_CHARS = 40          # a page with less text than this counts as image-only


def _page_text_ok(text: str) -> bool:
    """True when the text of a page reads like text: letters, digits and punctuation, words of a
    normal length, none of the replacement/private-use characters a broken font map produces."""
    chars = [c for c in text if not c.isspace()]
    if not chars:
        return False
    bad = sum(1 for c in chars if c == "\ufffd" or "\ue000" <= c <= "\uf8ff"
              or (ord(c) < 32) or 0x80 <= ord(c) < 0xa0)
    if bad / len(chars) > 0.01:
        return False
    # letters, digits, punctuation, symbols and combining marks (the vowel signs of Indic scripts
    # are marks: without them every Hindi page would look garbled)
    normal = sum(1 for c in chars if c.isalnum() or unicodedata.category(c)[0] in "PSM")
    if normal / len(chars) < 0.9:
        return False
    words = text.split()
    if words:
        long = sum(1 for w in words if len(w) > 30 and "/" not in w and "." not in w)
        if long / len(words) > 0.05:          # words run together: the spaces were lost
            return False
    return True


def text_layer_report(src: Path, sample: int = _SAMPLE_PAGES) -> dict:
    """Sample pages of a PDF with pypdfium2 (installed with docling) and say whether its own text
    layer can be trusted: ``{"reliable": bool, "reason": str, "pages": n, "sampled": n}``.
    Any failure means "not reliable" (the caller then OCRs every page)."""
    try:
        import pypdfium2 as pdfium
    except ImportError:
        return {"reliable": False, "reason": "pypdfium2 is not installed", "pages": 0, "sampled": 0}
    try:
        pdf = pdfium.PdfDocument(str(src))
    except Exception as exc:  # noqa: BLE001
        return {"reliable": False, "reason": f"cannot open for probing ({type(exc).__name__})",
                "pages": 0, "sampled": 0}
    try:
        n = len(pdf)
        if n == 0:
            return {"reliable": False, "reason": "no pages", "pages": 0, "sampled": 0}
        step = max(1, n // sample)
        picks = list(range(0, n, step))[:sample]
        texty = good = 0
        for i in picks:
            page = pdf[i]
            try:
                textpage = page.get_textpage()
                try:
                    text = textpage.get_text_range() or ""
                finally:
                    textpage.close()
            finally:
                page.close()
            if len("".join(text.split())) < _MIN_TEXT_CHARS:
                continue
            texty += 1
            good += _page_text_ok(text)
        sampled = len(picks)
        if texty < 0.7 * sampled:
            return {"reliable": False, "pages": n, "sampled": sampled,
                    "reason": f"{sampled - texty} of {sampled} sampled pages have (almost) no text: scanned or image-only"}
        if good < 0.9 * texty:
            return {"reliable": False, "pages": n, "sampled": sampled,
                    "reason": f"{texty - good} of {texty} sampled pages have garbled text"}
        return {"reliable": True, "pages": n, "sampled": sampled,
                "reason": f"clean text layer on {good} of {sampled} sampled pages"}
    except Exception as exc:  # noqa: BLE001
        return {"reliable": False, "reason": f"probe failed ({type(exc).__name__}: {exc})",
                "pages": 0, "sampled": 0}
    finally:
        pdf.close()


def resolve_ocr_mode(src: Path, cfg: dict) -> tuple[dict, str]:
    """Turn ``ocr=smart`` into force or auto for this file.  Returns (cfg, why); why is "" when
    the mode was not smart."""
    if cfg["ocr"] != "smart":
        return cfg, ""
    if src.suffix.lower() != ".pdf":
        return dict(cfg, ocr="auto"), ""
    rep = text_layer_report(src)
    if rep["reliable"]:
        return dict(cfg, ocr="auto"), f"smart: {rep['reason']}; OCR only for images"
    return dict(cfg, ocr="force"), f"smart: {rep['reason']}; OCR on every page"


def _accelerator_options(threads: int):
    """docling AcceleratorOptions with an explicit thread count (docling's default is 4)."""
    if threads <= 0:
        return None
    for module in ("docling.datamodel.accelerator_options", "docling.datamodel.pipeline_options"):
        try:
            import importlib
            cls = getattr(importlib.import_module(module), "AcceleratorOptions")
            return cls(num_threads=int(threads), device="auto")
        except (ImportError, AttributeError, TypeError, ValueError):
            continue
    return None


def _apply_batch_sizes(opts, n: int) -> None:
    """Larger batches keep the GPU / cores busy between pages (layout, table and OCR models take
    several pages per call).  They change speed and memory only, never the text.  Every name is
    optional: docling versions differ in which of them exist."""
    if n <= 0:
        return
    for name in ("layout_batch_size", "table_batch_size", "ocr_batch_size"):
        if hasattr(opts, name):
            try:
                setattr(opts, name, int(n))
            except (ValueError, TypeError, AttributeError):
                pass
    try:
        from docling.datamodel.settings import settings

        settings.perf.page_batch_size = max(int(n), int(getattr(settings.perf, "page_batch_size", 0)))
    except Exception:  # noqa: BLE001 - older/newer docling without this setting
        pass


# ── per-page facts for the conversion trace ───────────────────────────────────────────────
_SCRIPTS = (("Devanagari", 0x0900, 0x097F), ("Bengali", 0x0980, 0x09FF), ("Gurmukhi", 0x0A00, 0x0A7F),
            ("Gujarati", 0x0A80, 0x0AFF), ("Tamil", 0x0B80, 0x0BFF), ("Telugu", 0x0C00, 0x0C7F),
            ("Kannada", 0x0C80, 0x0CFF), ("Malayalam", 0x0D00, 0x0D7F), ("Arabic", 0x0600, 0x06FF),
            ("Arabic", 0x0750, 0x077F), ("Cyrillic", 0x0400, 0x052F), ("Greek", 0x0370, 0x03FF),
            ("Hebrew", 0x0590, 0x05FF), ("Thai", 0x0E00, 0x0E7F), ("CJK", 0x2E80, 0x9FFF),
            ("CJK", 0xAC00, 0xD7AF))


def script_counts(text: str) -> dict[str, int]:
    """Letters per writing system in *text* (Latin, Devanagari, Cyrillic, ...)."""
    counts: dict[str, int] = {}
    for ch in text:
        if not ch.isalpha():
            continue
        cp = ord(ch)
        if cp < 0x250:
            name = "Latin"
        else:
            name = next((n for n, lo, hi in _SCRIPTS if lo <= cp <= hi), "Other")
        counts[name] = counts.get(name, 0) + 1
    return counts


def dominant_script(text: str, min_letters: int = 10) -> str:
    """The writing system a text is mainly in: "" when it has too few letters, "Mixed" when no
    system dominates.  A non-Latin script is reported from 25 % of the letters on, because that
    is the case routing and OCR engines care about (a Hindi page with English headings)."""
    counts = script_counts(text)
    total = sum(counts.values())
    if total < min_letters:
        return ""
    other = {k: v for k, v in counts.items() if k != "Latin"}
    if other:
        name, n = max(other.items(), key=lambda kv: kv[1])
        if n / total >= 0.25:
            return name
    if counts.get("Latin", 0) / total >= 0.6:
        return "Latin"
    return "Mixed"


def page_text_ok(text: str) -> bool:
    """Public name of the text-layer check (see ``_page_text_ok``)."""
    return _page_text_ok(text)


def extract_confidence(res) -> dict[int, dict]:
    """docling's per-page quality scores from a conversion result, ``{page: {...}}``; {} when this
    docling version reports none.  Scores are 0..1; the grades are poor/fair/good/excellent."""
    out: dict[int, dict] = {}
    try:
        pages = getattr(getattr(res, "confidence", None), "pages", None)
        if not pages:
            return out
        items = pages.items() if hasattr(pages, "items") else enumerate(pages, 1)
        for no, sc in items:
            d: dict = {}
            for key, attr in (("parse", "parse_score"), ("layout", "layout_score"),
                              ("table", "table_score"), ("ocr", "ocr_score"),
                              ("mean", "mean_score"), ("low", "low_score")):
                try:
                    v = float(getattr(sc, attr, None))
                except (TypeError, ValueError):
                    continue
                if v == v:                                   # not NaN
                    d[key] = round(v, 3)
            for key, attr in (("grade", "mean_grade"), ("low_grade", "low_grade")):
                g = getattr(sc, attr, None)
                name = str(getattr(g, "name", g) or "").lower()
                if name and name != "none":
                    d[key] = name
            if d:
                out[int(no)] = d
    except Exception:  # noqa: BLE001 - a docling version without (or with another) report
        return {}
    return out


MIN_REAL_CHARS = 1                   # a page needs at least one letter or digit of its own (a short page is still a page)

# What docling prints for a picture it classified (its markdown export adds the class name under "<!-- image -->").
PICTURE_LABELS = frozenset({
    "other", "logo", "photograph", "icon", "engineering drawing", "line chart", "bar chart", "pie chart",
    "stacked bar chart", "scatter plot", "heat map", "box plot", "flow chart", "map", "stamp", "signature",
    "qr code", "bar code", "screenshot from computer", "screenshot from manual", "page thumbnail",
    "natural image", "full page image", "navigation chart", "table", "chart", "diagram", "picture", "image"})


def real_chars(md: str) -> int:
    """Letters, digits and combining marks of a page's Markdown, without markers, image placeholders and the
    class name docling prints under a picture ("Other").  A page that docling turned into ``<!-- image -->`` plus
    such a label has none; a page of text has hundreds."""
    s = re.sub(r"<!--.*?-->", "\n", md or "", flags=re.S)
    lines = [ln for ln in s.splitlines()
             if re.sub(r"[\s_\-*#>|]+", " ", ln).strip().lower() not in PICTURE_LABELS]
    return sum(1 for ch in "\n".join(lines) if unicodedata.category(ch)[0] in "LNM")   # letters, digits, vowel signs (Indic)


def has_real_text(md: str) -> bool:
    return real_chars(md) >= MIN_REAL_CHARS


def _picture_text(doc, page_no: int) -> str:
    """The text docling found *inside* pictures on a page.  docling's Markdown export leaves it out (a
    photographed page is one big "picture" to its layout model), but the OCR result is in the document."""
    try:
        items = doc.iterate_items(page_no=page_no, traverse_pictures=True)
    except TypeError:        # an older docling without those arguments
        try:
            items = doc.iterate_items()
        except Exception:  # noqa: BLE001
            return ""
    except Exception:  # noqa: BLE001
        return ""
    lines: list[str] = []
    try:
        for item, _level in items:
            text = getattr(item, "text", None)
            if not isinstance(text, str) or not text.strip():
                continue
            provs = getattr(item, "prov", None) or []
            if provs and not any(getattr(pv, "page_no", page_no) == page_no for pv in provs):
                continue
            lines.append(text.strip())
    except Exception:  # noqa: BLE001 - never let salvage break a conversion
        return ""
    return "\n\n".join(lines)


def export_page(doc, page_no: int) -> tuple[str, str]:
    """Markdown of one page plus a note ("" or what was done).  When the normal export holds no real
    text, the text inside pictures is used (a photographed document)."""
    try:
        md = doc.export_to_markdown(page_no=page_no).strip()
    except Exception:  # noqa: BLE001 - a page docling could not export is an empty page
        md = ""
    if has_real_text(md):
        return md, ""
    salvaged = _picture_text(doc, page_no)
    if has_real_text(salvaged):
        return salvaged, "text taken from inside the page's picture (docling's export had left it out)"
    return md, ""


_BIG_PICTURE = 1 / 3                  # a picture covering this share of a page is a scan candidate


def collect_page_stats(doc, page_texts: dict[int, str], confidence: dict[int, dict]) -> list[dict]:
    """Per page: characters, script, tables, pictures, large pictures and docling's confidence."""
    tables: dict[int, int] = {}
    pics: dict[int, int] = {}
    big: dict[int, int] = {}
    try:
        for t in getattr(doc, "tables", None) or []:
            for pv in (getattr(t, "prov", None) or [])[:1]:
                tables[pv.page_no] = tables.get(pv.page_no, 0) + 1
        for pic in getattr(doc, "pictures", None) or []:
            for pv in (getattr(pic, "prov", None) or [])[:1]:
                no = pv.page_no
                pics[no] = pics.get(no, 0) + 1
                try:
                    size = doc.pages[no].size
                    bb = pv.bbox
                    frac = abs((bb.r - bb.l) * (bb.t - bb.b)) / max(1.0, size.width * size.height)
                except Exception:  # noqa: BLE001
                    continue
                if frac >= _BIG_PICTURE:
                    big[no] = big.get(no, 0) + 1
    except Exception:  # noqa: BLE001 - never let tracing break a conversion
        pass
    stats = []
    for no in sorted(page_texts):
        text = page_texts[no]
        rec = {"page": no, "chars": len("".join(text.split())), "script": dominant_script(text),
               "tables": tables.get(no, 0), "pictures": pics.get(no, 0),
               "big_pictures": big.get(no, 0)}
        if no in confidence:
            rec["confidence"] = confidence[no]
        stats.append(rec)
    return stats


_LAST: dict = {}                      # facts about the conversion that just ran in this process


class ConversionTimeout(RuntimeError):
    """The document took longer than RAG_SEARCH_DOC_TIMEOUT."""


# A DocumentConverter loads its models on first use (seconds, and gigabytes): keep it for the next
# document instead of building one per file.  Two settings at most (e.g. the fallback backend).
_CONVERTERS: dict = {}


def _cached_converter(suffix: str, cfg: dict):
    key = repr(sorted((k, tuple(v) if isinstance(v, list) else v) for k, v in cfg.items()))
    conv = _CONVERTERS.get(key)
    if conv is None:
        while len(_CONVERTERS) >= 2:
            _CONVERTERS.pop(next(iter(_CONVERTERS)))
        conv = _CONVERTERS[key] = _build_converter(suffix, cfg)
    return conv


def _build_converter(suffix: str, cfg: dict):
    """A docling DocumentConverter for *cfg* (imports docling lazily)."""
    from docling.datamodel.base_models import InputFormat
    from docling.document_converter import DocumentConverter, PdfFormatOption

    try:
        from docling.document_converter import ImageFormatOption
    except ImportError:  # older docling: images use docling's own defaults
        ImageFormatOption = None  # noqa: N806

    if cfg["pipeline"] == "vlm":
        from docling.pipeline.vlm_pipeline import VlmPipeline

        fmt = {InputFormat.PDF: PdfFormatOption(pipeline_cls=VlmPipeline)}
        if ImageFormatOption is not None:
            fmt[InputFormat.IMAGE] = ImageFormatOption(pipeline_cls=VlmPipeline)
        return DocumentConverter(format_options=fmt)

    from docling.datamodel import pipeline_options as po

    opts = po.PdfPipelineOptions()
    opts.do_ocr = cfg["ocr"] != "off" or suffix in IMAGES
    opts.do_table_structure = True
    _apply_batch_sizes(opts, int(cfg.get("batch", 0) or 0))
    acc = _accelerator_options(cfg.get("threads", 0))
    if acc is not None:
        opts.accelerator_options = acc
    if cfg.get("timeout", 0) > 0:
        try:
            opts.document_timeout = float(cfg["timeout"])
        except (AttributeError, ValueError):
            pass  # this docling has no per-document timeout
    try:
        mode = po.TableFormerMode.ACCURATE if cfg["table"] == "accurate" else po.TableFormerMode.FAST
        opts.table_structure_options.mode = mode
    except AttributeError:
        pass  # this docling has no TableFormer mode switch

    if cfg["engine"] != "auto":
        cls = getattr(po, OCR_ENGINES[cfg["engine"]], None)
        if cls is None:
            raise RuntimeError(f"OCR engine {cfg['engine']!r} is not available in this docling "
                               "version (set RAG_SEARCH_OCR_ENGINE=auto)")
        opts.ocr_options = cls()
    if cfg["ocr"] == "force":
        _force_full_page_ocr(opts.ocr_options, po)
    if cfg["lang"]:
        opts.ocr_options.lang = cfg["lang"]

    backend = pdf_backend_class(cfg.get("pdf_backend", "default"))
    fmt = {InputFormat.PDF: PdfFormatOption(pipeline_options=opts, **({"backend": backend} if backend else {}))}
    if ImageFormatOption is not None:
        fmt[InputFormat.IMAGE] = ImageFormatOption(pipeline_options=opts)
    return DocumentConverter(format_options=fmt)


def describe_error(exc: BaseException) -> str:
    """'Type: message [at package/file.py:line in function]' - where inside docling (or a library
    it uses) the failure happened, which the bare message never says."""
    text = f"{type(exc).__name__}: {exc}"
    # EDEADLK (11) on macOS: an online-only file of a cloud-storage folder that could not be fetched
    if isinstance(exc, OSError) and exc.errno == 11 and sys.platform == "darwin":
        text = ("the file is kept online-only by a cloud-storage app (Box, Google Drive, iCloud, OneDrive) and "
                "could not be fetched: check that the app is running and online, or mark the folder "
                f"\"available offline\", then index again ({text})")
    try:
        last = traceback.extract_tb(exc.__traceback__)[-1]
        text += f" [at {'/'.join(Path(last.filename).parts[-2:])}:{last.lineno} in {last.name}]"
    except (IndexError, AttributeError):
        pass
    return text


EXIT_NO_TEXT = 3      # exit code of `python docling_convert.py` for a file without text


class NoTextError(RuntimeError):
    """The file was converted but holds no text (a photo, a blank or image-only page): not a
    failure of the tool, so the indexer reports the file as "skipped: no text"."""


class ProtectedPdfError(RuntimeError):
    """A PDF that needs a password (or is encrypted) and so cannot be read."""


_PROTECTED_HOWTO = "remove the protection (or save an unprotected copy) and index it again"


def protected_pdf_reason(src: Path) -> str:
    """Why *src* (a PDF that just failed to convert) cannot be read because of encryption,
    or "" when that is not the cause."""
    try:
        import pypdfium2 as pdfium

        try:
            pdfium.PdfDocument(str(src)).close()
        except Exception as exc:  # noqa: BLE001
            if "password" in str(exc).lower():
                return f"password-protected PDF, it cannot be opened without the password; {_PROTECTED_HOWTO}"
    except ImportError:
        pass
    try:
        size = src.stat().st_size
        with open(src, "rb") as fh:
            head = fh.read(4096)
            fh.seek(max(0, size - 262144))
            tail = fh.read()
        if b"/Encrypt" in head or b"/Encrypt" in tail:
            return f"encrypted PDF that the converters could not open; {_PROTECTED_HOWTO}"
    except OSError:
        pass
    return ""


# When a PDF cannot be read by the configured backend, the other one often can (a font or string
# that one parser rejects, e.g. non-UTF-8 text in the file): try it once before giving up.
_OTHER_BACKEND = {"pypdfium2": "docling-parse", "docling-parse": "pypdfium2", "default": "pypdfium2"}


def _run(src: Path, suffix: str, cfg: dict, page_range: tuple[int, int] | None = None):
    """One conversion attempt with a cached converter; enforces the per-document time limit."""
    t0 = time.perf_counter()
    conv = _cached_converter(suffix, cfg)
    res = conv.convert(str(src), page_range=page_range) if page_range else conv.convert(str(src))
    _LAST["confidence"] = extract_confidence(res)
    limit = cfg.get("timeout", 0)
    status = str(getattr(getattr(res, "status", None), "name", getattr(res, "status", "")))
    if limit > 0 and "PARTIAL" in status.upper() and time.perf_counter() - t0 >= limit * 0.98:
        raise ConversionTimeout(f"gave up on {src.name} after {int(limit)}s "
                                "(RAG_SEARCH_DOC_TIMEOUT); raise it, or make conversion faster "
                                "(RAG_SEARCH_OCR=smart)")
    return res.document


def _convert_document(src: Path, suffix: str, cfg: dict, page_range: tuple[int, int] | None = None):
    try:
        return _run(src, suffix, cfg, page_range)
    except ConversionTimeout:
        raise
    except Exception as first:  # noqa: BLE001
        other = _OTHER_BACKEND.get(cfg.get("pdf_backend", "")) if suffix == ".pdf" else None
        if not other or cfg.get("pipeline") == "vlm":
            raise
        try:
            doc = _run(src, suffix, dict(cfg, pdf_backend=other), page_range)
        except Exception as second:  # noqa: BLE001
            raise RuntimeError(f"{describe_error(first)} | retry with the {other} backend also "
                               f"failed: {describe_error(second)}") from first
        print(f"note: {src.name}: the {cfg['pdf_backend']} backend failed ({describe_error(first)}); "
              f"converted with the {other} backend instead", file=sys.stderr)
        return doc


class RangeUnsupported(RuntimeError):
    """This docling version does not honour ``page_range`` (it converted more pages than asked for):
    routing falls back to converting whole documents."""


def convert_range(src: Path, first: int, last: int, mode: str, ocr: bool | None = None) -> dict:
    """Read pages *first*..*last* (1-based, inclusive) of a PDF with docling.

    *mode* is ``digital`` (the pages have a text layer: OCR only for pictures, ``auto``) or ``scan``
    (the pages are images: full-page OCR).  ``RAG_SEARCH_OCR=off`` switches OCR off in both.  Returns
    ``{"pages": {n: markdown}, "stats": {n: per-page facts}, "seconds", "ocr"}`` with the page
    numbers of the *source* file.  docling versions differ in whether a result that was made from a
    page range keeps the original page numbers; both are handled."""
    if mode not in ("digital", "scan"):
        raise ValueError(f"mode must be digital or scan, got {mode!r}")
    t0 = time.perf_counter()
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    cfg = convert_settings(ocr)
    if cfg["ocr"] != "off":
        cfg = dict(cfg, ocr="auto" if mode == "digital" else "force")
    doc = _convert_document(src, src.suffix.lower(), cfg, (int(first), int(last)))
    want = list(range(int(first), int(last) + 1))
    try:
        have = sorted(int(k) for k in (getattr(doc, "pages", None) or {}))
    except (TypeError, ValueError):
        have = []
    if have and len(have) > len(want):
        raise RangeUnsupported(f"asked for pages {first}-{last} but docling returned {len(have)} pages")
    if have and set(have) <= set(want):
        to_docling = {n: n for n in want if n in have}                 # original page numbers kept
    elif have == list(range(1, len(want) + 1)):
        to_docling = {first + i: i + 1 for i in range(len(want))}      # renumbered from 1
    elif have:
        raise RangeUnsupported(f"asked for pages {first}-{last}, docling numbered its pages {have[:5]}...")
    else:
        to_docling = {n: n for n in want}                              # formats that do not say
    texts: dict[int, str] = {}
    notes: dict[int, str] = {}
    for real, dno in to_docling.items():
        texts[dno], note = export_page(doc, dno)
        if note:
            notes[dno] = note
    conf = _LAST.pop("confidence", {})
    stats = {st["page"]: st for st in collect_page_stats(doc, texts, conf)}
    back = {d: r for r, d in to_docling.items()}
    out_stats = {}
    for dno, st in stats.items():
        if dno in back:
            out_stats[back[dno]] = {**st, "page": back[dno]}
            if dno in notes:
                out_stats[back[dno]]["note"] = notes[dno]
    return {"pages": {back[d]: t for d, t in texts.items() if d in back}, "stats": out_stats,
            "seconds": round(time.perf_counter() - t0, 3), "ocr": cfg["ocr"]}


def convert_file(src: Path, out_md: Path, ocr: bool | None = None) -> dict:
    """Write Markdown for *src* to *out_md*.  Returns {"pages": int, "seconds": float}.

    OCR, table and pipeline settings come from the environment (``convert_settings``);
    *ocr*=True only makes sure PDFs are OCRed even if RAG_SEARCH_OCR=off.
    """
    t0 = time.perf_counter()
    out_md.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_md.with_name(out_md.name + ".partial")
    suffix = src.suffix.lower()

    if suffix in PASSTHROUGH:
        shutil.copyfile(src, tmp)
        os.replace(tmp, out_md)
        return {"pages": 1, "seconds": round(time.perf_counter() - t0, 2)}

    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")  # Apple GPU: fall back per-op
    cfg, why = resolve_ocr_mode(src, convert_settings(ocr))
    if why:
        print(f"note: {src.name}: {why}", file=sys.stderr)
    try:
        doc = _convert_document(src, suffix, cfg)
    except ConversionTimeout:
        raise
    except Exception as exc:  # noqa: BLE001
        reason = protected_pdf_reason(src) if suffix == ".pdf" else ""
        if reason:
            raise ProtectedPdfError(reason) from exc
        raise

    parts: list[str] = []
    n_pages = 0
    page_texts: dict[int, str] = {}
    try:
        n_pages = int(doc.num_pages())
    except Exception:  # noqa: BLE001 - formats without a page model
        n_pages = 0
    if n_pages > 0:
        for page_no in range(1, n_pages + 1):
            page_md, note = export_page(doc, page_no)
            page_texts[page_no] = page_md
            if note:
                print(f"note: {src.name} page {page_no}: {note}", file=sys.stderr)
            if page_md:
                parts.append(f"<!-- page {page_no} -->")
                parts.append(page_md)
    if parts and not has_real_text("\n".join(page_texts.values())):
        parts = []                    # only "<!-- image -->" and a label: nothing was read
    if not parts:
        whole = doc.export_to_markdown().strip()
        if has_real_text(whole):
            parts = ["<!-- page 1 -->", whole]
            n_pages = max(n_pages, 1)
            page_texts = {1: whole}
    if not parts:
        hint = (" (scanned PDF? make sure RAG_SEARCH_OCR is not 'off'; the default, "
                "RAG_SEARCH_OCR=force, reads every page as an image; for photographed pages install the "
                "document reader, see the Models tab)") if suffix == ".pdf" else ""
        raise NoTextError(f"no text extracted from {src.name}{hint}")

    tmp.write_text("\n\n".join(parts) + "\n", encoding="utf-8")
    os.replace(tmp, out_md)
    return {"pages": n_pages, "seconds": round(time.perf_counter() - t0, 2), "ocr": cfg["ocr"],
            "page_stats": collect_page_stats(doc, page_texts, _LAST.pop("confidence", {}))}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("input")
    ap.add_argument("output")
    ap.add_argument("--ocr", action="store_true", help="OCR PDFs even if RAG_SEARCH_OCR=off")
    ap.add_argument("--info", help="also write per-page facts (JSON) to this file")
    a = ap.parse_args(argv)
    try:
        info = convert_file(Path(a.input), Path(a.output), ocr=True if a.ocr else None)
        if a.info:
            import json

            Path(a.info).write_text(json.dumps(info), encoding="utf-8")
    except NoTextError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return EXIT_NO_TEXT
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"OK pages={info['pages']} seconds={info['seconds']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
