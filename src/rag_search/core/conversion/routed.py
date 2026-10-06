"""Per-page routed conversion of one PDF (docling is imported lazily, by the reader).

Every page takes the path its profile calls for (``router.decide``):

* ``digital`` -- the page has a text layer: docling reads it *without* forced OCR (OCR only for
  pictures), the way a born-digital page should be read;
* ``raster`` / ``unknown`` -- a scan, a garbled text layer, or a page that could not be profiled:
  the **document VLM** (``vlm.py``, a child process) reads the page image; when it is switched off,
  not installed, out of memory or fails on a page, docling reads that page as an image with
  full-page OCR instead (branch ``fallback``, the reason is in the trace);
* a digital page with a large picture inside (a photographed document, a screenshot) is read by
  docling and the picture is then read by the document VLM (branch ``embedded``);
* a blank page (no text, no ink) is not read at all;
* a page read as an image whose tables fail the arithmetic checks goes to **repair** (``repair.py``):
  the suspect cell is cut out, read again and replaced only when two reads and the arithmetic agree
  (outcome ``repaired``).

Image files take the VLM path page by page (``convert_image``; a multi-page TIFF has one page per
frame).  Nothing falls back for them: a reader that cannot read raises, and the indexer converts the
file whole with docling.

Pages are read in runs of consecutive pages of the same kind (docling's ``page_range``), at most
``READ_CHUNK`` pages per call, and each page's result is stored in the page cache the moment its run
is done: a cancelled or crashed run loses at most one run of pages, and a changed document re-reads
only its changed pages.  Every page then goes through the gate (``gate.py``) and ends with an outcome.

``convert_pdf`` writes the Markdown (page markers, like every reader) and returns one record per
page for the trace.  Any exception from a reader propagates: the indexer then converts the document
whole, so routing can fail without failing a document.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Callable

from ..docling_convert import NoTextError, convert_profile, convert_settings, has_real_text, page_text_ok
from . import (applevision, degenerate, gate, layer, pagecache, pagemd, profiler, reconcile, router, tables, tesseract,
               trace, vlm)
from .records import PROFILE_KEEP

READ_CHUNK = 20                      # pages per docling call at most


class DoclingReader:
    """The docling reader (digital pages: OCR auto; scanned pages: full-page OCR)."""

    id = "docling"

    def read(self, src: Path, first: int, last: int, mode: str) -> dict[str, Any]:
        from ..docling_convert import convert_range

        return convert_range(src, first, last, mode)


def default_reader() -> Any:
    return DoclingReader()


def _runs(pages: list[int], max_len: int = READ_CHUNK) -> list[tuple[int, int]]:
    """Consecutive page numbers as (first, last) runs of at most *max_len* pages."""
    runs: list[tuple[int, int]] = []
    for n in sorted(pages):
        if runs and n == runs[-1][1] + 1 and n - runs[-1][0] < max_len:
            runs[-1] = (runs[-1][0], n)
        else:
            runs.append((n, n))
    return runs


def _reader_info(cfg: dict[str, Any], mode: str, reader_id: str) -> dict[str, Any]:
    if cfg["ocr"] == "off":
        ocr = "off"
    else:
        ocr = "auto" if mode == "digital" else "full-page"
    return {"tool": reader_id, "mode": f"ocr-{ocr}" if ocr != "off" else "no-ocr",
            "engine": cfg["engine"], "backend": cfg["pdf_backend"], "table": cfg["table"]}


def _out_facts(md: str, stats: dict[str, Any] | None) -> dict[str, Any]:
    st = stats or {}
    out = {k: st[k] for k in ("chars", "script", "tables", "pictures", "big_pictures") if st.get(k)}
    if "chars" not in out:
        out["chars"] = len("".join(md.split()))
    return out


def plan_pages(profile: dict[str, Any]) -> list[dict[str, Any]]:
    """One entry per page: ``{"page", "branch", "mode", "why", "hash", "profile", "blank"}``.
    ``mode`` is what the reader will be asked (digital | scan); a blank page has ``blank`` set and no
    mode.  ``branch`` is the branch the page takes when the document VLM reads it (``raster``, or
    ``image`` for an image file); the converter changes it to ``fallback`` when docling has to."""
    prof_by_page = {int(p.get("page", 0)): p for p in profile.get("pages", [])}
    is_image = profile.get("kind") == "image"
    out = []
    for n, branch, why in profiler.route_pages(profile):
        prof = prof_by_page.get(n, {})
        entry = {"page": n, "branch": branch, "why": why, "profile": prof, "hash": prof.get("hash", ""),
                 "mode": "digital" if branch == "digital" else "scan", "blank": False}
        ink = prof.get("ink")
        if (branch in ("raster", "image") and isinstance(ink, (int, float)) and ink < profiler.BLANK_INK
                and int(prof.get("chars") or 0) == 0):
            entry.update(blank=True, mode="", why=f"blank page (almost no ink: {round(100 * ink, 3)} %)")
        if is_image and branch == "unknown":
            entry["mode"] = "scan"
        out.append(entry)
    return out


def _new_text(page_md: str, pic_md: str) -> str:
    """The part of a picture's text that the page does not already hold (docling's OCR also reads
    bitmaps): the whole text when it is not contained, or just the lines that are new."""
    have = " ".join(tables.plain_text(page_md).lower().split())
    whole = " ".join(tables.plain_text(pic_md).lower().split())
    if not whole or whole in have:
        return ""
    if "<table" in pic_md or "|---" in pic_md.replace(" ", ""):
        return pic_md.strip()                      # a table is kept whole
    keep = []
    for line in pic_md.split("\n"):
        norm = " ".join(tables.plain_text(line).lower().split())
        if norm and norm not in have:
            keep.append(line)
    return "\n".join(keep).strip()


# gate checks that a re-read by another model cannot cure (the page's resolution, docling's own grade)
NOT_ESCALATED = ("low_resolution", "docling_grade")


def layer_fill_mode() -> str:
    """``RAG_SEARCH_LAYER_FILL``: ``fill`` (default: a digital page that lost text gets the missing lines of its own text
    layer appended), ``report`` (compare and record only) or ``off``."""
    v = os.environ.get("RAG_SEARCH_LAYER_FILL", "fill").strip().lower()
    return v if v in ("fill", "report", "off") else "fill"


def ocr_first_mode() -> str:
    """``RAG_SEARCH_OCR_FIRST``: ``auto`` (a scanned PDF page whose image says clean print is read by docling's OCR first, and
    goes to the document reader only when the gate doubts the result) or ``off`` (default: the document reader reads
    every scanned page)."""
    v = os.environ.get("RAG_SEARCH_OCR_FIRST", "off").strip().lower()
    return "auto" if v == "auto" else "off"


class _Converter:
    """State of one document's conversion (shared by PDFs and image files)."""

    def __init__(self, src: Path, profile: dict[str, Any], *, cache: pagecache.PageCache | None,
                 reader: Any, scan_reader: Any, ocr: bool | None,
                 on_page: Callable[[dict[str, Any], int], None] | None, repairer: Any = None) -> None:
        self.src, self.cache, self.reader, self.on_page = src, cache, reader, on_page
        self.repairer = repairer
        self.tags: dict[int, str] = {}
        self.is_image = profile.get("kind") == "image"
        self.vlm = scan_reader
        self.cfg = convert_settings(ocr)
        self.settings = convert_profile(ocr, readers=False)
        self.plan = plan_pages(profile)
        self.total = len(self.plan)
        self.by_no = {e["page"]: e for e in self.plan}
        self.results: dict[int, dict[str, Any]] = {}
        self.keys: dict[int, str] = {}
        self.records: dict[int, dict[str, Any]] = {}
        self.hits = self.misses = self.runs = 0
        self.layer_mode = layer_fill_mode()
        self.layers = layer.LayerSource(src) if (self.layer_mode != "off" and not self.is_image) else None
        self.escalate: dict[int, list[str]] = {}        # page -> checks that sent a runway-b page on to the document reader
        self.first_try: dict[int, dict[str, Any]] = {}  # page -> the runway-b result kept while the document reader tries
        self.routing = ocr_first_mode() == "auto"

    def route_scans(self) -> None:
        """Runway b or d for every scanned page of a PDF (``router.decide_scan`` on the page image's facts).  Only when
        ``RAG_SEARCH_OCR_FIRST=auto`` and the document reader is there: without it docling's OCR reads every scan anyway."""
        if not (self.routing and not self.is_image and self.vlm_ok()):
            return
        from . import scanfacts                  # numpy: only when routing

        want = [e["page"] for e in self.plan if e["mode"] == "scan" and not e["blank"]]
        facts = scanfacts.pages_facts(self.src, want) if want else {}
        caps = {"ocr": self.cfg["ocr"] != "off"}
        for e in self.plan:
            if e["page"] in facts:
                runway, why = router.decide_scan(facts[e["page"]], e["profile"], caps)
                e["route"] = {"runway": runway, "reasons": why}

    def is_b(self, e: dict[str, Any]) -> bool:
        return (e.get("route") or {}).get("runway") == "b"

    def ocr_page(self, e: dict[str, Any], r: dict[str, Any]) -> bool:
        """Is this scanned page's text docling OCR's (so the OCR-side gate checks apply)?"""
        return (self.is_b(e) and r.get("via") not in ("vlm", "tesseract", "apple-vision")
                and (r.get("reader") or {}).get("tool") not in ("vlm", "tesseract", "apple-vision"))

    def doubts(self, e: dict[str, Any], md: str, conf: Any) -> list[str]:
        """The gate's reasons to send a runway-b page to the document reader (empty: keep it)."""
        g = gate.check_page(md, branch_kind="scan", profile=e["profile"], confidence=conf, ocr=True)
        names = [c for c in gate.failed(g) if c not in NOT_ESCALATED]
        if g["verdict"] != "empty" and not has_real_text(md) and "coverage" not in names:
            names.append("coverage")
        return names if (g.get("escalate") or names) else []

    def check_layer(self, n: int, e: dict[str, Any], r: dict[str, Any]) -> dict[str, Any] | None:
        """A digital page against its own text layer (``layer.py``): the gate's coverage check, and -- unless
        ``RAG_SEARCH_LAYER_FILL`` says report or off -- the lines docling left out are appended to the page.
        Returns the comparison to record (and to hand to the gate), or None when there is no layer to use."""
        prof = e["profile"]
        if self.layers is None or prof.get("hidden_ocr_layer") or not prof.get("text_ok", True):
            return None                       # a scanner's hidden layer or a garbled one proves nothing about the page
        text = self.layers.page(n)
        if not text or not page_text_ok(text):
            return None
        cmp = layer.compare(text, r["md"])
        if cmp["verdict"] != "intact" and cmp["layer_words"] >= layer.MIN_LAYER_TOKENS:
            lines = layer.missing_lines(text, r["md"])
            if self.layer_mode == "fill":
                new_md, added = layer.fill(r["md"], lines)
                if added:
                    r["md"] = new_md
                    r["stats"] = dict(r.get("stats") or {}, chars=len("".join(new_md.split())))
                    r["note"] = "; ".join(x for x in (r.get("note"), f"{added} line(s) the conversion left out were added "
                                                      "from the page's text layer") if x)
                    after = layer.compare(text, new_md)
                    if after["verdict"] != "intact" and (after["word_recall"] or 0) >= layer.LOST \
                            and (after["number_recall"] is None or after["number_recall"] >= layer.LOST):
                        after["verdict"] = "intact"    # what is left is lines that were mostly there: the page is complete
                    cmp = {**after, "added_lines": added, "before": cmp["verdict"],
                           "word_recall_before": cmp["word_recall"]}
            else:
                cmp = {**cmp, "would_add_lines": len(lines)}
        if self.layer_mode != "fill":
            return {**cmp, "mode": self.layer_mode, "verdict": cmp["verdict"], "report_only": True}
        return cmp

    # -- which reader takes a page, and the cache key that goes with it
    def vlm_ok(self) -> bool:
        return bool(self.vlm and self.vlm.usable())

    def pics(self, e: dict[str, Any]) -> list[list[float]]:
        return list(e["profile"].get("big_pics") or []) if (e["mode"] == "digital" and self.vlm_ok()) else []

    def reader_tag(self, e: dict[str, Any]) -> str:
        if e["mode"] == "scan" and self.is_b(e):
            return f"{self.reader.id}:scan"
        if e["mode"] == "scan" and self.vlm_ok():
            return f"{self.vlm.id}:scan"
        if e["mode"] == "digital" and self.pics(e):
            return f"{self.reader.id}:digital+{self.vlm.id}"
        return f"{self.reader.id}:{e['mode']}"

    def key(self, e: dict[str, Any], tag: str | None = None) -> str:
        return pagecache.cache_key(e["hash"], tag or self.reader_tag(e), self.settings) if e["hash"] else ""

    def branch_of(self, e: dict[str, Any], res: dict[str, Any]) -> str:
        if e["blank"]:                      # not read by anyone: it takes the branch its neighbours take
            return "image" if self.is_image else ("raster" if self.vlm_ok() else "fallback")
        return res.get("branch") or e["branch"]

    # -- the record of one page
    def finish(self, n: int) -> None:
        e, r = self.by_no[n], self.results[n]
        kind = "digital" if e["mode"] == "digital" else ("scan" if e["mode"] else "other")
        r["md"], n_tables = tables.normalize_html_tables(r["md"])        # one table format, whoever read the page
        if n_tables:
            r["note"] = "; ".join(x for x in (r.get("note"), f"{n_tables} HTML table(s) written as Markdown tables") if x)
        r["md"], n_joined = tables.join_split_pipe_tables(r["md"])
        if n_joined:
            r["note"] = "; ".join(x for x in (r.get("note"), "a table that the reader cut in two with a blank line was joined") if x)
        lay = self.check_layer(n, e, r) if (kind == "digital" and not e["blank"]) else None
        t_gate = time.perf_counter()
        conf = r.get("confidence") or (r.get("stats") or {}).get("confidence")
        g_layer = None if (lay is None or lay.get("report_only")) else lay      # report mode records, the gate keeps its old rule
        ocr = kind == "scan" and self.ocr_page(e, r)
        g = gate.check_page(r["md"], branch_kind=kind, profile=e["profile"], confidence=conf, layer=g_layer, ocr=ocr)
        gate_s = round(time.perf_counter() - t_gate, 4)
        repair_s, rep = 0.0, None
        flagged = [c for c in gate.failed(g) if c not in NOT_ESCALATED]
        if (kind == "scan" and not e["blank"] and self.repairer
                and (g.get("violations") or (flagged and getattr(self.repairer, "reread", False)))
                and self.repairer.usable()):
            prior = r.get("repair") if r["cache"] == "hit" else None
            if not (prior and prior.get("tag") == self.repairer.tag):
                rep = self._repair(n, e, r, g, kind, conf)
                repair_s = rep["seconds"]
                g = gate.check_page(r["md"], branch_kind=kind, profile=e["profile"], confidence=conf, ocr=ocr)
        if kind == "scan" and not e["blank"] and "degenerate" in gate.failed(g):
            if self._tesseract_last_resort(n, r):                       # the readers ran away: plain text instead
                ocr = False
                g = gate.check_page(r["md"], branch_kind=kind, profile=e["profile"], confidence=conf)
        fixed = int((r.get("repair") or {}).get("fixed") or 0) + (1 if (r.get("repair") or {}).get("tier") == "page" else 0)
        placeholder = not e["blank"] and not has_real_text(r["md"])       # "<!-- image -->" and a class label: nothing was read
        outcome = "no_text" if (g["verdict"] == "empty" or placeholder) else ("low" if g["verdict"] == "suspect" else "pass")
        if outcome == "pass" and fixed:
            outcome = "repaired"
        branch = self.branch_of(e, r)
        if r["cache"] == "hit":
            info = r.get("reader") or _reader_info(self.cfg, e["mode"] or "scan", self.reader.id)
        elif r.get("via") == "vlm":
            info = {"tool": "vlm", "model": r.get("model", ""), "mode": "page" if e["mode"] == "scan" else "picture"}
        elif r.get("via") == "apple-vision":
            info = {"tool": "apple-vision", "mode": "page"}
        elif r.get("via") == "tesseract":
            info = {"tool": "tesseract", "mode": "page"}
        else:
            info = _reader_info(self.cfg, e["mode"] or "scan", self.reader.id)
        was = r.get("was_branch") or branch
        notes = [x for x in (r.get("note"), (rep or {}).get("note"),
                             (f"read before by {was} ({r['was_s']} s)" if r["cache"] == "hit" and r.get("was_s")
                              else "result reused from the page cache" if r["cache"] == "hit" else "")) if x]
        out = _out_facts(r["md"], r.get("stats"))
        rec = trace.page_record(
            n, "cached" if r["cache"] == "hit" else branch, e["why"],
            profile={k: e["profile"][k] for k in PROFILE_KEEP if k in e["profile"] and e["profile"][k] not in (None, "")},
            reader=info if not e["blank"] else {"tool": "none"}, outcome=outcome, out=out,
            confidence=conf, note="; ".join(notes))
        rec["time_s"] = {"read": round(r["time_s"], 3), "gate": gate_s}
        if repair_s:
            rec["time_s"]["repair"] = round(repair_s, 3)
        rec["cache"] = r["cache"]
        if e.get("route"):
            rec["route"] = {"runway": e["route"]["runway"], "reasons": e["route"]["reasons"][:6],
                            "final": "b" if self.ocr_page(e, r) else "d"}
            if r.get("escalated_from"):
                rec["route"]["escalated_from"] = r["escalated_from"]
                if r.get("first_try_s"):
                    rec["time_s"]["first_try"] = round(r["first_try_s"], 3)
        if lay:
            rec["layer"] = {k: lay[k] for k in ("verdict", "word_recall", "number_recall", "added_lines", "before",
                                                "word_recall_before", "would_add_lines", "mode") if k in lay}
        if self.keys.get(n):
            rec["key"] = self.keys[n]
        if g.get("checks"):
            rec["gate"] = g
        if r["cache"] == "hit":
            rec["was"] = was
        elif r.get("via") == "vlm":
            if r.get("tokens"):
                rec["tokens"] = int(r["tokens"])
            if r.get("gpu_s"):
                rec["gpu_s"] = round(float(r["gpu_s"]), 3)
        if r.get("repair"):
            rp = r["repair"]
            rec["repair"] = {k: rp[k] for k in ("model", "second", "tier", "tried", "fixed", "cells") if rp.get(k) not in (None, "", [])}
            if rep:
                if rep.get("tokens"):
                    rec["tokens"] = int(rec.get("tokens") or 0) + int(rep["tokens"])
                if rep.get("gpu_s"):
                    rec["gpu_s"] = round(float(rec.get("gpu_s") or 0.0) + float(rep["gpu_s"]), 3)
        self.records[n] = rec
        if self.on_page:
            self.on_page(rec, self.total)

    def _repair(self, n: int, e: dict[str, Any], r: dict[str, Any], g: dict[str, Any], kind: str,
                conf: Any) -> dict[str, Any]:
        """Repair the suspect cells of a scanned page; the page's text and cache entry are replaced
        when something was fixed, and the attempt is kept in ``r["repair"]`` either way."""
        def page_ok(md: str) -> bool:
            g2 = gate.check_page(md, branch_kind=kind, profile=e["profile"], confidence=conf)
            return g2["verdict"] == "ok" or (not g2.get("violations") and set(gate.failed(g2)) <= {"low_resolution"})

        try:
            rep = self.repairer.run(self.src, n, self.is_image, r["md"], g.get("violations") or [], page_ok=page_ok)
        except Exception as exc:  # noqa: BLE001 - repair is optional: it never fails the page or the document
            why = f"{exc.reason}: " if isinstance(exc, vlm.ReaderError) else f"{type(exc).__name__}: "
            rep = {"md": r["md"], "cells": [], "fixed": 0, "tried": 0, "tier": "", "tokens": 0, "gpu_s": 0.0,
                   "seconds": 0.0, "model": self.repairer.reader.model, "second": "",
                   "note": f"repair not run ({why}{str(exc)[:120]})"}
        prior = r.get("repair") if r["cache"] == "hit" else None
        r["repair"] = {"tag": self.repairer.tag, "model": rep["model"], "second": rep["second"],
                       "tier": rep["tier"] or (prior or {}).get("tier", ""),
                       "tried": rep["tried"] + int((prior or {}).get("tried") or 0),
                       "fixed": rep["fixed"] + int((prior or {}).get("fixed") or 0),
                       "cells": list((prior or {}).get("cells") or []) + rep["cells"]}      # earlier fixes stay on record
        if rep["md"] != r["md"]:
            r["md"] = rep["md"]
            r["stats"] = dict(r.get("stats") or {}, chars=len("".join(rep["md"].split())))
        self.store(n, self.keys.get(n, ""), r, self.tags.get(n, ""))      # remember the attempt (and the fix)
        return rep

    def store(self, n: int, key: str, res: dict[str, Any], tag: str) -> None:
        self.tags[n] = tag
        if self.cache and key:
            self.cache.put(key, {"md": res["md"], "stats": res.get("stats") or {},
                                 "time_s": round(res["time_s"] or res.get("was_s") or 0.0, 3), "reader": tag,
                                 "settings": self.settings,
                                 "branch": res.get("branch") or res.get("was_branch") or self.by_no[n]["branch"],
                                 "reader_info": res.get("reader_info") or res.get("reader"),
                                 "tokens": res.get("tokens"), "gpu_s": res.get("gpu_s"),
                                 "model": res.get("model"), "repair": res.get("repair"),
                                 "guard": vlm.GUARD_VERSION if res.get("via") == "vlm" else None})

    # -- the work
    def plan_reads(self) -> dict[str, list[int]]:
        if self.vlm and any(not e["blank"] and (e["mode"] == "scan" or e["profile"].get("big_pics"))
                            for e in self.plan):
            self.vlm.check()                                 # a reader that cannot start is ruled out now
        self.route_scans()
        todo: dict[str, list[int]] = {"digital": [], "scan": []}
        for e in self.plan:
            n = e["page"]
            if e["blank"]:
                self.results[n] = {"md": "", "stats": {"chars": 0}, "time_s": 0.0, "cache": "none", "via": ""}
                continue
            self.keys[n] = key = self.key(e)
            hit = self.cache.get(key) if (self.cache and key) else None
            if self.is_b(e) and self.cache:                       # read by the document reader before, after OCR doubted it
                vkey = self.key(e, f"{self.vlm.id}:scan")
                vhit = self.cache.get(vkey) if vkey else None
                if vhit:
                    hit, self.keys[n] = vhit, vkey
            if (hit and (hit.get("reader_info") or {}).get("tool") == "vlm" and hit.get("guard") != vlm.GUARD_VERSION
                    and degenerate.assess(hit.get("md") or "")["bad"]):
                hit = None                    # read before the loop guard existed and it ran away: read it again
            if hit and self.is_b(e) and (hit.get("reader_info") or {}).get("tool") != "vlm":
                why = self.doubts(e, hit.get("md") or "", (hit.get("stats") or {}).get("confidence"))
                if why:                                           # the cached OCR text is not good enough: the document reader
                    self.escalate[n] = why
                    self.first_try[n] = {"md": hit["md"], "stats": hit.get("stats") or {}, "time_s": float(hit.get("time_s") or 0.0)}
                    hit = None
            if not hit and self.cache and e["mode"] == "scan" and not self.vlm_ok():     # a page Apple Vision read earlier
                for rescue_id in (applevision.ID, tesseract.ID):
                    av_key = self.key(e, f"{rescue_id}:scan")
                    hit = self.cache.get(av_key) if av_key else None
                    if hit:
                        self.keys[n] = av_key
                        break
            if hit:
                self.results[n] = {"md": hit["md"], "stats": hit.get("stats") or {}, "time_s": 0.0, "cache": "hit",
                                   "confidence": (hit.get("stats") or {}).get("confidence"),
                                   "was_s": hit.get("time_s"), "was_branch": hit.get("branch"),
                                   "reader": hit.get("reader_info"), "via": "cache", "repair": hit.get("repair"),
                                   "tokens": hit.get("tokens"), "gpu_s": hit.get("gpu_s"), "model": hit.get("model")}
                self.tags[n] = str(hit.get("reader") or "")
                self.hits += 1
            else:
                todo[e["mode"]].append(n)
                self.misses += 1
        return todo

    def read_b(self, pages: list[int]) -> list[int]:
        """Runway b: docling reads the pages with full-page OCR.  A page the gate doubts is kept aside and its number
        returned, for the document reader to read; the others are done."""
        sent: list[int] = []
        for first, last in _runs(pages):
            want = [n for n in range(first, last + 1) if n in pages]
            res = self.reader.read(self.src, first, last, "scan")
            self.runs += 1
            per = float(res.get("seconds") or 0.0) / max(1, len(want))
            for n in want:
                e = self.by_no[n]
                stats = (res.get("stats") or {}).get(n) or {}
                r = {"md": (res.get("pages") or {}).get(n, ""), "stats": stats, "time_s": per, "cache": "miss",
                     "confidence": stats.get("confidence"), "via": "docling", "branch": "fallback",
                     "note": stats.get("note") or ""}
                tag = self.reader_tag(e)
                self.keys[n] = self.key(e, tag)
                self.store(n, self.keys[n], r, tag)
                why = self.doubts(e, r["md"], r["confidence"])
                if why:
                    self.escalate[n] = why
                    self.first_try[n] = r
                    sent.append(n)
                else:
                    self.results[n] = r
                    self.finish(n)
        return sent

    def read_digital(self, pages: list[int]) -> None:
        for first, last in _runs(pages):
            want = [n for n in range(first, last + 1) if n in pages]
            res = self.reader.read(self.src, first, last, "digital")
            self.runs += 1
            per = float(res.get("seconds") or 0.0) / max(1, len(want))
            for n in want:
                e = self.by_no[n]
                stats = (res.get("stats") or {}).get(n) or {}
                r = {"md": (res.get("pages") or {}).get(n, ""), "stats": stats, "time_s": per, "cache": "miss",
                     "confidence": stats.get("confidence"), "via": "docling"}
                if stats.get("note"):
                    r["note"] = stats["note"]
                tag = f"{self.reader.id}:digital"
                key = self.keys.get(n, "")
                if self.pics(e):
                    self.add_pictures(n, r)
                    if r.get("via") == "vlm":
                        tag = self.reader_tag(e)
                    else:                                    # pictures not read: do not cache a half result
                        key = ""
                self.results[n] = r
                self.store(n, key, r, tag)
                self.finish(n)

    def add_pictures(self, n: int, r: dict[str, Any]) -> None:
        """The document VLM reads each large picture on a text page; what it finds that the page does
        not already have is appended.  Any failure leaves the docling result as it is."""
        e = self.by_no[n]
        added, tokens, gpu, t0 = [], 0, 0.0, time.perf_counter()
        for i, box in enumerate(e["profile"].get("big_pics") or [], 1):
            img = self.vlm._tmp_dir() / f"p{n}_{i}.png"
            try:
                vlm.render_pdf_page(self.src, n, img, crop=tuple(box))
                res = self.vlm.read_image(img, "picture")
            except vlm.ReaderError as exc:
                r["note"] = f"picture {i} not read by the document reader ({exc.reason}: {str(exc)[:120]})"
                return
            except Exception as exc:  # noqa: BLE001
                r["note"] = f"picture {i} not read ({type(exc).__name__}: {str(exc)[:120]})"
                return
            finally:
                img.unlink(missing_ok=True)
            tokens += int(res.get("tokens") or 0)
            gpu += float(res.get("seconds") or 0.0)
            new = _new_text(r["md"] + "\n\n" + "\n\n".join(added), str(res.get("md") or ""))
            if new:
                added.append(new)
        r.update(via="vlm", branch="embedded", model=self.vlm.model, tokens=tokens, gpu_s=gpu,
                 reader_info={"tool": "vlm", "model": self.vlm.model, "mode": "picture"},
                 time_s=r["time_s"] + (time.perf_counter() - t0))
        if added:
            r["md"] = r["md"].rstrip() + "\n\n" + "\n\n".join(added) + "\n"
            r["note"] = f"{len(added)} picture(s) read by the document reader"
            r["stats"] = dict(r["stats"], chars=len("".join(r["md"].split())))
        else:
            r["note"] = "pictures read by the document reader: nothing beyond the page's own text"

    def read_scan(self, pages: list[int]) -> None:
        failed: dict[int, str] = {}
        b_pages = [n for n in pages if self.is_b(self.by_no[n]) and n not in self.escalate]
        if b_pages:
            pages = [n for n in pages if n not in b_pages] + self.read_b(b_pages)
        if self.vlm_ok():
            for first, last in _runs(pages):
                want = [n for n in range(first, last + 1) if n in pages]
                tags = {n: f"{self.vlm.id}:scan" for n in want}        # before reading: the reader may die meanwhile
                for n in want:
                    self.keys[n] = self.key(self.by_no[n], tags[n])
                try:
                    res = self.vlm.read(self.src, first, last, "scan")
                except vlm.ReaderError as exc:
                    res = {"pages": {}, "stats": {}, "failed": {n: f"{exc.reason}: {exc}" for n in want}}
                self.runs += 1
                for n in want:
                    if n in (res.get("pages") or {}):
                        st = res["stats"].get(n) or {}
                        r = {"md": res["pages"][n], "stats": st, "time_s": float(st.get("read_s") or 0.0),
                             "cache": "miss", "via": "vlm", "model": self.vlm.model, "tokens": st.get("tokens"),
                             "note": st.get("note") or "",
                             "reader_info": {"tool": "vlm", "model": self.vlm.model, "mode": "page"},
                             "gpu_s": st.get("gpu_s"), "branch": "image" if self.is_image else "raster"}
                        if n in self.escalate:
                            ft = self.first_try.get(n) or {}
                            r["escalated_from"] = {"runway": "b", "checks": self.escalate[n]}
                            r["first_try_s"] = float(ft.get("time_s") or 0.0)
                        self.results[n] = r
                        self.store(n, self.keys.get(n, ""), r, tags[n])
                        self.finish(n)
                    else:
                        failed[n] = (res.get("failed") or {}).get(n, "the document reader did not return this page")
        else:
            why = (self.vlm.dead if self.vlm else "") or "the document reader is switched off"
            failed = {n: why for n in pages}
        if not failed:
            return
        if self.is_image:                                   # image files have no page fallback
            first = sorted(failed)[0]
            raise vlm.ReaderUnavailable(failed[first])
        for first, last in _runs(sorted(failed)):
            want = [n for n in range(first, last + 1) if n in failed]
            res = self.reader.read(self.src, first, last, "scan")
            self.runs += 1
            per = float(res.get("seconds") or 0.0) / max(1, len(want))
            for n in want:
                e = self.by_no[n]
                stats = (res.get("stats") or {}).get(n) or {}
                r = {"md": (res.get("pages") or {}).get(n, ""), "stats": stats, "time_s": per, "cache": "miss",
                     "confidence": stats.get("confidence"), "via": "docling", "branch": "fallback",
                     "note": "; ".join(x for x in (f"read by docling OCR: {failed[n][:200]}", stats.get("note")) if x)}
                if n in self.first_try:                         # OCR read it already, the document reader failed: keep OCR's text
                    r = dict(self.first_try[n], cache="miss", via="docling", branch="fallback", confidence=None,
                             note=f"the document reader could not take over ({failed[n][:120]}); docling OCR's text kept")
                if not e["blank"] and not has_real_text(r["md"]):
                    self.rescue(n, r)
                self.results[n] = r
                tag = {"docling": self.reader.id, "tesseract": tesseract.ID}.get(r["via"], applevision.ID) + ":scan"
                key = self.key(e, tag)
                self.keys[n] = key
                self.store(n, key, r, tag)
                self.finish(n)

    def _tesseract_last_resort(self, n: int, r: dict[str, Any]) -> bool:
        """The document reader (and the repair model) left a runaway on this page: read it with Tesseract.
        Kept only when the text is not a runaway and is real text; the page then is plain text, no tables."""
        why = tesseract.why_not()
        if why:
            r["note"] = "; ".join(x for x in (r.get("note"), f"Tesseract not used ({why})") if x)
            return False
        t0 = time.perf_counter()
        try:
            text = tesseract.read_page(self.src, n, is_image=self.is_image)
        except Exception as exc:  # noqa: BLE001 - a last resort never fails the page
            r["note"] = "; ".join(x for x in (r.get("note"), f"Tesseract failed ({type(exc).__name__}: {str(exc)[:120]})") if x)
            return False
        took = time.perf_counter() - t0
        if not has_real_text(text) or degenerate.assess(text)["bad"]:
            r["note"] = "; ".join(x for x in (r.get("note"), "Tesseract found no usable text either") if x)
            return False
        r.update(md=text.strip() + "\n", via="tesseract", branch="fallback", time_s=r["time_s"] + took,
                 reader_info={"tool": "tesseract", "mode": "page"}, repair=None,
                 stats=dict(r.get("stats") or {}, chars=len("".join(text.split()))),
                 note="; ".join(x for x in (r.get("note"), "the document reader repeated itself; the page was read "
                                            "by Tesseract (plain text, no tables)") if x))
        self.store(n, self.keys.get(n, ""), r, self.tags.get(n, ""))
        return True

    def _tesseract_rescue(self, n: int, r: dict[str, Any]) -> None:
        """docling's OCR and Apple Vision found nothing on a scanned page: Tesseract is the last reader."""
        if tesseract.why_not():
            return
        try:
            text = tesseract.read_page(self.src, n, is_image=self.is_image)
        except Exception:  # noqa: BLE001 - optional
            return
        if has_real_text(text):
            r.update(md=text.strip() + "\n", via="tesseract", branch="fallback",
                     reader_info={"tool": "tesseract", "mode": "page"},
                     stats=dict(r.get("stats") or {}, chars=len("".join(text.split()))),
                     note="; ".join(x for x in (r.get("note"), "read by Tesseract (plain text, no tables)") if x))

    def rescue(self, n: int, r: dict[str, Any]) -> None:
        """docling's OCR returned no text for this scanned page (a photographed page is one big picture to
        its layout model): read the page image with Apple Vision when this Mac can, and say so."""
        why = applevision.why_not()
        if why:
            r["note"] = "; ".join(x for x in (r.get("note"), f"no text from docling OCR; Apple Vision not used ({why})") if x)
            self._tesseract_rescue(n, r)
            return
        t0 = time.perf_counter()
        try:
            text = applevision.read_page(self.src, n, is_image=self.is_image, languages=self.cfg.get("lang") or None)
        except Exception as exc:  # noqa: BLE001 - a rescue never fails the page
            r["note"] = "; ".join(x for x in (r.get("note"), f"Apple Vision failed ({type(exc).__name__}: {str(exc)[:120]})") if x)
            return
        took = time.perf_counter() - t0
        if not has_real_text(text):
            r["note"] = "; ".join(x for x in (r.get("note"), "Apple Vision found no text either") if x)
            self._tesseract_rescue(n, r)
            return
        r.update(md=text.strip() + "\n", via="apple-vision", branch="fallback", time_s=r["time_s"] + took,
                 reader_info={"tool": "apple-vision", "mode": "page"},
                 stats=dict(r.get("stats") or {}, chars=len("".join(text.split()))),
                 note="; ".join(x for x in (r.get("note"), "docling OCR found no text on this page; read by Apple Vision "
                                            "(plain text, no tables)") if x))

    def reconcile(self) -> None:
        """Tables that run on across a page break: the continuation gets the table's header, the
        merged table is checked, and the pages concerned say so in their records (``reconcile.py``)."""
        res = reconcile.merge({n: r["md"] for n, r in self.results.items()})
        for n, md in res["pages"].items():
            r = self.results[n]
            if md == r["md"]:
                continue
            r["md"] = md
            r["stats"] = dict(r.get("stats") or {}, chars=len("".join(md.split())))
            rec, e = self.records.get(n), self.by_no[n]
            if rec is not None:
                kind = "digital" if e["mode"] == "digital" else ("scan" if e["mode"] else "other")
                g = gate.check_page(md, branch_kind=kind, profile=e["profile"], confidence=rec.get("docling"))
                if g.get("checks"):
                    rec["gate"] = g
                    if rec.get("outcome") in ("pass", "repaired"):
                        rec["outcome"] = "low"
                else:                                         # the header made the page's tables checkable and clean
                    rec.pop("gate", None)
                    if rec.get("outcome") == "low":
                        rec["outcome"] = "repaired" if (r.get("repair") or {}).get("fixed") else "pass"
        for t in res["tables"]:
            a, b = t["pages"]
            for page, role, other in ((a, "starts", b), (b, "continues", a)):
                rec = self.records.get(page)
                if rec is None:
                    continue
                item = {"role": role, "with": other, "header": t["header"], "rows": t["rows"], "ok": t["ok"]}
                bad = [v for v in t["violations"] if v["page"] == page]
                if bad:
                    item["violations"] = bad
                    gate_rec = rec.setdefault("gate", {"verdict": "suspect", "checks": []})
                    gate_rec["verdict"] = "suspect"
                    gate_rec.setdefault("checks", []).append(
                        {"name": "table_across_pages", "ok": False, "detail": bad[0]["why"]})
                    if rec.get("outcome") in ("pass", "repaired"):
                        rec["outcome"] = "low"
                rec["reconcile"] = item

    def run(self) -> dict[int, dict[str, Any]]:
        try:
            todo = self.plan_reads()
            for n in sorted(self.results):                  # cached and blank pages are done already
                self.finish(n)
            self.read_digital(todo["digital"])
            self.read_scan(todo["scan"])
            self.reconcile()
            return self.results
        finally:
            if self.layers is not None:
                self.layers.close()


def no_text_message(src: Path, results: dict[int, dict[str, Any]], reasons: str = "") -> str:
    """Why nothing could be read, in words that say what to do (shown in the Indexing tab)."""
    via = {r.get("via") or ("cache" if r.get("cache") == "hit" else "docling") for r in results.values()}
    msg = f"no text could be read from {src.name} ({len(results)} page(s))"
    if "vlm" not in via:
        msg += ("; the document reader (a vision model) did not read any page, so docling's OCR was the only "
                "reader and found nothing. For photographed or scanned pages install and select the reader "
                "(Models tab: Document reader), then index again")
    else:
        msg += "; the document reader read the pages and found no text (a photograph without writing?)"
    av = sorted({str(r["note"]) for r in results.values() if "Apple Vision" in str(r.get("note") or "")})
    if av:
        msg += "; " + av[0][:300]
    return msg + (f" [{reasons}]" if reasons else "")


def _write(md_path: Path, results: dict[int, dict[str, Any]], src: Path, reasons: str = "") -> None:
    md_all = pagemd.join_pages({n: results[n]["md"] for n in sorted(results)})
    if not has_real_text(md_all):      # empty, or only "<!-- image -->" placeholders and a label
        raise NoTextError(no_text_message(src, results, reasons))
    md_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = md_path.with_name(md_path.name + ".partial")
    tmp.write_text(md_all, encoding="utf-8")
    os.replace(tmp, md_path)


def _result(c: _Converter, t0: float) -> dict[str, Any]:
    return {"pages": c.total, "records": [c.records[n] for n in sorted(c.records)],
            "seconds": round(time.perf_counter() - t0, 2),
            "cache": {"hit": c.hits, "miss": c.misses}, "runs": c.runs}


def convert_pdf(src: Path, md_path: Path, profile: dict[str, Any], *, cache: pagecache.PageCache | None,
                reader: Any = None, scan_reader: Any = None, ocr: bool | None = None,
                on_page: Callable[[dict[str, Any], int], None] | None = None,
                repairer: Any = None) -> dict[str, Any]:
    """Convert *src* page by page.  Writes *md_path*; returns ``{"pages", "records", "seconds",
    "cache": {"hit", "miss"}, "runs"}``.  *reader* is docling (default), *scan_reader* the document
    VLM (``vlm.VlmReader``) or None.  Raises ``NoTextError`` when no page has any text."""
    t0 = time.perf_counter()
    c = _Converter(src, profile, cache=cache, reader=reader or default_reader(), scan_reader=scan_reader,
                   ocr=ocr, on_page=on_page, repairer=repairer)
    _write(md_path, c.run(), src)
    return _result(c, t0)


def convert_image(src: Path, md_path: Path, profile: dict[str, Any], *, cache: pagecache.PageCache | None,
                  scan_reader: Any, ocr: bool | None = None,
                  on_page: Callable[[dict[str, Any], int], None] | None = None,
                  repairer: Any = None) -> dict[str, Any]:
    """Convert an image file (one page per frame) with the document VLM.  Raises ``vlm.ReaderError``
    when the reader cannot read a frame (the caller converts the file with docling), ``NoTextError``
    for a picture without text (a photograph is not text)."""
    t0 = time.perf_counter()
    c = _Converter(src, profile, cache=cache, reader=default_reader(), scan_reader=scan_reader, ocr=ocr,
                   on_page=on_page, repairer=repairer)
    _write(md_path, c.run(), src)
    return _result(c, t0)
