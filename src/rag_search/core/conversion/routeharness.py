"""The routing harness (plan step R0): does the router's choice of runway agree with what OCR actually achieves, and does
the gate catch the pages where OCR was wrong?  Pillow + numpy; the OCR engine is injected.

Synthetic scanned pages with known text (``synth.py``, a ladder of damage) are examined (``scanfacts``), routed
(``router.decide_scan``), written as image-only PDFs, read by an OCR function, scored against the true text, and put
through the OCR-side gate (``gate.check_page(..., ocr=True)``).  Every page lands in one cell of this table:

=================  ===========================  ============================================================
router             OCR result (recall >= 0.97)  outcome
=================  ===========================  ============================================================
b (OCR first)      good                         ``saved``: the document reader was not needed
b                  bad, the gate escalates      ``caught``: OCR time wasted, the page is read properly
b                  bad, the gate passes it      ``false_pass``: a bad page enters the index (the one to drive to zero)
d (document        good                         ``missed_saving``: OCR would have done
reader first)      bad                          ``right``
=================  ===========================  ============================================================

``frontier`` lists, per damage level, how the router decides, what OCR recall is, and what the gate says; the thresholds
(``router.THRESHOLDS``, the gate's constants) are tuned until ``false_pass`` is empty and ``missed_saving`` small.
"""

from __future__ import annotations

import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

GOOD_RECALL = 0.97
FACT_DPI = 100                     # the resolution the facts are measured at (the same as ``scanfacts.page_facts``)

OcrFn = Callable[[Path], str]      # image-only one-page PDF -> the text OCR read


def tesseract_ocr(pdf: Path) -> str:
    from . import tesseract

    why = tesseract.why_not()
    if why:
        raise RuntimeError(f"Tesseract is not available: {why}")
    return tesseract.read_page(pdf, 1)


def docling_ocr(pdf: Path) -> str:
    from .routed import DoclingReader

    res = DoclingReader().read(pdf, 1, 1, "scan")
    return (res.get("pages") or {}).get(1, "")


ENGINES: dict[str, OcrFn] = {"tesseract": tesseract_ocr, "docling": docling_ocr}


def examine(case_lines: list[str], kw: dict[str, Any], tmp: Path, name: str, ocr: OcrFn, seed: int) -> dict[str, Any]:
    """One synthetic page through facts, router, OCR, scoring and gate."""
    from PIL import Image

    from . import gate, layer, profiler, router, scanfacts, synth

    kw = dict(kw)
    dpi = kw.setdefault("dpi", 200)
    im = synth.make_page(case_lines, seed=seed, **kw)
    small = im.resize((max(1, round(im.width * FACT_DPI / dpi)), max(1, round(im.height * FACT_DPI / dpi))), Image.LANCZOS)
    facts = scanfacts.facts_of_image(small, dpi=FACT_DPI)
    ink = profiler.ink_and_hash(im.resize((round(im.width * profiler.INK_SCALE * 72 / dpi),
                                           round(im.height * profiler.INK_SCALE * 72 / dpi)), Image.LANCZOS))["ink"]
    profile = {"dpi": dpi, "ink": ink}
    runway, reasons = router.decide_scan(facts, profile)
    pdf = tmp / f"{name}-{seed}.pdf"
    synth.write_pdf(pdf, [im], dpi)
    t0 = time.perf_counter()
    try:
        text, err = ocr(pdf), ""
    except Exception as exc:  # noqa: BLE001 - an OCR failure is a result of the case, not of the harness
        text, err = "", f"{type(exc).__name__}: {str(exc)[:100]}"
    seconds = round(time.perf_counter() - t0, 2)
    truth_w, truth_n = layer.tokens("\n".join(case_lines))
    got_w, got_n = layer.tokens(text)
    recall = layer.recall(truth_w, got_w)
    g = gate.check_page(text, branch_kind="scan", profile=profile, ocr=True)
    escalate = bool(g.get("escalate")) or not text.strip()
    good = (recall or 0.0) >= GOOD_RECALL
    if runway == "b":
        outcome = "saved" if good else ("caught" if escalate else "false_pass")
    else:
        outcome = "missed_saving" if good else "right"
    return {"level": name, "seed": seed, "runway": runway, "reasons": reasons[:4], "ink": ink,
            "recall": None if recall is None else round(recall, 3), "gate": gate.failed(g), "escalate": escalate,
            "outcome": outcome, "ocr_s": seconds, "error": err, "facts": {k: facts[k] for k in ("contrast", "sharp", "skew", "speckle", "bg_std", "text_lines", "h_rules", "v_rules")}}


def run(ocr: str | OcrFn = "tesseract", *, seeds: int = 3, levels: list[str] | None = None) -> dict[str, Any]:
    """The report: every case, counts per outcome, per damage level, and the false passes."""
    from . import router, synth

    fn = ENGINES[ocr] if isinstance(ocr, str) else ocr
    names = levels or list(synth.DAMAGE)
    cases: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="rag-search-route-") as t:
        tmp = Path(t)
        for name in names:
            for seed in range(1, seeds + 1):
                cases.append(examine(synth.text_lines(seed), synth.DAMAGE[name], tmp, name, fn, seed))
    by_outcome = Counter(c["outcome"] for c in cases)
    bad = [c for c in cases if (c["recall"] or 0.0) < GOOD_RECALL]
    gate_alone = {"bad_pages": len(bad), "caught": sum(1 for c in bad if c["escalate"]),
                  "missed": [f"{c['level']}#{c['seed']} (recall {c['recall']})" for c in bad if not c["escalate"]],
                  "false_alarms": sum(1 for c in cases if c not in bad and c["escalate"]), "good_pages": len(cases) - len(bad)}
    frontier: dict[str, dict[str, Any]] = {}
    per: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for c in cases:
        per[c["level"]].append(c)
    for name, cs in per.items():
        recalls = [c["recall"] for c in cs if c["recall"] is not None]
        frontier[name] = {"router": dict(Counter(c["runway"] for c in cs)), "min_recall": min(recalls) if recalls else None,
                          "gate_escalates": sum(1 for c in cs if c["escalate"]), "cases": len(cs),
                          "outcomes": dict(Counter(c["outcome"] for c in cs)), "why": cs[0]["reasons"][:2]}
    return {"thresholds": router.THRESHOLDS, "cases": cases, "by_outcome": dict(by_outcome), "gate_alone": gate_alone, "frontier": frontier,
            "false_pass": [c for c in cases if c["outcome"] == "false_pass"],
            "engine": ocr if isinstance(ocr, str) else getattr(ocr, "__name__", "custom")}


def markdown(rep: dict[str, Any]) -> str:
    """The report as text for a person."""
    out = [f"# Routing harness ({rep['engine']}, {len(rep['cases'])} synthetic pages)", ""]
    n = len(rep["cases"]) or 1
    labels = {"saved": "router chose OCR, OCR was good", "caught": "router chose OCR, OCR was bad, the gate sent it on",
              "false_pass": "router chose OCR, OCR was bad, THE GATE PASSED IT", "missed_saving": "router chose the document reader, OCR would have done",
              "right": "router chose the document reader, OCR was bad"}
    for k in ("saved", "caught", "false_pass", "missed_saving", "right"):
        out.append(f"* {rep['by_outcome'].get(k, 0):>4}  {k}: {labels[k]}")
    ga = rep["gate_alone"]
    out += ["", f"The gate alone, whatever the router chose: it escalated {ga['caught']} of {ga['bad_pages']} pages whose OCR "
            f"recall was under {GOOD_RECALL} and raised a false alarm on {ga['false_alarms']} of {ga['good_pages']} good ones."]
    if ga["missed"]:
        out.append("Bad pages it let through: " + ", ".join(ga["missed"]) + ".")
    out += ["", f"False passes: {len(rep['false_pass'])} of {n} pages.", "", "| damage | router | OCR recall (worst) | gate escalates | outcomes | why |", "|---|---|---|---|---|---|"]
    for name, f in rep["frontier"].items():
        out.append(f"| {name} | {f['router']} | {f['min_recall']} | {f['gate_escalates']}/{f['cases']} | {f['outcomes']} | {'; '.join(f['why'])} |")
    return "\n".join(out) + "\n"
