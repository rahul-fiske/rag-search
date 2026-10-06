#!/usr/bin/env python3
"""Sanity checks: can rag-search's own environment run its tools and models on this machine?

Run it with the Python of the installed tool (the one the dashboard and the daemons use), never with a
system Python or a separate virtualenv -- the point is to test what the user's dashboard really runs:

    "$(uv tool dir)/rag-search/bin/python" scripts/sanity_check.py              # quick: no model is loaded
    ... sanity_check.py --apple-vision      # + Apple Vision reads a generated image (Latin; Devanagari if a font exists)
    ... sanity_check.py --vlm               # + the document reader model reads a generated scan (loads the model)
    ... sanity_check.py --docling           # + docling converts a generated digital PDF (loads docling's models)
    ... sanity_check.py --e2e               # + a throw-away Playground experiment indexes a generated scan and PDF (routed pipeline)
    ... sanity_check.py --models            # + `rag-search models verify` (loads the embedding model and reranker)
    ... sanity_check.py --all               # everything above
    ... sanity_check.py --json              # machine-readable result

Every check prints PASS, FAIL or SKIP with a reason and its time; the exit code is 1 when any check failed.
The generated documents live in a temporary folder and contain no personal data.  Nothing is downloaded: a
model that is not in the cache is reported as missing (use the Models tab or `rag-search models download`).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable

HERE = Path(__file__).resolve().parent
SOURCE_INIT = HERE.parent / "src" / "rag_search" / "__init__.py"
NUMBER = "1234.50"                          # what the generated documents say; the reader must get it right
TEXT_LINES = ["Sanity Invoice 2026", f"Total amount {NUMBER}", "Paid by transfer on 31/03/2026"]
DEVANAGARI_FONTS = ("/System/Library/Fonts/Supplemental/DevanagariMT.ttc",
                    "/System/Library/Fonts/Supplemental/Devanagari Sangam MN.ttc",
                    "/System/Library/Fonts/Kohinoor.ttc")
LATIN_FONTS = ("/System/Library/Fonts/Helvetica.ttc", "/System/Library/Fonts/Supplemental/Arial.ttf",
               "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")

Result = dict[str, Any]


def result(name: str, status: str, detail: str = "", seconds: float = 0.0) -> Result:
    return {"check": name, "status": status, "detail": detail, "seconds": round(seconds, 1)}


def timed(name: str, fn: Callable[[], tuple[str, str]]) -> Result:
    """Run one check: *fn* returns (PASS|FAIL|SKIP, detail); an exception is a FAIL with its text."""
    t0 = time.perf_counter()
    try:
        status, detail = fn()
    except Exception as exc:  # noqa: BLE001 - a check never stops the others
        status, detail = "FAIL", f"{type(exc).__name__}: {str(exc)[:300]}"
    return result(name, status, detail, time.perf_counter() - t0)


# ── generated test documents ───────────────────────────────────────────────────────────────

def _font(candidates: tuple[str, ...], size: int):
    from PIL import ImageFont

    for c in candidates:
        if Path(c).exists():
            try:
                return ImageFont.truetype(c, size)
            except OSError:
                continue
    return ImageFont.load_default()


def scan_image(path: Path, lines: list[str] | None = None, *, fonts: tuple[str, ...] = LATIN_FONTS) -> Path:
    """A clean 200-dpi A4 'photograph' of a few printed lines (PNG, or PDF when *path* ends in .pdf)."""
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (1654, 2339), "white")
    draw = ImageDraw.Draw(img)
    font = _font(fonts, 56)
    y = 200
    for line in lines or TEXT_LINES:
        draw.text((160, y), line, fill="black", font=font)
        y += 110
    if path.suffix.lower() == ".pdf":
        img.save(path, "PDF", resolution=200.0)
    else:
        img.save(path)
    return path


def digital_pdf(path: Path, lines: list[str] | None = None) -> Path:
    """A one-page PDF with a real text layer (hand-written, Helvetica)."""
    body = "BT /F1 18 Tf 72 720 Td 28 TL " + " ".join(f"({ln}) Tj T*" for ln in (lines or TEXT_LINES)) + " ET"
    objs = ["<< /Type /Catalog /Pages 2 0 R >>",
            "<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] /Contents 4 0 R "
            "/Resources << /Font << /F1 5 0 R >> >> >>",
            f"<< /Length {len(body)} >>\nstream\n{body}\nendstream",
            "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"]
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for i, o in enumerate(objs, 1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n{o}\nendobj\n".encode("latin-1")
    xref = len(out)
    out += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode()
    for off in offsets:
        out += f"{off:010d} 00000 n \n".encode()
    out += f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    path.write_bytes(bytes(out))
    return path


def has_number(text: str) -> bool:
    return NUMBER in (text or "").replace(",", "").replace(" ", "")


# ── the checks ─────────────────────────────────────────────────────────────────────────────

def check_python() -> tuple[str, str]:
    import rag_search

    exe = Path(sys.executable)
    where = Path(rag_search.__file__).resolve()
    tool = ""
    try:
        tool = subprocess.run(["uv", "tool", "dir"], capture_output=True, text=True, timeout=20).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    notes = [f"python {sys.version.split()[0]} at {exe}", f"rag_search {rag_search.__version__} from {where}"]
    if "rag-search" not in exe.parts and not (tool and str(exe).startswith(tool)):
        return "FAIL", "; ".join(notes + ["this is not the installed tool's Python: run with "
                                           '"$(uv tool dir)/rag-search/bin/python"'])
    if SOURCE_INIT.exists():
        m = re.search(r'__version__\s*=\s*"([^"]+)"', SOURCE_INIT.read_text())
        if m and m.group(1) != rag_search.__version__:
            return "FAIL", "; ".join(notes + [f"the checkout is {m.group(1)}: the tool runs a different version "
                                               "(reinstall it editable: see 'Dev loop' in CONTRIBUTING.md)"])
        if SOURCE_INIT.parent not in where.parents:
            notes.append("installed as a copy, not editable: source edits will not take effect until you reinstall")
    return "PASS", "; ".join(notes)


def check_machine() -> tuple[str, str]:
    import platform

    from rag_search.core.conversion import vlm

    arm = sys.platform == "darwin" and platform.machine() == "arm64"
    free = vlm.available_memory_gb()
    detail = f"{platform.platform()} {platform.machine()}; free memory for models: " + (f"{free:.1f} GB" if free else "unknown")
    return ("PASS" if arm else "FAIL"), detail + ("" if arm else " (the document reader needs an Apple Silicon Mac)")


def check_packages() -> tuple[str, str]:
    from rag_search import models

    rt = models.runtime_state()
    rows = [(p["module"], p["installed"], p["required"]) for p in rt["packages"]]
    extra = []
    for mod in ("docling", "torch", "sentence_transformers"):
        try:
            __import__(mod)
            extra.append((mod, True, True))
        except Exception:  # noqa: BLE001
            extra.append((mod, False, True))
    missing = [m for m, ok, req in rows + extra if not ok and req]
    detail = ", ".join(f"{m}{'' if ok else ' MISSING'}" for m, ok, _ in rows + extra)
    return ("FAIL", f"{detail}  -> rag-search models runtime install") if missing else ("PASS", detail)


def check_gpu() -> tuple[str, str]:
    notes = []
    try:
        import mlx.core as mx

        a = mx.ones((256, 256))
        mx.eval(a @ a)
        notes.append(f"MLX ok on {mx.default_device()}")
    except Exception as exc:  # noqa: BLE001
        return "FAIL", f"MLX cannot run a matrix product: {type(exc).__name__}: {str(exc)[:200]}"
    try:
        import torch

        notes.append(f"torch MPS {'available' if torch.backends.mps.is_available() else 'NOT available'}")
    except Exception:  # noqa: BLE001
        notes.append("torch not importable")
    return "PASS", "; ".join(notes)


def check_models() -> tuple[str, str]:
    from rag_search import models
    from rag_search.paths import get_paths

    st = models.vlm_state(get_paths())
    bad, ok = [], []
    for kind in models.VLM_KINDS:
        row = next((r for r in st[kind]["models"] if r["active"]), None)
        if row is None:
            continue
        label = f"{kind}: {row['id']}"
        if row["cached"]:
            ok.append(label)
        else:
            bad.append(f"{label} ({row.get('why_not') or 'not downloaded'})")
    return ("FAIL", "not usable: " + "; ".join(bad) + (" | ok: " + "; ".join(ok) if ok else "")) if bad \
        else ("PASS", "; ".join(ok))


def check_tesseract() -> tuple[str, str]:
    """The last-resort page reader: the binary and its Marathi / Hindi / English data (install.sh adds them)."""
    from rag_search.core.conversion import tesseract

    why = tesseract.why_not()
    if why:
        return "FAIL", why
    have = tesseract.installed_languages()
    missing = [x for x in ("mar", "hin", "eng") if x not in have]
    return ("FAIL", f"tesseract lacks the language data {', '.join(missing)}") if missing else \
        ("PASS", f"tesseract {', '.join(sorted(have - {'osd'}))}")


def check_apple_vision(tmp: Path) -> tuple[str, str]:
    from rag_search.core.conversion import applevision

    why = applevision.why_not()
    if why:
        return "FAIL", why
    text = applevision.read_image(scan_image(tmp / "av.png"))
    if not has_number(text):
        return "FAIL", f"did not read {NUMBER}; got: {text[:120]!r}"
    detail = f"read {NUMBER} correctly"
    font = next((f for f in DEVANAGARI_FONTS if Path(f).exists()), "")
    if font:
        # Informational only, and with the languages the pipeline itself uses (none given): Apple Vision
        # has no Hindi / Marathi on every macOS, and asking for a language it lacks is an error.
        try:
            dev = applevision.read_image(scan_image(tmp / "av-dev.png", ["शाखा पता खाता क्रमांक"], fonts=(font,)))
            detail += f"; Devanagari line read as {dev.strip()[:60]!r} (informational)"
        except Exception as exc:  # noqa: BLE001
            detail += f"; Devanagari line not read ({type(exc).__name__}: {str(exc)[:80]}) (informational)"
    return "PASS", detail


def check_vlm(tmp: Path) -> tuple[str, str]:
    from rag_search.core.conversion import vlm

    reader = vlm.shared()
    if reader is None:
        return "SKIP", "the document reader is switched off (RAG_SEARCH_VLM=off)"
    try:
        reader.check()
        if not reader.usable():
            return "FAIL", reader.dead or "the document reader is not usable"
        src = scan_image(tmp / "vlm.pdf")
        t0 = time.perf_counter()
        res = reader.read(src, 1, 1, "scan")
        text = (res.get("pages") or {}).get(1, "")
        took = time.perf_counter() - t0
        if not has_number(text):
            return "FAIL", f"{reader.model}: did not read {NUMBER}; got {text[:120]!r}; failed={res.get('failed')}"
        return "PASS", f"{reader.model}: read {NUMBER} in {took:.0f}s (load {reader.load_s:.0f}s, peak {reader.peak_mb:.0f} MB)"
    finally:
        vlm.close_shared()


def check_docling(tmp: Path) -> tuple[str, str]:
    from rag_search.core.docling_convert import convert_file

    out = tmp / "docling.md"
    convert_file(digital_pdf(tmp / "digital.pdf"), out)
    text = out.read_text(errors="replace") if out.exists() else ""
    return ("PASS", f"converted a digital PDF ({len(text)} characters, contains {NUMBER})") if has_number(text) \
        else ("FAIL", f"docling output lacks {NUMBER}: {text[:160]!r}")


def _cli(*args: str, timeout: int = 1800) -> subprocess.CompletedProcess:
    exe = Path(sys.executable).parent / "rag-search"
    return subprocess.run([str(exe) if exe.exists() else "rag-search", *args], capture_output=True, text=True,
                          timeout=timeout)


def check_e2e(tmp: Path) -> tuple[str, str]:
    """The routed pipeline end to end, through the Playground (what its dashboard tab runs): a throw-away
    experiment gets a generated scanned PDF and a generated digital PDF, is indexed, and the Markdown and the
    per-page trace are read back.  The experiment is deleted at the end."""
    name = "sanity-check"
    _cli("playground", "rm", name, "--yes")                         # a leftover of an earlier run
    src = tmp / "sanity-sources"                                    # an experiment's source is a folder
    src.mkdir(exist_ok=True)
    scan_image(src / "sanity-scan.pdf"), digital_pdf(src / "sanity-digital.pdf")
    made = _cli("playground", "create", name, "--from", str(src), "--json")
    if made.returncode != 0:
        return "FAIL", f"playground create failed: {(made.stderr or made.stdout)[-300:]}"
    try:
        home = Path(json.loads(made.stdout)["home"])
        ran = _cli("playground", "index", name, "--json")
        if ran.returncode != 0:
            return "FAIL", f"playground index exited {ran.returncode}: {(ran.stderr or ran.stdout)[-400:]}"
        notes, bad = [], []
        for stem in ("sanity-scan", "sanity-digital"):
            mds = [m for m in (home / "workspace" / "markup").rglob("*.md") if m.name.startswith(stem)]
            text = mds[0].read_text(errors="replace") if mds else ""
            tr = next(iter((home / "workspace" / "markup").rglob(f"{stem}*.trace.json")), None)
            pages = (json.loads(tr.read_text()).get("pages") if tr else None) or []
            how = ", ".join(f"p{p.get('page')}:{p.get('branch')}/{p.get('outcome')}/lane {(p.get('route') or {}).get('final', '?')}"
                            for p in pages) or "no trace"
            notes.append(f"{stem}: {how}")
            want = "a" if stem.endswith("digital") else "d"          # a text page: lane a; a scan: the document reader
            if pages and any((p.get("route") or {}).get("final") != want for p in pages if p.get("branch") != "fallback"):
                bad.append(f"{stem}: expected lane {want}")
            if not has_number(text):
                bad.append(f"{stem} lacks {NUMBER} ({text[:80]!r})")
        return ("FAIL", "; ".join(bad) + " | " + "; ".join(notes)) if bad else ("PASS", "; ".join(notes))
    finally:
        _cli("playground", "rm", name, "--yes")


def check_models_verify() -> tuple[str, str]:
    notes, ok = [], True
    for kind in ("embedding", "reranker"):          # `models verify` takes one kind per call
        p = _cli("models", "verify", kind, timeout=1200)
        ok = ok and p.returncode == 0
        notes.append(f"{kind}: " + " ".join((p.stdout + p.stderr).split())[-200:])
    return ("PASS" if ok else "FAIL"), "; ".join(notes)


def run(a: argparse.Namespace) -> list[Result]:
    everything = a.all
    out = [timed("python and version", check_python), timed("machine", check_machine),
           timed("packages", check_packages), timed("GPU (MLX / torch)", check_gpu),
           timed("models downloaded", check_models), timed("Tesseract (last resort)", check_tesseract)]
    with tempfile.TemporaryDirectory(prefix="rag-sanity-") as d:
        tmp = Path(d)
        steps = [("Apple Vision reads an image", a.apple_vision, lambda: check_apple_vision(tmp)),
                 ("document reader (VLM) reads a scan", a.vlm, lambda: check_vlm(tmp)),
                 ("docling converts a digital PDF", a.docling, lambda: check_docling(tmp)),
                 ("Playground index: scan + digital PDF", a.e2e, lambda: check_e2e(tmp)),
                 ("rag-search models verify", a.models, check_models_verify)]
        for name, wanted, fn in steps:
            out.append(timed(name, fn) if (wanted or everything) else result(name, "SKIP", "not requested (see --help)"))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    for flag in ("apple-vision", "vlm", "docling", "e2e", "models", "all", "json"):
        ap.add_argument(f"--{flag}", action="store_true")
    a = ap.parse_args(argv)
    rows = run(a)
    if a.json:
        print(json.dumps(rows, indent=1))
    else:
        for r in rows:
            print(f"{r['status']:<5} {r['check']:<40} {r['seconds']:>6.1f}s  {r['detail']}")
        bad = sum(1 for r in rows if r["status"] == "FAIL")
        print(f"\n{bad} failed" if bad else "\nall requested checks passed")
    return 1 if any(r["status"] == "FAIL" for r in rows) else 0


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    sys.exit(main())
