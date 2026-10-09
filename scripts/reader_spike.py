#!/usr/bin/env python3
"""Does the CPU document reader work on this machine, and how fast?  (rag-search's own environment)

    python scripts/reader_spike.py [--model ibm-granite/granite-docling-258M] [--max-tokens 600]

Downloads the model if it is not on disk, draws a small page (a heading, a sentence with numbers, a table), reads it with
the ``transformers`` reader backend and prints the load time, the read time, the tokens per second and the Markdown.
Exit status 1 when the numbers of the page are not in the answer.  Run by hand or by the "reader spike" workflow.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import time
from pathlib import Path

NEEDLES = ("4,310,200", "12.5", "1,200.50", "3,109.70")


def draw(path: Path) -> None:
    from PIL import Image, ImageDraw, ImageFont

    img = Image.new("RGB", (1000, 700), "white")
    d = ImageDraw.Draw(img)
    f = ImageFont.load_default(size=28)
    d.text((60, 50), "Quarterly Report 2025", font=f, fill="black")
    d.text((60, 120), "Revenue grew 12.5% to $4,310,200 in the third quarter.", font=f, fill="black")
    for i, (a, b) in enumerate([("Region", "Sales"), ("North", "1,200.50"), ("South", "3,109.70")]):
        y = 220 + i * 50
        d.rectangle((60, y, 500, y + 50), outline="black")
        d.line((280, y, 280, y + 50), fill="black")
        d.text((75, y + 8), a, font=f, fill="black")
        d.text((295, y + 8), b, font=f, fill="black")
    img.save(path)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="ibm-granite/granite-docling-258M")
    ap.add_argument("--max-tokens", type=int, default=600)
    a = ap.parse_args()
    from huggingface_hub import snapshot_download

    snapshot_download(a.model)
    from rag_search.core.conversion.vlm_worker import TransformersBackend

    with tempfile.TemporaryDirectory() as tmp:
        page = Path(tmp) / "page.png"
        draw(page)
        t = time.perf_counter()
        backend = TransformersBackend(a.model)
        load_s = time.perf_counter() - t
        t = time.perf_counter()
        res = backend.read(str(page), "", a.max_tokens)
        read_s = time.perf_counter() - t
    missing = [n for n in NEEDLES if n not in res["md"]]
    print(f"device {backend.device}; load {load_s:.1f}s; read {read_s:.1f}s; {res['tokens']} tokens "
          f"({res['tokens'] / max(read_s, 1e-9):.2f}/s); stopped {res['stopped']!r}")
    print(res["md"])
    print("numbers missing:", missing or "none")
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
