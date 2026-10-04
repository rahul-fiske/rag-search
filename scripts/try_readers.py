#!/usr/bin/env python3
"""Try every page reader on one document, outside the indexing pipeline, and compare what they find.

Run it with rag-search's own Python, so it sees the same packages the dashboard does:

    "$(uv tool dir)/rag-search/bin/python" scripts/try_readers.py ~/Documents/passbook.pdf
    ... try_readers.py FILE --pages 1-2 --readers apple-vision,vlm      # a subset
    ... try_readers.py FILE --out /tmp/readers                          # where the results go
    ... try_readers.py FILE --readers vlm --pages 1 --show              # print the full text of the page
    ... try_readers.py FILE --readers vlm --model mlx-community/Qwen3-VL-8B-Instruct-4bit   # another reader model

For every page and reader it prints the time, the number of real characters and the first line, and writes
the text to ``<out>/<reader>-p<N>.md`` (default folder: ``~/.cache/rag-search-try-readers/<name>/``, never
next to the file: a folder of documents that rag-search indexes would pick the results up).  The source
file is only read.  Nothing is downloaded; a reader that cannot run says why and the others continue.

Readers:  apple-vision  Apple's text recognition through ocrmac (plain text, fast)
          vlm           the document reader: the vision model chosen on the Models tab (needs mlx-vlm and the weights)
          docling       docling's own full-page OCR, as the pipeline's fallback uses it (slow: loads layout models)
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

READERS = ("apple-vision", "vlm", "docling")


def parse_pages(spec: str, total: int) -> list[int]:
    if not spec:
        return list(range(1, total + 1))
    out: list[int] = []
    for part in spec.split(","):
        a, _, b = part.partition("-")
        out.extend(range(int(a), int(b or a) + 1))
    return [n for n in out if 1 <= n <= total]


def page_count(src: Path) -> int:
    from rag_search.core.conversion import profiler

    prof = profiler.profile_file(src)
    if prof.get("error"):
        raise SystemExit(f"cannot open {src.name}: {prof['error']}")
    return len(prof.get("pages") or []) or 1


def judge(text: str) -> str:
    """What the pipeline's own checks say about this text: tables found, the gate's verdict and how many
    running-balance / total checks failed.  No text is printed."""
    from rag_search.core.conversion import gate, tables

    found = tables.find_tables(text)
    g = gate.check_page(text, branch_kind="scan", profile={"ink": 0.1})
    bad = len(g.get("violations") or [])
    fence = ", wrapped in a code fence" if text.lstrip().startswith("```") else ""
    return f"{len(found)} table(s), gate {g['verdict']}" + (f", {bad} arithmetic violation(s)" if bad else "") + fence


def run_apple_vision(src: Path, pages: list[int], is_image: bool):
    from rag_search.core.conversion import applevision

    why = applevision.why_not()
    if why:
        yield None, why
        return
    for n in pages:
        t0 = time.perf_counter()
        try:
            yield (n, applevision.read_page(src, n, is_image=is_image), time.perf_counter() - t0), ""
        except Exception as exc:  # noqa: BLE001
            yield (n, "", time.perf_counter() - t0), f"{type(exc).__name__}: {exc}"


def run_vlm(src: Path, pages: list[int], is_image: bool):
    from rag_search.core.conversion import vlm

    reader = vlm.shared()
    if reader is None:
        yield None, "switched off (RAG_SEARCH_VLM=off)"
        return
    reader.check()
    if not reader.usable():
        yield None, reader.dead
        return
    print(f"  model: {reader.model}  (loading it takes a while the first time)")
    try:
        for n in pages:
            t0 = time.perf_counter()
            try:
                res = reader.read(src, n, n, "scan")
            except vlm.ReaderError as exc:
                yield (n, "", time.perf_counter() - t0), f"{exc.reason}: {exc}"
                continue
            yield (n, (res.get("pages") or {}).get(n, ""), time.perf_counter() - t0), \
                "" if n in (res.get("pages") or {}) else str((res.get("failed") or {}).get(n, "no result"))
    finally:
        vlm.close_shared()


def run_docling(src: Path, pages: list[int], is_image: bool):
    from rag_search.core.conversion.routed import DoclingReader

    try:
        import docling  # noqa: F401
    except ImportError as exc:
        yield None, f"docling is not importable here ({exc})"
        return
    reader = DoclingReader()
    for n in pages:
        t0 = time.perf_counter()
        try:
            res = reader.read(src, n, n, "scan")
        except Exception as exc:  # noqa: BLE001
            yield (n, "", time.perf_counter() - t0), f"{type(exc).__name__}: {str(exc)[:200]}"
            continue
        note = ((res.get("stats") or {}).get(n) or {}).get("note", "")
        yield (n, (res.get("pages") or {}).get(n, ""), time.perf_counter() - t0), note


RUNNERS = {"apple-vision": run_apple_vision, "vlm": run_vlm, "docling": run_docling}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("file", type=Path)
    ap.add_argument("--pages", default="", help="e.g. 1-3 or 2,4 (default: all)")
    ap.add_argument("--readers", default=",".join(READERS), help="comma-separated subset of: " + ", ".join(READERS))
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--model", default="", help="document reader model for this run (a Hugging Face id that is "
                    "already downloaded), instead of the one chosen on the Models tab")
    ap.add_argument("--show", action="store_true", help="print each page's full text (to judge, e.g., Devanagari)")
    a = ap.parse_args(argv)
    if a.model:
        os.environ["RAG_SEARCH_VLM_MODEL"] = a.model
    src = a.file.expanduser().resolve()
    if not src.is_file():
        raise SystemExit(f"not a file: {src}")
    from rag_search.core.conversion import profiler
    from rag_search.core.docling_convert import has_real_text, real_chars

    is_image = profiler.kind_of(src) == "image"
    pages = parse_pages(a.pages, page_count(src))
    out_dir = (a.out or Path.home() / ".cache" / "rag-search-try-readers" / src.stem).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    chosen = [r.strip() for r in a.readers.split(",") if r.strip()]
    bad = [r for r in chosen if r not in RUNNERS]
    if bad:
        raise SystemExit(f"unknown reader(s): {', '.join(bad)} (choose from {', '.join(READERS)})")
    print(f"{src.name}: {len(pages)} page(s); results go to {out_dir}\n")
    summary: dict[str, list[str]] = {}
    for name in chosen:
        print(f"== {name}")
        cells: list[str] = []
        got_any = False
        for item, problem in RUNNERS[name](src, pages, is_image):
            if item is None:
                print(f"  not available: {problem}")
                cells = [f"not available ({problem[:60]})"]
                break
            n, text, secs = item
            got_any = True
            chars = real_chars(text)
            first = next((ln.strip() for ln in text.splitlines() if has_real_text(ln)), "")
            quality = judge(text)
            print(f"  page {n}: {secs:5.1f} s, {chars} characters, {quality}" + (f"  [{problem}]" if problem else "")
                  + (f"\n          {first[:110]}" if first else "   (no text)"))
            if a.show:
                print("\n".join("          | " + ln for ln in text.splitlines()) + "\n")
            (out_dir / f"{name}-p{n}.md").write_text(text, encoding="utf-8")
            cells.append(f"p{n}: {chars}")
        summary[name] = cells if got_any or cells else ["no pages"]
        print()
    print("== summary (real characters per page)")
    for name, cells in summary.items():
        print(f"  {name:13s} " + "  ".join(cells))
    return 0


if __name__ == "__main__":
    sys.exit(main())
