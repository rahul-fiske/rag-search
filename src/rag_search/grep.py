"""Regex search over the converted Markdown (exact strings, IDs, numbers).

Safety properties: confined to the given markup root (collection names are plain
directory names, symlinks are not followed) and time-budgeted.  Per-client access is
applied by the caller, which passes the already-resolved list of collections.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


MAX_PATTERN_LEN = 500
MAX_FILE_BYTES = 64 * 1024 * 1024
_PAGE_RE = re.compile(r"<!-- page (\d+) -->")


def grep_markup(markup_root: Path | None, pattern: str, collections: list[str],
                context_lines: int = 2, max_matches: int = 20,
                time_budget_s: float = 15.0) -> dict[str, Any]:
    t0 = time.perf_counter()
    if not pattern or len(pattern) > MAX_PATTERN_LEN:
        return {"error": f"pattern must be 1-{MAX_PATTERN_LEN} characters", "matches": []}
    try:
        rx = re.compile(pattern, re.IGNORECASE)
    except re.error as exc:
        return {"error": f"invalid regex: {exc}", "matches": []}
    root = markup_root
    if root is None or not root.is_dir():
        return {"matches": [], "total_matches": 0, "truncated": False, "timed_out": False,
                "note": "nothing is published yet"}
    context_lines = min(max(int(context_lines), 0), 10)
    max_matches = min(max(int(max_matches), 1), 100)

    # exact directory names only: case-insensitive filesystems must not let "HR"
    # reach "hr" behind the caller's access check
    existing = {e.name for e in os.scandir(root) if e.is_dir(follow_symlinks=False)}
    coll_dirs = []
    for name in collections:
        if name in {".", ".."} or "/" in name or name.startswith("."):
            return {"error": f"invalid collection name: {name!r}", "matches": []}
        if name in existing:
            coll_dirs.append(root / name)

    matches: list[dict[str, Any]] = []
    files = 0
    truncated = False
    timed_out = False
    for cdir in coll_dirs:
        if truncated or timed_out:
            break
        for dirpath, dirnames, filenames in os.walk(cdir, followlinks=False):
            dirnames[:] = sorted(d for d in dirnames if not d.startswith("."))
            for fn in sorted(filenames):
                if not fn.endswith(".md"):
                    continue
                p = Path(dirpath) / fn
                if p.is_symlink():
                    continue
                if time.perf_counter() - t0 > time_budget_s:
                    timed_out = True
                    break
                try:
                    if p.stat().st_size > MAX_FILE_BYTES:
                        continue
                    lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
                except OSError:
                    continue
                files += 1
                page = ""
                for i, line in enumerate(lines):
                    pm = _PAGE_RE.fullmatch(line.strip())
                    if pm:
                        page = pm.group(1)
                        continue
                    if not rx.search(line):
                        continue
                    lo, hi = max(0, i - context_lines), min(len(lines), i + context_lines + 1)
                    matches.append({
                        "collection": cdir.name,
                        "doc": str(p.relative_to(cdir).with_suffix("")),
                        "page": page or "?",
                        "line": i + 1,
                        "text": line.strip()[:500],
                        "context": [x.strip()[:300] for x in lines[lo:hi]
                                    if not _PAGE_RE.fullmatch(x.strip())],
                    })
                    if len(matches) >= max_matches:
                        truncated = True
                        break
                if truncated:
                    break
            if truncated or timed_out:
                break
    return {
        "matches": matches,
        "total_matches": len(matches),
        "truncated": truncated,
        "timed_out": timed_out,
        "timing": {"scan_ms": int((time.perf_counter() - t0) * 1000), "files_scanned": files},
    }


def grep_isolated(markup_root: Path | None, pattern: str, collections: list[str],
                  context_lines: int = 2, max_matches: int = 20,
                  time_budget_s: float = 15.0) -> dict[str, Any]:
    """`grep_markup` in a child process that is killed on timeout.

    Python's `re` holds the GIL, so a pathological pattern (catastrophic backtracking) run
    inside a daemon would freeze every other request; the pattern comes from an agent.
    """
    args = {"root": str(markup_root) if markup_root else None, "pattern": pattern,
            "collections": collections, "context_lines": context_lines,
            "max_matches": max_matches, "time_budget_s": time_budget_s}
    pkg_parent = str(Path(__file__).resolve().parents[1])
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(
        [pkg_parent, *filter(None, [os.environ.get("PYTHONPATH")])]))
    t0 = time.perf_counter()
    try:
        proc = subprocess.run([sys.executable, "-m", "rag_search.grep"], input=json.dumps(args),
                              capture_output=True, text=True, timeout=time_budget_s + 5, env=env)
        res = json.loads(proc.stdout)
    except subprocess.TimeoutExpired:
        res = {"error": "grep timed out: the pattern is too expensive for these documents; "
                        "simplify it (avoid nested quantifiers such as (a+)+)", "matches": []}
    except (ValueError, OSError) as exc:
        res = {"error": f"grep failed: {exc}", "matches": []}
    # total_ms includes starting the isolated child process; scan_ms is the scan itself
    res.setdefault("timing", {})["total_ms"] = int((time.perf_counter() - t0) * 1000)
    return res


def _main() -> int:
    a = json.loads(sys.stdin.read())
    res = grep_markup(Path(a["root"]) if a["root"] else None, a["pattern"], a["collections"],
                      a["context_lines"], a["max_matches"], a["time_budget_s"])
    sys.stdout.write(json.dumps(res))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
