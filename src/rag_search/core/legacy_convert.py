"""Batch-convert legacy binary Office documents to modern formats via LibreOffice.

``.doc``/``.xls``/``.ppt``/``.rtf`` are not read by docling directly (or, for
``.doc``/``.xls``/``.ppt``, only by versions of docling that themselves shell out to
LibreOffice) -- converting them once, up front, with the well-tested ``soffice
--convert-to`` CLI means rag-search never has to depend on that machinery at all: it
just sees an ordinary ``.docx``/``.xlsx``/``.pptx`` sitting in the docs folder.

One file per ``soffice`` invocation, on purpose: a batched multi-file invocation is
faster, but a hang or crash on file N of a batch only shows up after the whole batch's
timeout, and you have to infer which file caused it. Per-file invocations map a timeout
and an error straight to one document, and (each with its own throwaway user-profile
directory) can never collide with each other the way concurrent LibreOffice processes
sharing the default profile can (the same bug docling's own LibreOffice integration
carried until it added a timeout and an isolated profile in v2.116.0).

Originals are only removed after the converted file is verified to be a real,
well-formed OOXML package -- not just "a file appeared at that path".
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator

# extension -> the format name `soffice --convert-to` should produce
LEGACY_FORMATS: dict[str, str] = {
    ".doc": "docx",
    ".xls": "xlsx",
    ".ppt": "pptx",
    ".rtf": "docx",
}

# every OOXML package `soffice` can hand back here must carry this member
_OOXML_MARKER = "[Content_Types].xml"

_LOCK_PREFIXES = ("~$", ".~lock")

_MAC_SOFFICE = Path("/Applications/LibreOffice.app/Contents/MacOS/soffice")


def find_soffice() -> str | None:
    """The ``soffice`` binary to run, or ``None`` if LibreOffice isn't findable."""
    for name in ("soffice", "libreoffice"):
        found = shutil.which(name)
        if found:
            return found
    if _MAC_SOFFICE.exists():
        return str(_MAC_SOFFICE)
    return None


def is_lock_file(p: Path) -> bool:
    """Office/LibreOffice leave these beside a file that's open elsewhere -- never touch them."""
    return p.name.startswith(_LOCK_PREFIXES)


def is_valid_ooxml(p: Path) -> bool:
    """*p* is a real, readable ``.docx``/``.xlsx``/``.pptx`` -- not just a path that exists."""
    try:
        if not zipfile.is_zipfile(p):
            return False
        with zipfile.ZipFile(p) as zf:
            return _OOXML_MARKER in zf.namelist()
    except (OSError, zipfile.BadZipFile):
        return False


def iter_legacy_files(root: Path, exts: Iterable[str]) -> Iterator[Path]:
    """Every file under *root* (recursively) whose extension is in *exts*, sorted, minus locks."""
    ext_set = {e.lower() for e in exts}
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix.lower() in ext_set and not is_lock_file(p):
            yield p


@dataclass
class ConvertResult:
    src: Path
    dest: Path
    ok: bool = False
    already_done: bool = False
    deleted: bool = False
    error: str = ""


@dataclass
class ConvertSummary:
    root: Path
    converted: list[ConvertResult] = field(default_factory=list)
    skipped: list[ConvertResult] = field(default_factory=list)   # already_done
    failed: list[ConvertResult] = field(default_factory=list)
    dry_run: bool = False


def _convert_one(soffice: str, src: Path, out_ext: str, timeout: float) -> tuple[bool, str]:
    """Run one file through LibreOffice in its own throwaway profile.  -> (ok, error)."""
    with tempfile.TemporaryDirectory(prefix="rag-search-lo-") as profile:
        cmd = [
            soffice, "--headless", "--norestore", "--nolockcheck", "--nodefault",
            f"-env:UserInstallation=file://{profile}",
            "--convert-to", out_ext, "--outdir", str(src.parent), str(src),
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return False, f"timed out after {timeout:.0f}s (LibreOffice may be stuck)"
        except OSError as exc:
            return False, f"could not run LibreOffice: {exc}"
        if result.returncode != 0:
            tail = (result.stderr or result.stdout or "").strip().splitlines()
            return False, tail[-1] if tail else f"soffice exited {result.returncode}"
        return True, ""


def convert_tree(
    root: Path,
    *,
    exts: Iterable[str] | None = None,
    soffice: str | None = None,
    dry_run: bool = False,
    keep_originals: bool = False,
    force: bool = False,
    timeout: float = 120.0,
    log_path: Path | None = None,
) -> ConvertSummary:
    """Convert every legacy file under *root* whose extension is in *exts* (default: every
    extension in `LEGACY_FORMATS`).

    Non-destructive by construction unless the conversion is verified: the original is only
    deleted (and only if ``keep_originals`` is false) once the new file is confirmed to be a
    valid OOXML package.  ``dry_run`` performs no conversion and no deletion at all.  Each
    processed file appends one JSON line to *log_path*, if given.
    """
    exts = set(exts) if exts else set(LEGACY_FORMATS)
    unknown = exts - set(LEGACY_FORMATS)
    if unknown:
        raise ValueError(f"unsupported extension(s): {', '.join(sorted(unknown))}")

    summary = ConvertSummary(root=root, dry_run=dry_run)
    log_fh = None
    if log_path and not dry_run:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_fh = open(log_path, "a", encoding="utf-8")

    try:
        for src in iter_legacy_files(root, exts):
            out_ext = LEGACY_FORMATS[src.suffix.lower()]
            dest = src.with_suffix("." + out_ext)
            res = ConvertResult(src=src, dest=dest)

            if dest.exists() and not force:
                res.already_done = True
                summary.skipped.append(res)
                continue

            if dry_run:
                summary.converted.append(res)  # "would convert"
                continue

            ok, error = _convert_one(soffice, src, out_ext, timeout)  # type: ignore[arg-type]
            if ok and not (dest.exists() and is_valid_ooxml(dest)):
                ok, error = False, "conversion reported success but no valid file was written"
            res.ok = ok
            res.error = error

            if ok and not keep_originals:
                try:
                    src.unlink()
                    res.deleted = True
                except OSError as exc:
                    res.error = f"converted, but could not delete the original: {exc}"

            (summary.converted if ok else summary.failed).append(res)

            if log_fh:
                log_fh.write(json.dumps({
                    "ts": round(time.time(), 3), "src": str(src),
                    "dest": str(dest) if ok else None, "ok": ok,
                    "deleted": res.deleted, "error": error or None,
                }, ensure_ascii=False) + "\n")
                log_fh.flush()
    finally:
        if log_fh:
            log_fh.close()

    return summary
