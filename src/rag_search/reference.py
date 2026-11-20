"""Reference data: lookup tables that indexing consults to describe what it reads (stdlib only).

A reference dataset is a few files that come from outside rag-search and are needed only while documents are
converted: GeoNames for the name of the place a photograph was taken, and later, say, a face gallery or an object
vocabulary for what a photograph shows.  The answer is written into the document's Markdown, so search never reads
the dataset.

Datasets live in ``<home>/reference/<name>/`` with a ``manifest.json`` (where each file came from, when, its size and
checksum).  Nothing is fetched while indexing: a dataset that is not installed is simply not used, and
``rag-search reference download NAME`` is the one place that goes to the network.  The ``ref`` string of an installed
dataset (``places@2026-11-20``) goes into what was derived from it, so it is visible which data a record used.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import shutil
import tempfile
import time
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from .paths import Paths

MANIFEST = "manifest.json"
CHUNK = 1 << 20


@dataclass(frozen=True)
class Dataset:
    name: str
    title: str
    used_for: str
    license: str
    files: tuple[tuple[str, str], ...]          # (file name, URL), saved as that name; a ".zip" is unpacked
    size_hint: str = ""


DATASETS: dict[str, Dataset] = {d.name: d for d in (
    Dataset(
        "places", "GeoNames: towns and villages with 1,000 or more people",
        "the place a photograph was taken near, from its GPS position",
        "CC BY 4.0, https://www.geonames.org",
        (("cities1000.zip", "https://download.geonames.org/export/dump/cities1000.zip"),
         ("admin1CodesASCII.txt", "https://download.geonames.org/export/dump/admin1CodesASCII.txt"),
         ("countryInfo.txt", "https://download.geonames.org/export/dump/countryInfo.txt")),
        "about 10 MB"),
)}


class ReferenceDataError(RuntimeError):
    """A dataset cannot be used or fetched; the message says what to do."""


def root(paths: Paths) -> Path:
    return paths.home / "reference"


def folder(paths: Paths, name: str) -> Path:
    return root(paths) / name


def manifest(paths: Paths, name: str) -> dict:
    try:
        return json.loads((folder(paths, name) / MANIFEST).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def installed(paths: Paths, name: str) -> bool:
    spec = DATASETS.get(name)
    man = manifest(paths, name)
    if not spec or not man:
        return False
    return all((folder(paths, name) / _stored_name(fn)).is_file() for fn, _ in spec.files)


def ref(paths: Paths, name: str) -> str:
    """``name@date`` of the installed dataset, or "" when it is not installed."""
    return f"{name}@{manifest(paths, name).get('fetched', '')}" if installed(paths, name) else ""


def path(paths: Paths, name: str, filename: str) -> Path | None:
    """A file of an installed dataset (an archive's member under its own name), or None."""
    if not installed(paths, name):
        return None
    p = folder(paths, name) / filename
    return p if p.is_file() else None


def _stored_name(fn: str) -> str:
    return fn[:-4] + ".txt" if fn.endswith(".zip") else fn


def status(paths: Paths) -> list[dict]:
    out = []
    for d in DATASETS.values():
        man = manifest(paths, d.name)
        out.append({"name": d.name, "title": d.title, "used_for": d.used_for, "license": d.license,
                    "size": d.size_hint, "installed": installed(paths, d.name), "fetched": man.get("fetched", ""),
                    "bytes": sum(f.get("bytes", 0) for f in (man.get("files") or {}).values()),
                    "folder": str(folder(paths, d.name))})
    return out


def _fetch(url: str, dest: Path) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": "rag-search"})
    with urllib.request.urlopen(req, timeout=120) as resp, open(dest, "wb") as out:          # noqa: S310 - fixed https URLs
        shutil.copyfileobj(resp, out, CHUNK)


def download(paths: Paths, name: str, *, fetch: Callable[[str, Path], None] = _fetch) -> dict:
    """Fetch every file of the dataset into a new folder and put it in place when all of them arrived (an interrupted
    download leaves the installed copy as it was).  Returns the manifest."""
    spec = DATASETS.get(name)
    if spec is None:
        raise ReferenceDataError(f"unknown reference dataset {name!r}; choose from {', '.join(DATASETS)}")
    root(paths).mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix=f".{name}-", dir=root(paths)))
    try:
        files: dict[str, dict] = {}
        for fn, url in spec.files:
            raw = work / fn
            try:
                fetch(url, raw)
            except Exception as exc:  # noqa: BLE001 - say which file, keep the old copy
                raise ReferenceDataError(f"could not download {url}: {exc}") from exc
            if fn.endswith(".zip"):
                import zipfile

                member = _stored_name(fn)
                with zipfile.ZipFile(raw) as z, z.open(member) as src, open(work / _stored_name(fn), "wb") as dst:
                    shutil.copyfileobj(src, dst, CHUNK)
                raw.unlink()
                fn = _stored_name(fn)
            digest = hashlib.sha256()
            with open(work / fn, "rb") as fh:
                for block in iter(lambda: fh.read(CHUNK), b""):
                    digest.update(block)
            files[fn] = {"url": url, "bytes": (work / fn).stat().st_size, "sha256": digest.hexdigest()}
        man = {"name": name, "fetched": time.strftime("%Y-%m-%d"), "license": spec.license, "files": files}
        (work / MANIFEST).write_text(json.dumps(man, indent=1), encoding="utf-8")
        target = folder(paths, name)
        old = None
        if target.exists():
            old = work.with_name(work.name + ".old")
            os.replace(target, old)
        os.replace(work, target)
        if old is not None:
            shutil.rmtree(old, ignore_errors=True)
        return man
    finally:
        with contextlib.suppress(OSError):
            shutil.rmtree(work, ignore_errors=True)
