"""The test corpus (``tests/data/corpus/``) and its manifest (``tests/data/corpus.json``), stdlib only.

One small synthetic file per kind of input rag-search handles; ``corpus.json`` says what each must produce.
See ``tests/data/README.md``.  Tests copy what they need into their temporary data folder: the corpus itself
is never written to.
"""

from __future__ import annotations

import json
import shutil
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

DATA = Path(__file__).resolve().parent / "data"
CORPUS = DATA / "corpus"


@lru_cache(maxsize=1)
def manifest() -> tuple[dict[str, Any], ...]:
    return tuple(json.loads((DATA / "corpus.json").read_text(encoding="utf-8"))["files"])


def entries(kind: str | None = None, *, has: str | None = None) -> list[dict[str, Any]]:
    """Manifest entries, optionally of one *kind* and/or with a key *has* (``route``, ``real``, ``skip``)."""
    return [e for e in manifest() if (kind is None or e["kind"] == kind) and (has is None or has in e)]


def entry(rel: str) -> dict[str, Any]:
    return next(e for e in manifest() if e["path"] == rel)


def path(rel: str) -> Path:
    """The corpus file *rel* (read it, never write it)."""
    p = CORPUS / rel
    if not p.is_file():
        raise FileNotFoundError(f"{rel} is not in the test corpus (see tests/data/README.md)")
    return p


def copy(rel: str, dest: Path) -> Path:
    """Copy the corpus file *rel* to *dest* (a file path) and return it."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(path(rel), dest)
    return dest


def copy_tree(dest: Path, rels: Iterable[str] | None = None) -> Path:
    """Copy the whole corpus (or the files *rels*, keeping their folders) under *dest*: a docs folder
    whose first-level folders (pdf, office, text, images, unsupported, collision) are collections."""
    if rels is None:
        shutil.copytree(CORPUS, dest, dirs_exist_ok=True)
    else:
        for rel in rels:
            copy(rel, dest / rel)
    return dest


def doc_name(rel: str) -> tuple[str, str]:
    """(collection, document) of a corpus file as rag-search names it: ``pdf/text.pdf`` -> (pdf, text)."""
    parts = Path(rel).with_suffix("").parts
    return parts[0], "/".join(parts[1:])


def expand_branches(spec: Any) -> list[str]:
    """``["digital", ...]`` as it is, or the short form ``"digital*23"``."""
    if isinstance(spec, str):
        name, _, n = spec.partition("*")
        return [name] * int(n or 1)
    return list(spec or [])
