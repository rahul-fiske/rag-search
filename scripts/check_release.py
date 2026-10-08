#!/usr/bin/env python3
"""Checks that a release is what it claims to be (standard library only; run by the release workflow).

    check_release.py tag vX.Y.Z [--root DIR]     the tag is a release version and equals rag_search.__version__
    check_release.py dist DIR [--root DIR]       DIR holds exactly the wheel and the source archive of that version,
                                                 with the license, and nothing that must stay on the author's machine

The version is written once, in ``src/rag_search/__init__.py``; the git tag only labels the commit that has it.
Exit status 0 when every check passes, 1 with one line per problem.
"""

from __future__ import annotations

import argparse
import re
import sys
import tarfile
import zipfile
from pathlib import Path

TAG = re.compile(r"^v(\d+)\.(\d+)\.(\d+)((?:a|b|rc)\d+)?$")
PRIVATE = re.compile(r"(^|/)(hosts_[^/]*\.py|INTERNAL[^/]*\.md|\.release-deny)$")


def read_version(root: Path) -> str:
    text = (root / "src" / "rag_search" / "__init__.py").read_text(encoding="utf-8")
    m = re.search(r'^__version__\s*=\s*"([^"]+)"', text, re.M)
    if not m:
        raise SystemExit("no __version__ in src/rag_search/__init__.py")
    return m.group(1)


def check_tag(tag: str, version: str) -> list[str]:
    errors = []
    if not TAG.match(tag):
        errors.append(f"the tag {tag!r} is not a release: it must look like v1.2.3 (or v1.2.3rc1)")
    elif tag[1:] != version:
        errors.append(f"the tag is {tag} but the package version is {version}: change __version__ and tag that commit")
    if re.search(r"dev|\+", version):
        errors.append(f"the package version {version} is a development version")
    return errors


def check_dist(dist: Path, version: str) -> list[str]:
    errors = []
    wheels = sorted(dist.glob("*.whl"))
    sdists = sorted(dist.glob("*.tar.gz"))
    want_wheel, want_sdist = f"rag_search_local-{version}-py3-none-any.whl", f"rag_search_local-{version}.tar.gz"
    if [w.name for w in wheels] != [want_wheel]:
        errors.append(f"expected exactly {want_wheel} in {dist}, found {[w.name for w in wheels]}")
    if [s.name for s in sdists] != [want_sdist]:
        errors.append(f"expected exactly {want_sdist} in {dist}, found {[s.name for s in sdists]}")
    if errors:
        return errors
    with zipfile.ZipFile(wheels[0]) as z:
        names = z.namelist()
        meta = next((n for n in names if n.endswith(".dist-info/METADATA")), "")
        text = z.read(meta).decode("utf-8") if meta else ""
        if f"\nVersion: {version}\n" not in "\n" + text:
            errors.append(f"the wheel's metadata does not say Version: {version}")
        if not any(n.endswith("/licenses/LICENSE") or n.endswith(".dist-info/LICENSE") for n in names):
            errors.append("the wheel has no LICENSE")
        for dep in ("mcp", "docling", "torch"):
            if not re.search(rf"^Requires-Dist: {dep}\b", text, re.M):
                errors.append(f"the wheel does not require {dep}")
        errors += [f"the wheel contains {n}, which must never be published" for n in names if PRIVATE.search(n)]
    with tarfile.open(sdists[0]) as t:
        members = t.getnames()
        if not any(m.endswith("/LICENSE") for m in members):
            errors.append("the source archive has no LICENSE")
        errors += [f"the source archive contains {m}, which must never be published" for m in members if PRIVATE.search(m)]
    return errors


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="what", required=True)
    for name in ("tag", "dist"):
        p = sub.add_parser(name)
        p.add_argument("value")
        p.add_argument("--root", default=".", type=Path)
    a = ap.parse_args(argv)
    version = read_version(a.root)
    errors = check_tag(a.value, version) if a.what == "tag" else check_dist(Path(a.value), version)
    for e in errors:
        print(f"check_release: {e}", file=sys.stderr)
    if not errors:
        print(f"check_release: {a.what} ok (version {version})")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
