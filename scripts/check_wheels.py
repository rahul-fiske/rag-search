#!/usr/bin/env python3
"""Does every package of rag-search have a wheel on each platform we support?  (standard library + uv)

    check_wheels.py [--platform P ...] [--python-version 3.12] [--project DIR]

A package that has no wheel for a platform is built from source there: that needs a compiler (and minutes) on the
user's machine, and fails on a machine without one.  This resolves the project's dependencies for each platform
(``uv pip compile --python-platform``) and asks PyPI whether every pinned version has a wheel that fits.  Pure-Python
packages are fine as sdists (``ALLOW``); anything else is reported, and the exit status is 1.  The platforms are the
ones in ``PLATFORMS``; run weekly by .github/workflows/wheels.yml because upstream releases change the answer (Intel
Mac wheels disappeared from docling-parse and cryptography in 2026).
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import urllib.request
from collections.abc import Callable
from pathlib import Path

# uv's name -> (markers that a wheel's platform tag must satisfy)
PLATFORMS = {
    "aarch64-apple-darwin": ("macosx", ("arm64", "universal2")),
    "x86_64-apple-darwin": ("macosx", ("x86_64", "universal2", "intel", "fat")),
    "x86_64-unknown-linux-gnu": ("manylinux", ("x86_64",)),
}
# pure-Python packages that only publish an sdist: building one needs no compiler
ALLOW = {"antlr4-python3-runtime"}


def wheel_fits(filename: str, platform: str, python: str) -> bool:
    """Does the wheel *filename* install on *platform* (a key of PLATFORMS) with Python *python* ("3.12")?"""
    if not filename.endswith(".whl"):
        return False
    py, abi, plat = filename[:-4].split("-")[-3:]
    cp = "cp" + python.replace(".", "")
    minor = int(python.split(".")[1])
    pure = plat == "any"
    if not (pure or py in ("py3", "py2.py3", cp) or (abi == "abi3" and py.startswith("cp3") and int(py[3:] or 0) <= minor)):
        return False
    if pure:
        return True
    family, archs = PLATFORMS[platform]
    return family in plat and any(a in plat for a in archs)


def missing(pins: list[tuple[str, str]], platform: str, python: str,
            fetch: Callable[[str, str], list[str]]) -> list[tuple[str, str]]:
    """The pins that have no wheel for *platform*; *fetch(name, version)* gives the file names PyPI has."""
    out = []
    for name, version in pins:
        if name.lower().replace("_", "-") in ALLOW:
            continue
        if not any(wheel_fits(f, platform, python) for f in fetch(name, version)):
            out.append((name, version))
    return out


def pypi_files(name: str, version: str) -> list[str]:
    url = f"https://pypi.org/pypi/{name}/{version}/json"
    with urllib.request.urlopen(url, timeout=30) as resp:                      # noqa: S310
        return [f["filename"] for f in json.load(resp)["urls"]]


def resolve(project: Path, platform: str, python: str) -> list[tuple[str, str]]:
    cp = subprocess.run(["uv", "pip", "compile", str(project / "pyproject.toml"), "--python-platform", platform,
                         "--python-version", python, "--no-header", "--quiet"], capture_output=True, text=True, check=False)
    if cp.returncode:
        raise SystemExit(f"uv could not resolve for {platform}:\n{cp.stderr}")
    return [(m.group(1), m.group(2)) for m in re.finditer(r"^([A-Za-z0-9_.\-]+)==([^\s;]+)", cp.stdout, re.M)]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--platform", action="append", choices=sorted(PLATFORMS))
    ap.add_argument("--python-version", default="3.12")
    ap.add_argument("--project", default=".", type=Path)
    a = ap.parse_args(argv)
    bad = 0
    for platform in a.platform or sorted(PLATFORMS):
        pins = resolve(a.project, platform, a.python_version)
        lacking = missing(pins, platform, a.python_version, pypi_files)
        print(f"{platform}: {len(pins)} packages, {len(lacking)} without a wheel")
        for name, version in lacking:
            print(f"  {name}=={version}", file=sys.stderr)
        bad += bool(lacking)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
