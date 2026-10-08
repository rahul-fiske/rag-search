#!/usr/bin/env bash
# Build the shareable release folder: wheel + sdist + install scripts + SHA256SUMS + zip.
# Requires `uv` (or `python -m build`).
#
#   scripts/build_release.sh               internal build: everything in the tree
#   scripts/build_release.sh --external    build for people outside: optional host modules,
#                                          their tests and internal notes are left out, marked
#                                          installer blocks are removed, and the build fails if a
#                                          denied word (RAG_SEARCH_RELEASE_DENY=word1,word2 or the
#                                          private file .release-deny) is anywhere in the result.
#                                          Output: dist/rag-search-<version>-external.zip
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXTERNAL=0
for arg in "$@"; do
  case "$arg" in
    --external) EXTERNAL=1 ;;
    -h|--help) awk 'NR > 1 { if (/^#/) print; else exit }' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "unknown option: $arg" >&2; exit 2 ;;
  esac
done
cd "$ROOT"
VER="$(python3 -c 'import re,pathlib; print(re.search(r"__version__ = \"(.+?)\"", pathlib.Path("src/rag_search/__init__.py").read_text()).group(1))')"

# ── external: work on a cleaned copy of the tree ─────────────────────────────
SRC="$ROOT"; OUT="$ROOT/dist"; NAME="rag-search-$VER"
if [[ $EXTERNAL == 1 ]]; then
  if [[ -z "${RAG_SEARCH_RELEASE_DENY:-}" && ! -s "$ROOT/.release-deny" ]]; then
    echo "ERROR: --external needs the words to keep out: RAG_SEARCH_RELEASE_DENY=word1,word2 or a .release-deny file" >&2
    exit 2
  fi
  NAME="rag-search-$VER-external"
  SRC="${TMPDIR:-/tmp}/rag-search-external-stage"
  rm -rf "$SRC"; mkdir -p "$SRC"
  ( cd "$ROOT" && tar -cf - \
      --exclude=./dist --exclude=./build --exclude=./.git --exclude='*.egg-info' \
      --exclude=__pycache__ --exclude='*.pyc' --exclude=.DS_Store --exclude=./.ruff_cache --exclude=./.pytest_cache --exclude=./.venv \
      --exclude=./.release-deny --exclude='./INTERNAL*.md' --exclude=./docs/design --exclude=./src/rag_search/ui/static/docs \
      --exclude='./src/rag_search/hosts_*.py' --exclude='./tests/portable/test_hosts_*.py' . ) | tar -xf - -C "$SRC"
  # the installer's marked blocks are internal-only
  python3 - "$SRC" <<'PY'
import re, sys
from pathlib import Path
root = Path(sys.argv[1])
pat = re.compile(r"^[ \t]*(?:#|<!--)[ \t]*>>> internal-host\b.*?^[ \t]*(?:#|<!--)[ \t]*<<< internal-host\b[^\n]*\n",
                 re.S | re.M)
for f in list(root.joinpath("scripts").glob("*")) + [root / "README.md"]:
    if f.is_file() and f.suffix in (".sh", ".md", ""):
        text = f.read_text(encoding="utf-8")
        new = pat.sub("", text)
        if new != text:
            f.write_text(new, encoding="utf-8")
PY
  python3 "$ROOT/scripts/release_guard.py" "$SRC"
  cd "$SRC"; OUT="$SRC/dist"
fi

REL="$OUT/$NAME"
# the dashboard shows README and ARCHITECTURE; ship exactly the current copies
mkdir -p src/rag_search/ui/static/docs
cp README.md ARCHITECTURE.md src/rag_search/ui/static/docs/
rm -rf build src/*.egg-info "$REL" "$OUT/$NAME.zip" "$OUT"/rag_search-"$VER"-py3-none-any.whl "$OUT"/rag_search-"$VER".tar.gz
mkdir -p "$REL"

if command -v uv >/dev/null; then uv build --out-dir "$OUT" ${BUILD_ARGS:-}
else python3 -m build --outdir "$OUT"; fi
# older setuptools name the sdist after the project as written ("rag-search-…"); install.sh and
# SHA256SUMS expect the normalised name
if [[ -f "$OUT/rag-search-$VER.tar.gz" && ! -f "$OUT/rag_search-$VER.tar.gz" ]]; then
  mv "$OUT/rag-search-$VER.tar.gz" "$OUT/rag_search-$VER.tar.gz"
fi

cp "$OUT"/rag_search-"$VER"-py3-none-any.whl "$OUT"/rag_search-"$VER".tar.gz "$REL"/
cp scripts/install.sh scripts/uninstall.sh scripts/run_sample.sh README.md ARCHITECTURE.md "$REL"/
chmod +x "$REL"/*.sh
( cd "$REL"
  if command -v shasum >/dev/null; then shasum -a 256 rag_search-* > SHA256SUMS; else sha256sum rag_search-* > SHA256SUMS; fi )
( cd "$OUT" && zip -qr "$NAME.zip" "$NAME" )

if [[ $EXTERNAL == 1 ]]; then
  # look inside everything that will be handed out: folder, wheel, sdist and the zip itself
  python3 "$ROOT/scripts/release_guard.py" "$REL" "$OUT/$NAME.zip" || { echo "ERROR: external build rejected (see above); nothing was copied to dist/" >&2; exit 1; }
  mkdir -p "$ROOT/dist"
  rm -rf "$ROOT/dist/$NAME" "$ROOT/dist/$NAME.zip"
  cp -R "$REL" "$ROOT/dist/$NAME"; cp "$OUT/$NAME.zip" "$ROOT/dist/$NAME.zip"
  echo "Clean copy of the tree (for running the tests): $SRC"
  OUT="$ROOT/dist"; REL="$OUT/$NAME"
fi
echo "Release folder: $REL"; echo "Zip:            $OUT/$NAME.zip"; cat "$REL/SHA256SUMS"
