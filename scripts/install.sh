#!/usr/bin/env bash
# rag-search installer (macOS / Linux).  Run from the unpacked release folder.
#
#   ./install.sh                 install the wheel, download models, start the daemons,
#                                register with Claude Desktop / Claude Code
#   ./install.sh --dev [DIR]     unpack the sdist into DIR (default ./rag-search-src) and
#                                install it *editable* so source edits take effect
#   options: --home PATH   data folder (default: ~/Library/Application Support/rag-search)
#            --python VER  Python to use (default 3.12)
#            --skip-models don't download models now (they download on first use)
#            --models P    use a model preset instead of the default: default, qwen3-small or
#                          qwen3-large (see: rag-search models). Downloaded now, from Hugging Face.
#            --no-register don't touch Claude Desktop / Claude Code config
#            --tool-prefix P  add a prefix to the MCP tool names (they already start with rag_)
#            --service     start the daemons at login (macOS launchd)
#            --no-mcp      same as --no-register (the MCP adapter is always installed)
#            --no-tesseract don't check for / install Tesseract (the last-resort page reader, with its
#                          Marathi and Hindi language data). Installed with Homebrew on macOS.
#   The Python packages come with the wheel (pyproject.toml); everything after the install is `rag-search setup`.
#            --import-only for using collections others exported (rag-search collection import):
#                          skips docling's document-conversion models, the document reader and
#                          Tesseract, which only indexing your own documents needs (the embedding
#                          model and the reranker are still downloaded -- search needs them).
#                          Indexing your own text documents later still works (the conversion
#                          models download on first use), but scanned PDFs and photos get no OCR
#                          until you run: rag-search setup
#            --verify-only just check SHA256SUMS and exit
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY_VER="3.12"; DEV=0; DEV_DIR=""; HOME_OPT=""; SKIP_MODELS=0; MODELS_PRESET=""; NO_REGISTER=0; VERIFY_ONLY=0
PREFIX=""; SERVICE=0; NO_MCP=0; IMPORT_ONLY=0; NO_TESSERACT=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dev) DEV=1; if [[ ${2:-} && ${2:0:2} != -- ]]; then DEV_DIR="$2"; shift; fi ;;
    --home) HOME_OPT="$2"; shift ;;
    --python) PY_VER="$2"; shift ;;
    --skip-models) SKIP_MODELS=1 ;;
    --models) MODELS_PRESET="${2:?--models needs a preset name}"; shift ;;
    --no-register) NO_REGISTER=1 ;;
    --tool-prefix) PREFIX="$2"; shift ;;
    --service) SERVICE=1 ;;
    --no-mcp) NO_MCP=1 ;;
    --no-tesseract) NO_TESSERACT=1 ;;
    --import-only) IMPORT_ONLY=1 ;;
    --verify-only) VERIFY_ONLY=1 ;;
    -h|--help) awk 'NR > 1 { if (/^#/) print; else exit }' "$0" | grep -v '^# *[<>][<>][<>] '; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
  shift
done

say() { printf '\n==> %s\n' "$*"; }
die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

# 1. integrity ---------------------------------------------------------------
say "Checking release integrity"
if [[ -f "$HERE/SHA256SUMS" ]]; then
  if command -v shasum >/dev/null; then (cd "$HERE" && shasum -a 256 -c SHA256SUMS)
  else (cd "$HERE" && sha256sum -c SHA256SUMS); fi
else
  echo "warning: no SHA256SUMS next to install.sh; skipping verification"
fi
[[ $VERIFY_ONLY == 1 ]] && exit 0

# 2. uv ----------------------------------------------------------------------
if ! command -v uv >/dev/null; then
  cat >&2 <<'MSG'
ERROR: 'uv' (the Python installer used here) was not found.
Install it with ONE of:
    brew install uv
    curl -LsSf https://astral.sh/uv/install.sh | sh
then open a new terminal and re-run ./install.sh
MSG
  exit 1
fi

# 3. install -----------------------------------------------------------------
# An upgrade replaces the tool environment: stop daemons that run the old code first.
OLD_EXE="$(uv tool dir --bin)/rag-search"
UI_PORT=""
if [[ -x "$OLD_EXE" ]]; then
  say "Stopping running daemons (if any)"
  "$OLD_EXE" ${HOME_OPT:+--home "$HOME_OPT"} daemon stop >/dev/null 2>&1 || true
  # the web dashboard keeps running the old code too: note its port, stop it, start it again below
  UI_URL="$("$OLD_EXE" ${HOME_OPT:+--home "$HOME_OPT"} ui --url 2>/dev/null || true)"
  if [[ -n "$UI_URL" ]]; then
    UI_PORT="$(printf '%s' "$UI_URL" | sed -E 's#.*127\.0\.0\.1:([0-9]+)/.*#\1#')"
    "$OLD_EXE" ${HOME_OPT:+--home "$HOME_OPT"} ui --stop >/dev/null 2>&1 || true
    if "$OLD_EXE" ${HOME_OPT:+--home "$HOME_OPT"} ui --url >/dev/null 2>&1; then
      echo "note: the dashboard on port $UI_PORT was started in a terminal: press Ctrl-C there and start it again with '$OLD_EXE ui'"
      UI_PORT=""
    fi
  fi
fi
# Every Python package (the MCP adapter, Apple Vision OCR, the MLX document reader, HEIC support, the Intel-Mac pins)
# is a dependency of the wheel itself (pyproject.toml, with platform markers): uv resolves them for this machine.
# before 1.1 the tool environment was called rag-search (the PyPI name is rag-search-local): remove the old one
uv tool list 2>/dev/null | grep -q '^rag-search ' && uv tool uninstall rag-search >/dev/null 2>&1 || true
if [[ $DEV == 1 ]]; then
  SDIST="$(ls "$HERE"/rag_search_local-*.tar.gz 2>/dev/null | head -1 || true)"
  [[ -n "$SDIST" ]] || die "no rag_search_local-*.tar.gz next to install.sh"
  DEV_DIR="${DEV_DIR:-$PWD/rag-search-src}"
  say "Unpacking source to $DEV_DIR"
  mkdir -p "$DEV_DIR"
  tar -xzf "$SDIST" -C "$DEV_DIR" --strip-components=1
  say "Installing editable from $DEV_DIR (Python $PY_VER)"
  uv tool install --force --python "$PY_VER" --editable "$DEV_DIR"
else
  WHEEL="$(ls "$HERE"/rag_search_local-*.whl 2>/dev/null | head -1 || true)"
  [[ -n "$WHEEL" ]] || die "no rag_search_local-*.whl next to install.sh"
  say "Installing $(basename "$WHEEL") (Python $PY_VER) - this downloads PyTorch and docling, a few GB"
  uv tool install --force --python "$PY_VER" "$WHEEL"
fi

BIN_DIR="$(uv tool dir --bin)"
if [[ $DEV == 0 ]]; then
  WANT="$(basename "$WHEEL" | sed -E 's/^rag_search_local-([^-]+)-.*/\1/')"
  GOT="$("$BIN_DIR/rag-search" --version 2>/dev/null | awk '{print $2}')"
  say "Installed rag-search $GOT (release $WANT)"
  [[ "$GOT" == "$WANT" ]] || die "installed version ($GOT) is not the release ($WANT): run  uv cache clean rag-search  and try again"
fi
EXE="$BIN_DIR/rag-search"
[[ -x "$EXE" ]] || die "expected $EXE after install"
case ":$PATH:" in *":$BIN_DIR:"*) ;; *) echo "note: $BIN_DIR is not on your PATH; run: uv tool update-shell";; esac

HOME_ARGS=(); [[ -n "$HOME_OPT" ]] && HOME_ARGS=(--home "$HOME_OPT")

# 4. first-time setup: models, the document reader, Tesseract, health check, daemons, Claude registration ------------
# (the same steps as after `uv tool install rag-search-local`; see: rag-search setup --help)
SETUP_ARGS=()
[[ -n "$MODELS_PRESET" ]] && SETUP_ARGS+=(--models "$MODELS_PRESET")
[[ $SKIP_MODELS == 1 ]] && SETUP_ARGS+=(--skip-models)
[[ $IMPORT_ONLY == 1 ]] && SETUP_ARGS+=(--skip-docling)
[[ $NO_TESSERACT == 1 ]] && SETUP_ARGS+=(--no-tesseract)
[[ $SERVICE == 1 ]] && SETUP_ARGS+=(--service)
[[ $NO_REGISTER == 1 || $NO_MCP == 1 ]] && SETUP_ARGS+=(--no-register)
[[ -n "$PREFIX" ]] && SETUP_ARGS+=(--tool-prefix "$PREFIX")
say "Setting up (the default models are about 5 GB, once; change them later with: rag-search models)"
"$EXE" ${HOME_ARGS[@]+"${HOME_ARGS[@]}"} setup ${SETUP_ARGS[@]+"${SETUP_ARGS[@]}"} \
  || echo "setup reported problems (see above); run it again with:  $EXE setup"

# 5. dashboard: bring it back if it was running before the upgrade ---------------
if [[ -n "$UI_PORT" ]]; then
  say "Restarting the web dashboard (port $UI_PORT)"
  "$EXE" ${HOME_ARGS[@]+"${HOME_ARGS[@]}"} ui --detach --no-browser --port "$UI_PORT" || echo "could not restart the dashboard; start it with: $EXE ui"
fi

cat <<MSG

Done.
  * Fully quit and reopen Claude Desktop (Cmd-Q), then ask: "list my document collections".
MSG
if [[ $IMPORT_ONLY == 1 ]]; then
cat <<MSG
  * Import a collection someone exported for you:
        $EXE collection import FILE.rag.tgz
    (it must have been embedded with the same model as this installation: $EXE models)
MSG
fi
cat <<MSG
  * Register each source folder with  $EXE location add NAME FOLDER  , then run
        $EXE index new --follow
    or ask Claude to "index my new documents".
  * Search from the shell:  $EXE search "your question"
  * Daemon status:          $EXE daemon status
  * Web dashboard:          $EXE ui        (status, indexing, models, search, access, architecture, help)
  * Models:                 $EXE models    (list, download and switch the embedding model and reranker)
  * Every collection is open to every client by default. To limit one to some clients:
        $EXE access restrict COLLECTION claude      (see: $EXE access)
  * Self-test:              $EXE doctor --roundtrip
MSG
