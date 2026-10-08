#!/usr/bin/env bash
# Remove rag-search: unregister from Claude and uninstall the tool.
# Your documents are always kept; indexes and settings are deleted only with --purge-data.
set -euo pipefail
PURGE=0; HOME_OPT=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --purge-data) PURGE=1 ;;
    --home) HOME_OPT="$2"; shift ;;
    *) echo "usage: $0 [--purge-data] [--home PATH]" >&2; exit 2 ;;
  esac
  shift
done
command -v uv >/dev/null || { echo "uv not found; nothing to uninstall via uv"; exit 1; }
EXE="$(uv tool dir --bin)/rag-search"
if [[ -x "$EXE" ]]; then
  [[ -n "$HOME_OPT" ]] && export RAG_SEARCH_HOME="$HOME_OPT"
  "$EXE" unregister || true
  "$EXE" ui --stop || true           # the web dashboard, if it runs in the background
  "$EXE" service uninstall || true   # also stops the daemons
  "$EXE" daemon stop || true
  DATA="$("$EXE" paths home)"
  uv tool uninstall rag-search-local
  uv tool uninstall rag-search >/dev/null 2>&1 || true    # the name before 1.1
else
  DATA="${HOME_OPT:-${RAG_SEARCH_HOME:-$HOME/Library/Application Support/rag-search}}"
fi
if [[ $PURGE == 1 ]]; then
  echo "About to delete the indexes and settings under: $DATA"
  echo "  (indexer_workspace, serving, run, jobs, config.json)."
  echo "Your documents in $DATA/docs are NOT touched."
  read -r -p "Type 'delete' to confirm: " ans
  if [[ "$ans" == "delete" ]]; then
    for d in indexer_workspace serving run jobs config.json; do rm -rf "${DATA:?}/$d"; done
    echo "deleted."
  else
    echo "kept."
  fi
else
  echo "Data kept at: $DATA   (re-run with --purge-data to delete indexes and settings)"
fi
echo "Restart Claude Desktop to drop the server from its tool list."
