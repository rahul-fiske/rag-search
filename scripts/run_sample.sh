#!/usr/bin/env bash
# End-to-end run on one document: convert -> index (indexer daemon) -> list -> search
# (search daemon) -> grep, with timings.
# Writes everything to <data>/sample-report/ (report.md plus one JSON per query).
#
#   ./run_sample.sh /path/to/document.pdf [--home DIR] [--queries FILE] [--grep PATTERN]...
#
# The queries file has one query per line (lines starting with # are ignored).
set -euo pipefail

PDF="${1:-}"; [[ -n "$PDF" && -f "$PDF" ]] || { echo "usage: $0 FILE [--home DIR] [--queries FILE] [--grep RE]..." >&2; exit 2; }
shift
HOME_DIR=""; QUERIES=""; GREPS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --home) HOME_DIR="$2"; shift ;;
    --queries) QUERIES="$2"; shift ;;
    --grep) GREPS+=("$2"); shift ;;
    *) echo "unknown option $1" >&2; exit 2 ;;
  esac; shift
done

if command -v rag-search >/dev/null; then EXE=rag-search
elif command -v uv >/dev/null && [[ -x "$(uv tool dir --bin)/rag-search" ]]; then EXE="$(uv tool dir --bin)/rag-search"
else echo "rag-search is not installed (run install.sh first)" >&2; exit 1; fi
[[ -n "$HOME_DIR" ]] && export RAG_SEARCH_HOME="$HOME_DIR"

DATA="$("$EXE" paths home)"
OUT="$DATA/sample-report"; mkdir -p "$OUT" "$DATA/docs/samples"
REPORT="$OUT/report.md"
DEFAULT_QUERIES=$'how is access to the cluster controlled\nwhat authentication methods are supported\nhow do I create a role\nmulti-factor authentication setup'
[[ -n "$QUERIES" ]] && QLIST="$(grep -v '^\s*#' "$QUERIES" | sed '/^\s*$/d')" || QLIST="$DEFAULT_QUERIES"

now() { python3 -c 'import time; print(time.time())'; }
elapsed() { python3 -c "print(round($(now) - $1, 1))"; }

{
  echo "# rag-search sample run"; echo
  echo "- date: $(date -u +%FT%TZ)"; echo "- machine: $(uname -sm), $(sysctl -n machdep.cpu.brand_string 2>/dev/null || true)"
  echo "- file: $(basename "$PDF") ($(du -h "$PDF" | cut -f1))"; echo "- data folder: $DATA"; echo
} > "$REPORT"

echo "== doctor"; ("$EXE" doctor 2>&1 | tee "$OUT/doctor.txt" | tail -25) || true

cp -n "$PDF" "$DATA/docs/samples/" 2>/dev/null || true
DOC="$DATA/docs/samples/$(basename "$PDF")"

echo "== convert"; t=$(now)
"$EXE" convert "$DOC" -o "$OUT/converted.md" 2>&1 | tail -3
echo "- convert: $(elapsed "$t")s ($(grep -c '<!-- page' "$OUT/converted.md" || true) page markers)" >> "$REPORT"

echo "== index"; t=$(now)
"$EXE" index new --follow > "$OUT/index.txt" 2> "$OUT/index.log" || { echo "index failed; see $OUT/index.txt"; tail -20 "$OUT/index.txt"; exit 1; }
tail -8 "$OUT/index.txt"
echo "- index new (conversion + embedding + publish + search-daemon reload): $(elapsed "$t")s" >> "$REPORT"
"$EXE" index status --json > "$OUT/index.json"
"$EXE" list --json > "$OUT/list.json"
"$EXE" daemon status | tee "$OUT/daemons.txt"

echo "== searches (first one is cold: loads models)"
n=0
while IFS= read -r q; do
  n=$((n+1)); t=$(now)
  "$EXE" search "$q" --top-k 5 --json > "$OUT/search_$n.json"
  echo "- search $n ($(elapsed "$t")s): $q" >> "$REPORT"
  python3 - "$OUT/search_$n.json" >> "$REPORT" <<'PY'
import json, sys
for r in json.load(open(sys.argv[1])).get("results", []):
    print(f"    - p.{r['page']} [{r['score']}] {r['heading'][:60]} :: {r['text'][:110].replace(chr(10),' ')}")
PY
done <<< "$QLIST"

for g in ${GREPS[@]+"${GREPS[@]}"}; do
  echo "- grep '$g':" >> "$REPORT"
  "$EXE" grep "$g" --max 5 --json > "$OUT/grep.json"
  python3 - "$OUT/grep.json" >> "$REPORT" <<'PY'
import json, sys
for m in json.load(open(sys.argv[1])).get("matches", []):
    print(f"    - p.{m['page']} line {m['line']}: {m['text'][:120]}")
PY
done

echo; echo "Report: $REPORT"
