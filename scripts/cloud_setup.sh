#!/usr/bin/env bash
# Set up a Linux machine (a cloud session) to run ALL of rag-search's tests except the Apple-only ones.
#
#   scripts/cloud_setup.sh            # from the repository root: creates .venv, installs everything, checks it
#
# What it installs, and why this is the smallest set that exercises the real tools:
#   * the project and its real dependencies (docling, torch, sentence-transformers, mcp) into ./.venv
#     (or $VENV).  torch comes from PyTorch's CPU-only index when that is reachable (about 200 MB); from
#     PyPI otherwise, which on Linux drags in about 4 GB of CUDA libraries that a CPU machine never uses.
#     Allow download.pytorch.org and download-r2.pytorch.org in the sandbox's network policy to get the small one.
#   * tesseract (apt): docling's own default OCR engine downloads its models from a host that cloud
#     sandboxes often block; tesseract needs nothing from the network.  The tests select it themselves.
#   * pillow-heif and reportlab: only to read .heic and to rebuild the corpus (tests/data/make_corpus.py)
#   * coverage and ruff
# Models are NOT installed here: the real tests download two small ones on first use
# (all-MiniLM-L6-v2, ms-marco-MiniLM-L-2-v2; about 100 MB) from huggingface.co, so that host must be reachable.
# Not available on Linux, so left to the Mac (tests/machine/): MLX, Apple Vision, the document reader (VLM).
set -euo pipefail
cd "$(dirname "$0")/.."
VENV="${VENV:-.venv}"

say() { printf '\n== %s\n' "$*"; }

say "uv"
command -v uv >/dev/null || { echo "uv is needed: https://docs.astral.sh/uv/"; exit 1; }

say "OCR engine (tesseract)"
if command -v tesseract >/dev/null; then
  echo "tesseract already installed: $(tesseract --version 2>&1 | head -1)"
elif command -v apt-get >/dev/null; then
  SUDO=""; [ "$(id -u)" -eq 0 ] || SUDO="sudo"
  $SUDO apt-get install -y -q --no-install-recommends tesseract-ocr tesseract-ocr-eng
else
  echo "no tesseract and no apt-get: install tesseract with your package manager (the real tests need an OCR engine)"
fi

say "virtual environment (.venv) and dependencies"
[ -d "$VENV" ] || uv venv --python 3.12 "$VENV"
export VIRTUAL_ENV="$(cd "$VENV" && pwd)"
CPU_INDEX="https://download.pytorch.org/whl/cpu"
if [ "$(uname -s)" = "Linux" ] && ! "$VIRTUAL_ENV/bin/python" -c "import torch" 2>/dev/null \
   && curl -fsS -m 15 -o /dev/null -I "$CPU_INDEX/torch/" 2>/dev/null \
   && [ "$(curl -sS -m 15 -o /dev/null -w '%{http_code}' https://download-r2.pytorch.org/ 2>/dev/null)" != "000" ]; then
  echo "installing CPU-only torch from $CPU_INDEX"
  uv pip install --index-url "$CPU_INDEX" torch torchvision
else
  echo "using torch from PyPI (the CPU-only index is not reachable, or torch is already installed)"
fi
uv pip install -e ".[mcp]" pillow-heif reportlab coverage ruff

say "check"
"$VIRTUAL_ENV/bin/python" - <<'EOF'
import importlib.util, shutil, sys
need = ["docling", "torch", "sentence_transformers", "pypdfium2", "PIL", "mcp", "numpy"]
gone = [m for m in need if importlib.util.find_spec(m) is None]
print("python", sys.version.split()[0], "| missing:", gone or "nothing", "| tesseract:", bool(shutil.which("tesseract")))
sys.exit(1 if gone else 0)
EOF
cat <<'EOF'

Ready.  Run, from the repository root (with the environment above as $VENV, default .venv):
  .venv/bin/ruff check src tests scripts --select E4,E7,E9,F
  PYTHONPATH=src .venv/bin/python -m unittest discover -s tests/portable -t .   # tier A: fakes, ~4 min
  PYTHONPATH=src .venv/bin/python -m unittest discover -s tests/real -t .       # tier B: real tools, ~2 min
EOF
