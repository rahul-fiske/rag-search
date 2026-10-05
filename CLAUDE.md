# rag-search: notes for coding agents

Before making any change, read the design record in this repository and follow it:

* `ARCHITECTURE.md` -- how the code works today (source of truth).
* `docs/design/README.md` -- index of the design documents (plan with status and open items,
  critical review, location/export/import analysis, containerisation analysis and plan), and the
  working rule: read the relevant ones first, keep them updated in the same change.
* `CONTRIBUTING.md` -- code map, layering conventions, tests (three tiers: `tests/portable/` anywhere, `tests/real/` with the real libraries, `tests/machine/` on the Mac), lint (`ruff check src tests`), release (`scripts/build_release.sh`).

Since 0.9.9 this checkout (the Mac) is the source of truth: work and test here. (Earlier versions were
developed in a cloud session and synced to the Mac; `dist/` holds those releases.)

Tests have three tiers, chosen by what they need (`docs/design/test-strategy.md` is the plan and the reasons):

| Tier | Folder | Needs | Run it |
|---|---|---|---|
| A portable | `tests/portable/` | Python, numpy, pypdfium2, Pillow, mcp. No model, no docling, no torch, no network | `PYTHONPATH=src python3 -m unittest discover -s tests/portable -t .` (about 4 minutes) |
| B real | `tests/real/` | + docling, torch, sentence-transformers, an OCR engine, two small models (about 100 MB) | `PYTHONPATH=src python3 -m unittest discover -s tests/real -t .` (about 2.5 minutes) |
| C machine | `tests/machine/` | an Apple Silicon Mac, the default models, MLX | on the Mac, in the `uv tool` environment: `scripts/sanity_check.py --all` |

Run A for every change, on any machine; B when a change touches conversion, embedding, the daemons' real
start-up or the packages' versions; C before trusting anything about MLX, Apple Vision or the document reader.
A is hermetic on purpose (`tests/guard.py`): `docling`, `torch`, `sentence_transformers`, `transformers` and
`huggingface_hub` cannot be imported by it and Hugging Face is offline, so it gives the same result on a machine
that has them and on one that has not. A test that needs a real library belongs in B.

Tests that need documents use the corpus in `tests/data/` (one small synthetic file per kind of input, and
`corpus.json` saying what each must produce; see its README), not a file made up inline, unless the test needs
text that varies or geometry tied to a fake reader.

Test in rag-search's own environment, the `uv tool` environment the dashboard runs, not in a separate
virtualenv, when the question is about the real tools on the Mac: see "Dev loop" in `CONTRIBUTING.md` and run
`scripts/sanity_check.py` with `"$(uv tool dir)/rag-search/bin/python"` before trusting any real-model result.

Never copy personal data from the user's test documents (account numbers, addresses) into docs, tests,
commits or outputs. After editing `README.md` or `ARCHITECTURE.md`, copy both into
`src/rag_search/ui/static/docs/` (a test compares them).

## Working away from the Mac (a cloud session, Cowork, any other machine)

Everything except Apple's hardware can be run there. One command sets a Linux machine up (a virtualenv with the
real stack, tesseract for OCR; CPU-only torch when PyTorch's CPU index is reachable):

```bash
scripts/cloud_setup.sh          # from rag-search/; `VENV=/some/path` to put the environment elsewhere
```

Then `.venv/bin/ruff check src tests scripts --select E4,E7,E9,F` and the tier A and B commands above (with
`.venv/bin/python`). For tier A alone, `pip install numpy pypdfium2 pillow "mcp>=1.12,<2"` is enough.

What a cloud session needs from its network policy: PyPI; for tier B also `huggingface.co` (and its CDN hosts,
`*.hf.co`) for the small models and docling's own models (about 600 MB, once); and, to avoid 4 GB of CUDA
libraries, `download.pytorch.org` and `download-r2.pytorch.org`. Without Hugging Face, tier B cannot run and tier A
still can.

In the result of a change, say which tiers ran. Tier A proves the framework's logic, tier B that the real docling,
OCR and embedding work on the whole corpus, neither proves MLX, Apple Vision or the document reader: say so, and
leave `tests/machine/` for the Mac, instead of assuming it works.
