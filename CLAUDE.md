# rag-search: notes for coding agents

Before making any change, read the design record in this repository and follow it:

* `ARCHITECTURE.md` -- how the code works today (source of truth).
* `docs/design/README.md` -- index of the design documents (plan with status and open items,
  critical review, location/export/import analysis, containerisation analysis and plan), and the
  working rule: read the relevant ones first, keep them updated in the same change.
* `CONTRIBUTING.md` -- code map, layering conventions, tests (`tests/portable/` anywhere, `tests/machine/` on the Mac), lint (`ruff check src tests`), release (`scripts/build_release.sh`).

Since 0.9.9 this checkout (the Mac) is the source of truth: work and test here. (Earlier versions were
developed in a cloud session and synced to the Mac; `dist/` holds those releases.)

Test in rag-search's own environment, the `uv tool` environment the dashboard runs, not in a separate
virtualenv: see "Dev loop" in `CONTRIBUTING.md` and run `scripts/sanity_check.py` with
`"$(uv tool dir)/rag-search/bin/python"` before trusting any real-model result. The unit tests
(`PYTHONPATH=src python3 -m unittest discover -s tests/portable -t .`, about 200 s) use fakes and prove nothing about
docling, MLX or Apple Vision.

Never copy personal data from the user's test documents (account numbers, addresses) into docs, tests,
commits or outputs. After editing `README.md` or `ARCHITECTURE.md`, copy both into
`src/rag_search/ui/static/docs/` (a test compares them).

## Working away from the Mac (a cloud session)

The repository holds everything needed to change and unit-test the code: `src/`, `tests/`, `scripts/`, the
design record. `dist/` (built releases), the data folder and the models are not in it. The unit tests need
only `numpy` (plus `pypdfium2` and `Pillow` for the PDF and image tests and `mcp` for the MCP test, which skip
themselves when missing): `pip install numpy pypdfium2 pillow "mcp>=1.12,<2"`, then
`PYTHONPATH=src python3 -m unittest discover -s tests/portable -t .` and `ruff check src tests scripts --select E4,E7,E9,F`.
Nothing that needs docling, MLX, Apple Vision or a downloaded model can be checked there: say so in the
result instead of assuming it works, and leave those checks (`tests/machine/`, or `scripts/sanity_check.py --all`) for the Mac.

