# Working on rag-search

## Setup

```bash
uv sync                      # creates .venv with all dependencies (torch, docling, ... ~5 GB)
uv run python -m unittest discover -s tests/portable -t .      # or: uv run pytest
uv run ruff check src tests
```

On a Linux machine that is not the Mac (a cloud session, Cowork, CI), `scripts/cloud_setup.sh` does the same
with CPU-only torch and installs tesseract; see "Working away from the Mac" in `CLAUDE.md`.

## Tests: three tiers, chosen by what they need

The plan and its reasons are in `docs/design/test-strategy.md`.

| Tier | Folder | What it needs | Run it |
|---|---|---|---|
| A portable | `tests/portable/` | Python, numpy, pypdfium2, Pillow, mcp: no model, no docling, no torch, no network. Runs anywhere | `python -m unittest discover -s tests/portable -t .` |
| B real | `tests/real/` | A + docling, torch, sentence-transformers, an OCR engine (tesseract, or Apple Vision on the Mac) and two small models downloaded on first use. Runs on the Mac and on any machine that has them | `python -m unittest discover -s tests/real -t .` |
| C machine | `tests/machine/` | an Apple Silicon Mac, rag-search's own `uv tool` environment, the default models; skipped elsewhere | `python -m unittest discover -s tests/machine -t .` |
| all | the Mac | | `python -m unittest discover -s tests -t .` |

**A** is where the framework's logic is tested, and where coverage comes from (92 % of the statements: `coverage run -m unittest discover -s tests/portable -t .`,
then `coverage combine && coverage report -m`; the configuration in `pyproject.toml` counts the daemons and
workers the tests start as well). It uses a deterministic fake
embedder and reranker and **real daemon and worker processes**, so sockets, files, the CLI and the dashboard are
exercised for real. It is hermetic (`tests/guard.py`): importing `docling`, `torch`, `sentence_transformers`,
`transformers` or `huggingface_hub` fails and Hugging Face is offline, so a portable test gives the same result
on every machine; when it is about the path taken *without* the document reader it calls
`helpers.no_real_reader(self)` instead of relying on the reader being absent. Where the code's glue around a real
library needs testing (device choice, batching, page numbering of a docling result), the library is a small fake
put in `sys.modules`.

**B** runs the whole corpus once through the real pipeline (docling with OCR, a real small embedder and reranker,
real daemons) and checks every file against `tests/data/corpus.json`, then search, grep, export/import, the MCP
tools and the CLI views on that index. It tests the seams, not the framework again, and it is the only check that
the real docling, OCR and embedding behave as the fakes assume. It skips itself, with the reason, when a tool is
missing.

**C** is only what needs Apple's hardware and the default models: MLX, Apple Vision, the document reader, the
routed pipeline that uses them, `models verify`. They are the checks of `scripts/sanity_check.py` as unit tests
(generated documents, nothing downloaded).

`tests/helpers.py` is shared; `TempHome.tearDown` stops any daemon a test left running in its temporary data
folder. `tests/corpus.py` and `tests/data/` are the corpus (README there): one small synthetic file per kind of
input, and what each must produce. `uv run rag-search doctor --roundtrip` is the quick real-model smoke test.

## Dev loop: test in rag-search's own environment

The unit tests use fakes. To test the real tools and models, run everything in the environment the dashboard
and daemons use -- the `uv tool` environment -- not in a separate virtualenv or the system Python:

```bash
./install.sh --dev                       # from an unpacked release folder: installs the tool EDITABLE from a source tree
# or, in this checkout, with the same extras install.sh adds on Apple Silicon:
uv tool install --force --python 3.12 --with "mcp>=1.12,<2" --with ocrmac \
    --with "mlx-vlm>=0.3.4" --with "pillow-heif>=0.18" --editable .
PY="$(uv tool dir)/rag-search/bin/python"
"$PY" -c "import rag_search; print(rag_search.__version__, rag_search.__file__)"   # must be this checkout
"$PY" scripts/sanity_check.py            # quick: environment, packages, MLX/GPU, models downloaded
"$PY" scripts/sanity_check.py --all      # + Apple Vision, the document reader, docling, a Playground index, models verify
```

`scripts/sanity_check.py` prints PASS / FAIL / SKIP per check and exits 1 on a failure; it generates its own
documents (no personal data). After a code change in the editable install, restart what runs the old code:
`rag-search daemon stop && rag-search daemon start`, and the dashboard (`rag-search ui --stop`, then `rag-search ui`).
Real documents are tried in a Playground experiment (`rag-search playground create/config/index`, or the
dashboard's Playground tab), never in a production collection.

## Design record

`ARCHITECTURE.md` (how the code works) and the notes in `docs/design/` (plan and status,
reviews, analyses) are the design record. Read the relevant ones before a change and update them
in the same change -- see `docs/design/README.md`.

## Code map (`src/rag_search/`)

| Module | Responsibility |
|---|---|
| `paths.py`, `config.py` | folder layout and env, `config.json` + defaults |
| `policy.py`, `access.py` | per-client collection access: rules + enforcement helpers (used by daemons/api), and the `rag-search access` management code (CLI only; the MCP adapter must never import it) |
| `protocol.py`, `client.py` | wire format; sockets, on-demand daemon start (`spawn`, `request_sync`, `stop`) |
| `api.py` | high-level operations shared by CLI and MCP (search, grep, list, index_*, daemon_*, publish_and_reload) |
| `jobs.py` | read side of the indexing job records |
| `publish.py`, `catalog.py`, `grep.py` | generations under `serving/`, listings, confined regex search |
| `cli.py`, `service.py`, `register.py` | command line, launchd agents, Claude Desktop/Code registration (optional extra hosts: `hosts_*.py`) |
| `core/daemon_base.py` | single instance, socket server, idle exit (stdlib) |
| `core/indexer_daemon.py`, `core/worker.py` | run supervisor (stdlib) and the killable worker process |
| `core/search_daemon.py`, `core/search.py` | resident search process; engine with generation hot-swap |
| `core/indexer.py`, `core/chunker.py`, `core/bm25.py` | scan, freshness, per-document prepare/embed, merge; chunking; BM25 |
| `core/embedding.py`, `core/docling_convert.py`, `core/diagnostics.py` | models, converter (also reports per-page facts), `setup`/`doctor` |
| `core/conversion/*` | conversion tracking, all stdlib at import: `trace` (page/document records, run totals), `costs`, `router` (branch decision + reason), `profiler` (per-page profile, pypdfium2), `records`, `applevision` (last-resort page reader: Apple Vision through `ocrmac`), `runview` (worker lanes), `estimate`, `pageimage`, `pagemd` (page-marked Markdown split/join), `tables` (pipe/HTML tables, numbers), `validators` (running balance, totals), `metrics` (CER, numeric cells, table similarity), `engines` (page readers for the benchmark: `current`, `routed`, `vlm`), `bench` (gold sets and runs), `pagecache`, `gate`, `routed` (per-page conversion), `vlm` (document reader client, child process, memory guard) and `vlm_worker` (the child: backends `mlx` / `module:attr`), `repair` (re-reads a suspect table cell; second reader + accept rule) and `reconcile` (tables that continue across a page break) |
| `stages.py`, `effective.py` | the numbered pipeline registry (indexing 1, 2, 3, 3.1-3.5, 4-8; search S1-S6: what each stage does, which settings it reads, where it runs) and the resolver that reports each setting's real value and source (default < `config.json` < daemon environment); `GET /api/pipeline` serves both. A new setting is listed under exactly one stage (a test enforces it); every place that names a phase uses these numbers |
| `playground_runs.py`, `core/playground.py` | Playground experiments as background jobs (job record, `.events.jsonl` event log, pid file, cancel) running the same per-document conversion and indexing code as production in a child process; per-experiment settings and reader/repair pins |
| `spec.py` | the retrieval constants (RRF k, BM25 k1/b, pool sizes, limits, batch sizes) in one stdlib module: the engine and the dashboard's Architecture tab both import it |
| `ui/server.py`, `ui/info.py`, `ui/markdown.py`, `ui/static/*` | web dashboard: HTTP + SSE server over `api`, facts for the Architecture/Help tabs, safe Markdown renderer, plain JS pages (no build step). `pipeline.js` renders the numbered stages for the Indexing, Settings and Playground tabs (its `STAGE_KEYS` must equal the registry), `overview.js` is the Overview, `playground.js` the experiments |
| `mcp/server.py`, `mcp/profiles.py` | MCP tools (plain async functions in `make_tools`), host profiles |

`ARCHITECTURE.md` explains the processes, the on-disk format and the protocol.

`scripts/try_readers.py FILE` runs every page reader (Apple Vision, the document reader, docling OCR) on one
document outside the pipeline and compares what each finds; use it to judge a reader before wiring it in.
`scripts/sanity_check.py` checks that the installed tool can run the tools and models at all (see *Dev loop*).

## Conventions

* Keep the layering (ARCHITECTURE.md §4): the light modules and the indexer daemon must not import
  numpy/torch/docling (this includes `ui/`); only `mcp/` imports `mcp`. Tests enforce this.
* The dashboard's numbers come from `spec.py` and the real files: when you change a retrieval constant, change it in `spec.py`; when you add a field to `nodes.json`, `index.meta.json` or `catalog.json`, document it in `ui/info.py` (a test compares them). Edit README.md / ARCHITECTURE.md at the repo root; `scripts/build_release.sh` copies them into `ui/static/docs/` (a test fails if the copies differ).
* MCP adapters and daemons must never write to stdout unless they are the protocol (stdout is the
  MCP transport); log to stderr / `run/*.log`.
* Anything that can block (sockets, file walks, subprocesses) runs in a worker thread with a bound.
* Errors on one document are reported in the run summary; they must not abort the run.
* Bump `protocol.PROTOCOL_VERSION` on incompatible wire changes, `chunker.CHUNKER_VERSION` /
  `bm25.TOKENIZER_VERSION` when indexing output changes (existing indexes become stale and are
  rebuilt by the next `index new`).

## Building a release

```bash
scripts/build_release.sh     # dist/rag-search-<version>/ (wheel, sdist, install.sh, SHA256SUMS) + .zip
```
Bump `__version__` in `src/rag_search/__init__.py` first.
