# rag-search

Private, local document search. Point it at a folder of PDFs, Word/PowerPoint/Excel files, HTML,
Markdown or images; it converts and indexes them on your Mac and answers questions with the
**source file and page number**. Use it from the shell, from Claude Desktop / Claude Code, or from
any other MCP host — all of them share the same indexes and the same warm models.

Everything runs on your machine. The only network traffic is the one-time download of the models
from Hugging Face (about 5 GB) and of the Python packages at install time.

## How it works

```
                     ┌─────────────────────────── one data folder ───────────────────────────┐
 rag-search CLI ─┐   │                                                                        │
 Claude Desktop ─┤   │  indexer daemon ──spawns──▶ worker ──writes──▶ indexer_workspace/      │
 Claude Code    ─┼─▶ │  (one run at a time)                                   │ publish       │
 Other hosts    ─┘   │                                                        ▼               │
   (thin clients,    │  search daemon ◀── reload ───────────────────── serving/current ──▶ gen-000042
    MCP adapters)    │  (always on, models + indexes in RAM)                                  │
                     └────────────────────────────────────────────────────────────────────────┘
```

* **Indexer daemon** – a light supervisor. Guarantees exactly one indexing run at a time, runs it
  in a killable worker process, and when the run succeeds publishes a new *generation* and tells
  the search daemon. `index new --restart` kills the run and starts over (finished documents are
  skipped by SHA-256, so this is cheap).
* **Search daemon** – always on. Keeps the embedding model, the reranker and every published index
  in memory; switches to a new generation without downtime.
* **Front-ends are thin socket clients**: the `rag-search` CLI and the MCP adapter
  (`rag-search-mcp`) call the same functions (`rag_search.api`). See `ARCHITECTURE.md`.
* **Pipeline** – [docling](https://github.com/docling-project/docling) → Markdown with
  `<!-- page N -->` markers → chunks → BM25 + `BAAI/bge-m3` vectors, fused with reciprocal-rank
  fusion, then reranked with `BAAI/bge-reranker-v2-m3`.

## Requirements

* macOS on Apple Silicon (M1 or newer) with 16 GB RAM or more (Linux works too; slower on CPU).
* Intel Macs are supported on a best-effort basis: CPU only (no Metal), so indexing and reranking are
  much slower, and `install.sh` pins `numpy<2` because the last PyTorch build for Intel Macs needs it.
* About 15 GB free disk (packages ≈ 5 GB, models ≈ 5 GB, plus your indexes).
* [`uv`](https://docs.astral.sh/uv/) (`brew install uv`). It fetches Python 3.12 itself.
* Optional: Claude Desktop and/or Claude Code (any other MCP host can use the adapter too).

## Install

Unzip the release folder, open Terminal in it and run:

```bash
./install.sh
```

This verifies `SHA256SUMS`, installs the wheel with `uv tool install` (including the MCP adapter and,
on macOS, `ocrmac` for on-device OCR of scanned PDFs and photographed documents), downloads the
models, runs `rag-search doctor`, starts both daemons and registers the adapter with Claude Desktop
and Claude Code. **Fully quit and reopen Claude Desktop afterwards.**

Options: `--home PATH` (custom data folder), `--service` (daemons start at login via launchd),
`--tool-prefix P` (prefix tool names to
avoid clashes with other servers), `--no-mcp` (CLI and daemons only), `--skip-models`,
`--models PRESET` (install with another model pair: `default`, `qwen3-small`, `qwen3-large`; see "Models"),
`--no-register`, `--python 3.11`, `--import-only` (for using collections someone else exported:
skips the OCR engine and docling's document-conversion models; see "Sharing a collection").

Self-test (real daemons and models, three tiny documents): `rag-search doctor --roundtrip`.

Uninstall: `./uninstall.sh` (keeps your data; add `--purge-data` to delete indexes too).

## Use

1. Tell rag-search where your documents are: `rag-search location add NAME FOLDER` (a notes vault,
   a synced drive, any folder). **Each registered folder is one collection**, indexed where it is and
   never changed (see "Source locations"). There is no built-in documents folder.
2. Index: `rag-search index new --follow` (or ask Claude: *"index my new documents"*). The run
   happens in the indexer daemon; when it finishes the new documents are searchable automatically.
3. Search: `rag-search search "how do I create a role?"`, or ask Claude: *"According to my
   manuals, how do I … ? Cite pages."*

```
rag-search search "query" [-c collection] [-k 5] [--json]     meaning-based search, cites file + page
  [--stages bm25,dense,rerank] [--retrieval-pool N] [--rerank-pool N] [--rrf-k N] [--explain]
rag-search grep "regex" [-c collection]                        exact text / IDs / numbers
rag-search list [--client NAME]                               what is published: sizes, when built, build time
rag-search describe [COLLECTION [TEXT|--clear]]                view or set a collection's short description
rag-search access [restrict|grant COLLECTION CLIENT...]        who may use which collection (see "Access per client")
rag-search ui [--port N] [--read-only] [--detach|--stop|--url] local web dashboard (see "Web dashboard")
rag-search index new [PATH] [--restart] [--rebuild] [-f]       index new/changed documents
rag-search index all [PATH] [--restart] [-f]                   wipe and rebuild everything in scope
rag-search index status [JOB] [-f] [--docs N] [--history N]   progress, timings per document, or -f live
rag-search index status --doc-branch B --doc-outcome O         only documents with a page of that branch / outcome
rag-search index estimate [PATH]                               dry run: pages per branch and time, converts nothing
rag-search trace COLLECTION/DOC [--page N] [--md]              how a document was converted: branch + outcome per page; --md prints the converted Markdown
rag-search index cache [--clear]                               the page cache (pages already read): size, empty it
rag-search index cancel | publish | foreground                 stop / publish now / run without daemon
rag-search daemon status|start|stop|restart [search|indexer]   state, warm-up time and memory; control
rag-search service install|uninstall|status                    launchd: daemons start at login
rag-search config show|init|path                               settings file (see below)
rag-search doctor [--roundtrip] | setup | paths [name] | convert FILE
rag-search location list|add NAME FOLDER|remove NAME [-y]     register the folders documents are read from
rag-search collection info NAME                                documents, folders, sizes, dates and state of one collection
rag-search collection export NAME [-o FILE]                    one .rag.tgz file to share (see "Sharing")
rag-search collection import FILE [--as NAME] [--replace]      add someone's exported collection
rag-search collection delete NAME [-y]                         delete a collection's Markdown + index (never documents)
rag-search register [--desktop] [--code] | unregister | mcp-config [--profile NAME]
```

A file that cannot be indexed for its own reasons (a password-protected PDF, a file with no text at all) is
converted once: later runs still list it with its reason ("not tried again") but only compute its checksum.
Change the file, change a conversion setting, or tick *re-convert to Markdown* (`--force-md`) to try again.


Every command accepts `--json`, `--home PATH` and `--client NAME`. `PATH` arguments are files or
folders inside a registered location (`vault/projects` = the `projects` folder of location
`vault`). Exit codes: 0 ok, 1 error, 2 usage, 3 daemon not ready/unavailable.
`list` and `grep` keep working when the search daemon is down (they read the published files
directly); `search` starts the daemon on demand.

### What each command tells you about time and size

| Question | Command | Fields (`--json`) |
|---|---|---|
| How big is each index, when was it built, how long did it take? | `rag-search list` | per collection: `index_bytes`, `markdown_bytes`, `source_bytes`, `built_at`, `build_seconds`; per document: `build_s`, `convert_s`, `embed_s`, `index_bytes`; `totals` |
| Since when is indexing running, and what is it doing? | `rag-search index status` (add `-f` to watch) | `job.started_at`, `job.elapsed_s`, `job.progress` (`phase`, `phase_started_at`, `current`, `current_since`) |
| How long did each document take? | `rag-search index status [--docs N]` | `documents.items[]`: `status`, `chunks`, `convert_s`, `chunk_s`, `embed_s`, `total_s`; `summary.phase_s` |
| How was each page of a document read, and what did conversion cost? | `rag-search trace COLLECTION/DOC`, `index status` (run totals), `collection info` | `documents.items[].conversion`, `job.summary.conversion` (pages per `branches` / `outcomes`, `strip`, `time_s`, `cost`); `trace` per page: `branch`, `why`, `outcome`, profile and docling confidence |
| How well is a page read, compared with checked text? | `rag-search bench gold init SET`, `bench run SET [--engine current|routed|vlm|module:attr]`, `bench compare SET A B`, Playground tab | per run: `summary.all` / `summary.by_class`: `cell_exact`, `cell_bag`, `cer`, `table_sim`, `balance_ok`, `query_hit`, `s_per_page`; gold sets live in `<home>/conversion_gold/` |
| How long will indexing take? | `rag-search index estimate [PATH]` | `pages`, `branches`, `docling.seconds`, `planned_vlm` |
| How long did a search take, and where? | `rag-search search` (last line) | `timing`: `total_ms`, `embed_query_ms`, `keyword_ms`, `dense_ms`, `rerank_ms`, `queue_ms`, `server_ms`, `round_trip_ms` |
| How long did a grep take? | `rag-search grep` (last line) | `timing`: `total_ms`, `scan_ms`, `files_scanned`, `server_ms`, `round_trip_ms` |
| Is the search daemon warm, how long did warm-up take, how much memory? | `rag-search daemon status` | `warm`, `warmup` (`status` warming_up / warm / error, `models_s`, `index_s`, `total_s`, `ready_at`), `memory` (`rss_bytes`, `peak_rss_bytes`, `embeddings_bytes`, `text_bytes`, `models_bytes`, per collection), `last_reload` |

The MCP tools return the same JSON. Documents indexed by 0.2.0 have no recorded build time until
they are indexed again (`list` says so). `rag-search doctor` also shows the warm-up time and
resident memory of a running search daemon.

### Troubleshooting and tuning the search pipeline

Every search already runs BM25 and dense vector retrieval, fuses them with Reciprocal Rank Fusion
(RRF), then reranks the fused pool with a cross-encoder (or an LLM reranker, depending on the
configured model) — see [`ARCHITECTURE.md`](ARCHITECTURE.md#52-search-flow-search-daemon-coresearchpy)
or the dashboard's Architecture tab. To see *why* a result ranked where it did, or to isolate one stage while
debugging, `rag-search search` and the dashboard's Search tab both expose the pipeline directly:

* **Per-result scores.** Every hit now carries `bm25_score`/`bm25_rank`, `dense_score`/`dense_rank`,
  `rrf_score`, and `rerank_score` alongside the final `score` (a `null` means that stage didn't
  retrieve the hit at all). `--explain` prints them; `--json` always includes them.
* **`--stages bm25,dense,rerank`** runs only the listed stages (at least one of `bm25`/`dense` must
  stay on). Dropping `dense` skips the embedding call entirely; dropping `rerank` skips the
  cross-encoder. With only one retrieval stage active there's nothing to fuse, so results are
  ranked by that stage's own raw score instead of RRF (`rrf_score` is `null` in that case).
* **`--retrieval-pool N` / `--rerank-pool N` / `--rrf-k N`** override the pool-size and
  reciprocal-rank-fusion formulas in `spec.py` for one search, clamped server-side to safe
  ceilings (`retrieval_pool` ≤ 300, `rerank_pool` ≤ 100) so a debug request can never make the
  cross-encoder run on an unreasonable number of candidates. The RRF constant defaults to 60;
  a smaller `k` weights rank position more steeply (a strong #1 in one retriever matters more
  relative to a mediocre-but-consensus hit), a larger `k` flattens it towards pure consensus.
* **`--explain`** adds a second line with the effective `stages`/pool sizes/`rrf_k` used, how many
  candidates each retriever found (`bm25_candidates`/`dense_candidates`), how much they agreed
  (`overlap_count`/`bm25_only_count`/`dense_only_count`), and, per stage, the score gap between
  rank 1 and rank 2 (`bm25_gap`/`dense_gap`/`rerank_gap`) — a large gap is a sign that stage found
  one clearly-best passage that fusion or reranking may be burying.

None of this changes default behaviour: a plain `rag-search search "query"` is identical to before
these fields existed. The MCP `rag_search` tool intentionally does not expose the overrides — this
is a human-troubleshooting surface, not something to hand an LLM caller extra knobs for.

### Playground: try models and tunables, and benchmark them

`rag-search playground` is a sandbox for trying a different embedding model, reranker, chunk size
or pool/RRF tunable against a small sample of documents, and for benchmarking the result —
structurally separate from your real collections. Everything lives under
`<home>/playground/<experiment>/`: its own source folders (registered the way a production collection's are), its own index, its own `config.json`. It is
never read by production search, indexing or publish, and there is no daemon — every command loads
the small index and the experiment's chosen models in its own process and exits (from the dashboard, an
index or benchmark run is started in the background and watched live, see below).

```bash
rag-search playground create demo --from ~/Downloads/samples      # a folder, read where it is (never copied)
rag-search playground source demo add ~/Notes/more --as notes      # more folders: list | add | remove
rag-search playground config demo --embedding-model Qwen/Qwen3-Embedding-0.6B --rerank-model BAAI/bge-reranker-base
rag-search playground index demo
rag-search playground search demo "how is a session token refreshed" --explain
```

`config` pins an experiment's embedding model, reranker, **document reader (`--reader-model`) and repair
model (`--repair-model`)** (blank / `production` = production's choice from the Models tab), chunk size/overlap, default
stages/pool-sizes/RRF-k, and every docling/OCR/table/PDF-backend tunable
(`--ocr`/`--ocr-engine`/`--ocr-lang`/`--table-mode`/`--pdf-backend`/`--pipeline`/`--doc-timeout`/
`--docling-batch` — the same knobs and choices as `rag-search config set`, since both are generated
from the same tunables registry); changing any of these does not touch anything until you `index`
again (with `--rebuild` or `--force-md` to redo documents already converted). `search` takes the
same `--stages`/`--retrieval-pool`/`--rerank-pool`/`--rrf-k`/`--explain` flags as production
`rag-search search` — a technique you learn here (say, "dense-only finds it but hybrid buries it")
transfers directly.

**The production bridge.** `rag-search playground create NAME --from-production` seeds the new
experiment's embedding/reranker model, chunk size/overlap and search tunables from today's
*effective* production settings (config.json, env vars and all — the same precedence "Pipeline
tunables" above describes) instead of this module's own defaults, so a benchmark starts from what
real searches actually get. Going the other way, `rag-search playground promote NAME` writes an
experiment's combo back into production's `config.json` — `--dry-run` shows what would change
without writing anything, and a change to the embedding model or chunk size/overlap (which makes
the *existing* production index stale until it is rebuilt) needs an explicit `--confirm`, with the
`documents`/`estimated_s` cost from the same estimate `rag-search models set` shows before an
embedding-model switch:

```bash
rag-search playground create tuned --from-production
rag-search playground config tuned --chunk-size 768 --retrieval-pool 60
rag-search playground index tuned && rag-search playground bench tuned --label wider-chunks
rag-search playground promote tuned --dry-run     # see what would change, and the reindex cost
rag-search playground promote tuned --confirm     # apply it; then: rag-search index --rebuild
```

**Benchmarking.** Put labeled questions in `<experiment>/bench/queries.jsonl` (one JSON object per
line — a template is created for you: `bench/queries.jsonl.example`):

```json
{"query": "how is a session token refreshed", "relevant": [{"file": "sample.pdf", "page": 4}]}
```

Labels are pinned to *document + page*, not an exact chunk, because chunk boundaries shift whenever
you change chunk params or the embedding model — a label written today should still mean the same
thing after you switch models. `rag-search playground bench demo --label baseline` replays every
labeled query and records Recall@k, MRR, nDCG@k, and latency (mean/p50/p95), tagged with the exact
combo that produced them (model, chunk params, stages, pool sizes, RRF k), to
`bench/runs/<run-id>.json`. `rag-search playground compare demo` lists every recorded run side by
side, so you can see which combo actually moved the needle rather than eyeballing one search at a
time.

The dashboard's Playground tab does the same things with a form instead of flags: create and open
an experiment (with a "copy from production" checkbox for the bridge above), choose its source folders (read in place, like a production collection's), edit its
model/tunables and its own "Conversion (docling)" card (the same OCR/table/PDF-backend fields as
the Settings tab's Indexing pipeline card), build the index — with the same two independent
checkboxes as production's Indexing tab, "re-embed even if unchanged" (`--rebuild`, bypasses only
the chunk/embed freshness check) and "re-convert to Markdown" (`--force-md`, also redoes the
docling conversion step even when nothing detectably changed; changing a docling/OCR/table/
PDF-backend setting to something actually different reconverts on its own either way) — search it
with the same stage-toggle and advanced-pool panel as the Search tab, run a benchmark, see every
recorded run compared, open any document of a build in the same side panel as the Indexing tab (page records, the source page, the converted Markdown), and promote a combo to production with one button (it previews the diff
and, when it would leave the index stale, asks for confirmation before writing anything) —
promotion never includes the docling settings, only the embedding/reranker/chunk/search combo.
**Live progress.** *Build index* and *Run benchmark* start a background process and the tab follows it with the same stage numbers as everywhere else (2 Fingerprint, 3 Convert with 3.1–3.5, 4 Chunk, 5 Embed, 6 Write, 7 Merge): a strip with counts per stage, the pages being read right now, the workers, and a row per document with its stage chips and a grid of its pages (hover for branch, outcome, time and model; *page table* lists the gate checks). A benchmark shows every query with the rank at which the answer was found and the time of each search stage (S2–S5). The run continues if you leave the tab, can be cancelled, and earlier runs are listed. Like `doctor`, every playground action runs `rag-search playground ...` in a child process, so the dashboard's own process never loads a model.

`rag-search playground settings NAME` prints what the next run really uses, stage by stage, and where each value comes from; `rag-search playground status NAME` prints the latest run (stages, documents, pages) in a terminal.

`rag-search playground list` / `rm NAME --yes` manage experiments; deleting one only removes that
sandbox — nothing in production is ever touched.

### MCP tools (Claude Desktop, Claude Code, other MCP hosts)

| Tool | Purpose |
|---|---|
| `rag_list_collections(documents=false)` | What is indexed, per collection: name, description, document count, sizes. Pass `documents=true` for the full per-document listing |
| `rag_describe_collection(collection, description)` | Set (or, with `description=""`, clear) a collection's short description |
| `rag_search(query, collection="", top_k=5)` | Hybrid search + rerank; returns file, page, heading, text (and `"confidence": "low"` for a hit on a page the converter could not fully verify) |
| `rag_grep(pattern, collection="", context_lines=2, max_matches=20)` | Regex over the converted Markdown, with page numbers |
| `rag_index_update(path="", restart=false, rebuild=false)` | Incremental indexing in the indexer daemon (idempotent while a run is active) |
| `rag_index_rebuild(path="", confirm=false, restart=false)` | Wipe and rebuild (needs `confirm=true`) |
| `rag_index_status(job_id="", wait_seconds=0)` | Progress / result of the current or a given run |
| `rag_index_cancel` | Stop the running run |

`rag_list_collections` used to always include every collection's full document listing, which for a
large collection could be a lot of text handed to the calling LLM on every call. By default it now
returns just each collection's name, `description`, `document_count`, and sizes; pass
`documents=true` when you actually need individual file names. Since a collection is just whatever
folder of documents someone indexed, `description` can't be generated automatically -- it starts
empty. After exploring an undescribed collection (a search, a grep, or a `documents=true` listing),
call `rag_describe_collection` once to save a short summary for future calls to see, yours or
another agent's; `rag-search describe` is the human-facing equivalent, to read what an agent set or
to fix one that drifted (see "Collection descriptions" below).

(Before 0.2.4 the tools were called `docs_list_collections`, `docs_search`, `docs_grep`,
`docs_index_new`, `docs_index_all`, `docs_index_status`, `docs_index_cancel`; hosts pick up the new
names when they reload the server.)

The adapter is `rag-search-mcp [--profile NAME] [--tool-prefix P] [--client-id NAME]`. `NAME` is the host's
client identity (`claude`, or any name you choose), see "Access per client". Each
host launches its own copy (like any stdio MCP server); the copies are
tiny — they never load a model — and all talk to the same two daemons.


**Other MCP hosts.** Any host that can start a stdio MCP server can use the adapter. Print the
entry with `rag-search mcp-config --profile NAME` (add `--home PATH` for a non-default data folder)
and put it into that host's MCP settings; `NAME` becomes the host's client identity, so
`rag-search access restrict COLLECTION claude NAME` can give it its own set of collections. The tool
names all start with `rag_`, so they do not clash with other servers; `--tool-prefix P` still adds a
prefix if you ever need one.

## Web dashboard

```bash
rag-search ui              # starts it (default http://127.0.0.1:8765) and opens your browser
rag-search ui --detach     # keep it running in the background;  rag-search ui --stop  ends it
rag-search ui --url        # print the link again;  --read-only shows everything but changes nothing
```

The dashboard is part of the package, so it works wherever `rag-search` works (Apple-Silicon and Intel
Macs, Linux) and needs nothing extra: no Node, no build step, no internet. Everything updates live.

| Tab | What it shows and does |
|---|---|
| Overview | answers "is it working, and what next?": a status banner with the next action (Search / Index new and changed documents / Watch the run), the path of a document drawn once -- documents → index (stages 1–6) → publish (7–8) → search daemon (S1–S6) → clients -- with the live numbers of each step, a *Needs attention* list where every item links to where it is fixed (failed or empty documents, an unpublished run, a model chosen but not downloaded, the document reader that cannot run), a card per collection, how the last run read its pages, and the two daemons with Start / Restart / Stop |
| Indexing | the **Sources** of every collection (registered folders, imports), start a run (new & changed, or a full rebuild after confirmation), cancel, publish; the pipeline's numbered stages with progress and ETA, **Settings in effect, by pipeline stage** (read-only: the value every stage will really use and where it comes from), long lists of failed or unsupported files folded away (the filtered document list has them all), per-document convert / chunk / embed times, run history; a *Phase detail* card with one tab per phase (Convert: pages per branch -- digital, scanned, image, Office, text --, CPU time and memory; Embed: documents, chunks, chunks per second, model; Merge: collections; Publish: generation and search reload) that follows the run or shows the phase you click, a *Workers* card (every process, the phase, document and stage it is in), a branch strip, cost and page-by-page drawer for every document (click a row; a page shows its record, the source page image and the converted Markdown, and the whole document's Markdown opens in a new tab), branch / outcome filters, and an **Estimate** button (dry run) |
| Collections & access | **Add collection…** (register a folder as a location) and **Import…** (a `.rag.tgz` export); every collection with documents, sizes and build times; who may use it, edited with a click (same as `rag-search access`); its description (**Describe…**). Click a row for its details (same as `rag-search collection info NAME`): a state (*up to date*, *needs indexing*, *not published yet*, *source unreachable*, *imported*), tiles for documents indexed vs. found in the source folder, chunks, space on disk, source size, last indexed and build time, then collapsible sections -- **Where it lives** (source, Markdown, index and published folders, each with a Copy button and its size), **Indexing & publishing** (model and weights commit, vectors, chunking, first/last indexed, merged and published state, access, the last run's counts and errors), **Needs attention** (documents not indexed yet with the reason, changed since indexed, interrupted), **Conversion** (pages per branch and outcome, time, cost, documents docling itself graded poor), **Origin** for an imported collection, and the **Documents** list with a filter -- plus **Export…** (save a `.rag.tgz`, then download it), **Delete…** (its Markdown and index only, never the documents; type the name to confirm) and, for a location, **Remove location…** |
| Models | the embedding models and rerankers you can use, which one is in use, what is downloaded and what fits this computer (a model is *in use* only when it is chosen **and** downloaded **and** able to run: otherwise it says what is missing); download and switch with a click (the same as `rag-search models`); the **Document reader** card has a readiness checklist (platform, the optional Apple-only runtime `mlx-vlm`, the weights) with one-click fixes, including **Install the runtime** (the same as `rag-search models runtime install`). The batch sizes, sequence length, precision and device are on the Settings tab (stages 5 and S5) |
| Search | try semantic search and regex grep, optionally *as* another client to see exactly what that host would get, with a breakdown of where the time went; toggle the BM25/dense/rerank stages, override pool sizes and the RRF constant, and see each hit's per-stage scores, for troubleshooting and tuning the pipeline (see "Troubleshooting and tuning the search pipeline" above) |
| Playground | create sandbox experiments with their own sample documents, models (embedding, reranker, **document reader, repair model**) and tunables, arranged by pipeline stage (optionally seeded from production); build their (tiny, isolated) index and **watch it live** -- the numbered stages, every document, every page with the reader that read it and how it ended, the workers -- search it with the same stage/pool controls as Search, run a labeled-query benchmark query by query, compare every recorded run, see the settings the next run really uses, and promote a combo back to production — never touches production on its own (see "Playground" above) |
| Settings | production defaults, **arranged by the stage of the pipeline that reads them** (the numbers 1–8 and S1–S6 of the Architecture tab) — pool sizes, RRF k, default stages, chunk size/overlap, docling and document-reader settings, batch sizes, device — each with a "what it means" line, an ⓘ impact note and the value *really in effect* and where it comes from (default, `config.json`, or an environment variable the daemon was started with, which wins); the same values `rag-search config set` changes (see "Configuration") |
| System | health check (`rag-search doctor`), where things live, the daemon logs (with timestamps). The *Indexer daemon (files & pipeline)* log follows every file through convert → chunk → embed → INDEXED (or FAILED, with the cause) while a run is going |
| Architecture | pictures of the whole system (who talks to whom, which side reads and which writes), the indexing pipeline, how each page is read (profiler, docling, vision reader, quality gate, repair; CPU/GPU chips) and the search pipeline, with live daemon status; where each collection's documents come from; how indexing, the index files, the search pipeline (BM25, vectors, RRF, cross-encoder rerank), the models, the daemons and clients, and where every tunable is actually stored (which `config.json` section, what wins when several are set, who reads it) work, with the numbers read from the running engine |
| Help & CLI | the command reference generated from the installed CLI, and this guide |

Security: the server listens on `127.0.0.1` only, refuses other `Host` headers and cross-origin
requests, and needs a secret token. `rag-search ui` prints and opens a link containing it; the
browser then keeps it in a cookie. The token lives in `run/ui.token` (mode 0600); delete that file
and restart the dashboard to change it. Anyone who can read your files can use the dashboard, just
as they can use the CLI. The dashboard acts as the administrator (`cli`), so it can edit access
rules; use `--read-only` if you only want to look. Set the port with `--port` or `RAG_SEARCH_UI_PORT`
(`0` = any free port). It never loads a model itself; searches go through the search daemon.

After an upgrade the old dashboard keeps running the old code (the page says so). `rag-search ui`
replaces it; `install.sh` restarts it by itself if it was running. `rag-search ui --stop` works for a
dashboard in the foreground too (since 0.3.3).

## Access per client

By default **every collection is available to every client**. To keep a collection away from some
hosts, restrict it to the clients that may use it. This is managed only with the CLI; the MCP tools
cannot read or change it, they just return what their host is authorised for: `rag_list_collections`
lists only those collections, and a collection the host may not use is reported as *unknown*, exactly
like one that does not exist (also in error messages and in `rag_index_status`).

```bash
rag-search access                              # every collection, who may use it, known clients
rag-search access restrict hr claude           # only claude may use "hr" (everybody else: no)
rag-search access restrict finance claude agent  # exactly these two clients; run it again with fewer names to remove one
rag-search access restrict hr                  # no client names = nobody (only your own terminal)
rag-search access grant hr agent               # add a client to a restricted collection
rag-search access restrict hr all              # back to "every client" (`grant hr all` does the same)
rag-search list --client agent                 # exactly what "agent" would see (also: search/grep --client agent)
```

`restrict` always states the complete list of clients, so it also removes clients; `grant` only adds. `all` is
not a client name: it means every client and removes the restriction.

Changes take effect immediately (the daemons re-read `access.json`; no restart). The dashboard's *Collections & access* tab edits the same rules. You can restrict a
collection before it is indexed, so nothing is exposed while the first run is in progress.

**How a host is identified.** Each host starts its own adapter with a name:
`rag-search-mcp --profile claude` (Claude Desktop and Claude Code; `rag-search register` writes it)
or `rag-search-mcp --profile NAME` (any other host; `rag-search mcp-config --profile NAME` prints the entry). The adapter sends that name with every request, and the
daemons decide from it. Any new name is a new client: register another host with
`rag-search-mcp --profile NAME` in its MCP config, then `rag-search access restrict ... NAME`. The
terminal (`cli`) is the administrator and can always use everything. The last line of `rag-search access` lists the
client names known so far (built in, named in a rule, or seen by the running search daemon).

This is a guard against an agent reaching documents it should not, **not** a security boundary: the name
is declared by whoever starts the adapter and all files belong to your macOS user. For hard isolation use
a separate `RAG_SEARCH_HOME` (or macOS user) per audience.

Upgrading from 0.2.4 or older: `personal` and `private` used to be private by default; now nothing is
restricted until you say so. `rag-search doctor` warns if such collections exist, and ignores the old
`policy` section of `config.json` (also `RAG_SEARCH_PRIVATE_COLLECTIONS`). Restore the old behaviour with
`rag-search access restrict personal claude`.

## Source locations, deleted documents and unreachable folders

A collection is a **registered location**: a folder whose whole tree becomes one collection, indexed
where it is (nothing is copied). Without a registered location there is nothing to index:

```bash
rag-search location add vault ~/Documents/Notes     # collection "vault" = that folder
rag-search location add "" "/Users/me/My Drive/Notes"   # no name given: the folder's own name (Notes); spaces, @ etc. are fine
rag-search location list                            # every location, and whether it can be read now
rag-search index new --follow                       # indexes it with everything else
rag-search index new vault/projects                 # just one folder of it
rag-search location remove vault --yes              # unregister + delete its Markdown and index (not the folder)
```

A location's name must be free (not an imported collection or another location) and its folder
may not overlap rag-search's data folder or another location. A collection is always one of two things -- a registered location or an import -- and registering
one means giving its folder. An index that is neither (a leftover of an older layout, or a hand-edited
`locations.json`) is not listed and is removed, with its Markdown, by the next full indexing run.

**Source documents are read-only to rag-search.** Indexing never writes, moves or deletes
anything in a registered location; what it deletes is its own derived data.
No rag-search command writes into a source folder, whatever you ask it to.

What indexing concludes from a source folder:

| the document... | what happens |
|---|---|
| changed | re-converted and re-embedded (its SHA-256 changed) |
| unchanged, but the folder moved (registered again at its new place, or re-mounted elsewhere) | kept as is; its recorded location is updated |
| deleted | its converted Markdown and index are deleted, and it leaves search with the next publish |
| in a folder that cannot be read right now (unmounted drive, share down, a sub-folder that cannot be listed) | **nothing**: the collection keeps its last index and stays searchable; the run reports it as *not reachable* |
| in a location that is suddenly **completely empty** | **nothing** either -- an unmounted mount point looks exactly like that. To drop such a collection on purpose: `rag-search location remove NAME` |

A document only counts as deleted when the run read its whole collection: `index new` (everything)
or `index new COLLECTION`. A run over one sub-folder or one file never removes anything outside
what it read. Indexing refuses to run at all while `locations.json` cannot be read (without it every
collection would look like a leftover). On a case-insensitive disk (the macOS default), renaming a folder only in case
(`security` → `Security`) keeps its index.

## Sharing a collection: export and import

```bash
rag-search collection export manuals                 # -> ./manuals.rag.tgz (or -o FILE|FOLDER)
rag-search collection import manuals.rag.tgz         # on another machine (or --as NAME)
rag-search collection import manuals.rag.tgz --replace   # a newer export of an earlier import
rag-search collection delete manuals --yes           # remove a collection's Markdown and index (any kind)
```

An export is **one compressed file** with the collection's merged index (chunks + vectors), its
converted Markdown (so `grep` works), per-document metadata and a `manifest.json`: embedding model
**and the exact commit of its weights**, vector size, chunking and tokenizer versions, conversion
settings, the collection's description, and a SHA-256 for every file. It does not contain the
source documents, absolute paths from your machine, or your access rules — whoever imports decides
who may use it (`rag-search access restrict …`).

Import refuses what it cannot use as-is: an export embedded with a **different model** than this
installation's (vectors of different models cannot be searched together, so there is no partial
or keyword-only import; when both sides record the weights commit, that must match too; and
while a model switch is still re-embedding your own documents, imports wait until it is done), a
damaged file (checksums), anything but plain files at safe paths, and a name that is already
taken (a location or an indexed collection is never overwritten; `--as NAME`
imports under another name, `--replace` replaces only an earlier import of the same name).

An imported collection has no source documents: indexing never scans, prunes, re-merges or
rebuilds it (`index new security` on it explains why), it is published and searched like any
other, and `collection delete` removes it. If you later switch this installation's embedding model,
an imported collection embedded with the old one blocks publishing until you delete it (and
import an export made with the new model).

`collection delete NAME` works on **every** collection -- a location or an
import -- and removes only what rag-search built for it in its workspace: the converted Markdown
(`indexer_workspace/markup/NAME/`) and the index (`indexer_workspace/index/NAME/`), then publishes,
so it leaves search at once. It **never touches source documents**, and it keeps the collection's
access rule (a rebuilt collection keeps its restrictions), its description and a location's
registration. A collection whose documents are still in place is therefore built again by the next
indexing run: deleting is how to throw away a broken or unwanted index and start over. To stop
indexing a folder for good, `rag-search location remove` it (unregister + delete). An imported collection has no documents, so deleting it is final.

Add a location, import, export and delete are also on the dashboard's Collections tab (**Add
collection…**, **Import…**, and **Export…** / **Delete…** / **Remove location…** in each collection's
details; an export can be downloaded straight from the browser). MCP hosts cannot call any of them.

## Collection descriptions

`rag_list_collections` shows each collection's `description` alongside its document count, instead
of always listing every document (see "MCP tools" above). A description can't be generated at index
time -- a collection is just whatever folder of documents someone indexed -- so it starts empty, and
is normally filled in by the calling LLM itself: `rag_describe_collection(collection, description)`
lets it save a short summary right after it has looked at a collection (a search, a grep, or a
`documents=true` listing), for that call and every future one, from any host, to see.

`rag-search describe` (and the dashboard's **Describe…** button on the Collections tab) is the
human side of the same file: a way to see what an agent has set, or to fix one that's wrong or gone
stale. All three go through the same function, with the same checks: `rag-search describe --client
agent …` is refused for a collection `agent` may not see, exactly as the MCP tool would be. Your own
terminal can also describe a collection that is not published yet.

```bash
rag-search describe                             # every collection that has a description
rag-search describe manuals                     # just this one's current description
rag-search describe manuals "Product manuals and setup guides for the X100 series"
rag-search describe manuals --clear             # remove it
```

Descriptions are stored in `<home>/descriptions.json`, capped at 500 characters each (it is returned
on every `rag_list_collections` call, so a long one defeats the point), and are shown to every
client the collection is visible to -- there is no per-client description. `rag_describe_collection`
itself is subject to the same access rules as everything else: a host can only describe a collection
it is authorised to see.

## Models

`rag-search models` lists what this build can use, what is in use, what is already downloaded and
what fits this computer; the dashboard's **Models** tab does the same with buttons.

| Role | Models (Hugging Face ids) |
|---|---|
| embedding (turns text into vectors) | `BAAI/bge-m3` (default, MIT), `Qwen/Qwen3-Embedding-0.6B`, `-4B`, `-8B` (Apache-2.0) |
| reranker (orders the best hits) | `BAAI/bge-reranker-v2-m3` (default, MIT), `BAAI/bge-reranker-base`, `cross-encoder/ms-marco-MiniLM-L-12-v2`, `mixedbread-ai/mxbai-rerank-base-v2`, `-large-v2`, `Qwen/Qwen3-Reranker-0.6B`, `-4B` |

Any other Hugging Face id can be given too (`rag-search models set embedding ORG/NAME`): it is loaded as a
plain sentence-transformers model (embedding) or cross-encoder (reranker) and tested first. Presets set both
roles at once: `default`, `qwen3-small` (0.6B + 0.6B), `qwen3-large` (4B + 4B).

```bash
rag-search models                                   # list, with size, licence, on-disk state and memory fit
rag-search models set reranker mixedbread-ai/mxbai-rerank-base-v2
rag-search models set embedding Qwen/Qwen3-Embedding-0.6B
rag-search models use qwen3-small                   # both at once
rag-search models download [MODEL_ID ...]           # fetch without switching (default: the two in use)
rag-search models verify [embedding|reranker]       # load the models in use and run a relevance test
rag-search models limit 8                           # memory the models may use, for the fit check (off = none)
rag-search models status | cancel                   # the running download / switch
```

What a switch does, in order: shows what it will cost and asks (`--yes` skips the question); downloads the
model from Hugging Face into its cache (progress is shown; an interrupted download resumes); loads it once in a
separate process and checks that it puts the relevant passage first for three test questions (nothing is
changed if it does not, `--no-verify` skips this); writes the choice to `config.json` (`models.embedding`,
`models.reranker`, so the daemons, the indexer and the CLI all agree; `RAG_SEARCH_MODEL` /
`RAG_SEARCH_RERANK_MODEL` still override it); then applies it.

"Downloaded" means a usable copy is in the cache: a weights file, a `config.json`, and every shard that the
weights index names (an index that names shards the repository does not contain, while it holds a different
weights file, is ignored: `mlx-community/Qwen3-VL-4B-Instruct-4bit` ships such a leftover index). The Models
tab and `rag-search models` say why a copy is not usable, and a download that finishes without producing a
usable copy fails with that reason instead of reporting success.

* **Reranker**: the running search daemon loads the new one next to the old one and swaps it in. No
  re-indexing, no restart, searches are never interrupted. If it cannot be loaded, the old one stays and the
  choice is undone.
* **Embedding model**: every stored vector changes, so **every document is embedded again**. The switch shows
  how many documents and roughly how long (scaled from the last embedding times), then starts an indexing run;
  conversion is reused (only the vectors are rebuilt). Search keeps using the old index *and the old model*
  until every document is done: only then is the new index published and the search daemon picks up the new
  model together with it. If the run is cancelled half way nothing is published; `rag-search index new` finishes
  it, and `rag-search doctor` and the Models tab say so. `--no-reindex` only saves the choice.
* **Memory**: each row shows the estimated memory of the search daemon with that model (weights + index +
  about 1.5 GB). Half precision is used on Apple GPUs; cross-encoders and Intel Macs run in full precision.
  A model that does not fit 60% of the installed memory (or `models limit`) is marked and needs `--force`.
  Limits are used for this check only; nothing caps a running process.
* The Qwen3 models need `transformers >= 4.51`; a row says so, and a switch is refused, when the installed
  version is older (`rag-search doctor` shows it too). Qwen3-Embedding gets its documented query instruction automatically. Qwen3-Reranker
  is an LLM that answers "yes"/"no"; its score is the probability of "yes".
* Only the two defaults have been tested on real documents by the author; the others are catalogue entries
  based on the model cards. Try one on your own questions and switch back if it is not better. Models are
  downloaded only when you ask (or by `install.sh` / `rag-search setup`, which fetch the two in use); search
  itself never uses the network.
* `RAG_SEARCH_DTYPE=float32|float16|bfloat16` forces the weight precision if a model gives invalid numbers on
  your device.

## Configuration

`rag-search config init` writes `<data>/config.json` with the defaults (every key optional):

| Key | Default | Meaning |
|---|---|---|
| `search.idle_exit_seconds` | `0` | `0` = the search daemon never exits by itself |
| `search.prewarm` | `true` | load every published index at start / reload (else lazily) |
| `indexer.idle_exit_seconds` | `0` | same for the indexer daemon (never exits while a run is active) |
| `indexer.jobs` | `0` | parallel conversion workers; `0` = 1 on ≤ 17 GB RAM, else 2 |
| `indexer.auto_publish` | `true` | publish + reload the search daemon after a successful run |
| `models.embedding` / `models.reranker` | `""` (= the defaults) | the models in use; set with `rag-search models set` or the dashboard rather than by hand |
| `models.memory_limit_gb` | `0` | memory the models may use for the fit check; `0` = 60% of the installed memory |

### Pipeline tunables: `rag-search config set`, the Settings tab, and the Models tab's Advanced section

Every other knob that shapes a search or an indexing run — pool sizes, RRF k, default stages and
top_k for search; chunk size/overlap and the docling conversion settings for indexing; batch
sizes, max sequence length, dtype and device for the models — lives in the same `config.json`,
under `search`, `indexer` or `models`. `0` (numbers) or `""` (choices/text) always means "use the
built-in default", the same convention as the keys above.

```bash
rag-search config set --retrieval-pool 60 --rrf-k 80      # search: applies to the very next search
rag-search config set --chunk-size 700 --ocr smart         # indexer: applies to the next indexing run
rag-search config set --embed-batch 16 --device cpu        # models: needs `rag-search daemon restart`
rag-search config show                                     # the merged config, as JSON
```

Every one of these tunables — its default, a one-line "what it means", a longer "impact" note,
and which of the three tiers above it belongs to — is described once, in `spec.TUNABLES`, and
`rag-search config set --help`, this table's longer cousin in [ARCHITECTURE.md](ARCHITECTURE.md),
and the dashboard all read from that same list, so they cannot drift apart. The dashboard's new
**Settings** tab has one card each for the search and indexing tunables (label and "what it
means" always visible; "impact" behind an ⓘ icon next to the label); the **Models** tab's
*Advanced: hardware / runtime* section has the model-runtime ones, next to the model they affect.
Both tabs, and `config set`, write the same `config.json` — there is exactly one place these
values live, however you change them.

Precedence, for any one tunable: an actual environment variable of the matching name (e.g.
`RAG_SEARCH_OCR`, `RAG_SEARCH_EMBED_BATCH` — see the tables below) wins over `config.json`, which
wins over the built-in default. Search tunables have no environment-variable form (they are read
fresh from `config.json` on every search, so there is nothing an env var would add); indexer and
model tunables do, since they end up as a subprocess/process environment variable at the point a
worker or a daemon starts.

Environment variables (override `config.json` where both exist):

| Variable | Default | Meaning |
|---|---|---|
| `RAG_SEARCH_HOME` | `~/Library/Application Support/rag-search` | data folder (workspace, serving, run, jobs) |
| `RAG_SEARCH_CLIENT` | `cli` | client identity used by the CLI (`cli` = administrator; set e.g. `agent` to test what that client sees) |
| `RAG_SEARCH_JOBS` | auto | parallel conversion workers (each ~1–2 GB RAM) |
| `RAG_SEARCH_IDLE_SECONDS` / `RAG_SEARCH_PREWARM` | `0` / `1` | search daemon overrides |
| `RAG_SEARCH_RERANK` | `1` | `0` = skip the reranker (faster, less precise) |
| `RAG_SEARCH_OCR` | `force` | `force` = render every PDF page and read it with OCR instead of trusting the PDF's own text layer (best for tables and odd fonts; slower); `smart` = `force` only for PDFs whose own text layer looks unreliable (scans, garbled fonts), `auto` for the rest (much faster on born-digital PDFs); `auto` = OCR only the images inside PDFs (scanned pages, figures); `off` = use only the PDF's own text. **With `RAG_SEARCH_ROUTING=pages` (the default since 0.9.0) this setting only switches OCR on or off**: pages with a clean text layer are read without forced OCR, scanned or garbled pages with full-page OCR |
| `RAG_SEARCH_PDF_BACKEND` | `pypdfium2` | how docling opens PDFs: `pypdfium2` (recommended with `force`), `docling-parse` (docling's own parser, newest installed generation) or `default` (leave the choice to docling) |
| `RAG_SEARCH_OCR_ENGINE` | `auto` | `auto` (docling picks), `ocrmac` (Apple Vision, macOS), `rapidocr`, `easyocr`, `tesseract`, `tesserocr` |
| `RAG_SEARCH_OCR_LANG` | engine default | comma-separated languages in the engine's own spelling (`en-US` for ocrmac, `en` for easyocr, `eng` for tesseract) |
| `RAG_SEARCH_TABLE_MODE` | `accurate` | TableFormer mode; `fast` trades table accuracy for speed |
| `RAG_SEARCH_THREADS` | auto | threads docling may use for one document; the indexer divides up to 8 cores between the parallel workers (docling's own default is 4) |
| `RAG_SEARCH_DOC_TIMEOUT` | `2700` (45 min) | seconds one document may take; longer means it is reported as failed and skipped (retried on the next run). `0` = no limit |
| `RAG_SEARCH_LAYER_FILL` | `fill` | a digital PDF page is compared with the PDF's own text layer; with `fill` the lines docling left out (table cells, notes, diagram labels) are appended to the page, with `report` the comparison is only recorded, `off` skips it |
| `RAG_SEARCH_OCR_FIRST` | `off` | `auto` = a scanned PDF page whose image says clean print (resolution, contrast, sharpness, skew, speckle, no ruled table) is read by docling's OCR first and goes on to the document reader only when the scan-side gate doubts the text (`expected_size`, `plausibility`, `column_types`, empty, runaway); the page record says `route` and `escalated_from`. Needs the document reader; default `off` until `rag-search bench route` and a real run agree |
| `RAG_SEARCH_ROUTING` | `pages` | `pages` = each PDF page is read the way it needs (text-layer pages without forced OCR, scanned pages with full-page OCR, a page cache so nothing is read twice, a quality gate); `document` = one docling call per file as before (also the automatic fallback if routing fails for a document) |
| `RAG_SEARCH_VLM` | `auto` | `auto` = scanned pages, large pictures in PDFs and image files are read by a vision-language model (the *document reader*, Apple Silicon, `pip install "rag-search[mac-vlm]"`, model downloaded with `rag-search models download --reader`) when it can run, with docling OCR as the per-page fallback; `off` = never start it |
| `RAG_SEARCH_VLM_MODEL` / `RAG_SEARCH_REPAIR_MODEL` | catalogue default | the document reader / repair model (Hugging Face ORG/NAME); `rag-search models reader MODEL_ID` stores the choice |
| `RAG_SEARCH_TESSERACT` / `RAG_SEARCH_TESSERACT_LANG` | `auto` / `mar+hin+eng` | the last-resort page reader (plain text, no tables) for a page the document reader and the repair model could not read; `off` switches it off, a language that is not installed is left out. `scripts/install.sh` installs Tesseract and its Marathi / Hindi data with Homebrew (`--no-tesseract` to skip) |
| `RAG_SEARCH_REPAIR` | `auto` | `auto` = a table cell on a scanned page that breaks the table's arithmetic (a running balance, a total) is cut out, read again and replaced when a second, independent reader (Apple Vision, `ocrmac`) and the arithmetic agree; `off` = suspect pages are only flagged low-confidence |
| `RAG_SEARCH_REPAIR_SECOND` | `auto` | the independent second reader: `auto` (ocrmac when it can run), `off`, or `module:attr` (for tests) |
| `RAG_SEARCH_VLM_PAGE_TIMEOUT` | `300` | seconds one page may take in the document reader before it is stopped and the page is read by docling OCR |
| `RAG_SEARCH_VLM_BACKEND` / `RAG_SEARCH_VLM_FREE_GB` | `mlx` / measured | backend of the reader (`mlx` or `module:attr`, for tests) / free memory to assume (the reader needs the model plus 2 GB free) |
| `RAG_SEARCH_DOCLING_BATCH` | `8` | pages per batch for docling's layout, table and OCR models (`0` = docling's own 4); changes speed and memory, never the text |
| `RAG_SEARCH_PIPELINE` | `standard` | `vlm` = docling's own vision-language pipeline reads whole pages (experimental, slow on CPU; turns page routing, the document reader, gate and repair off) |
| `RAG_SEARCH_ALLOW_UNSAFE_TORCH_LOAD` | `0` | `1` = allow `torch.load` for non-BAAI models on PyTorch < 2.6 (Intel Macs) |

The conversion settings (`RAG_SEARCH_OCR*`, `PDF_BACKEND`, `TABLE_MODE`, `PIPELINE`) are recorded with every
converted document: changing one converts and re-indexes the affected documents again on the next
`index new`. The daemons read the environment they were started with, so set the variables and
then run `rag-search service install` (or `daemon restart`). Check the active settings with
`rag-search doctor` (lines "conversion settings" and "PDF backend").

**Photographed pages that come out empty.** A page photographed with a phone (a passbook, a letter) is, to
docling alone, one big picture: its Markdown export keeps `<!-- image -->` and the picture's class name
("Other") and leaves out the text inside it. Since 0.9.1 the text docling did find inside such a picture is
used, and since 0.9.4 a page that docling still cannot read is read by Apple Vision (`ocrmac`, installed with the
document reader runtime) as plain text, with the page record naming `apple-vision` as its reader. A document whose
Markdown holds nothing but those placeholders is reported as *no text* with the
reason ("the document reader did not read any page …; install it / choose it on the Models tab") instead of being
indexed as an empty document. The proper fix is the document reader: the Models tab's *Document reader* card
lists what is missing (an Apple Silicon Mac, the optional runtime `mlx-vlm` -- `rag-search models runtime
install` or the *Install* button --, the model weights) and the Overview shows it under *Needs attention*. The
runtime is an optional extra (`rag-search[mac-vlm]`) because it is Apple-only and large. `install.sh` adds it on
Apple Silicon Macs (so an upgrade, which rebuilds the `uv tool` environment, keeps it); the *Install* button and
`rag-search models runtime install` add it later with `uv pip install` into the environment rag-search runs in
(that environment has no pip; uv is looked for on the PATH and in `~/.local/bin`, `~/.cargo/bin`, Homebrew). The
embedding model and the reranker need no extra package, only their weights.

The defaults (`RAG_SEARCH_OCR=force`, `RAG_SEARCH_PDF_BACKEND=pypdfium2`) are the same as running
`docling convert --force-ocr --pdf-backend pypdfium2`. To compare on one file:
`rag-search convert file.pdf -o new.md`, then
`RAG_SEARCH_OCR=auto RAG_SEARCH_PDF_BACKEND=default rag-search convert file.pdf -o old.md`. To get the pre-0.3.1
behaviour back (faster on PDFs with a good text layer), set those two variables and run `rag-search index new`.

**Conversion is slow?** `force` is the expensive part: every page is rendered as an image and read by OCR,
typically several times slower than `auto` (the PDF backend itself is not slower). In order of effect:

1. `RAG_SEARCH_OCR=smart` - looks at a sample of pages with pypdfium2; a PDF whose text layer is clean
   is converted like `auto`, a scan or a PDF with garbled text is still OCRed on every page. The decision
   is printed (`note: x.pdf: smart: ...`) and shown as `OCR auto|force` by `rag-search convert`.
   It gives up nothing for PDFs that are really scans; it can only differ from `force` on PDFs whose
   text layer looks clean but reads tables worse than OCR does. Check on your own documents (below).
2. `RAG_SEARCH_OCR_ENGINE=ocrmac` (macOS) - Apple's Vision OCR is much faster than the engines docling
   uses otherwise, and `install.sh` installs it by default on macOS since 0.7.4 (`rag-search doctor`
   line "OCR" says what is installed). Without it, a document that needs OCR doesn't just fall back to a
   slower engine - it fails to convert at all, so this isn't purely a speed knob. On an install from
   before 0.7.4, add it with `uv tool install --force --with ocrmac <path to the wheel/zip you installed
   from>` or by re-running the current `install.sh`.
3. `RAG_SEARCH_TABLE_MODE=fast` - only if the tables come out well enough.
4. Since 0.3.2 the models are loaded once per process instead of once per document, cores are shared between
   the workers (`RAG_SEARCH_THREADS`), and a document that runs past `RAG_SEARCH_DOC_TIMEOUT` (45 minutes)
   is given up. Since 0.4.0 docling's layout, table and OCR models also take 8 pages per call instead of 4
   (`RAG_SEARCH_DOCLING_BATCH`; only speed and memory change, never the text). The three switches above
   change the text, so they stay opt-in: measure them on your own files first.

Measure instead of guessing: `rag-search convert file.pdf --compare` converts the file once with your current
settings and once with each faster option that applies here (OCR `smart`; `smart` + Apple Vision when `ocrmac`
is installed; `smart` + fast tables), each in a fresh process, and prints the time, the speed-up and how much of
the text is identical to the current settings. The outputs are kept in `file-compare/` for a `diff`.

Compare settings on one document; each run prints its time and the OCR decision:

```
rag-search convert file.pdf --mode force -o force.md
rag-search convert file.pdf --mode smart -o smart.md
rag-search convert file.pdf --mode smart --engine ocrmac -o smart-mac.md
diff force.md smart.md | head -50
```
| `RAG_SEARCH_DEVICE` | auto | force `mps`, `cuda` or `cpu` |
| `RAG_SEARCH_EMBED_BATCH` | `32` | embedding batch size |
| `RAG_SEARCH_MODEL` / `RAG_SEARCH_RERANK_MODEL` | from `config.json`, else bge-m3 / bge-reranker-v2-m3 | model ids; they override `models set` |
| `RAG_SEARCH_DTYPE` | auto | force the weight precision: `float32`, `float16` or `bfloat16` |
| `RAG_SEARCH_DOCLING_PYTHON` | unset | python of a separate environment that has docling |

The daemons started on demand inherit the environment of whoever starts them; with
`service install` the launchd job captures the variables listed above at install time.
To change the models use `rag-search models` (see "Models"): the daemon keeps serving the index it has,
with the model that built it, until a re-embedded index is complete.

## Supported files

`.pdf .docx .pptx .xlsx .html .htm .csv .adoc .md .txt .png .jpg .jpeg .tif .tiff .bmp .webp`.
Legacy `.doc`/`.xls`/`.ppt`/`.rtf` are not read (docling only reads modern Office formats reliably); a
file with any other extension is skipped and shown in the Indexing tab / `index status` as
`unsupported_extension`, never silently dropped. To index a legacy file, save a modern copy yourself (Word,
LibreOffice) in a folder that is registered as a location: rag-search never changes your files.
PDF pages are OCRed by default (`RAG_SEARCH_OCR`, `RAG_SEARCH_PDF_BACKEND`). Images inside
documents are not stored: docling writes a placeholder, so only text found in them by OCR is
searchable. Two files with the same name in one folder (e.g. `a.pdf` and `a.docx`) collide; the
second is reported as an error, so rename one.

Changing a file is picked up by `index new` (SHA-256 of the source decides). Deleting a file
removes it from search after the next `index new`.

## Troubleshooting

* **`rag-search doctor`** lists problems with imports, devices, models, folders and daemons.
* **`rag-search daemon status`** shows both daemons; logs are in `<data>/run/search.log` and
  `indexer.log` (`rag-search paths search_log`), per-run worker logs in `<data>/jobs/`.
* **Claude doesn't show the tools** – quit Claude Desktop completely (Cmd-Q) and reopen; check
  `~/Library/Logs/Claude/mcp-server-rag-search.log`; `rag-search mcp-config` shows what is registered.
* **Search says `warming_up`** – the search daemon is still loading its models (first start after
  install can take minutes); `list` and `grep` work meanwhile.
* **Indexing seems stuck** – `rag-search index status -f`; `index new --restart` starts over. If one
  file takes very long, see "Conversion is slow?" above; a file that passes `RAG_SEARCH_DOC_TIMEOUT` is
  reported as failed and the run goes on with the next one.
* **The indexer daemon stopped** – `rag-search daemon status`, then the last lines of `indexer.log`
  (`rag-search paths indexer_log`): since 0.3.2 it says why ("received SIGTERM", "shutdown requested by a
  client"), and the interrupted run's record says the same (`rag-search index status`). A run whose
  worker was killed by the system shows "worker was killed by SIGKILL", which usually means out of memory:
  set `RAG_SEARCH_JOBS=1`. The daemon does not stop by itself (`idle_exit_seconds` is 0), and starting any
  `rag-search index ...` command brings it back.
* **"Operation not permitted" on documents** – daemons started by launchd (`service install`) do
  not inherit Terminal's access to `~/Documents`, `~/Desktop` or `~/Downloads`. Grant Full Disk Access to the tool's Python.
* **Out of memory while indexing** – set `indexer.jobs` to 1 and lower `RAG_SEARCH_EMBED_BATCH`.
* **"A module that was compiled using NumPy 1.x cannot be run in NumPy 2.x" / "Failed to initialize
  NumPy: _ARRAY_API not found"** – Intel Mac with NumPy 2 installed next to the old PyTorch. Re-run
  `./install.sh` from release 0.2.1 or later (it pins `numpy<2`), or by hand:
  `uv tool install --force --python 3.12 --with "numpy<2" --with "transformers>=4.44,<5" --with "huggingface_hub>=0.30,<1" --with "mcp>=1.12,<2" rag_search-*.whl`.
* **"PyTorch >= 2.4 is required but found 2.2.2" / `import sentence_transformers` fails with
  `NameError: name 'nn' is not defined`** – Intel Mac with transformers 5. Same fix as above
  (release 0.2.1 or later does it automatically).
* **"we now require users to upgrade torch to at least v2.6" while indexing (CVE-2025-32434)** –
  Intel Mac (PyTorch 2.2.2). Release 0.2.1 and later allow `torch.load` for the official `BAAI/*`
  models only; set `RAG_SEARCH_ALLOW_UNSAFE_TORCH_LOAD=1` to allow other models at your own risk.
* **Start over** – `rag-search daemon stop`, then delete `<data>/indexer_workspace` and `<data>/serving`.

## Modify the source

The wheel is pure Python and the release also contains the full source distribution.

```bash
./install.sh --dev ~/src/rag-search     # unpack the sdist and install it editable
cd ~/src/rag-search
python -m unittest discover -s tests/portable -t .
```

See `ARCHITECTURE.md` (design, on-disk format, protocol) and `CONTRIBUTING.md` (code map,
conventions). No license has been chosen yet: all rights are reserved by the author.
