# Document conversion: implementation plan

Status: plan (2026-10-02), not started. The *why* and the evaluated options are in
`document-conversion-analysis.md`. This file is the *how*: modules, data model, tracking, UI,
phases. Pictures: `document-conversion-flow.svg` (the flow), `document-conversion-processes.svg`
(processes and hardware), `document-conversion-ui-mockup.svg` (the dashboard).

## 1. Goals and decisions

**Goals**
1. Every page takes the right branch:
   - born-digital → docling (2a);
   - scanned, photographed or garbled pages and image files → document VLM (2b);
   - pictures embedded in Office/PDF files → the same VLM reader (2c);
   - then the quality gate (3). Clean pages go straight on. Suspect or failed pages go to
     **repair (4, Qwen3-VL)**, which ends in *pass (repaired)* or *low confidence*;
   - then reconcile (5).

   2b and 2c are separate branches with their own counts, timing and results. They share one
   VLM reader process (one model load, one GPU queue). They differ in the unit read (a whole
   page vs a picture inside a page or an Office file) and in where the result goes (2b's
   result is the page; 2c's is merged back into a page that 2a already converted).
2. **Everything is traceable.** For every document and every page: which branch it took, which
   tool and model read it, what the checks said, how long each step took and what it cost
   (CPU time, GPU time, tokens, memory). Live while a run is going, and kept afterwards.
3. The dashboard shows this visually: run totals, a live pipeline, CPU/GPU lanes, a per-page
   branch strip for every document, and a drill-down to one page. The Architecture tab shows
   the flow chart with the tools behind each step.
4. It fits the existing indexing pipeline: same daemons, same job records and event log, same
   publish. Nothing changes for search clients, except a low-confidence flag on hits.

**Decisions already taken**
- Local only. GPU work runs natively on the Mac (MLX). No model server, no container for it.
- **No laptop modes**: no plugged-in/battery switch, no background priority. The run uses the
  CPU for 2a and the GPU for 2b and 2c. Knobs: the existing worker count (`indexer.jobs`) and which
  reader/model to use.
- Source documents stay read-only. All new artefacts live under `indexer_workspace/`.
- Tracking comes first (phase P0). Today's pipeline is instrumented before any new engine
  exists, so the dashboard and the baseline numbers are ready when the new branches arrive.

## 2. What changes in the indexing run

| Today (`core/indexer.py`) | After |
|---|---|
| Phase 1, pool of 2 processes: convert (docling) → chunk → `nodes.json`, per document | Phase 1, pool of 2: **profile pages (1)**, convert digital pages with docling (2a), queue scanned pages (2b) and embedded pictures found by 2a (2c). Documents with only digital pages finish in the pool, as today. |
| — | **VLM reader** (2b and 2c): one child process of the worker, started when the first scanned page or picture is queued, `mlx-vlm` + document VLM, model loaded once, reads pages and pictures as they arrive (§3, analysis §12). |
| — | **Per-document barrier** in the worker. When all of a document's pages are read: assemble the `DoclingDocument`, run the **gate (3)**, mark pages for repair. |
| — | **Repair (4)**: a second child process after the reader has finished (one GPU model at a time), only if something failed. Then gate again. |
| — | **Reconcile (5)**: page order, tables continued across pages, provenance. Markdown is rendered from the `DoclingDocument`. |
| chunk inside the pool task | chunk after reconcile (pool task, pure Python, unchanged chunker plus page ranges for merged tables) |
| Phase 2 embed, Phase 3 merge, publish | unchanged |
| events: `stage`, `doc`, progress `phase` | + `page` events, branch/outcome/cost fields on `doc`, a `conversion` block in progress (§4) |

The process picture is `document-conversion-processes.svg`. Cancelling a run still kills the
whole process group, the reader included. Pages already read stay in the **page cache**, so a
cancelled or crashed run loses no VLM work.

## 3. Modules

New package `rag_search/core/conversion/`. Heavy modules import docling, mlx and PIL lazily,
only inside the worker processes. The light modules (record types, aggregation, API views) are
stdlib-only and covered by the layering test.

| Module | Role | Weight |
|---|---|---|
| `trace.py` | Page and document records, branch/outcome codes, cost counters, JSON (de)serialisation, aggregation for runs and collections | light |
| `profiler.py` | Step 1: per-page signals (text-layer quality, image coverage, hidden OCR layer, script, rotation, photo/scan, dpi, content hash); images and TIFF frames as pages | heavy-ish (pypdfium2, Pillow) |
| `router.py` | Branch decision from the profile + settings; explains *why* (stored in the trace) | light |
| `docling_reader.py` | 2a: docling standard pipeline on a page range, OCR only for small pictures; finds embedded pictures for 2c; captures docling's per-page confidence | heavy (docling) |
| `raster_reader.py` | 2b/2c child process (shared reader): queue consumer, image preparation (orientation, straightening), layout regions, VLM calls, result → page record | heavy (mlx-vlm) |
| `fallback_reader.py` | 2b without a VLM: docling + OCR with the engine chosen by script | heavy (docling) |
| `gate.py` | Step 3 checks 1–4 (coverage, garbage/script, docling grade, table shape) | light |
| `validators/` | Running balance, column totals; each recognises its own tables | light |
| `repair.py` | Step 4 child process: cell crops + header strip → repair model; independent second read (ocrmac); acceptance rule | heavy (mlx-vlm) |
| `reconcile.py` | Step 5: merge pages, tables continued across pages, Markdown rendering with page markers | medium (docling-core) |
| `page_cache.py` | `indexer_workspace/page_cache/<hh>/<hash>.json`, keyed by page content hash + reader/model version + settings; garbage-collected after each run | light |
| `costs.py` | Wall/CPU timers, peak RSS, MLX peak memory, token counts | light |

`core/docling_convert.py` stays as the standalone entry point (`RAG_SEARCH_DOCLING_PYTHON`) and
becomes a thin wrapper over the docling-only route.

## 4. Tracking: the data model

### 4.1 Branch and outcome codes (one vocabulary for engine, CLI, API and UI)

| Branch | Meaning | Flow-chart path |
|---|---|---|
| `copy` | md/txt used as they are | md/txt → chunk |
| `office` | Office/HTML file read by docling | Office → 2a |
| `digital` | PDF page with a good text layer | 1 → 2a |
| `raster` | scanned/garbled PDF page | 1 → 2b |
| `image` | image file or TIFF frame | 1 → 2b |
| `embedded` | large picture inside an Office/PDF file, found by 2a | 2a → 2c |
| `fallback` | scanned page, image or picture read by docling + OCR (no VLM, or the VLM failed) | 1 → 2b-fallback, or 2a → 2c-fallback |
| `cached` | page result reused from the page cache | (any) → cache |

| Outcome | Meaning |
|---|---|
| `pass` | passed the gate |
| `repaired` | failed a check, fixed in 4, passed again |
| `low` | still failing after 4: indexed, flagged low-confidence |
| `no_text` | nothing to read (blank page, natural photo) |
| `error` | the page could not be read (with the reason) |

### 4.2 Page record (stored in the document's trace)

```json
{"page": 102, "branch": "raster", "outcome": "repaired", "cache": "miss",
 "profile": {"text_chars": 0, "image_cover": 0.98, "hidden_ocr_layer": false,
             "script": "Devanagari", "rotation": 0, "kind": "scan", "dpi": 300},
 "why": "no usable text layer (0 chars, 98 % image)",
 "reader": {"tool": "mlx-vlm", "model": "PaddleOCR-VL-1.6", "rev": "a1b2c3", "mode": "regions",
            "regions": 9, "tables": 2},
 "gate": {"verdict": "suspect", "checks": [
            {"name": "coverage", "ok": true}, {"name": "script", "ok": true},
            {"name": "table_shape", "ok": true},
            {"name": "running_balance", "ok": false, "detail": "row 6: 44,699 + 281.00 ≠ 47,518"}]},
 "repair": {"cells": 1, "accepted": 1, "model": "Qwen3-VL-4B",
            "changes": [{"row": 6, "col": "amount", "from": "2,81.00", "to": "2,819.00"}]},
 "time_s": {"profile": 0.012, "read": 7.2, "gate": 0.004, "repair": 1.9},
 "cost": {"cpu_s": 0.6, "gpu_s": 9.1, "tokens_in": 1830, "tokens_out": 1412, "peak_mb": 2410},
 "chunks": 11}
```

### 4.3 Document summary

The summary goes into `index.meta.json`, so the Collections tab and `collection info` see it
without reading the trace. It is also sent on the `doc` event:

```json
"conversion": {"pages": 166,
  "branches": {"raster": 163, "digital": 0, "embedded": 0, "cached": 3},
  "outcomes": {"pass": 163, "repaired": 1, "low": 2},
  "low_pages": [41, 152], "repaired_cells": 1, "scripts": {"Devanagari": 141, "Latin": 25},
  "time_s": {"profile": 2.1, "read_2a": 0, "read_2b": 1110, "gate": 4.0, "repair": 30, "chunk": 0.4, "embed": 21},
  "cost": {"cpu_s": 48, "gpu_s": 1140, "tokens_out": 236000, "peak_mb": 2410},
  "readers": ["mlx-vlm:PaddleOCR-VL-1.6@a1b2c3"], "trace": "markup/contracts/sample/sample_agreement.trace.json"}
```

### 4.4 Run totals (job record `progress.conversion`, updated live, frozen at the end)

- Pages total, done and per lane.
- Counts per branch and per outcome.
- The VLM queue (2b and 2c): waiting, done, rate in pages/min (rolling median), ETA.
- Per-lane state: what each docling worker and the reader is doing now, and how busy it is.
- Time and cost totals per step.
- Memory per process.

The existing `progress.phase`, `done` and `total` stay, so the current stepper and history keep
working.

### 4.5 What "cost" means here, and how it is measured

| Measure | How | Where shown |
|---|---|---|
| Wall time per step | `perf_counter` around each step per page/document | everywhere (time split bars) |
| CPU time | `time.process_time()` deltas in the process doing the step | page, document, run |
| GPU busy time | wall time inside the model's `generate` in reader/repair (the GPU is the bottleneck there) | page, document, run, GPU lane |
| Tokens in/out | returned by `mlx-vlm` per call | page, document, run |
| Peak memory | `resource.getrusage` RSS per process; MLX's peak-memory counter for the model | lanes, run, document |
| Throughput / ETA | pages per minute per lane; queue × rolling median | live pipeline, VLM queue |

Energy (watts) is **not** measured: on macOS that needs root (`powermetrics`). The analysis'
30–60 W figure stays an estimate.

### 4.6 Events and storage

- `events.jsonl` gets one compact `page` event per page (≈ 250 bytes: page, branch, outcome,
  times, gpu_s, tokens). That is ≈ 1.3 MB for a 5,000-page run.
- The server aggregates the log incrementally: it remembers the file offset per job, so a
  dashboard tick costs only the new lines.
- `markup/<coll>/<doc>.trace.json`: the full per-page trace, written next to the Markdown
  (atomic, like the Markdown).
- `markup/<coll>/<doc>.doc.json`: the `DoclingDocument`, the canonical form.
- Deleting a collection removes both, as part of its workspace.
- Export bundles carry the per-document summary in the manifest, not the full traces.
- Page cache: `indexer_workspace/page_cache/`. Garbage-collected after each run (entries no
  current trace refers to). Its size is shown under System.

## 5. Interfaces

**Python API (`api.py`, stdlib).**
- `conversion_run(paths, job_id)` → run totals + per-lane state (§4.4).
- `conversion_documents(paths, job_id, branch=, outcome=, q=, limit=)` → the document list with
  summaries. It extends `jobs.documents`, with filters by branch and outcome.
- `conversion_trace(paths, collection, doc)` → the full trace (admin only, like
  `collection_info`).
- `conversion_estimate(paths, target)` → profiler-only dry run: pages per branch and
  estimated time and cost, using recent rates or published defaults.

**HTTP (dashboard).**
- `GET /api/conversion/run?job=`
- `GET /api/conversion/documents?job=&branch=&outcome=&q=`
- `GET /api/conversion/trace?collection=&doc=`
- `GET /api/conversion/page-image?collection=&doc=&page=` renders the source page on demand,
  as a small PNG, from the read-only source, through the same allow-list as the other admin
  views.
- `POST /api/conversion/estimate {target}`.
- Read-only mode allows the GETs.

**CLI.**
- `rag-search index status` adds branch/outcome counts and the VLM queue line.
- `rag-search index estimate [PATH]` is the dry run.
- `rag-search collection info NAME` adds a *Conversion* block.
- New `rag-search trace COLLECTION/DOC [--page N]`: the drawer, as text.

**MCP.** No new tools. `rag_search` hits from low-confidence pages carry
`"confidence": "low"`, and the formatted text says "(low-confidence page)", so the model can
caveat a citation. Grep is unchanged.

## 6. Dashboard

![Conversion view mockup](document-conversion-ui-mockup.svg)

### 6.1 Indexing tab: current run

1. **Run totals.** One row of tiles:
   - pages done;
   - pages per branch: digital (2a), scanned (2b), embedded pictures (2c);
   - repaired cells and low-confidence pages;
   - GPU time with tokens, and CPU time.
   Each branch tile carries its colour swatch.
2. **Live pipeline.** The flow chart as a live diagram:
   - Nodes for files, profiler, 2a, 2b, 2c, gate, **4 Repair** and the two outcomes (*Pass*
     and *Low confidence*).
   - 2b and 2c sit inside one dashed box labelled "one shared VLM reader · GPU". 2c is fed by a
     thin band from 2a (pictures found inside Office/PDF files), not from the profiler.
   - The gate sends clean pages straight to *Pass*. Only suspect cells and failed pages (a
     thin band) enter Repair. Repair ends in *Pass* (counted as "after repair") or *Low
     confidence*, so what Repair costs and achieves is visible on its own node.
   - Bands between nodes, as wide as the number of pages taking that path.
   - The running node pulses. The 2b node shows progress, pages/min and time left; the Repair
     node shows pages repaired of pages queued, cells fixed and the second-read model.
   - Repair starts after the reader has finished (one GPU model at a time), so while 2b/2c are
     running the Repair node shows its queue and "waits for the reader".
   - **Clicking a node or band filters the document table** to documents that took that
     branch.
   - Drawn with the same DOM-SVG helpers as the Architecture tab, so it needs no library.
3. **CPU/GPU lanes.**
   - One bar per docling worker and one for the GPU (reader, then repair): busy %, what it is doing now
     (document and page), and the model with its memory.
   - A memory line against the 12–14 GB budget.
   - The VLM queue: pages and pictures waiting, and the next one.
4. **Documents table.** Today's table, extended:
   - **Branch strip**: one cell per page, coloured by branch (embedded pictures in a lighter
     orange), hatched for low confidence, grey for cache hits. Documents with more than 300 pages draw the strip on a `<canvas>`,
     aggregated to the strip's width.
   - **Time split**: profile / docling / VLM / repair / embed, scaled to the slowest document.
   - **Cost**: CPU s, GPU s, tokens.
   - Filters by branch and outcome next to the existing status, collection and search filters.
5. **Document drawer** (click a row):
   - a page grid (hover for a summary, click for details);
   - branch counts, scripts, and time and cost totals;
   - the selected page: profile, branch and *why*, reader and model, every gate check with
     its result, repair changes (old → new), result, and links to the source page image, the
     converted Markdown and the trace JSON.

### 6.2 Indexing tab: run history

- Each finished run keeps its frozen totals: pages per branch and outcome, time per step, GPU
  time, tokens.
- The history table gets a mini branch bar per run, a GPU-time column and a "low" column.
- Selecting two runs shows the difference: per-step time, and pages that changed branch or
  outcome. This is useful after switching the VLM model.

### 6.3 Collections tab

The expanded row (from 0.8.1) gains a **Conversion** section:
- pages per branch and outcome;
- conversion time and cost;
- the readers and models used;
- the documents with low-confidence pages, each opening the same drawer.

*Needs attention* lists low-confidence pages and documents still waiting for the VLM.

### 6.4 Architecture tab: the flow chart with its tools

The *Indexing pipeline* picture is replaced by the new flow chart. Every node lists the tool
behind it, where it runs, and — after the first run — what it typically costs. The model names
come from the live configuration, so the picture cannot drift:

| Node | Tools shown | Runs on | Live figures |
|---|---|---|---|
| 1 Profiler | pypdfium2, Pillow / pillow-heif | CPU | ms per page (median, last run) |
| 2a Docling | docling (Heron layout, TableFormer), pypdfium2, ocrmac for small pictures | CPU, 2 workers | s per page |
| 2b Scanned pages | mlx-vlm + `<configured model>` (size), docling layout for regions, OpenCV straightening | GPU (MLX) | s per page, tokens per page |
| 2c Embedded pictures | the same reader and model as 2b, reading the picture crop | GPU (MLX) | s per picture, tokens per picture |
| 2b / 2c fallback | docling + `<OCR engine per script>` | CPU | s per page or picture |
| 3 Gate | built-in checks, validators (running balance, totals), docling confidence | CPU | share passing |
| 4 Repair | mlx-vlm + `<repair model>` (Qwen3-VL-4B, 8B opt-in), ocrmac (Apple Vision) second read | GPU | s per cell, acceptance rate, pages ending *repaired* vs *low* |
| 5 Reconcile | docling-core (`DoclingDocument`) | CPU | — |
| Chunk / embed / publish | chunker, bge-m3 (MPS), hard-link publish | CPU / GPU | as today |

- A second picture shows processes and hardware, the same as `document-conversion-processes.svg`.
- A node whose tool is not installed (no `mlx-vlm`) is drawn dashed and labelled "not
  installed: fallback in use".

### 6.5 Models tab

The document VLM and the repair model join the embedding model and the reranker:
- download/switch buttons;
- size, and whether it fits (the existing fits-this-computer check);
- the extra it needs (`rag-search[mac-vlm]`).

Switching the document VLM explains the cost: "pages read by the VLM are read again; N pages,
about M minutes", using the estimate API.

### 6.6 Colour and accessibility

- Branch colours are the first three slots of the validated reference palette, one per branch
  across the whole dashboard:
  - digital/docling `#2a78d6`;
  - scanned/VLM `#eb6834`;
  - repaired `#1baf7a`.
- They pass the colour-blind checks for any pair, in light and dark.
- **Hardware is marked with a chip, not a colour.** Every node and lane that runs on the Mac GPU
  (2b, 2c, 4, embed) carries a dark `GPU` chip; CPU nodes (1, 2a, 3, 5) carry an outlined `CPU`
  chip. Colour keeps meaning *what happened to the page* (digital, VLM-read, repaired), so
  Repair stays green although it is a VLM, and the chip says it uses the GPU. The same chips
  appear in the Architecture tab and in the lane labels.
- Cache hits are neutral grey. Low confidence is the status red **plus a hatch and a "!"
  label**, never colour alone.
- Every strip cell and band has a text tooltip, and every chart has a table view (the
  documents table).

## 7. Phases

Sizes: S ≈ 1–2 days, M ≈ 3–5, L ≈ 6–10 of focused work.

| Phase | Content | Exit criteria | Size |
|---|---|---|---|
| **P0 Tracking on today's pipeline** (**done in 0.8.3**, see "P0 as built") | `trace.py`, `costs.py`; page/doc records for the current docling path (branches `office`/`digital`/`raster` from a minimal profiler, all read by docling); docling confidence captured; `page` events; `conversion` progress block; API + HTTP; documents table with branch strip, time split, cost; drawer; Collections *Conversion* section; Architecture tab flow chart (future nodes dashed); `index estimate` | dashboard shows true per-page branches and costs for a real run; baseline numbers for the corpus | M |
| **P1 Measurement (C0)** | gold set from this corpus (passbooks with their hand transcriptions, statements, deeds, images, digital controls); conversion benchmark in the Playground (sandbox, never production); metrics: numeric-cell exact match, balance pass rate, CER, TEDS, s/page, memory | baseline report; model shortlist chosen with numbers | M |
| **P2 Option A: routing** | full profiler (script, photo, dpi, hidden OCR layer); per-page routing; docling without forced OCR on digital pages; raster pages via `fallback_reader` with OCR chosen by script; gate checks 1–4; `DoclingDocument` JSON; page cache; live pipeline view | no forced-OCR damage on digital gold pages; Devanagari readable; failures visible | L |
| **P3 Option B: document VLM** | `raster_reader` child process; `mlx-vlm` extra; model management in the Models tab; image files (§13 of the analysis); embedded pictures (2c); CPU/GPU lanes, queue, ETA, tokens | numeric-cell exact match on scanned gold pages ≥ target set in P1; peak memory within budget | L |
| **P4 Option C: validators and repair** | running balance, totals; `repair` child process; acceptance rule; reconcile tables across pages; chunk page ranges; low-confidence flag in search results; *Needs attention* | passbooks need no hand transcription; balance checks pass on gold; zero accepted repairs that disagree with the second read | L |
| **P5 Hardening and release** | performance tuning (batch sizes, regions vs whole page), export/import of the new metadata, upgrade path, docs, screenshots; release 0.9.0 | full test suite, Mac sync, design docs updated | M |

The order gives value early: P0 makes today's pipeline transparent and measurable, and every
later phase lands in a dashboard that already shows it.

### P0 as built (0.8.3)

Implemented as planned, with these deviations (all recorded in ARCHITECTURE.md 5.1.1):

- **Branches are observational.** Every page is still read by docling; the branch is what
  `router.decide` implies from the profile. Outcomes are only `pass`, `no_text`, `error`.
- **No per-page times and no `page` events.** docling converts a whole document, so a page has
  no time of its own. Times are per document (profile, convert, chunk, embed); `doc` events carry
  the document's conversion *summary* (pages per branch/outcome, a run-length strip, times, cost)
  and the job's progress carries running totals. Per-page events come with P2, when pages are
  read one by one.
- **Cost** = CPU time (own + child processes) and the process's peak memory; no GPU or energy figures yet.
- **Run-length strip** (`d566r2d12`) instead of one record per page in events and meta; the full
  per-page record is `markup/<coll>/<doc>.trace.json`.
- **Lanes** come from `stage` events (now carrying `pid`) read incrementally; there is no embedding
  lane yet because embedding does not write stage events (adding one would change the stage-event
  order the tests pin down).
- **Page image** endpoint (`conversion/page-image`) added for the drawer; confined to sources in
  the docs folder / registered locations.
- Extra: `index status --doc-branch/--doc-outcome`, `rag-search trace`, `collection info` Conversion block.
- Not done in P0 (belongs to later phases): baseline numbers *for this corpus* need one real run on the Mac
  (`rag-search index new --force-md`, then `rag-search index status` / the Conversion card) -- the
  exit criterion "true per-page branches and costs for a real run" is verified on synthetic PDFs
  in the tests and a headless-browser pass over the dashboard; the real-run check is the owner's first run.

### P1 as built (0.9.0 development)

- Gold set at `<home>/conversion_gold/<set>/gold.json` (+ `images/`), runs at
  `<home>/conversion_bench/<set>/<run>.json`; CLI `rag-search bench gold init|list|show`, `bench run|list|show|compare`;
  `api.bench_*`; read-only listing and two-run comparison in the Playground tab. Details: ARCHITECTURE.md 5.1.2.
- `gold init` **pre-fills** the truth with the pipeline's own text and marks it unverified; measuring
  unverified pages is refused unless `--drafts` is given. The owner's labelling work is therefore the
  one step of P1 that code cannot do: **no baseline numbers exist until a gold set has been checked
  against the images.**
- Measures: numeric-cell exact / found-anywhere, CER, table similarity (a cell-sequence edit distance, *not*
  TEDS), balance-validator pass rate, search phrases found, s/page, CPU and peak memory. The
  "grep-hit proxy" of the plan is the *search phrases* list on each gold page.
- Engines are `current` (whole-document docling, today's settings) and `module:attr`; routed and VLM engines register
  in later phases. Engine time per page = engine seconds / pages the engine read.
- Only the stdlib parts (parsing, validators, metrics, gold/run storage, CLI/API/HTTP) are verified by tests,
  with a fake engine; the `current` engine's docling call is not run in CI.

### P2 as built (0.9.0 development)

- Implemented as `conversion/routed.py` + `pagecache.py` + `gate.py`, `docling_convert.convert_range`,
  profiler `ink`/`hash`, `indexer.convert_source_routed`; details in ARCHITECTURE.md 5.1.1.
- **Page Markdown is the canonical intermediate, not `DoclingDocument` JSON** (`doc.json` is not written):
  readers produce one Markdown string per page (`<!-- page N -->`), which is what the chunker, the gate, the
  validators, the cache and the benchmark all consume. This keeps every reader (docling, VLM, repair) on one
  interface and keeps the cache and trace small. Cross-page table reconciliation (P4) therefore works on
  Markdown tables, not on docling's table items.
- New setting `routing` (`pages` | `document`, `RAG_SEARCH_ROUTING`, Settings tab). With `pages` the OCR setting
  only switches OCR on or off (digital pages `auto`, scanned pages full-page). **Fallback is automatic**: a
  document whose routing raises is converted whole, with a note in its trace.
- Scanned PDF pages are branch `fallback` (docling OCR); image files stay branch `image`. The VLM reader
  (P3) will produce branches `raster` / `image`.
- OCR engine / language are not chosen per page by script: the profile has no script for a page without a text
  layer. The configured engine and language apply, and the gate's `script` check flags results that are garbled
  or in the wrong script (for pages with a text layer that disagrees). Choosing the engine from the *result's*
  script and re-reading is a repair-phase decision (P4).
- Per-page times are the run's seconds divided over its pages (docling reports none per page).
- `RAG_SEARCH_DOCLING_PYTHON` (a separate docling environment) keeps whole-document conversion.
- **Not verified here**: docling's real `page_range` behaviour (page numbering of a ranged result, memory over
  repeated calls, OCR-auto vs force results on real pages) -- the tests use a fake reader. `convert_range`
  raises `RangeUnsupported` and the document falls back when the numbering is not as expected. First check on
  the Mac: index a collection with `rag-search index new --force-md`, then `rag-search trace COLL/DOC` and
  the Indexing tab; compare with `RAG_SEARCH_ROUTING=document`.

### P3 as built (0.9.0 development)

- Implemented as `conversion/vlm.py` (parent: `Worker`, `VlmReader`, memory guard, page rendering) and
  `vlm_worker.py` (child, backends `mlx` and `module:attr`), `routed._Converter` (scan pages, embedded
  pictures, image files, docling fallback), `models.VLM_CATALOG`, setting `indexer.vlm`, extra `mac-vlm`;
  details in ARCHITECTURE.md 5.1.3.
- **Whole-page reading only.** The default prompt asks for Markdown with HTML tables for the whole page
  image; reading by layout regions (what PaddleOCR-VL is built for) is not implemented. The catalogue says so
  for that model; the benchmark (`--engine vlm`) decides on your pages.
- **No separate "document finalised after the reader" step.** Reading is inline in the document's conversion
  (the reader process stays loaded between documents), so a document is complete when its conversion returns.
  With `jobs > 1` each worker has its own reader (memory: model + 2 GB each; the guard falls back to docling
  when it is not free).
- **Embedded pictures are PDF-only** (crops of the page render; boxes from the profile). Pictures inside
  Office files stay docling's. Duplicated text is filtered line by line against the page's own text.
- **Image files:** multi-page TIFF frames are pages, EXIF orientation is applied, HEIC/HEIF are supported with
  `pillow-heif` (extensions added to the supported list), a low-resolution image (< 150 dpi, or < 1000 px when
  the file gives no dpi) is read but gated `low` (`low_resolution`), a photograph without text is "no text".
  The same `low_resolution` check applies to scanned PDF pages.
- **Cost fields:** `tokens` and `gpu_s` (seconds inside the model) per page / document / run; there is no
  energy figure. The reader's peak memory is recorded in benchmark runs (`cost.peak_mb`).
- Models: IDs checked on Hugging Face (October 2026): `mlx-community/Qwen3-VL-4B-Instruct-4bit` (3.09 GB),
  `mlx-community/PaddleOCR-VL-1.5-8bit` (about 1.1 GB), `-bf16`. Sizes in the catalogue come from the model
  pages; memory needs are estimates (weights + 2 GB). MinerU2.5 and GLM-OCR from the analysis are **not**
  offered: no MLX build was checked.
- **Not verified here (no Apple Silicon, no model):** the MLX backend itself, real output quality on
  scans/tables, speed and memory (the 3-10 s per page in the estimate is the analysis's assumption, not a
  measurement), the prompt wording, HEIC reading. The child-process protocol, memory guard, timeouts, crash
  restart, fallback, caching, pictures, image files, models and dashboard/CLI/HTTP are tested with a fake backend.
  **First check on the Mac:** `pip install "rag-search[mac-vlm]"`, `rag-search models download --reader`,
  `rag-search index estimate`, index a folder with one scanned PDF and look at the trace (`rag-search trace`)
  and the *Pages read right now* card; compare `rag-search bench run --engine current` / `routed` / `vlm` on
  your gold set.

### P4 as built (0.9.0 development)

- Implemented as `conversion/repair.py` (second-reader interface, `locate`, the accept rule, `repair_cells`,
  `Repairer`), `conversion/reconcile.py` (tables across pages), `tables.replace_cell`, gate violations carrying the
  table number, `vlm.PROMPT_CELL` / `repair_shared`, the hook in `routed._Converter.finish` (and the page cache),
  setting `indexer.repair`; details in ARCHITECTURE.md 5.1.4. The validators (running balance, totals) were built
  in P1/P2.
- **No separate `repair` process.** Repair uses the document reader's process when `models.repair` is the same
  model (the default: Qwen3-VL 4B), so one model is in memory; another repair model gets a second reader process
  (`VlmReader` in `vlm.repair_shared`), under the same memory guard and fallback rules.
- **The second reader is Apple Vision (`ocrmac`),** not a second VLM: it is a different kind of reader (a trained
  recogniser, not a generative model), so a hallucinated digit is unlikely to be shared. It returns phrases; they
  are cut into words in proportion to their characters, which is exact enough to find a cell and is not used for
  the new text (the *model's* crop reading is, and must equal the second reader's). Without it (not a Mac,
  `RAG_SEARCH_REPAIR_SECOND=off`) cells are not repaired and the page is flagged; the whole-page re-read by a
  *different* repair model still runs.
- **Acceptance rule as designed:** a number, not the original text, equal to the second reader, and the arithmetic
  fixed by it. Tested: the rule rejects `unchanged`, `disagree`, `not_a_number`, `not_confirmed`, a cell that
  cannot be found or addressed, and a reader that errors; a rejected attempt is cached with the page so it is not
  retried with the same models.
- **Deviation: no chunk page ranges.** The plan listed `page_end` metadata for tables that span pages. The chunker
  is unchanged: a table across a break stays two page-marked pieces, the continuation is given the header, and
  both pages record the link. A chunk keeps one page number (the citation stays exact) and `CHUNKER_VERSION`
  does not change. A chunk that spans a page break would need a search-result format change for little gain.
- **Cross-page violations are flagged, not repaired** (the boundary figure could be on either page; the crop would
  need both page images). The pages become `low`, the trace names the cell and the page.
- **Low-confidence flag** is stored per chunk (`metadata.confidence = "low"`, only on flagged pages) and shown by
  search results, the CLI, the MCP `rag_search` description and the Search playground chip; the *Needs attention*
  list is the collection's conversion block (documents and pages with `low` outcome, links to the page record),
  with repaired cells and merged tables beside it.
- **Not verified here:** Apple Vision's real output on scans (word/phrase granularity, the origin of its boxes:
  Vision's is bottom-left and `OcrMacSecond` converts it, but this was not run on a Mac), the crop readings of a
  real model, and how many suspect cells real scans yield. **First check on the Mac:** `pip install
  "rag-search[mac-vlm]"`, index a scanned statement, then `rag-search trace COLL/DOC` and open a page with a
  repair record; a number that is wrong in the source and kept (`status` not `fixed`) is the expected safe outcome.

### Status after 0.9.1-0.9.9 (real-document findings; the Mac is now the place to test)

Built on top of P0-P4 (all unit-tested with stand-in readers; **real docling / MLX / ocrmac runs only
through the Mac's own tool environment, see CONTRIBUTING.md "Dev loop"**):

- 0.9.1: one numbered pipeline (`stages.py`, `effective.py`, `GET /api/pipeline`), Settings by stage, Playground
  runs as background jobs with event logs.
- 0.9.2-0.9.3: model "downloaded" is only shown when the cache copy is complete and usable
  (`models.cache_state`); the runtime installs with `uv` (no pip needed); `install.sh` bundles ocrmac, mlx-vlm,
  pillow-heif on Apple Silicon.
- 0.9.4-0.9.5: a photographed page is one Picture to docling and its Markdown export drops the text inside, so
  docling can return placeholders. Placeholder detection plus **Apple Vision as the last resort**
  (`applevision.py`): plain text, no tables, garbles Devanagari. `scripts/try_readers.py` compares readers
  outside the pipeline.
- 0.9.6: VLM replies wrapped in code fences are cleaned (`clean_reply`, prompt version p2); Qwen3-VL 8B (4/8-bit)
  added as reader options.
- 0.9.7-0.9.8: **the `vlm` value of "Conversion pipeline" (docling's own whole-page pipeline) turns page routing
  off**, so the document reader, gate and repair never run; on a photographed passbook only the Apple Vision
  rescue read the pages. Wording of the setting and of the "not read page by page" reason now says so. A
  Playground field cleared with Reset only counts after "Save settings" (a second button was added at the top).
- 0.9.9: gate/validators stack multi-line table headers (Marathi/Hindi/English) before recognising the balance and
  amount columns, and two new `table_shape` findings: a balance column that is mostly empty (numbers slid into
  another column) and consecutive repeated rows.

- 0.9.12 (first full run of `documents` on the Mac, 1386 files): **the run never reached embedding.** Three
  PDFs hit the document timeout; docling abandoned its OCR / layout threads, one of them stuck in an Apple Vision
  text request, so that pool process could not exit and `ProcessPoolExecutor`'s shutdown waited for it forever
  (all documents done, 0 % CPU, "run active" for nine hours). Fixed: the pool is closed with a bound
  (`indexer._shutdown_pool`, 20 s, then terminate) and the worker and CLI child processes leave with `os._exit`
  once everything is written (`worker.leave`). Also from that run: the drawer's "Show the source page" failed for
  every document not embedded yet (the source path was only read from `index.meta.json`, which is written after
  embedding; it now falls back to the trace), the drawer shows the converted Markdown of a page and links the
  whole document's (`conversion_markdown`), and collapsible sections no longer fall shut on the next live update.
  Not fixed here: why the Vision request hangs, and the three documents still time out (retried next run).

- 0.9.13: the first run after installing 0.9.12 failed every document in 16 s (`FileNotFoundError` at
  `posixpath.abspath`, then "partially initialized module 'torch' has no attribute '_logging'" for every
  embedding). Cause: the daemons had been started by `install.sh` from inside `dist/rag-search-0.9.12/`, and that
  folder was deleted and rebuilt a minute later, so their working directory no longer existed. Nothing was wrong
  with the package or the environment, and the workspace was untouched (the Markdown was reused; only embedding
  failed). Fixed at the root: long-lived processes start in the data folder (`paths.detached_start`).
  `scripts/sanity_check.py` asked Apple Vision for Hindi/Marathi, which this macOS does not offer, and called
  `models verify` without a kind; both corrected. The tests are now two folders: `tests/portable/` (fakes; the same result in the cloud and on
  the Mac) and `tests/machine/` (the real tools, Mac only). Nine tests that assumed a machine without the
  document reader now say so themselves (`helpers.no_real_reader`), one that crashed the interpreter by
  importing torch twice is fixed, and the suite no longer leaves daemon processes behind.

- 0.9.14: adding a collection failed for a path pasted with quotes around it (the quoted path had also been
  put in the name field, which is what the error was about). Pasted paths are now cleaned in one place
  (`paths.pasted_path`: surrounding quotes, shell escapes), a missing name is taken from the folder, and a path
  in the name field is taken as the folder. A folder under `~/Library/CloudStorage` (Box) refused its first
  listing once and answered a moment later: the reachability probe now retries within its 10 s and the error
  says why a folder cannot be read. Open: files that such a folder keeps online-only are fetched by the
  cloud app when they are read; how a long run behaves over many of them is not measured.

- 0.9.15: every online-only file of the Box folders failed with `[Errno 11] Resource deadlock avoided` when the
  run was started by the launchd daemon, and read fine from a terminal. Measured with a temporary launchd job:
  a launchd-started process has the macOS I/O policy "materialize dataless files" off (1), a terminal has it
  on (2). `paths.allow_cloud_files` switches it on in every rag-search process (children inherit it); the same
  launchd job then read a placeholder file in 3.9 s. This also explains the folder listing that was refused in
  0.9.14. Consequence to know: indexing such a collection makes the cloud app download every file it reads.

- 0.9.16: a document that cannot be indexed for its own reasons (password-protected PDF, no text after every
  reader had its turn) is remembered by checksum and conversion settings (`outcome.json` in its index folder)
  and not converted again until it changes; it is still reported, with its reason. Decision: timeouts are
  *not* remembered, because the page cache lets a later run get further, and neither is any result where a
  reader did not get its turn. Open: whether to remember a timeout after it has happened several times.

- 0.9.17: `scripts/sanity_check.py --e2e` (and the machine test that runs it) gave the Playground two files with
  `--from`; since an experiment's source is a folder, that failed. It now puts the two generated PDFs in a
  folder. Nothing in the product changed; the Mac-only tier passes (5 tests).

- 0.9.18: three findings from the real documents, fixed together.
  (1) The reader is asked for HTML tables, so some pages came back as `<table>` markup and others (the model's
  own choice) as pipe tables. Every page now goes through `tables.normalize_html_tables`: a table without merged
  cells becomes a pipe table, cell for cell; the chunker keeps HTML tables whole and cuts an oversized one by rows
  with the header repeated (`CHUNKER_VERSION` v2).
  (2) The gate passed a table whose last four columns all held the same balance, and one whose balances sat in the
  deposit column: new column checks (one amount in three or more columns, amounts in the cheque column, `Cr`/`Dr` in a
  debit or credit column).
  (3) **The 4B reader falls into loops** on dense pages (a 36-page scanned deed: 7 pages, 150-195 s each, one line or
  phrase repeated hundreds of times, one page in Bengali script; the gate passed six). By a rough test 56 of 578
  image-read pages in `documents` were runaways (49 passed the gate). New: `degenerate.py` (repeated line or
  phrase, near-total compressibility, empty-table-row runaway, stray script), a `degenerate` gate check, a loop stop
  inside the worker (the text is checked every 64 tokens), and a retry ladder in the reader (repetition penalty 1.2
  and a 3072-token limit, then three strips cut at blank rows). Measured with the real 4B model on the deed: pages
  20, 21 and 24 came out clean after the penalty retry, 21 in 115 s instead of 190 s; healthy page 13 was read once,
  unchanged. Page 22 (Marathi read as Bengali) is a runaway the retry cannot fix: it needs another reader.
  Cached pages that ran away before the guard are read again once (`guard` marker in the cache entry).
  The document-level conversion profile gained `post=2`, which is not part of the page cache's key: installing
  re-converts every document from the page cache, re-reads only the runaway pages, and (with the chunker bump)
  re-embeds everything once. `convert-legacy` was removed: rag-search writes nothing into a source folder.

- 0.9.19: the first real run of the loop guard stopped a healthy page by mistake (a list of repeated "field: value"
  lines is not a loop) and the penalty retry came back shorter than the page it replaced. Fixes: the live test looks
  at the text without tags and needs the last 100 characters seen 15 times in 6000 (or 300 identical lines of
  empty table rows), measured on the 578 stored image-read pages of `documents`: it would stop 39 of the 56
  runaways live (the rest are caught by `assess` after the full generation) and 3 healthy pages; and among the
  readings that are not runaways the most complete is kept, so a false alarm costs time, not text.

- 0.9.20 and the reader experiments (4B and 8B 4-bit, PaddleOCR-VL bf16, Tesseract `mar+hin+eng`) on three PPF
  passbook pages and nine pages of the scanned deed, scored without hand transcription (gate, running-balance
  arithmetic, the share of English words that are real words, Devanagari share, time; the 4B and 8B readers also
  agree on every money amount of the passbook, so the digits were right and only the *columns* were wrong).
  * **Page 2 of the passbook was never verified**: the reader wrote a blank line between the header lines and
    the data rows, so the table was three rows long and the balance check ran over 0 rows. Joined (0.9.20), 24 rows
    are checked and the arithmetic holds; two rows are one cell short (a shifted narration) and the page is now
    flagged.
  * **8B 4-bit vs 4B 4-bit** (machine otherwise idle; the 4B's first timings were taken while two indexing workers
    shared the GPU): the 8B puts the passbook's values in the right columns (pages 2 and 4 pass the gate, page 3 fails
    one balance), needs a retry on 3 of the 9 deed pages against 8 for the 4B, and reads the Marathi on deed
    page 24 that the 4B leaves out. It is about twice as slow per page, drops the Devanagari header lines of the
    passbook, and neither reader can read deed page 22 (a sheet of photographed cards).
  * **PaddleOCR-VL 1.5 bf16** is not usable here: whole pages come back as 16 to 37 characters on three pages and as
    runaways on others, and on a cropped passbook table it writes OTSL cells of junk until the token limit.
  * **Tesseract** (`mar+hin+eng`, psm 4) takes 1 to 13 s a page, cannot loop, matches the 4B on English-word
    validity on the healthy pages (0.79 to 0.90) and finds the Devanagari the VLMs miss on pages 22 to 24; it writes
    no tables. Candidate: a last-resort reader for a page the VLM ladder cannot read, next to the Apple Vision rescue.
  * Candidate for a decision: escalate a scanned page that the gate flags to the 8B (the repair step already
    re-reads a page with a different repair model) instead of re-reading everything with it.

- 0.9.21: the two decisions of the experiments. (1) The repair model is now Qwen3-VL 8B 4-bit by default and
  re-reads any scanned page the gate flags (shifted columns, a runaway, a failed balance), not only a page with a suspect
  cell; the 4B stays the reader, so only the pages that need it pay for the 8B. (2) Tesseract is the last-resort
  reader for a page that is still a runaway, and after Apple Vision in the no-text rescue. `install.sh` checks for / installs
  Tesseract and its Marathi and Hindi data and downloads the reader and repair models; `doctor` and
  `scripts/sanity_check.py` report both.

**First real result** (3-page scanned Marathi/Hindi/English passbook, standard pipeline, Qwen3-VL 4B 4-bit):
Devanagari headings and the cover page are correct; page 2's table has real rows but one row is missing, one
merges two rows, and narration pushes numbers one column left; page 3's table is scrambled (shifted and
duplicated rows). The gate flags both pages (`low`); repair cannot fix a scrambled table.

**Open items:** (1) compare the 8B reader on the dense table pages (`scripts/try_readers.py ... --readers vlm
--pages 2,3 --model mlx-community/Qwen3-VL-8B-Instruct-4bit --show`); (2) if still scrambled, re-read a suspect
table page in horizontal strips of a few rows, or ask for a Markdown pipe table instead of HTML, and keep either
only if it wins on the real pages; (3) check repair on a page with a real single-cell arithmetic violation;
(4) a Playground warning when a pinned `vlm` pipeline disables the reader (today only the help text and the
conversion note say so).

## 8. Testing

- **Fakes, like the existing `FakeEmbedder`.** `tests.helpers:FakeVlmReader` returns
  deterministic text and tables per page image hash. `FakeRepair` does the same for repairs.
  The whole pipeline therefore runs in CI without MLX or a GPU.
- **Fixtures made by the tests.**
  - A digital PDF; a "scanned" PDF made by rendering text to an image with Pillow; a mixed
    PDF.
  - A synthetic passbook image with known balances and one wrong digit, for the validator and
    repair.
  - A multi-page TIFF; a HEIC sample (when pillow-heif is available); a docx with an
    embedded scan.
- **Unit tests:**
  - router decisions and their *why*;
  - each gate check and validator;
  - the acceptance rule;
  - page-cache keys and garbage collection;
  - trace aggregation;
  - cost counters.
- **Integration tests:**
  - branch counts and outcomes in the job progress, `doc` events and `index.meta.json`;
  - cancel during 2b keeps cached pages;
  - fallback when the reader crashes.
- **Layering:** `trace`, `router`, `gate`, `validators`, `page_cache`, `costs` and the API
  views stay stdlib-only. MCP still cannot reach admin views (trace, page images).
- **UI:** endpoint tests (filters, read-only mode, path safety for page images); `node
  --check`; Playwright screenshots of the conversion view and the Architecture tab, light and
  dark, during development.
- **Benchmarks (P1 harness)** run by hand, not in CI. Results are stored with the run so they
  can be compared later.

## 9. Rollout

- Each phase bumps `CONVERT_VERSION`, so affected documents are re-converted.
- Before the first big run, `rag-search index estimate` (or *Estimate* on the Indexing tab)
  shows the number of pages per branch and the expected time.
  - The digital collections (manuals, product_docs, books, guides) re-convert in about
    their current time.
  - `documents` is the long one. Its estimate decides whether to run it overnight.
- Without the `mac-vlm` extra, everything works with the fallback reader (option A quality).
  The Architecture tab says so.
- Export bundles keep the same index format. The new conversion metadata is optional in the
  manifest, so bundles stay compatible in both directions (to be confirmed with a test in P5).

## 10. Risks

| Risk | Mitigation |
|---|---|
| `mlx-vlm` / model API changes | Pin versions in the extra. The reader is behind one interface (`read(page_image, regions) → page record`). A smoke test downloads nothing and checks the import and signatures. |
| VLM invents text or numbers | Region mode + gate + second-read agreement. Accepted repairs must satisfy the invariant **and** agree. Low confidence is visible in search results. |
| Devanagari quality unknown | Measured in P1 before P3 is committed; Tesseract `mar+hin` fallback. |
| Event log / UI too heavy for 5,000-page runs | Compact `page` events, incremental aggregation by file offset, canvas strips for long documents, document list paged as today. |
| Memory pressure with other apps open | Memory check before each model load; one GPU model at a time; fallback reader when short; memory line on the dashboard. |
| Long first run on `documents` | Estimate first. The page cache makes interrupted runs resumable; nothing is lost on cancel. |
| Licences for the external build | Model and tool licences recorded in the Models tab; Surya/Marker/Chandra excluded; MinerU toolkit reviewed before any use. |

## 11. Documents to update as each phase lands

- `ARCHITECTURE.md`: §5.1 (indexing flow), a new conversion section, §2 (data folder:
  traces, `doc.json`, page cache), §7 (failure behaviour).
- `README.md`: Indexing/Collections/Architecture/Models tabs, `index estimate`, `trace`, the
  `mac-vlm` extra.
- `CONTRIBUTING.md`: code map for `core/conversion/`.
- This plan's status and `document-conversion-analysis.md`.
- Copies in `src/rag_search/ui/static/docs/`.
