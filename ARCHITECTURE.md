# rag-search architecture (v0.9)

## 1. Processes

```
┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│ rag-search   │ │ rag-search-  │ │ rag-search-  │ │ scripts /    │   front-ends: short-lived,
│ CLI          │ │ mcp (Claude) │ │ mcp (other)  │ │ other tools  │   stdlib (+ mcp for adapters),
└──────┬───────┘ └──────┬───────┘ └──────┬───────┘ └──────┬───────┘   never load a model
       └────────────────┴───── rag_search.api ─────┴───────┘
                              │  client.py  (JSON lines over Unix sockets, protocol v1)
         ┌────────────────────┴─────────────────────┐
         ▼                                          ▼
┌─────────────────────────┐   reload    ┌─────────────────────────────┐
│ indexer daemon          │────────────▶│ search daemon               │
│ stdlib only, supervisor │             │ numpy/torch, always on      │
│ start/restart/cancel/   │             │ list · grep · search · reload
│ status/follow/publish   │             │ access rules per client     │
└───────┬─────────────────┘             └──────────────▲──────────────┘
        │ spawn (own process group)                     │ reads (hard-linked files)
        ▼                                               │
┌─────────────────────────┐  publish   ┌────────────────┴─────────────┐
│ worker (docling, bge-m3)│ ─────────▶ │ serving/gen-000042/  ◀─ current
│ writes ONLY             │  (indexer) │  index/<coll>/_all, markup/,  │
│ indexer_workspace/      │            │  catalog.json                 │
└─────────────────────────┘            └───────────────────────────────┘
```

Both daemons are single-instance (`flock` on `run/<kind>.alive`), listen on a Unix socket
(`run/<kind>.sock`, mode 0600, directory 0700) and are **always on**: they start on demand the
first time a client needs them (`client.spawn`, serialised by `run/<kind>.start.lock`, which is held until the new daemon owns its
alive-lock, so concurrent callers never start duplicates) or at login
via launchd (`rag-search service install`). `idle_exit_seconds` in `config.json` can make either exit
when idle; the default is 0 = never. Daemons, the detached dashboard, the indexing worker, Playground runs and
model tasks are all started with the data folder as their working directory (`paths.detached_start`), never the
caller's: a process whose working directory is deleted later cannot call `os.getcwd()`, and then path handling,
the process pool and `import torch` fail.

### Indexer daemon (`core/indexer_daemon.py`, stdlib only)

* `start {mode: new|all, path, rebuild, force_md, restart}` – idempotent: while a run is active it
  returns that run (`already_running`), unless `restart=true`, which kills the worker's whole
  process group (SIGTERM, then SIGKILL) and starts a new run. Finished documents are skipped by
  SHA-256, so a restart loses at most the document that was being embedded.
* `cancel`, `status {job_id?, history?}`, `follow {job_id?}` (streamed JSON lines, ends with
  `{"event":"end"}`), `publish` (refused while a run is active).
* Each run is a **worker subprocess** (`python -m rag_search.core.worker <job-id>`) started with
  `start_new_session=True`. The worker reads its spec from `jobs/<id>.json`, writes only to
  `indexer_workspace/` and to `jobs/<id>.events.jsonl`; the daemon tails that file and is the only
  writer of the job record. `run/index.lock` (flock, released by the kernel when a process dies)
  guards the workspace even against a `rag-search index foreground` run started by hand.
* On success (`indexer.auto_publish`) the daemon calls `api.publish_and_reload`: build a generation,
  switch `serving/current`, send `reload` to the search daemon (starting it if it is not running).
* On start it marks stale `running` records `interrupted` and kills orphaned workers (matched by
  pid **and** command line). On shutdown it cancels the active run.
* Job statuses: `queued running succeeded partial failed cancelled interrupted` (`partial` = some
  documents failed; the rest is published).

### Search daemon (`core/search_daemon.py`)

* Binds its socket first, then loads models in a background thread: state
  `starting → loading_models → loading_index → ready` (or `error`). `ping`, `list` and `grep`
  answer immediately; `search` waits up to the client's `wait_s`, then replies `warming_up`.
* `reload` (also triggered by a 5 s watcher on `serving/current`, so a lost message is harmless)
  loads the new generation **off to the side** and swaps it in with one pointer assignment under a
  short lock. Collections whose `manifest_sha` (in `catalog.json`) and embedding model are unchanged
  are **reused from memory**, not re-read or re-tokenized; only changed or new collections are
  loaded (the reply lists `reused` and `loaded`). The embedding model always **follows the published
  index**: a generation built with another model brings its own embedder, which is loaded off to the side and
  swapped in with the generation (an engine given an explicit embedder, as in the tests, refuses such a generation:
  `model_mismatch`). `reload` also applies a changed reranker setting the same way (`prepare_reranker` /
  `install_reranker`); if the new reranker cannot be loaded the old one stays and the reply carries `reranker_error`.
* Enforces the per-client access rules (`policy.py`, read from `access.json`, re-read when the file
  changes) for `list`, `grep` and `search`, and remembers which client names it has served
  (`clients_seen` in `ping`).
* Serves searches **concurrently** (one thread per connection, no daemon-wide lock): each query
  takes one consistent snapshot of (generation, embedder, reranker) from the engine, and only the
  model calls themselves -- embedding the query and reranking -- are serialised (one model on one
  device). BM25 scoring, the vector product and fusion of different queries overlap, so a slow
  many-collection query no longer queues every other client behind it. A collection that is
  loaded lazily is loaded once even when several queries need it at the same moment.

### Front-ends

`rag_search.api` is the only place that knows which socket to call, whether to auto-start, and how
to fall back: `list` and `grep` read `serving/current` directly when the search daemon is down
(same access rules); `index status` hides restricted collections from the client asking (documents
filtered, strings that name them blanked); `index status/cancel` never start the indexer daemon just to say "idle".
The CLI and the MCP adapter are formatting layers over `api`; MCP tools run `api` calls in
`asyncio.to_thread`, and every start lock is taken and released inside one thread function, so a
cancelled MCP call cannot leave a lock held.

`list`'s per-collection reply (`catalog.list_view`, `api.list_collections`) defaults to a compact
shape -- `document_count` and a `description` instead of every document -- since the full listing
used to be sent unconditionally to whichever caller asked, MCP included, and a collection can hold
hundreds of files. `full=True` (the CLI, the dashboard, and the MCP `rag_list_collections`'s
`documents=true`) adds the per-document `documents` list back. `description` comes from
`descriptions.py` (stdlib only): `<home>/descriptions.json`, `{name: text}`, read through a
process-wide cache that re-reads it only when it changes. Nothing writes it automatically -- a
collection is just whatever folder of documents someone indexed, so there is no text to summarize
at index time -- `api.describe_collection` is the only writer (the MCP `rag_describe_collection`
tool, `rag-search describe` and the dashboard's Describe… button all call it), gated by the same
`policy.resolve_scope` a `search`/`grep` request is, so a client can only describe a collection it
is authorised to see (the administrator's `cli` may also describe one not published yet).

**Small metadata files** (`access.json`, `descriptions.json`, `locations.json`) share one set of
helpers in `paths.py`: `write_json_atomic` (temp file + `os.replace`, optional mode -- 0600 for
`access.json`), `file_lock` (an exclusive `flock` on `.<name>.lock` held across a whole
read-modify-write, so two concurrent `rag-search access …` commands cannot lose each other's
change) and `CachedFile` (re-parse only when mtime/size/inode change; `policy.AccessStore` is one).
A path a person types or pastes (a location's folder, an index target, an export or import file, a
Playground source) goes through `paths.pasted_path` first: one pair of surrounding quotes and shell escapes
are shell syntax, not part of the name; spaces, `@` and the like are ordinary characters. `locations.add`
takes the folder's own name when no collection name is given (`paths.name_from_folder`), and a path put in
the name field is taken as the folder.
Collection names are resolved in one place, `catalog.canonical_name`: exact match first, else
case-insensitive, against every collection rag-search knows (published, docs-folder folders,
locations, workspace). `policy.resolve_scope` matches the same way and always returns the real
spelling, de-duplicated -- a request naming a collection twice searches it once.

**Web dashboard** (`ui/`, stdlib only, `rag-search ui`) is a third front-end over `api`, acting as `cli`.
`ui/server.py` is a `ThreadingHTTPServer` on 127.0.0.1 (default port 8765): a sampler thread polls
`api.daemon_status` and `api.index_status` every second (5 s when nobody is watching) and the catalog
and access rules every 2 s, and pushes changes to browsers as Server-Sent Events (`/api/events`; the
page falls back to polling `/api/state`). Actions are POSTs that call the same `api` functions as the
CLI (`index/start|cancel|publish`, `access`, `describe`, `daemon`, `search`, `grep`, and `collection/add-location|import|export|delete`
+ `location/remove`, where delete and remove require the collection's name typed back as `confirm`); an export's file is
offered for download through `GET /api/download?id=` with an id the server minted for that export (an in-memory
registry of the last 50; never an arbitrary path); `GET /api/collection/info?name=`
serves the expanded row of the Collections tab from `inventory.collection_info` (the same as `rag-search collection
info`), fetched only when a row is opened -- never in the polling loop, since it walks the collection's source and
workspace folders -- and refreshed when the generation or the indexing run changes; `--read-only` refuses all but
search, grep and doctor. Its own searches carry `origin: "ui"` so the search daemon does not record
"viewing as agent" as that agent having connected. Doctor runs in a child process, so the dashboard never loads
a model. Access control: loopback-only, `Host` must be loopback:port, POST must be same-origin
`application/json` (body ≤ 1 MB), and every request except `/api/ping` needs the token from
`run/ui.token` (0600) as an HttpOnly SameSite=Strict cookie or `X-RagSearch-Token`; a CSP forbids
external resources. `run/ui.pid` and `run/ui.port` belong to `--detach`.
The pages are plain HTML/CSS/JS in `ui/static/` (no build step; views register in `core.js`).
Live areas are updated in place by `patch()` (`core.js`): new content is merged into the existing nodes.
A `<details>` section keeps the state the user gave it across those updates unless the view passes an
`open` property, in which case the view owns the state.
Everything on the Architecture tab comes from `ui/info.py`, which reads the constants in `spec.py` that
the engine itself imports (RRF k, BM25 k1/b, pool sizes, limits) plus the effective chunking and model
runtime settings, so the diagrams cannot drift. The Indexing tab's *Conversion* card, branch strips,
document drawer (page grid, page detail, source-page image) and the Collections tab's *Conversion*
section come from `ui/static/conversion.js` (colour = branch, GPU/CPU chips = where it runs). The system, indexing, conversion and search pictures are hand-drawn SVG
built with the DOM in `architecture.js` (no library, nothing external, colours from the theme variables);
the daemon boxes carry live status dots; the
field lists of `nodes.json` / `index.meta.json` / `catalog.json` there are checked against real files by
the tests. The Help tab renders the README and this file (copied into `ui/static/docs/` by
`build_release.sh`, checked for drift by a test) with a small escaping Markdown renderer, and builds the
command reference from the argparse tree.

## 2. Data folder

```
<home>/                       $RAG_SEARCH_HOME or ~/Library/Application Support/rag-search
  config.json                 optional settings
  access.json                 collections restricted to specific clients (written by `rag-search access` only)
  descriptions.json           short per-collection descriptions ({name: text}; written only through
                               api.describe_collection)
  locations.json              registered source locations ({name: folder}; `rag-search location`): the
                               only place documents come from -- there is no built-in documents folder
  .<file>.lock                read-modify-write locks of the three files above
  indexer_workspace/          written by the worker (and by collection import/delete, under index.lock)
    markup/<coll>/<doc>.md    page-annotated Markdown  (+ .md.sha256 of the source)
    markup/<coll>/<doc>.trace.json   conversion trace: one record per page (branch, outcome, profile,
                              docling confidence), see 5.1.1; removed together with its Markdown
    index/<coll>/<doc>/       nodes.json  embeddings.npy  index.meta.json (written last)
                              outcome.json instead, for a document that cannot be indexed (see Freshness)
    index/<coll>/_all/        merged: nodes.json embeddings.npy merge.manifest.json
    index/<coll>/collection.origin.json   only in an imported collection (see 5.5)
  serving/
    gen-000041/ gen-000042/   immutable published generations (last 3 kept)
      catalog.json            generation, content_sha, model, collections + documents
      index/<coll>/_all/…     hard links into the workspace (copy fallback)
      markup/…
    current -> gen-000042     symlink switched atomically (os.replace)
  run/                        0700: <kind>.sock .alive .start.lock .pid .log, index.lock,
                              ui.token (0600) ui.pid ui.port (dashboard)
  jobs/                       <id>.json (record) <id>.events.jsonl <id>.log
```

**Publishing** (`publish.py`) collects every workspace collection with a complete `_all/` index,
refuses to mix embedding models, computes a `content_sha` (no-op if it equals the live one), builds
`.tmp-gen-N`, renames it to `gen-N` and switches `current`. Hard links make it O(files); a running
search daemon keeps reading the old generation's files until it swaps. `rollback` re-points
`current` at the previous generation.

**Freshness** – a document is re-indexed when its source SHA-256, the chunk parameters,
`CHUNKER_VERSION`, `TOKENIZER_VERSION`, the index format or the model differ from
`index.meta.json`; the Markdown is reused only when the `.md.sha256` sidecar matches. An unchanged
document found at a new path (a location registered again where the folder now is) keeps its index and gets its `src_path` updated.
Merging concatenates stored embeddings; nothing is re-embedded.
A document that **cannot** be indexed for a reason of its own gets the same treatment: `outcome.json` in its
index folder records the source SHA-256, the conversion settings (`convert_profile`) and the reason, and the
next update run does not convert it -- only the checksum is computed. Remembered: a password-protected or
encrypted PDF, a PDF whose bytes cannot be opened at all (`damaged_pdf_reason`; a copy that syncs completely later
has another checksum), and "no text" when every reader had its turn. Never remembered: anything that may pass by
itself (a cloud file that could not be fetched, a timeout, a crash, a "no text" where the document reader was not
used). A changed file, a changed conversion setting, `--force-md` ("re-convert to Markdown") or a complete run
(`index all`, `--rebuild`) tries again; success or a deleted source removes the record.
**How a remembered document is reported (0.9.28).** The run that finds the failure reports it in `errors` /
`no_text`. An update run after that lists it under `known` (document status `known`, "not tried again", with the
reason and what it was), **outside** `errors`, so the run ends `succeeded` when nothing new failed; until 0.9.27 it
was put into `errors` again and every update run of a folder with one protected PDF ended `partial`. A file left
out because another file has its document name (`a.pdf` next to `a.jpg`) is handled the same way: the names a run
reported are kept in `left_out.json` in the workspace, an update run lists them as `known`, a complete run reports
them again, and a renamed or deleted file drops out. `not_retried` in the summary is the number of `known`
documents. In the dashboard's Documents list the filter **Skipped** shows everything a run did not convert, each
with its reason: unchanged, not tried again, unsupported format. Unsupported formats never reach this step: they are
recognised by extension when the folder is listed and are not opened at all. A document whose source was deleted
loses its Markdown and index (`prune_orphans`, see 5.1) -- but only when the run read its whole
collection and the collection's folder could be read; an unreachable folder changes nothing.
`index.meta.json` also records `model_revision`, the commit of the model's weights in the local
Hugging Face cache: not part of the freshness key (advisory), but carried into `catalog.json` and
collection exports and compared on import.

## 3. Wire protocol (`protocol.py`)

One request per connection, newline-delimited JSON:

```
→ {"v":1, "client":"claude", "action":"search", "query":"…", "top_k":5, "collections":[…], "wait_s":40}
← {"ok":true, "result":{…}}                      # single reply
← {"event":"progress", …} … {"event":"end", …}   # streaming actions (follow)
← {"ok":false, "code":"forbidden", "error":"…"}
```

`search`, `grep` and `list` results, `index status` and the search daemon's `ping` carry timings,
sizes and warm-up/memory figures (see the README table "What each command tells you about time and
size"); per-document timings travel as `doc` events in `jobs/<id>.events.jsonl` and in `follow`.
`list` also takes an optional `full` flag (default false): false returns `document_count` +
`description` per collection, true adds the per-document `documents` list (see "Front-ends" above).

Codes: `bad_request protocol_mismatch warming_up unavailable forbidden model_error model_mismatch busy
internal`. (`forbidden` is no longer sent for collections.) `PROTOCOL_VERSION` is bumped on incompatible changes; a daemon rejects newer clients
with `protocol_mismatch` and tells them to upgrade. The `client` field is the host identity
(`claude`, or any other valid name; `cli` = administrator; else `unknown`) that selects the
collections it may use. The MCP adapter fixes it from `--profile NAME`; it is declared, not authenticated.

## 4. Layering (enforced by tests)

| Layer | Modules | May import |
|---|---|---|
| light (stdlib) | `paths config policy access descriptions locations bundle lifecycle inventory spec protocol client jobs publish catalog grep api service register cli models model_tasks`, `ui/*`, `core/daemon_base`, `core/indexer_daemon`, `core/conversion/{trace,costs,router,profiler,records,runview,estimate,pageimage}` | stdlib only (`profiler` and `pageimage` import pypdfium2 / Pillow inside functions) |
| core (heavy) | `core/{bm25,chunker,embedding,indexer,search,search_daemon,worker,docling_convert,diagnostics,playground}` | numpy, torch, docling (lazily) |
| adapter | `mcp/*` | the `mcp` package, light layer |

`tests` check that importing the light modules never pulls in numpy/torch/docling/mcp, that
only `rag_search/mcp/` imports `mcp`, and that the adapter can neither manage access nor reach
collection management (`bundle`, `lifecycle`, location changes). The adapter is installed with the
package (a dependency since 1.1; `rag-search register` adds it to Claude).

## 5. Indexing and search flows

### 5.0 The numbered pipeline (`stages.py`)

One numbered list of stages describes a document's way into the index and a query's way through it. The
numbers are the same everywhere: the progress events of a run (`stage` events carry `id`), the indexer's
log lines ("3 Convert started", "2 Fingerprint unchanged: skipped"), `rag-search index` output, the
Architecture tab's diagram, the Indexing tab ("Settings in effect, by pipeline stage"), the Settings tab
(one card per stage; the steps 3.1-3.5 are cards inside 3 Convert, and 3.2 Read has one card for the router and one
for each lane), the page traces, the Playground and this document.
`stages.py` (stdlib only) is the registry: each `Stage` has its number, key, name, scope, where it runs, a
one-line description, the settings it owns (`section.key` of `config.json`, the keys the Settings tab edits)
and its environment-only variables. A stage whose settings belong to several parts lists them as `groups`
(`stages.Group`: id, name, what it does and what else its settings reach): 3.2 Read has `3.2` (the router: routing,
pipeline), `3.2a` (docling: table mode, PDF backend, batch, timeout, text-layer fill, hand-over to the reader), `3.2b`
(OCR first, OCR mode, engine, languages; the Tesseract variables), `3.2c` (residue regions) and `3.2d` (document
reader switch, model, memory limit; its page timeout and backend variables). `/api/pipeline` carries the groups, and
the Settings tab, the Indexing tab's "Settings in effect" and a Playground experiment's settings all draw one block
per group, so it is plain which setting acts on which lane. `tests/test_stages.py` checks that every tunable belongs
to exactly one stage, that every setting and variable of a grouped stage is in exactly one group, that every environment variable listed is really read, and that the browser's copy of the keys
(`ui/static/pipeline.js`) equals the registry. Conversion is not a side branch: 3 Convert is a stage like
the others, with its steps 3.1-3.5 numbered inside it.

Indexing (one run; 3.2 has four lanes, chosen per page by the router: 3.2a docling on the text layer, 3.2b
OCR for a clean scan, 3.2c docling on the text layer plus the document reader on its pictures and regions, 3.2d
the document reader for every other scan and image; a page a cheaper lane doubts goes on to 3.2d):

| Stage | Name | Runs on | Once per | What it does | Settings it owns (`section.key`) |
|---|---|---|---|---|---|
| 1 | Discover | CPU | run | list the files of every collection (registered folders, imports); formats the pipeline cannot read are skipped and counted | – |
| 2 | Fingerprint | CPU | document | SHA-256 of the file plus the chunk, model and conversion settings; unchanged documents are skipped in seconds | – |
| 3 | Convert | CPU+GPU | document | turn the file into page-marked Markdown, page by page, through the steps below; several documents are converted side by side when the machine has the memory for it; a conversion process that goes silent is stopped and its document reported, so one stuck document never holds the run | `indexer.jobs`, `indexer.stall_timeout` |
| 3.1 | Profile | CPU | document | look at every page once: text layer, scan or photo, pictures, ink, resolution, script | – |
| 3.2 | Read | CPU+GPU | document | each page goes down the lane it needs: 3.2a docling on the text layer (and Office files), 3.2b OCR for a clean scan (docling's OCR, or Tesseract for a skewed page and image files), 3.2c docling on the text layer plus the document reader on its pictures and regions, 3.2d the document reader (a vision model) for every other scan, photo and image; a page a cheaper lane doubts goes on to 3.2d | `indexer.routing`, `indexer.ocr`, `indexer.ocr_engine`, `indexer.ocr_lang`, `indexer.table_mode`, `indexer.pdf_backend`, `indexer.pipeline`, `indexer.vlm`, `indexer.ocr_first`, `indexer.residue`, `indexer.escalate_digital`, `indexer.layer_fill`, `models.reader`, `models.memory_limit_gb`, `indexer.docling_batch`, `indexer.doc_timeout`; env `RAG_SEARCH_THREADS`, env `RAG_SEARCH_VLM_PAGE_TIMEOUT`, env `RAG_SEARCH_VLM_FREE_GB`, env `RAG_SEARCH_VLM_BACKEND`, env `RAG_SEARCH_TESSERACT`, env `RAG_SEARCH_TESSERACT_LANG` |
| 3.3 | Gate | CPU | document | deterministic checks on every page: coverage, script, tables, resolution, running balances and totals, runaway output | – |
| 3.4 | Repair (optional) | GPU | document | a table cell that breaks the arithmetic is cut out, read again and replaced only when a second, independent reader and the arithmetic agree | `indexer.repair`, `models.repair`; env `RAG_SEARCH_REPAIR_SECOND` |
| 3.5 | Reconcile | CPU | document | a table that runs across a page break is joined (the continuation gets the header) and checked across the break | – |
| 4 | Chunk | CPU | document | split the Markdown into passages that never cross a page; tables and code stay whole where they fit | `indexer.chunk_size`, `indexer.chunk_overlap` |
| 5 | Embed | GPU | run | turn every new passage into a vector; the model is loaded once per run, after all documents are converted | `models.embedding`, `models.embed_batch`, `models.max_seq`, `models.dtype`, `models.device` |
| 6 | Write | CPU | document | nodes.json and embeddings.npy, then index.meta.json last, so a half-written document is never taken for a finished one | – |
| 7 | Merge | CPU | collection | concatenate the documents' vectors into the collection's index for search (nothing is re-embedded) | – |
| 8 | Publish | CPU | collection | hard-link the new generation, switch the "current" symlink, tell the search daemon to reload; the last three generations are kept | `indexer.auto_publish` |

Search (one query):

| Stage | Name | Runs on | Once per | What it does | Settings it owns (`section.key`) |
|---|---|---|---|---|---|
| S1 | Access | CPU | query | which collections this client may search; an unknown name is refused like a typo | – |
| S2 | Keyword | CPU | query | BM25 over the chunk text: exact terms, identifiers, versions; the pool size below applies to both retrievers | `search.retrieval_pool` |
| S3 | Vectors | GPU | query | cosine similarity of the query vector with every passage vector; the models are loaded when the search daemon starts | `search.prewarm` |
| S4 | Fuse | CPU | query | reciprocal rank fusion of the two lists into one pool | `search.rrf_k`, `search.stages` |
| S5 | Rerank (optional) | GPU | query | a cross-encoder reads query and passage together and re-orders the best candidates | `models.reranker`, `search.rerank_pool`, `models.rerank_batch`, `models.rerank_max_len` |
| S6 | Top k | CPU | query | the best passages with page, heading, snippet and score | `search.top_k` |

**Settings really in effect (`effective.py`).** `config.json` holds the defaults; an environment variable the
daemon was started with wins over it (`settings_env`: an actual environment variable is never overwritten, 0 /
blank in the file means "no override"). `effective.resolve(cfg, env)` computes, per stage and per setting, the
value the code will read, where it comes from (default, `config.json` or environment) and which function reads
it (`READ_BY`), using the same parsers as the pipeline (`docling_convert.convert_settings`, `vlm.mode`,
`repair.mode`, ...), so what the dashboard shows cannot differ from what a run does. `GET /api/pipeline`
resolves the indexing stages with the environment of the indexer daemon and the search stages with that of
the search daemon (what each really runs with, reported by its ping as `env_overrides`). The Indexing,
Settings, Architecture and Playground tabs and `rag-search playground settings` all render this one answer.

### 5.1 Indexing flow (worker process, `core/indexer.py`)

```
 <location folder>/<file>     pdf docx docm rtf pptx xlsx html csv adoc md txt png jpg tif bmp webp
                              (a registered location: its whole tree is one collection)
        │  plan (locations.plan_scan): refuses to run while locations.json is unreadable; every
        │  source folder is probed for reachability first (in a thread, 10 s bound; a listing
        │  that fails is retried within it, for cloud-storage folders that answer late); one that is
        │  missing, cannot be listed, is completely empty while its collections have an index
        │  (an unmounted mount point), or has a sub-folder the walk could not open, is *frozen*
        │  -- not pruned, not re-merged.  A run over everything, or a whole collection, *covers* it.
        │  scan: hidden files and ~$ lock files skipped; two files that map
        │  to the same document name (a.pdf + a.docx) -> only one is indexed (the one already
        │  indexed under that name, else the first), the other is reported -- also when the one
        │  already indexed is not part of this run (a single-file run cannot replace it)
        │  prune: for every covered collection, documents not found lose their Markdown and
        │  per-document index (files removed one by one: a.pdf + a/b.pdf nest b inside a;
        │  compared by path and by file identity, so a case-only folder rename on a
        │  case-insensitive disk is not mistaken for a deletion)
        ▼
  1  DISCOVER is the plan and scan above (it runs once per run); the boxes below are 2-7.
┌─ PHASE 1 · per document · parallel (`indexer.jobs` processes) · watched: a process that goes ─┐
│  silent is stopped, its document reported, the others go on (stallwatch.py, section 7)       │
│                                                                                               │
│  2  FINGERPRINT: SHA-256 of the source file                                                   │
│     fresh?  same source SHA + chunk size/overlap + CHUNKER/TOKENIZER version + index format   │
│     + embedding model + conversion settings, as recorded in index.meta.json ─yes▶ SKIP        │
│  3.1 PROFILE pages (core/conversion/profiler.py, pypdfium2 only, 120 s budget): characters,   │
│     picture cover, script, dpi per page -> a branch + reason per page (5.1.1); skipped when   │
│     the Markdown and its trace are current                                                    │
│  3  CONVERT -> markup/<coll>/<doc>.md  (3.2 Read, 3.3 Gate, 3.4 Repair, 3.5 Reconcile)        │
│       PDFs and image files page by page, one lane per page (5.1.1): a text layer -> docling   │
│       without forced OCR; a scan -> the document reader (or OCR first for a clean one, or     │
│       docling OCR when there is no reader); pictures on a text page -> the document reader.   │
│       Office / HTML files whole, by docling (layout + TableFormer; pypdfium2 backend).        │
│       converter kept per process, threads shared, RAG_SEARCH_DOC_TIMEOUT per docling call;    │
│       every reader call is announced (`step` event) and every page reported when it is done;  │
│       <!-- page N --> marks                                                                   │
│       (formats without pages become page 1; .md/.txt are read as text, whatever encoding;     │
│        an .md is reused when its .md.sha256 sidecar has the same source SHA + settings)       │
│       no text at all (a photo, a blank page) -> "skipped: no text", not a failure;            │
│       a password-protected / encrypted PDF fails, with that reason in the message             │
│  4  CHUNK  (pure stdlib, core/chunker.py)                                                     │
│       split at the page markers ─▶ blocks (paragraphs; code fences and tables kept whole)     │
│       ─▶ oversized blocks are split: tables by rows with the header row repeated,             │
│          prose by sentences, then by words                                                    │
│       ─▶ greedy packing to ~512 estimated tokens with ~64 tokens of block-level overlap       │
│       every chunk carries: page, nearest heading, file, collection, source path               │
│  6  WRITE nodes.json   (index.meta.json is removed first, so a half-done document is never    │
│                         mistaken for a complete one)                                          │
└──────────────────────────────────────────────┬────────────────────────────────────────────────┘
                                               ▼
┌─ PHASE 2 · embeddings · one process · model loaded once ──────────────────────────────────────┐
│                                                                                               │
│5 chunk texts ─▶ BAAI/bge-m3 (SentenceTransformer; fp16 on mps/cuda; max_seq 1024; batch 32)   │
│              ─▶ one dense vector per chunk (1024 dimensions), L2-normalised                   │
│              ─▶ embeddings.npy (float32)                                                      │
│6 then write index.meta.json LAST: source SHA, chunk params, chunker/tokenizer versions,       │
│  model (+ weights commit), dim, chunk count, timestamp ==> marks the document as complete     │
└──────────────────────────────────────────────┬────────────────────────────────────────────────┘
                                               ▼
┌─ PHASE 3 · merge per collection · cheap · nothing is re-embedded ─────────────────────────────┐
│                                                                                               │
│7 concatenate nodes + embeddings of all COMPLETE documents whose source file still exists      │
│  ─▶ index/<coll>/_all/{ nodes.json, embeddings.npy, merge.manifest.json }                     │
│     manifest_sha = hash of the member documents; unchanged manifest = merge skipped           │
│  imported and frozen collections are not merged (left exactly as they are); a generated     │
│  collection left with no documents disappears from the workspace                             │
└──────────────────────────────────────────────┬────────────────────────────────────────────────┘
                                               ▼
   8 PUBLISH  indexer daemon: publish -> serving/gen-N (hard links) -> `reload` to the search daemon,
   which reuses in-memory collections whose manifest_sha is unchanged and loads the rest

 The keyword index is NOT stored on disk.  The search daemon builds BM25 in memory when it loads a
 collection:  tokens = lower-cased words; compound tokens ("svm-name", "9.16.1", "a/b") are kept
 whole AND split into parts;  postings + idf per term;  Okapi BM25 with k1 = 1.5, b = 0.75.
```

#### 5.1.1 Conversion tracking (`core/conversion/`, phase P0 of the conversion plan)

Every converted document gets a *conversion record*, in one vocabulary shared by the engine, the CLI,
the HTTP API and the dashboard (design: `docs/design/document-conversion-plan.md`):

* a page takes one **branch** -- `copy` (.md/.txt), `office` (Office/HTML; `.rtf` is read by `docling_convert.rtf_to_markdown` with the standard library, `.docm` as the `.docx` it is -- docling needs LibreOffice for RTF, and neither is allowed to need a tool that is not already installed), `digital` (PDF page with
  a text layer), `raster` (scanned page or one without a usable text layer), `image` (image file or
  TIFF frame), `embedded`, `fallback`, `cached`, `unknown` (not profiled) -- decided by
  `router.decide(kind, page_profile)`, which also returns the reason; and ends with one **outcome**:
  `pass`, `repaired`, `low`, `no_text`, `error`.
* **Routing (phase P2, `RAG_SEARCH_ROUTING=pages`, the default)**: a PDF with a usable profile is read
  page by page (`conversion/routed.py`). `digital` pages go to docling *without* forced OCR (OCR only for
  pictures), `raster` / `unknown` pages (scans, garbled text layers, unprofiled pages) to the **document
  VLM** (5.1.3; branch `raster`) or, when it is off, not installed, short of memory or fails on the page,
  to docling with full-page OCR (branch `fallback`, the reason is in the page's `note`), blank pages (no
  text, almost nothing darker than the ground **and almost nothing lighter**: since 0.9.28 a slide, a screenshot in
  dark mode or a photograph with light writing on a dark ground is no longer taken for a blank page) are not read
  at all. docling reads pages in runs of consecutive
  pages of one kind via its `page_range` (at most 20 pages per call); the document reader reads **one page per
  call**, so each of its pages is stored, gated and reported the moment it is read. The OCR setting only decides
  whether OCR is on or off. Image files are read page by page by the document VLM (5.1.3); Office/HTML
  files and Markdown are converted whole as before,
  and so is any PDF whose profile failed, when `RAG_SEARCH_DOCLING_PYTHON` is set, or with
  `RAG_SEARCH_ROUTING=document` (the escape hatch; also `rag-search config set indexer.routing document`).
  **If routing fails for a document (an exception from a reader, a docling that ignores `page_range`) it
  is converted whole and the trace says so** -- routing can degrade, it cannot fail a document.
  `convert_range` handles docling versions that keep or renumber the original page numbers of a range.
  `CONVERT_VERSION` is `c4` and the conversion profile includes `route=`, so the upgrade re-converts
  everything once, and switching the setting converts again.
* **Profile** (`profiler.py`): per page characters, text-layer quality, picture coverage, rotation,
  script, and for pages without text a small greyscale render (36 dpi) from which `ink` (the share of
  clearly dark pixels) and a content `hash` come; text pages are hashed from their text layer and size.
  The profile is the page's identity for the cache and the source of the blank-page rule.
  What a review of the profiler and the gate corrected in 0.9.29 (each found on generated pages, most confirmed on
  the pages of the full run):
  * **Where a picture is.** The boxes of large pictures (`big_pics`, cut out for the document reader in lane c) and
    the text rectangles of the residue finder were taken from the page's own coordinates, which ignore `/Rotate`
    and the origin of the page box: on a rotated page the reader was handed another part of the page.
    `profiler.page_box` maps them with PDFium to the page as it is displayed; a picture hanging over the edge
    counts for the part on the page.
  * **Whose text it is.** `hidden_ocr_layer` meant "text over a full-page picture". Of 556 such pages in the run
    506 were born-digital pages on a background picture (statements on a letterhead, web pages printed with their
    backdrop), and lost the comparison with their own text layer for it. It now needs **invisible** text (text
    render mode 3 or 7, which is how a scanner writes its OCR): 50 pages. Those are read from that layer as before
    and get `plausibility` and `column_types` from the gate, because the layer is an OCR reading nobody else has
    seen (with `RAG_SEARCH_ESCALATE_DIGITAL=auto` a failure sends the page to the document reader).
  * **Which text layer is garbled** (`docling_convert._page_text_ok`; a garbled layer sends the page to the
    document reader). Three kinds of sound layers were called garbled, 81 of the run's 94 such pages: long tokens
    that are values, not words run together (UUIDs, identifiers with `_` and `:`, rows of figures: only long runs
    of letters count now, and none in a script written without spaces); bullets and icons from a symbol font
    (one or two private-use or control codes standing alone; damage inside words and pages set in a symbol font
    still count); and invisible format characters (zero-width spaces, joiners). Those pages keep their exact
    text instead of a model's reading of their identifiers.
  * **Image files.** A picture with a transparent ground (a PNG without a background) reached every reader as
    black on black; `profiler.opaque` puts it on white (on black when its content is light), for the ink figure,
    the document reader, Tesseract and Apple Vision (which now also gets the picture upright). Only a TIFF has
    pages: the second frame of a phone's JPEG (a depth or gain map) and the frames of an animation are not read
    as page 2. A resolution tag with two values counts by the coarser one (a fax).
  * **Skew** (`scanfacts`): the search went to 6 degrees while pages up to 8 are straightened for Tesseract, so
    a page at 8 or 12 degrees measured 5 or 6 and passed as straightenable. It goes to 10.
  * **Gate.** The script of a page's text layer is not held against the reading when that layer is garbled (a
    broken font map can come out as Greek letters). A Tesseract last resort is kept only when its text is words
    (`plausibility`, `script`): on a page in a script Tesseract was not given it writes letters, not words.
    "No text" for a page served from the page cache names the reader that read it (it said no reader had, and
    the file was tried again on every update run).
* **Page cache** (`pagecache.py`, `indexer_workspace/page_cache/<hh>/<key>.json`): the Markdown and facts
  of one page, keyed by hash of (page content hash, reader id + mode, the full conversion profile string);
  any change of page, tool or setting is a different key, so a hit is always valid. One file per page,
  written atomically after each docling run of pages and after every page of the document reader: a cancelled or
  crashed run loses at most one docling run (seconds) or one reader page, a changed document re-reads only its changed pages, and two documents that contain the same
  page share the work. Cached pages are branch `cached` (`was` keeps the original branch). After every
  run entries that no stored trace refers to (`page.key`) and that are older than six hours are
  deleted; `rag-search index cache [--clear]`, `api.page_cache`, `GET /api/conversion/page-cache` and
  the System tab show the size.
* **Digital pages against their own text layer** (`layer.py`, 0.9.23). docling leaves out table cells it could not
  structure, sidebars, footnotes, labels inside vector drawings and a page's running header and footer. The PDF's
  text layer has all of it. For every digital page the layer (read with pypdfium2, running headers and footers --
  lines repeated in the top or bottom 7 % of at least three pages -- and lone page numbers left out) is compared with
  the Markdown: the share of its words and numbers that the Markdown holds (`intact` from 97 % / 98 %, `lost text`
  below 90 %, in between `uncertain`; a layer of under 15 words is not judged). When text is missing, the layer lines
  the Markdown does not hold are appended under `<!-- text layer: lines the conversion left out -->` (at most 20,000
  characters a page; deterministic, milliseconds, no model), and the page record says so (`layer`: verdict, recalls,
  `added_lines`, `before`). Not applied when the page's layer is a scanner's hidden OCR layer (invisible text) or looks garbled
  (`text_ok`, `page_text_ok`): such a layer proves nothing and the page belongs to an image reader.
  `RAG_SEARCH_LAYER_FILL`: `fill` (default), `report` (compare and record `would_add_lines`, change nothing, gate as
  before) or `off`. Measured on the first full run (16,842 pages): 16 % had lost text, 2,212 of those had passed the gate.
  Two rules since 0.9.28, from the full run of 0.9.27 where 1,021 digital pages were marked `low` by this check, 808
  of them in one collection of specifications: numbers of **one digit do not count** in the number recall (they
  were the bit rulers over register diagrams, list markers and footnote signs; words count from two letters on for
  the same reason; a missing line of digits is still put back), and a page of which **no line is missing** and at
  least 90 % of the words and numbers are there is `intact` (the rule a page already got after a fill). Evaluated
  again on that run's pages: 621 of the 1,021 are no longer marked; the 400 that stay miss a number of two digits
  or more or more than a tenth of their words.
* **The four lanes of 3.2** (`routed.py`, `router.py`, `residue.py`, `scanfacts.py`, `tesseract.py`; 0.9.24). Every
  page that is read goes down exactly one lane, chosen once per page by the router and recorded in the page's `route`
  (`runway`: the lane chosen, `reasons`, `final`: the lane whose text the page ended with, `engine`, and
  `escalated_from` with the gate checks when a lane handed the page on):

  | Lane | Reads | When | Gate checks that catch its failures | Default |
  |---|---|---|---|---|
  | **a** text layer | docling on the text layer (OCR only for embedded bitmaps) | the page has a text layer and nothing the layer does not explain | `coverage` against the PDF's own text layer (`layer.py`, below), `script`, tables, arithmetic | always on |
  | **b** OCR | docling's full-page OCR for a straight PDF page; Tesseract (after straightening the page) for a skewed page (2-8 degrees) and for every image file | a scan whose page image says clean print (`router.decide_scan`) | `expected_size`, `plausibility`, `column_types`, plus the common checks | **off**: `indexer.ocr_first` = `auto` (`RAG_SEARCH_OCR_FIRST`) |
  | **c** text layer + pictures | docling on the text layer, and the document reader on each large picture (profile `big_pics`) and, with the switch, on each *residue* region | a text page with large pictures; with `RAG_SEARCH_RESIDUE=auto` also one with regions of ink the text layer does not explain | `residue_read`: every picture or region asked of the reader was read | pictures **on**; regions **off**: `indexer.residue` = `auto` (`RAG_SEARCH_RESIDUE`) |
  | **d** document reader | a vision model on the whole page image (guarded, then repaired, 5.1.3-5.1.4); Tesseract as the last resort for a runaway | every other scan, photo and image, a page whose text layer is garbled, and every page a cheaper lane doubts | `degenerate`, table arithmetic, repair | always on |

  *Router* (`router.py`): `decide` (the page's kind: text layer, scan, image, Office, text) is as before;
  `decide_digital` makes a text page lane c when it has large pictures (and, with the residue switch, regions) and a
  document reader is there, otherwise lane a; `decide_scan` turns the page-image facts (`scanfacts.py`: one render at
  100 dpi, numpy + Pillow, 50-150 ms: contrast, stroke sharpness, skew, speckle, paper texture, text lines, ruled
  lines, ink share) and the resolution into lane b or d through `router.THRESHOLDS` (version 2; unknown resolution, no
  facts, no OCR engine mean d; a ruled table is d, because plain text loses its structure); `b_engine` picks docling or
  Tesseract. With no document reader every scan is lane b by necessity (`route.forced`; docling's OCR is all there is).
  An image file goes through the same router (its resolution is estimated from its pixels as if it were an A4 page, since
  its dpi tag is usually 72 or 96) and is read by Tesseract in lane b.

  *Escalation*: a lane-b page the gate doubts (`gate.check_page(..., ocr=True)` returns `escalate`: any failed check but
  `low_resolution` and `docling_grade`, or no real text) gets one more cheap try, **Tesseract** on the straightened
  page, unless the doubt is about structure (`table_shape`, `column_types`, `totals`, `running_balance`: plain text would
  lose it); only if that is doubted too does the document reader read the page, and the OCR text is kept for when the
  reader fails. With `indexer.escalate_digital` = `auto` (`RAG_SEARCH_ESCALATE_DIGITAL`) a text page (lane a or c) that still has lost
  text or garbled text after the text-layer fill is read as an image by the document reader (d); when the reader
  cannot take over, the text-layer result is kept and flagged. Both are off by default. The lane switches and the
  text-layer fill are ordinary settings (`indexer.ocr_first`, `indexer.residue`, `indexer.escalate_digital`,
  `indexer.layer_fill`): the Settings tab, `rag-search config set` and a Playground experiment's own configuration
  set them, an environment variable of the daemon wins as for every tunable. An OCR text the reader could not replace
  (it failed on the page) is kept as it is -- the page is not read by docling's OCR a second time -- and stays
  flagged. A page that has a text layer and is read as an image (a garbled layer, or handed on by lane a or c) is
  drawn from vectors, so the gate does not judge it by the resolution of a picture on it (`gate_profile`). The page cache keys follow the reader (`docling:scan`,
  `tesseract:scan`, `<vlm>:scan`, `docling:digital+<vlm>[+res1]`), a cached document-reader result is used before a
  cached OCR one, and a cached OCR result the gate now doubts goes on. The lane switches that are *on* are part of the
  document's conversion profile (`|ocrfirst=auto`, `|residue=auto`, `|updigital=auto`), so switching one on
  converts the documents again; the defaults add nothing.

  *Residue* (`residue.py`): the regions of a text page that have ink the text layer cannot hold (a stamp, a signature, a
  drawing): one render at 60 dpi, the text layer's rectangles and long straight lines (rules, table borders) taken away,
  the cells of a coarse grid that still hold ink grouped; a region is at least 4 % of the page and 10 % of its width and
  height and not an embedded picture the profile already has. Measured on 99 real text pages: 37 % had a region at
  2 % (too many), 10 % at 4 %.

  *Gate constants measured on real pages* (`scripts/mine_traces.py`, 1,107 scanned pages that passed): characters per
  unit of ink had a median of 15,105 (p05 1,664, p95 48,566), so `CHARS_PER_INK` is 15,000 and `SIZE_LOW`, `SIZE_HIGH` 0.25
  and 3.0. `scripts/measure_lanes.py` reads a sample of real scans with docling's OCR and Tesseract and scores both
  against the document reader's Markdown: on 38 scanned pages the router sent 3 to lane b (none wrong), the OCR gate
  alone kept 29 pages of which 17 were not as good as the reader's text (partial loss, numbers missed: this is why the
  router stays strict), and two cheap readers agreeing did not make a page safe (numbers differ between engines), so
  OCR first stays off by default. `plausibility` judges only words in Latin letters: other scripts have other vowels.
  `rag-search bench route [--ocr tesseract|docling]` runs a ladder of synthetic damaged scans (`synth.py`) through facts,
  router, OCR and gate (`routeharness.py`).
* **Quality gate** (`gate.py`): every page's Markdown is checked -- `coverage` (a digital page: the layer
  comparison above, or with no layer the share of characters of the text layer that survived; a scanned page with
  clear ink came out empty), `script` (garbled text, OCR noise, a script
  that differs from the text layer's), `docling_grade` (poor confidence), `table_shape` (ragged or
  empty tables, a statement whose balance column is mostly empty because its numbers slid into a
  neighbouring column, consecutive rows that repeat each other, one amount repeated in three or more
  columns of most rows, amounts in the cheque / reference column, `Cr`/`Dr` balances in a debit or credit
  column), `low_resolution` (an image page under 150 dpi, or a small image of unknown dpi; an image file's default dpi tag of 72 or 96 counts as unknown, so its pixels decide), `degenerate` (a
  page read as an image that is a runaway of the reader: one line or phrase repeated hundreds of times, text
  that is nothing but repetition, a script that is not on the page; `degenerate.py`; not applied to a text
  layer), for a page read by OCR `expected_size`, `plausibility` and `column_types`, for a page whose pictures the reader
  was to read `residue_read`, and the table validators (`running_balance`, `totals`) -- and ends `pass`, `low`
  (a check failed: indexed, flagged) or `no_text`. For a digital page whose text is all there (layer verdict `intact`),
  `table_shape`, `totals`, `running_balance` and `docling_grade` no longer make it low: they are listed under the
  record's `gate.notes`, because for search the text is what counts. Failed checks and the validators' suspect cells (with
  a hypothesis value each, and the number of the table) are stored in the page record's `gate`. Suspect cells are repaired by `repair.py` (5.1.4).
  (A bug fixed on the way: `_page_text_ok` counted the combining vowel signs of Indic scripts as
  garbage, so every Hindi/Marathi text layer looked broken to the profiler.)
* **Per-page times and events**: `time_s.read` (the run's seconds split evenly over its pages) and
  `time_s.gate` per page; `step_s` (read, gate) in the document summary and the run totals; one compact
  `page` event per finished page in the job's event log (`file`, `page`, `of`, `branch`, `kind`, `outcome`,
  `cache`, `read_s`, `chars`, failed `gate` checks, `runway` (the lane that finished the page), `moved` (the lane that
  handed it on) and `engine`), written by the pool processes themselves like
  `stage` events, one `step` event before every reader call (`{file, page, what}`: "document reader", "docling, pages
  1-20", "repair", "tesseract" ...; a call can take minutes, and this is what the Workers card shows as "now" and
  what the stall watch takes as a sign of life), and one `plan` event per document once its pages are profiled (or
  known from its stored trace):
  `{pages, branches}`, what kinds of pages it has, any format (an Office file is one page of kind `office`).
  A page's **kind** (`trace.page_kind`) is what it is -- `digital`, `raster`, `image`, `office`, ... -- and its
  `branch` is how this run got it: a page reused from the page cache has the branch `cached` and keeps its kind
  in `was`. Everything that shows a distribution (summaries, run totals, the live view, the Collections tab)
  counts by kind; the number of reused pages is a separate figure (`cached_pages`, `cached`). Pages are also counted by the
  lane that finished them (`runways`, with `moves` such as `b>d`, and the lane-b `b_engines`, in the summaries, the run
  totals as `runways` / `ok_runways`, and the live view with `runway_outcomes`). `runview`
  aggregates the events incrementally into `live`: pages finished by kind and outcome, how many were read now
  (`read`) and how many reused (`cached`), pages read per minute (reused pages are not a rate), the documents
  being read with `done`/`of`, and `active`, the pages of the files being converted right now by kind (from their
  `plan` events; a file not profiled yet is counted in `unprofiled`) -- and into per-lane page progress. The
  Indexing tab's Convert panel shows two bars: *Pages in active files* (`live.active`) and *Pages in
  successfully converted files* (`totals.ok_branches`: files that converted, not failed or empty ones).
* `docling_convert.py` returns `page_stats` per page (characters, script, tables, big pictures and
  docling's own confidence scores and grade, read defensively with `getattr`); `records.build_pages`
  merges them with the profile.
* **Storage**: the full trace in `markup/<coll>/<doc>.trace.json` (version 1; reused untouched when the
  document is re-embedded without being re-converted); a *summary* (pages, `branches` (by kind), `outcomes`,
  run-length `strip` such as `d566r2d12`, scripts, tables, `low_pages`, `poor_pages`, `time_s`, `step_s`,
  `cached_pages`, `gate_failed`, `cost`,
  `trace` path) in `index.meta.json["conversion"]` -- carried into collection exports
  automatically -- and on the document's `doc` event. A failed or empty document keeps no trace
  but its pages still count in the run totals.
* **Run totals** (`trace.RunTotals`) are live in the job's `progress.conversion` and frozen in
  `summary.conversion`.
* **Run events and the run view** (`runview`, stdlib). Every phase of a run reports in the same way, through
  four kinds of line in the job's event log, each carrying the writing process id (`pid`): `phase` (Convert,
  Embed, Merge start / end, with their totals: files, documents, chunks, model, collections), `work` (a
  process starts / ends one unit of work -- a document in Convert (`indexer.prepare_document`, framed in a
  `try/finally`, so a failed document closes its work too) or Embed, a collection in Merge -- with its
  outcome; the run itself closes the work of a process it stopped (`stalled`) or that ended with the pool
  (`interrupted`: not counted, the document goes to a new process)), `stage` (the numbered stage reached inside that
  work), `step` (the reader call about to start) and `page`. A finished document is a `doc` event with its
  `collection`, file name (`source`) and `path` inside the collection; the run's document list and its counts are keyed
  by collection and path (by file name alone two `statement.pdf` in two folders were one row, and the tiles counted
  1,945 of 2,059 documents). Publish is a phase of the
  job record (`progress.phase = "publish"`, then `publish`, `publish_s`, `search_reload`). Workers are
  generic processes, not docling's: `worker N` is a conversion-pool process, `main process` embeds and merges
  (with `jobs=1` it does everything), and a lane shows the phase, document (its path inside the collection,
  the one spelling used everywhere), pipeline stage, pages done, the reader call in progress (`step`: page, reader,
  since when), how long the process has been quiet (`quiet_s`; the dashboard warns from ten minutes and says when the
  stall watch will stop it) and its state (`working`, `idle` between documents, `done` after the run, `gone` when it
  was stopped), plus its documents per phase. `runview.run_view` returns one snapshot -- `now` (the one active work of the current phase), `phases`
  (status, counts, timing and per-phase figures: chunks per second for Embed, collections for Merge, generation
  and reload for Publish), `lanes`, `live` (pages), `totals` -- read from a remembered file offset, so a
  dashboard tick costs only the new lines. The Indexing tab's *Current run* card takes its "now:" from `now`
  (not from the throttled `progress.current`, which `EventWriter` also no longer drops when the document changes),
  *Phase detail* shows the picked or current phase, and *Workers* is its own card. Logs written before these
  events existed still show their convert / embed lanes from `stage` events.
* **API** (`api.conversion_*`): `conversion_run` (totals, now, phases, lanes), `conversion_documents` (a run's
  documents filtered by branch / outcome; `index_status` takes the same `doc_branch` / `doc_outcome`),
  `conversion_trace` (summary + a line per page, or one page's full record; confined to the
  workspace), `conversion_page_image` (PNG of a source page; only for sources inside a
  registered location; the source is `src_path` of `index.meta.json`, and for a document
  that is converted but not embedded yet -- no meta -- the file the trace names, looked for in the
  collection's own folder), `conversion_markdown` (the converted Markdown of a document or of one
  page, read from `markup/`; at most 2 million characters per reply), `conversion_estimate` (dry run: profiles the sources and
  estimates pages per branch and docling time; converts nothing). HTTP: `GET /api/conversion/{run,
  documents,trace,page-image,markdown}` (`markdown` with `raw=1` answers plain text, which the document
  drawer opens in a new tab; `rag-search trace COLL/DOC --md [--page N]` prints the same), `POST /api/conversion/estimate` (allowed with `--read-only`).
  `inventory.collection_info` adds a `conversion` block (totals over the stored summaries,
  `poor_documents`, `low_documents`).

Everything above writes only inside `indexer_workspace/` -- sources are opened for reading only, and
a test runs indexing (with pruning and a full rebuild), export and deletion over read-only source
folders and checks every source file's bytes, mode and mtime afterwards. A change to the source, the chunk
parameters, `CHUNKER_VERSION`, `TOKENIZER_VERSION` or the model makes step 2 fail, and only that
document is redone.

#### 5.1.3 The document reader (`conversion/vlm.py`, `vlm_worker.py`, phase P3)

A vision-language model (VLM) reads page *images* and writes Markdown (tables as HTML). It is used for
scanned PDF pages (branch `raster`), image files (branch `image`, one page per frame of a multi-page TIFF;
EXIF orientation applied; HEIC with `pillow-heif`) and large pictures on text pages (branch `embedded`).

* **A child process.** `Worker` starts `python -m rag_search.core.conversion.vlm_worker --backend B --model M`
  and talks JSON lines over stdin/stdout (`{"event":"ready"}`, `{"op":"read","image":...}` ->
  `{"ok":true,"md":...,"tokens":N,"seconds":S,"rss_mb":R}`, `{"op":"quit"}`); library output goes to stderr.
  The model's memory never sits in the indexer worker next to docling and torch, and goes back to the system
  when the child exits. A reader reading thread gives every wait a timeout.
* **Backends.** `mlx` (mlx-vlm, Apple Silicon only, a dependency of the package there) or `module:attr` of your own
  class (`RAG_SEARCH_VLM_BACKEND`; the tests use `tests.helpers:FakeVlmBackend`). The real backend **never
  downloads**: `VlmReader.preflight` reports a model that is not in the Hugging Face cache (the child runs with
  `HF_HUB_OFFLINE=1`), a missing mlx-vlm, or a machine that is not an Apple Silicon Mac.
* **Lifecycle.** `vlm.shared()` is one reader per process, created on first use (`RAG_SEARCH_VLM=off` gives
  none), its process is started when the first page is queued and kept for the run (`close_shared` at the end
  of `run_index` and at exit; a closed pipe also stops it). With `jobs > 1` every worker has its own reader.
* **Guards.** Free memory (`/proc/meminfo`, `vm_stat`, `RAG_SEARCH_VLM_FREE_GB`) must cover the model plus
  2 GB headroom, checked before every start (*not* sticky: memory may be back for the next page). A model
  that fails to load, or a reader that has died more than twice in the run, is ruled out for the run
  (`dead`). One page is limited to `RAG_SEARCH_VLM_PAGE_TIMEOUT` seconds (300): the child is killed and the
  page falls back. A crash or timeout costs one page. The child's file descriptor 1 is redirected to stderr, so even a
  native library that writes to it cannot corrupt the protocol. The memory check is per process: with `jobs > 1` two workers
  can both pass it before either has loaded the model (the cost is swapping, not a wrong result; use `jobs=1` for
  scan-heavy runs on a small machine). The reader is closed as soon as conversion is over, before the embedder loads.
* **Loop guard.** A small model decoding greedily can fall into a loop on a dense page (an annexure table,
  close Devanagari print): the same line hundreds of times until the token limit, minutes per page, or a stray
  script. The worker checks the text every 64 tokens (`degenerate.looping`) and stops a loop after a few
  hundred tokens (`stopped: "loop"`); the page is then read again (`VlmReader.read_page_guarded`) with a
  repetition penalty of 1.2 and a limit of 3072 tokens, and if that also runs away in three horizontal strips
  cut at blank rows. Of the readings that are not runaways (`degenerate.assess`) the most complete (most text) is kept, so a
  false alarm of the live test costs time and not text; when none is, the least repetitive one is, the page's
  note says so and the gate marks the page `low`. A healthy page is read
  once. A page cached by a version without the guard (`guard` of the cache entry < `vlm.GUARD_VERSION`)
  whose text is a runaway is read again once instead of being served from the cache.
* **Table format.** The reader is asked for HTML tables (they can carry merged cells), but every page goes
  through `tables.normalize_html_tables` in the router's finish step: a table without merged cells, nesting
  or unclosed markup becomes a Markdown pipe table, cell for cell (a `|` is escaped); the rest stays HTML. The
  chunker keeps an HTML table whole across blank lines and, when it is larger than a chunk, cuts it by `<tr>`
  rows into valid `<table>` pieces that each carry the header rows (`CHUNKER_VERSION` v2).
  `tables.join_split_pipe_tables` (same step) joins the data rows of a pipe table that a reader cut from its
  header with a blank line (rows with the header's number of cells, or up to two fewer): the parser, the chunker
  and the running-balance check otherwise see a table of three header rows and nothing to verify. In a real
  passbook page this was 28 rows whose arithmetic had never been checked. `table_shape` accepts a row that
  leaves out the header's last (empty) column; any other difference in the number of cells is a shifted row.
  The conversion profile of a document carries `post=<POST_VERSION>` (not the page cache's key), so a change
  to this kind of post-processing re-converts documents from the page cache and reads no page again.
  (`POST_VERSION` is 4 since 0.9.26: a page the repair model read again kept its HTML tables until then, about 92
  pages of the first full run; the bump converts every document again from the page cache and embeds it again once.)
* **Fallback, always per page.** Whatever the VLM cannot read goes to docling with full-page OCR: branch
  `fallback`, and the page's `note` says why (`crashed: ...`, `timeout: ...`, `unavailable: only 1.5 GB of
  memory is free ...`). Image files have no page fallback: the reader raises and the indexer converts the file
  whole with docling (`note`: "document reader not used (...)"). A photograph without text comes back empty
  and is "no text".
* **Last resort: Tesseract (`tesseract.py`).** When a page is still a runaway after the reader's retries and the
  repair model, or docling's OCR and Apple Vision found no text, the page image (3000 px) is read by the
  `tesseract` binary with `mar+hin+eng` (the installed subset; `RAG_SEARCH_TESSERACT_LANG`, `RAG_SEARCH_TESSERACT=off`),
  psm 4: plain text in reading order, no tables, 1 to 13 s a page, and it cannot loop. Kept only when it is real
  text and not a runaway; the page record says `tesseract` (branch `fallback`) and the note why. A missing
  binary or language is a note, not a failure (`rag-search doctor` checks both; `rag-search setup` installs them).
* **Last resort: Apple Vision.** docling's OCR cannot read a photographed page (its layout model calls the
  whole photo one picture and drops the text inside it). When a `fallback` page comes back without real text
  (only `<!-- image -->` and a class label), or a document converted whole has no text at all, and this Mac
  has `ocrmac`, the page image is read by Apple's text recognition (`conversion/applevision.py`) and the
  result is plain text in reading order (no tables). The page record says `apple-vision` as its reader; where
  Apple Vision cannot run (not a Mac, no `ocrmac`) the note says so. A document that is converted whole
  always says why it was not read page by page (`note`: "not read page by page: RAG_SEARCH_ROUTING=...").
* **Pictures on text pages.** The profile keeps the boxes of pictures covering 25-85 % of a text page
  (`big_pics`, at most 4). docling reads the page, the VLM reads each crop, and only the text the page does not
  already hold is appended (`_new_text`: whole text already present -> nothing; tables are kept whole). The
  page is branch `embedded`. If a picture cannot be read the page stays a plain `digital` page and is not
  cached as complete. (Pictures inside Office files stay docling's.)
* **Page cache.** The key's reader id is `vlm:<model>#<prompt version>`, so another model, another prompt
  (`PROMPT_VERSION`) or switching the reader off is a different key; a fallback result is cached under the
  docling key, so a run without a usable reader still reads nothing twice, and a later run with a working one
  reads those pages again. Cache entries remember the branch, reader info, tokens and model.
* **Records.** A VLM page's record has `reader {"tool":"vlm","model","mode":"page|picture"}`, `tokens`,
  `gpu_s` (seconds in the model) and `time_s.read` (render + model); summaries and run totals add `tokens`,
  `gpu_s`, `models`; `page` events carry tokens/gpu_s/model, the live view `tokens` and `tokens_per_s`.
* **Models.** `models.VLM_CATALOG` (kinds `reader` and `repair`, not part of `KINDS`: choosing one never
  re-embeds): `mlx-community/Qwen3-VL-4B-Instruct-4bit` (default, 3.1 GB), `mlx-community/PaddleOCR-VL-1.5-8bit`
  (1.1 GB; plain text from whole pages) and `-bf16`. Selection in `config.json` `models.reader` / `models.repair`
  or `RAG_SEARCH_VLM_MODEL` / `RAG_SEARCH_REPAIR_MODEL`; any other ORG/NAME is allowed (prompted as an instruct
  model). `rag-search models reader|repair [ID]`, `models download --reader`, `POST /api/models/reader`, the
  Models tab's *Document reader* card (the download uses the ordinary model task). Prompts per style are in
  `vlm.py` (`instruct` asks for Markdown with HTML tables and exact numbers; `paddleocr` uses `OCR:`).
* **Settings.** `indexer.vlm` (`auto` | `off`, `RAG_SEARCH_VLM`, `--doc-reader`).
* **Benchmark.** Engines `vlm` (the reader alone; a page it cannot read fails the file, so a measurement never
  silently measures docling) and `routed` (the production path without the page cache) next to `current`.
* **Not verified in CI:** everything above runs against a fake backend (the child process and protocol are
  real); the MLX backend, real model output, memory use and speed are measured only on an Apple Silicon Mac.

#### 5.1.4 Repair and tables across pages (`conversion/repair.py`, `reconcile.py`, phase P4)

**Repair** fixes one kind of damage cheaply and safely: a digit the document reader got wrong in a table whose
arithmetic is checkable (a bank statement's running balance, a Total row). It runs inside `_Converter.finish`,
only for pages read as images (`raster`, `image`, `fallback`; a born-digital text layer is exact, so an
arithmetic failure there is the document's own) and only when the gate reports suspect cells:

1. The validators (`validators.py`) name the suspect cell (table, row, column, role, what is written, and a
   *hypothesis*: what the cell would have to be for the sums to hold).
2. An **independent second reader** (`repair.second_reader()`; Apple Vision through `ocrmac` on the Mac,
   `RAG_SEARCH_REPAIR_SECOND` = `auto` | `off` | `module:attr`) reads the page image and returns words with boxes.
   `repair.locate` finds the row by matching the row's *other* cells to the words of one text line (the line that
   matches most; a tie, as with repeated rows, is refused), and the cell as the words between the row's
   neighbouring cells (several candidates are told apart by similarity to what was written). Its box is the crop.
3. The **repair model** (`vlm.repair_shared()`: the document reader's own process when `models.repair` is the same
   model, else a second reader) reads the crop alone (`PROMPT_CELL`: only the number, as printed).
4. The new text is **accepted only if** it is a number, it is not the text already there, it equals what the
   second reader saw (`norm_number_text`), *and* `validators.recheck_with` finds the cell no longer suspect with no
   new suspect. Anything else is logged and the page keeps the reader's text: two models agreeing on a digit that
   also makes the arithmetic hold is as good as certain; less than that is left to a person (the page stays `low`).
5. The cell is replaced in place (`tables.replace_cell`: pipe tables and HTML tables without merged cells; only
   that cell's text changes). Suspects are recomputed after every accepted fix (one fix can explain the next row),
   at most 10 cells per page.
6. **Escalation (0.9.21).** The repair model is by default **Qwen3-VL 8B 4-bit**, the reader the 4B. Any scanned page
   the gate flags -- not only one with a suspect cell: shifted columns, a runaway, a failed balance, an empty
   page -- goes to the repair model as a whole (the cell step is skipped when there is no suspect cell), through
   `read_page_guarded`, so a loop of the 8B is stopped and retried like the 4B's. The re-read replaces the page when it
   passes the gate and keeps the tables, 90 % of the *different* numbers and 80 % of the *different* words of the
   first reading (different, because a page with mixed-up columns repeats its figures and the correct reading is
   shorter); a runaway is replaced by any clean reading. `ALGO_VERSION` r2: pages whose repair was tried before are tried
   again. With the same model for both (`models.repair` set to the reader's) there is no escalation. In
   the original design: If cells remain suspect and the repair model is *not* the reader model, the whole page is read once more by the
   repair model and kept only if it passes the gate (ignoring `low_resolution`) **and** is not smaller than the first
   reading (`repair.not_smaller`: at least as many tables, 90 % of the numbers, 80 % of the text; a page that lost its
   table would otherwise pass the gate trivially). Any error in repair, an unreadable source included, is a note on the
   page, never a failure of the document.

The page is then gated again: `repaired` when it now passes and something was fixed, otherwise `low`. A page repaired earlier
stays repaired from the page cache even if repair is switched off later (the cache key is the reader's, not the repair
setting's); a second attempt on a cached page adds its cells to the first one's record. A scan under
150 dpi stays `low` even when repaired (the flag is about the page, the fix about a cell). The record gets
`repair {model, second, tier: cells|page, tried, fixed, cells:[{table,row,col,role,before,after,status,why,second,
read,expected}]}` (`status`: `fixed`, `unchanged`, `disagree`, `not_a_number`, `not_confirmed`, `not_located`,
`not_editable`, `error`), `time_s.repair`, and the tokens / GPU seconds of the cell reads are added to the page's.
The repaired Markdown replaces the cached page (the cache entry carries `repair.tag` = method version, repair
model, second reader), so a page is not tried again with the same models, a failed attempt included; another
repair model or second reader tries again. An attempt that could not run at all -- no memory free just then, the
reader crashed (`complete: false`) -- is not remembered as an attempt: the tag is left empty and the next run
tries again (a page that timed out is remembered: it would time out again). Without a second reader (not a Mac, ocrmac missing, `off`) cells are
not repaired and the page's note says why; sources are never touched. `indexer.repair` (`auto` | `off`,
`RAG_SEARCH_REPAIR`, `--repair`) switches the step off.

**Tables across pages** (`reconcile.merge`, run by the converter after every page of the document is read, on cached
pages too): when the last thing on page N is a table and the first thing on page N+1 is a table of the same
width whose first row is the same header again or plainly data (an amount with decimals or digit grouping, or a date; not a row of years, of 1 2 3 column numbers or
column titles), they are one table. A headerless continuation gets the header (so its chunks say what the columns
are, and its own checks work); the *merged* table runs through the validators; suspect cells that only the merged
table shows (the first row of page N+1 against the last balance of page N, a Total that spans the break) are
recorded on the page they sit on (`reconcile {role: starts|continues, with, header: added|repeated, rows, ok,
violations}`, gate check `table_across_pages`) and the page becomes `low`. They are flagged, not repaired (a suspect is identified by its text *and its row*, so a boundary suspect is not hidden
by a local one with the same figure; the `page` event of the live view is sent before this step, so the live counters can
show `pass` for a page the trace and the summary then call `low`). Pages stay
separate (page markers untouched), so a chunk still has one page number and the chunker is unchanged.

**Flagged pages reach search.** A chunk of a `low` page gets `metadata.confidence = "low"` (absent otherwise,
written at index time from the page records), search results carry `"confidence": "low"`, the CLI prints
"(low-confidence page: check the source)", the MCP `rag_search` description tells the model to caveat such a hit,
and the Search playground shows a chip. The collection view's conversion block lists the documents and pages that
need attention. Documents indexed before this version get the flag when they are next indexed.

* **Not verified in CI:** the cell locator and acceptance rule are tested against fake readers with synthetic
  boxes; Apple Vision's real output (phrase granularity, the bottom-left origin of its boxes), the crop reads of a
  real model, and the repair rate on real scans are measured only on a Mac with the gold set (P1).

#### 5.1.2 Measuring conversion (`core/conversion/bench.py`, phase P1)

Before any reader is changed, how well a page is read is *measured* on pages with known text:

* A **gold set** is `<home>/conversion_gold/<set>/gold.json` plus `images/<id>.png`: pages of your own
  documents, each with a failure `class` (`digital-table`, `scan-table`, `devanagari-text`, `image`, ...,
  from `bench.classify` on the trace records), the checked `truth` Markdown, optional `truth_tables`
  and search `queries`, and the source's SHA-256 (a changed source is skipped, not mis-scored).
  `rag-search bench gold init SET` samples pages per class from the stored traces (one page per
  document first), renders them and **pre-fills `truth` with what the pipeline read**; such entries are
  `"verified": false` and are not measured until a person has corrected the text against the image and
  set `"verified": true` (`--drafts` overrides: comparing the pipeline with itself).
* A **run** (`rag-search bench run SET [--engine E] [--name N]`) reads the verified pages with an
  *engine* (`engines.py`: `read_pages(src, pages) -> {pages: {n: markdown}, seconds, pages_read}`;
  `current` = whole-document docling with today's settings; `module:attr` for another) and stores
  `<home>/conversion_bench/<set>/<run-id>.json`: per page and per class **numeric cells exact** (same row
  and column; rows are matched by their text cells, so one lost row does not shift the rest) and *found
  anywhere*, **CER** (bit-parallel Levenshtein on `tables.plain_text`), **table similarity** (cell
  sequence edit distance -- not the published TEDS), **balance checks** (the P4 validators on the
  predicted tables), **search phrases found**, s/page and the process's CPU time and peak memory.
  `bench compare SET A B` shows the change of every measure per class and the pages that got worse.
* Sources are resolved through registered locations only and opened read-only; runs
  write nothing outside `conversion_bench/`, never the index, markup or serving folders. The dashboard's
  Playground tab lists sets and runs and compares two (`GET /api/conversion/bench`, `bench-run`,
  `bench-compare`); creating and running are CLI-only because they load readers and take minutes.
* `tables.py` (pipe and HTML tables with row/column spans, Indian and western number grouping,
  `Dr`/`Cr`, parentheses) and `validators.py` (running balance in either row order -- one damaged amount
  fails one row, a wrong balance fails two consecutive rows and names the balance cell -- and totals; each
  violation carries a *hypothesis* value; titles printed on two or three lines, as in a Marathi/Hindi/English passbook, are stacked into one header before the columns are recognised) are stdlib-only and shared by the metrics, the gate and repair.

### 5.2 Search flow (search daemon, `core/search.py`)

```
 request: query, top_k (1..25, default 5), collections (optional), client identity,
          stages (default bm25+dense+rerank), retrieval_pool/rerank_pool/rrf_k (optional
          troubleshooting overrides, clamped server-side -- see below)
        │
        ▼  access rules: no collections named -> every collection this client may use;
        │  a named one must exist AND be allowed, else `bad_request: unknown collection`
        │  (a restricted collection is indistinguishable from a missing one)
        ▼  models still loading? -> wait up to wait_s, then `warming_up`
        │
        │   for every collection in scope (indexes are already in memory: nodes, vectors, BM25)
        ├───────────────────────────────────────────┬──────────────────────────────────────────┐
        ▼                                           ▼                                          │
  KEYWORD BRANCH (skipped if "bm25" not          DENSE BRANCH (skipped if "dense" not           │
  in `stages`)                                   in `stages` -- the embed call itself           │
  tokenize(query)                                is skipped, not just the dot product)          │
  BM25 score of every chunk                      bge-m3 encode(query)  ─▶ q (unit vector)        │
  keep the best pool_n chunks with score > 0     cosine = embeddings @ q                         │
                                                  keep the best pool_n chunks                     │
        │  ranked list A                            │  ranked list B                           │
        └─────────────────────┬─────────────────────┘        pool_n = max(4 x top_k, 20)        │
                              ▼                              by default, or `retrieval_pool`     │
      both stages active?  ──yes──▶  RECIPROCAL RANK FUSION (RRF)                                │
              │no                    score(chunk) = 1/(k + rank in A) + 1/(k + rank in B)        │
              ▼                      (k = 60 by default, or `rrf_k`; rank starts at 1; a chunk   │
      rank by that one stage's       missing from a list adds nothing for that list; chunks are  │
      raw score instead -- RRF       keyed by (collection, id), so all collections share one     │
      over one list is just its      pool)                                                       │
      rank order, discarding                        ▼                                            │
      the score magnitude            keep the top max(3 x top_k, 15) chunks, at most 60 by       │
                                      default, or `rerank_pool` (clamped to <= 100)               │
                              ▼                                                                 │
              CROSS-ENCODER RERANK   BAAI/bge-reranker-v2-m3   (skipped if "rerank" not in       │
              reads (query, chunk) TOGETHER for each pair -> relevance in [0, 1] (sigmoid)       │
              `stages`, or if the reranker is disabled -- disabled is reported as `rerank_error`,│
              it is never lazily loaded mid-search just because a request asked for it)          │
                              ▼                                                                 │
              sort by rerank score, keep top_k                                                  │
              (reranker off or failing -> keep the prior order and report `rerank_error`)        │
                              ▼
 result per hit: rank, score (the order key: rerank_score, else rrf_score, else the single
   active stage's raw score), rrf_score, bm25_score/bm25_rank, dense_score/dense_rank,
   rerank_score (each null when that stage didn't produce the hit or wasn't run), collection,
   file, source, page, heading, text (up to 1200 characters), and `confidence: "low"` when the chunk's
   page was flagged by the quality gate (5.1.4)
 timing: retrieve/rerank/total (as before) plus the effective stages/retrieval_pool/rerank_pool/
   rrf_k used, bm25_candidates/dense_candidates, overlap_count/bm25_only_count/dense_only_count,
   and a top1/gap per stage (bm25_gap/dense_gap/rerank_gap) -- a large gap between the best and
   second-best candidate in one stage is a sign that stage found a clearly-best passage that
   fusion or reranking might bury
```

Why two stages: BM25 finds exact terms (identifiers, command names, version numbers) that
embeddings blur; embeddings find paraphrases that share no words with the question. RRF merges the
two rankings without having to calibrate their very different score scales (BM25 is unbounded,
cosine similarity is `[0, 1]`). The cross-encoder is too slow to run on every chunk, so it only
re-orders the few dozen candidates that survived fusion. `rag_grep` / `rag-search grep` is a
separate path: a regular-expression scan of the converted Markdown (in an isolated child process
with a timeout), for exact text.

**Troubleshooting/tuning knobs** (`rag-search search --stages/--retrieval-pool/--rerank-pool/--rrf-k
--explain`, and the dashboard's Search tab): every one of the overrides above is optional and
additive -- a request that sets none of them gets exactly the response shape and pool sizes this
diagram already describes by default, so nothing that reads today's `search()` reply breaks. The
overrides exist to isolate one retriever (does BM25 alone find it? does dense alone?), or to see
whether a different RRF constant or pool size changes the outcome, without needing a code change
or a daemon restart. They are validated and clamped at the request boundary (`search_daemon.py`
for the socket protocol; `spec.clamp_retrieval_pool`/`clamp_rerank_pool`/`clamp_rrf_k` for the
ceilings) before ever reaching the engine, so a bad or oversized value from the CLI or the
dashboard can never touch the model. The MCP `rag_search` tool does not expose these overrides --
they are a human-debugging surface, not extra knobs for an LLM caller.

Backends are pluggable through `RAG_SEARCH_EMBEDDER` / `RAG_SEARCH_RERANKER` (`module:Class`); the
tests use a hashed bag-of-words embedder so the whole stack, including real daemon and worker
processes, runs without downloading models.

### 5.3 Models (`models.py`, `model_tasks.py`)

* `models.py` (stdlib only) holds the catalogue (`ModelSpec`: size, memory, licence, backend, query prefix,
  library requirements; the document reader models are a separate catalogue, 5.1.3), the selection (`selection()` = `$RAG_SEARCH_MODEL` / `$RAG_SEARCH_RERANK_MODEL`, else
  `config.json` `models.*`, else the default; `paths.model_name()` / `rerank_model_name()` use it, so every process
  agrees and reads the file each time), the fit estimate (weights, index, 1.5 GB overhead against 60% of RAM or
  `models.memory_limit_gb`), what is in the Hugging Face cache (`cache_state`), and what a switch costs
  (`reindex_estimate` from the per-document `index.meta.json` files).
* `model_tasks.py` runs one task at a time (`run/models.task.json` + a lock held by the runner; a record that
  says "running" without the lock is reported as failed): **download** (`snapshot_download` in a thread,
  progress = bytes in the cache blobs / the size the hub reports), **verify** (a subprocess loads the model and
  checks three test questions each rank their own passage first), **switch** (plan → download → verify → write
  `config.json` → apply). Applying a reranker calls `reload` on the search daemon; applying an embedding model
  starts `index new` (every document is stale because `is_fresh` includes the model; the converted Markdown is
  reused through its `.sha256` sidecar). The CLI runs a task in the foreground; the dashboard starts it detached
  (`GET /api/models` polls the record).
* `core/embedding.py` has three backends chosen from the catalogue: `Embedder` (sentence-transformers, with the
  optional query instruction for Qwen3-Embedding), `Reranker` (cross-encoder) and `Qwen3Reranker` (a causal LM
  scored as P("yes")). Outputs are checked for NaN/inf, and `RAG_SEARCH_DTYPE` forces the precision. Both
  rerankers honour `RAG_SEARCH_RERANK_BATCH` / `RAG_SEARCH_RERANK_MAX_LEN` (`spec.py`: `rerank_batch`,
  `rerank_max_len`).
* Loading is **offline** when it can be (`embedding._load_model`, `embedding.offline`): a model whose complete
  copy is in the Hugging Face cache is loaded with `local_files_only=True` *and* with the libraries in offline mode
  (`HF_HUB_OFFLINE`, `TRANSFORMERS_OFFLINE`, set for the load in the environment and in the already-imported
  settings, and put back afterwards). `local_files_only` alone was not enough: transformers starts a background
  check for a converted copy of the weights on the Hub, four requests naming the model at every daemon start and
  every embed phase (seen in the logs until 0.9.24). With both, no request leaves the machine, and one with blocked
  or no internet starts as quickly as any other; `HF_HUB_DISABLE_TELEMETRY=1` is set as well; if that fails (an older library, an incomplete cache) the normal
  load runs, and a failed download of a model that is not cached is explained with
  `model_tasks.explain_download_error`. On PyTorch < 2.6 the transformers `torch.load` safety check is relaxed
  for trusted `BAAI/*` checkpoints **only while that one model loads** (`trusted_load`) and restored afterwards
  (in every transformers module), so a later, untrusted model in the same long-lived process is checked again;
  model loads in one process are serialised so two overlapping loads never see each other's relaxation.
* `models.cached_revision(id)` reads the commit `refs/main` points at in the cache; the embedder records it and
  the indexer writes it to `index.meta.json` as `model_revision` (a model id names a repository, not fixed weights).
* Publishing cannot mix embedding models (`publish._collect`), so a half-finished re-embedding never goes live.

### 5.4 Playground (`core/playground.py`)

A playground experiment is a sandbox for trying a different embedding model, reranker, chunk size
or pool/RRF tunable against a small sample of documents, and for benchmarking the result, without
any of it touching production.

```
 <home>/playground/<name>/
   locations.json               its own source folders (registered like production's with `playground source`
                                 or `create --from`; read in place, never copied)
   workspace/index/<coll>/_all/ nodes.json, embeddings.npy, merge.manifest.json (own indexer_workspace)
   config.json                  embedding_model, rerank_model, reader_model, repair_model,
                                 chunk_size, chunk_overlap, rerank, stages, retrieval_pool,
                                 rerank_pool, rrf_k, and every conversion tunable (ocr, table_mode, ...)
   jobs/<run-id>.json           one index / bench run: status, progress, summary (its own, never production's)
   jobs/<run-id>.events.jsonl   the run's event log: stage, page, progress, doc and query events
   jobs/<run-id>.{log,pid}      the child process's output and pid
   bench/queries.jsonl          labeled queries (user-authored; a .example template is created)
   bench/runs/<run-id>.json     one recorded benchmark run
```

**Structural isolation.** `paths.get_playground_paths(base, name)` builds a full, independent
`Paths` (own `home`, `docs`, `workspace`/`index`, `serving`, `run`, `jobs`) rooted at
`playground/<name>/` inside the real `$RAG_SEARCH_HOME`. `serving/` and `run/` exist on
that dataclass for structural completeness but nothing in the playground code path writes to them
(`jobs/` is the experiment's own, see "Runs" below):
there is no generation/publish machinery (no `serving/gen-N`, no `serving/current` symlink swap,
no `catalog.json`) and no daemon. Every `rag-search playground ...` command loads the small index
and the experiment's own models in that one process and exits — the same execution shape as
`rag-search index foreground` — so a playground run can never be mistaken for, or interfere with, a
production generation. `test_production_untouched_by_playground` snapshots production's `serving/`
tree, `serving/current` target, `config.json` and `run/` listing before and after a full
create→index→search→bench→remove cycle and asserts byte-for-byte equality.

**Reuse, not reimplementation.** `build_index()` calls `core.indexer.run_index()` directly — the
exact same function production indexing uses — pointed at the experiment's own `Paths` and
documents. `search()` and `bench()` construct a `core.search.SearchEngine` and populate
`engine.gen = Generation(indexes={...})` straight from the experiment's
`workspace/index/<coll>/_all/` directories, bypassing `prepare_generation()`/`catalog.json`/
`serving/current` entirely, then call the engine's real `search()` — the same BM25/dense/RRF/
rerank/stage-toggle logic described in §5.2, with the same `--stages`/`--retrieval-pool`/
`--rerank-pool`/`--rrf-k`/`--explain` overrides, runs unmodified. Nothing about ranking or fusion is
duplicated for the playground, so there is no drift risk between what an experiment shows and what
production would do with the same models and tunables.

**Model resolution.** `paths.configured_model()` (and therefore the bare `model_name()`/
`rerank_model_name()`) always reads `default_home()/config.json` — production's config — regardless
of what `Paths.home` a caller actually built with. Naively calling `run_index()` or constructing
`SearchEngine()` for a playground experiment would therefore silently pick up production's model
instead of the experiment's own. This is avoided two ways: `run_index()` takes an additive
`model: str | None = None` override (defaults to `model_name()` when omitted, so production
behaviour is unchanged) which playground code always sets explicitly from the experiment's
`config.json`; and `SearchEngine` is always constructed with pre-built `embedder=`/`reranker=`
objects (never `None`), which makes it skip its internal `model_name()` calls altogether. Because
`SearchEngine.search()`'s lazy-load guard (`if self.embedder is None: self.load_models()`) is
skipped once the embedder is already set, playground code calls `engine.load_models()` explicitly
right after construction.

**Same pipeline, same numbers.** An experiment's index build is `run_index()` with the experiment's own
`Paths`, so it goes through exactly the stages of 5.0 (2 Fingerprint, 3 Convert with 3.1-3.5, 4 Chunk,
5 Embed, 6 Write, 7 Merge); 8 Publish does not exist in a sandbox. The experiment's settings are grouped by
those stages in the Playground tab: the document reader model (3.2) and the repair model (3.4) -- both
pinnable per experiment through `reader_model` / `repair_model`, blank meaning production's choice from the
Models tab --, the conversion tunables (3.2 / 3.4, the same `spec.py` registry the Settings tab uses), chunk
size and overlap (4), the embedding model (5), and the search defaults (S2-S6). `experiment_env()` builds the
environment a run applies with the same `effective.settings_env` the production indexer daemon uses for its
workers: an environment variable the dashboard was started with still wins, a blank / 0 value adds nothing,
the pinned reader/repair model becomes `RAG_SEARCH_VLM_MODEL` / `RAG_SEARCH_REPAIR_MODEL` (which is what
`models.reader_choice()` reads), and the *hardware* settings of the Models section (device, precision, batch
sizes, memory limit) follow production because they describe this computer. The environment is applied for
the one call (`_experiment_env`) and removed afterwards. `effective_settings(base, name)` /
`rag-search playground settings NAME` / `POST /api/playground/settings` report what the next run really
uses, per stage, with where each value comes from (this experiment, production config, environment, default)
-- the same `effective.resolve` as production, so the Playground shows the truth, not a copy of the form.
Promoting carries a pinned reader / repair model (`models.set_vlm_selection`) along with the models, chunk and
search settings; conversion tunables are not promoted (they are production's Settings tab).

**Runs and live progress (`playground_runs.py`, stdlib only).** An index build or a benchmark takes minutes, so
the dashboard starts it detached (`POST /api/playground/index|bench` answers at once with the run id) and
watches it the way production watches an indexing run -- with the same machinery. The child
(`rag-search playground index|bench NAME --job ID`) keeps a job record `jobs/<id>.json` (status queued ->
running -> done / failed / cancelled, progress, summary, its pid) and an event log `jobs/<id>.events.jsonl`:
`stage` events with the stage number, one `page` event per converted page (branch, outcome, cache, read time,
characters, tokens, model, failed gate checks), `progress` and `doc` events from `run_index`, and for a
bench one `query` event per query with the time each search stage took. `jobs.documents` and
`conversion.runview` read these exactly as they do for production, `playground_runs.view()` adds a per
document timeline (its stage events and pages), and `POST /api/playground/run` returns it all: the Playground
tab polls it once a second and shows the numbered stage strip (1-7, 3.1-3.5) with counts, the pages read
right now, the workers, every document with its stage chips and a page grid (a page table with branch,
outcome, time, model and gate checks opens under it), and for a bench every query. A run whose process died
is marked failed with the tail of its log (`_settle`), one run per experiment at a time is enforced,
`POST /api/playground/cancel` stops it, `POST /api/playground/runs` lists the history and `rag-search
playground status NAME` prints the same view in a terminal. Search is short and still answers at once (in a
child process, with the S2-S5 timing of that search). The dashboard never loads a model: runs and searches
are `rag-search playground ...` child processes (`ui/server.py`'s `_playground_cli` / `playground_runs.start`),
so a long-lived server can never race a per-experiment environment against another request.

**Benchmarking.** A label in `bench/queries.jsonl` is pinned to *document + page*
(`{"query": ..., "relevant": [{"file": "a.pdf", "page": 4}, ...]}`), not an exact chunk id, because
chunk boundaries shift whenever chunk params or the embedding model change — a label should still
mean the same thing after a model switch. `bench()` replays every labeled query through `search()`
and computes, per query and averaged:

* **Recall@k** — 1 if any of the top-k hits matches a labeled `(file, page)`, else 0.
* **MRR** — `1 / rank` of the first matching hit (0 if none in the top-k).
* **nDCG@k** — binary relevance, `dcg = Σ rel_i / log2(i + 2)` over the top-k, divided by the ideal
  DCG over `min(n_relevant_labels, k)` ideal positions.
* **latency** — mean, p50, p95 over the replayed queries.

Each run is tagged with the exact combo that produced it (embedding model, rerank model, chunk
size/overlap, stages, retrieval/rerank pool sizes, RRF k, k) and written to
`bench/runs/<run-id>.json`; `compare()` lists every recorded run for an experiment side by side so a
change in a tunable can be judged by its effect on the metrics rather than by eyeballing one search
at a time.

**The production bridge.** `production_snapshot(base)` reads production's *effective* settings
(`config.load_config(base)`, plus the same env-var-wins precedence `paths.model_name()` /
`rerank_model_name()` use) into the same shape as an experiment's `config.json`. Two entry points
build on it, and neither ever touches an index:

* `create_experiment(base, name, from_production=True)` seeds a new experiment's `config.json`
  from the snapshot instead of this module's own `DEFAULT_CONFIG`, so a benchmark starts from what
  real searches actually get (`rag-search playground create NAME --from-production`, or the
  Playground tab's "copy from production" checkbox).
* `promotion_preview(base, name)` diffs an experiment's config against the snapshot (only the
  fields that actually differ appear in `changes`), and calls `models.reindex_estimate()` -- the
  same estimate `rag-search models set` shows before an embedding-model switch -- whenever the
  diff touches `embedding_model`, `chunk_size` or `chunk_overlap`: those three change what a
  stored chunk/vector *means*, so the existing production index would be stale until it is
  rebuilt; a different reranker or search tunable takes effect on the next search/run with nothing
  to rebuild. `promote_to_production(base, name, confirm=False)` calls the preview first and
  raises `PlaygroundError` (never writing anything) when it needs a reindex and `confirm` was not
  passed; once confirmed (or when nothing needs a reindex), it writes only the fields that changed
  -- `models.set_selection()` for the embedding model/reranker (so a custom, non-catalogue id is
  validated the same way `rag-search models set` validates one), `spec.validate_section()` +
  `config.update_config()` for chunk size/overlap and the search tunables. `rag-search playground
  promote NAME --dry-run` / `--confirm` and the Playground tab's "Promote to production" button
  (which shows the same diff and reindex cost in a confirmation dialog) are the two front-ends.

### 5.5 Source locations, collection export/import and deletion

**Locations** (`locations.py`, stdlib). `locations.json` maps a collection name to a folder;
it is the only source of documents (no registered location = nothing to index, `locations.NO_LOCATIONS`).
`paths.SourceRoots` (the locations, plain data so it travels to the conversion processes) is what
`mirror_rel`, `index_dir_for` and `markup_path_for` take, so a document gets the workspace path
`<name>/<path inside the folder>`; a file in no location raises `ValueError`. `add` refuses taken names
(a location, an import) and folders that overlap the data folder or another location. A
collection is a registered location or an import and nothing else; an index folder that is neither is not
listed (`catalog.known_names`) and a full indexing run removes its derived data (`ScanPlan.orphans`,
`summary.orphans_removed`). A playground experiment has its own `locations.json` (its
`Paths` is rooted in the experiment), so the same module, planner (`plan_scan`/`run_plan`) and
document/page/Markdown APIs serve it (`?exp=NAME` on the dashboard's `/api/conversion/*`).
`resolve_target` turns an `index new PATH` argument into a folder (a path starting with a location's
name means a folder inside it; an imported collection's name is refused with an explanation).

**Export/import** (`bundle.py`, stdlib). An export is one `.rag.tgz`: `manifest.json`,
`index/_all/{nodes.json,embeddings.npy,merge.manifest.json}`, `index/docs/<doc>/index.meta.json`,
`markup/<doc>.md`. The manifest records the embedding model and weights commit, vector size,
chunk/tokenizer/index-format versions, conversion settings, document and chunk counts, the
description and a SHA-256 per file; `src_path` is reduced to the file name everywhere and
`access.json` is never included. Export takes `index.lock` (a consistent workspace) and refuses a
collection whose documents disagree on model or chunking. Import, also under `index.lock`:
manifest format/version/type check → **model gate** (the configured embedding model must equal
the manifest's and, when something is published, the model it was built with -- a switch in
progress blocks imports; when both sides know the weights commit it must match too; no partial
import) → name
check (a location, an indexed or published collection is never overwritten; an
earlier import only with `replace`) → unpack into a hidden staging folder in the workspace, refusing
anything but plain files at safe relative paths that are listed in the manifest, and verifying every
checksum and the `.npy` shape against `nodes.json` (read from the header, no numpy) → assemble the
collection folder → write `collection.origin.json` → rename into `index/<name>/` + `markup/<name>/`
(an earlier import is parked and put back if either rename fails) → publish. The marker makes
indexing skip the collection (no scan, prune, merge or wipe); publish lists it with
`origin: imported`; after a later model switch an imported collection blocks publishing with an
explanation, as any mixed-model workspace does.

**Deletion** (`lifecycle.py`, stdlib, administrator: CLI and dashboard, never MCP). Works for every
kind of collection. Under `index.lock`: remove `index/<name>/` and `markup/<name>/` (asserted to lie
inside the workspace) and nothing else, then publish with `allow_drop`. Source documents are never
touched; the access rule, the description and a location's registration are kept, so a collection
whose documents are still in place is rebuilt by the next run with its restrictions intact (the
result says so in `note`/`sources_remain`). `location remove` = delete + unregister.

**Inventory** (`inventory.py`, stdlib, administrator view: shows folder paths, so CLI/dashboard only,
never MCP). `collection_info(name)` combines: the source folder (reachability; one walk counting
supported/unsupported files and bytes -- sources are listed and stat'ed, never opened), the workspace
(Markdown and index folders with file counts and sizes, merged vs. per-document index bytes, complete
and interrupted per-document indexes, the merge manifest), the published generation (documents,
chunks, folder; hard links, so no extra space), the build metadata from `index.meta.json` (model and
weights commit, dimensions, chunking, first/last indexed, build/convert/embed time), the newest job
that touched the collection (per-status counts and error messages from its event log), documents not
indexed yet (with the last run's reason) and documents modified since they were indexed (source mtime
newer than `built_at`; no hashing). From these it derives one state: `ok`, `pending`, `unpublished`,
`unreachable`, `not_indexed` or `imported`.

## 6. The tunables registry (`spec.py`) and `config.json`

Every knob that shapes a search or an indexing run beyond the small set in §2's `config.json`
table -- pool sizes, RRF k, default stages and `top_k` for search; chunk size/overlap and the
docling conversion settings for indexing; batch sizes, max sequence length, dtype and device for
the models -- is described exactly once, as a `spec.Tunable` in the `spec.TUNABLES` tuple:
section (`search`/`indexer`/`models`), key, kind (`int`/`choice`/`text`/`stages`), the `0`/`""`
"use the built-in default" sentinel, a label, a one-line "what it means" (always shown), a longer
"impact" note (shown behind an info icon), which of three tiers it belongs to, its CLI flag, and,
for indexer/model tunables, the environment variable it feeds at the point of use. Every consumer
below reads this one list instead of re-describing or re-validating the same knob:

* **`config.py`** extends `DEFAULTS` with every tunable's key (documentation/defaults only --
  it does not import `spec`, to avoid a cycle).
* **`cli.py`**'s `config set` generates one flag per tunable from `TUNABLES` (`--chunk-size`,
  `--ocr`, `--embed-batch`, ...) and validates with `spec.validate_section()`, so a new tunable
  needs no parser change.
* **`core/search_daemon.py`** reads `search.*` fresh from `config.json` on every request
  (`stages`/`retrieval_pool`/`rerank_pool`/`rrf_k`/`top_k` all fall back to it when a caller does
  not ask for something different), and applies `models.*` to the process environment
  (`apply_model_env()`, `os.environ.setdefault`, so an already-set real env var always wins) once,
  right before constructing a real `SearchEngine` -- never for an injected fake one, so tests are
  unaffected.
* **`core/indexer_daemon.py`** reads `indexer.chunk_size`/`chunk_overlap` into each job's spec at
  `_validate_spec()`, and turns every `indexer.*`/`models.*` tunable with an `env` into a
  subprocess environment variable for the worker it spawns (`_worker_env()`, same
  `setdefault` precedence).
* **`ui/server.py`** exposes `GET /api/config` (values + the full `tunables()` list from
  `ui/info.py`) and `POST /api/config/set` (validate + `config.update_config` + nudge the live
  feed), read by the dashboard's Settings tab (search + indexer tunables) and the Models tab's
  *Advanced* section (model-runtime tunables) -- both build their form once per tunable
  (`core.js`'s `tunableField()`/`tunablesForm()`) from the same list `config set --help` uses, and
  the Search tab's advanced panel shows the same values as placeholders instead of a hardcoded
  formula, so a debug override is visibly relative to the production default.
* **`core/playground.py`** validates a `chunk_size`/`stages` override with the same
  `spec.parse_stages()`, and `promote_to_production()` (§5.4) validates what it writes back with
  `spec.validate_section()` -- one registry, one set of rules, whichever surface changes a value.
* **`ui/info.py`**'s `config_storage()` (which keys live in which `config.json` section, the precedence built-in
  default → `config.json` → environment variable, who reads each section) stays in `/api/architecture`; the
  Architecture tab no longer shows it as a card since 0.9.27 (the Settings tab shows every value and its source).

**What a value may be (0.9.28).** `spec.LIMITS` gives every whole-number tunable a range (a chunk of 32 to 8,192
tokens, a stall limit of one minute to seven days, a batch of at most 1,024 ...); `validate_tunable` refuses a value
outside it, and `spec.check_chunking` refuses an overlap of more than half a chunk, checked against the default of
whichever of the two is not given (the chunker clamps to the same half, so a value that reaches it another way still
makes progress). A `config.json` edited by hand is read through `config._sane`: a value of the wrong kind (text where
a number belongs, a negative number, a section that is not an object) is left out, the default is used, and
`load_config` returns one message that names each of them; a daemon no longer fails to start on such a file. Keys
the program does not know are kept as they are. `update_config` holds the file lock while it reads and writes.
`tests/portable/test_tunables.py` walks the registry: every tunable takes its sentinel and its good values and
refuses the bad ones, reaches the environment of the process that reads it, loses to a variable that is already
set, and is shown with the right source; every `RAG_SEARCH_*` name in the code is in `README.md`; and a launchd
service keeps every variable a tunable or a stage reads (`service.PASS_ENV` is derived from the registry; until
0.9.27 it was a list written by hand that lacked the reader, repair, Tesseract, lane and stall variables, so those
were lost when the daemons ran as services).

"When does a change take effect" is one of three tiers (`spec.IMMEDIATE`/`NEXT_RUN`/`RESTART`),
shown as a pill next to each tunable and next to each group on the Settings tab:

| Tier | Meaning | Tunables |
|---|---|---|
| immediate | applies to the very next search | all of `search.*` |
| next-run | applies to the next indexing run; no daemon restart | all of `indexer.*` |
| restart | needs `rag-search daemon restart` | all of `models.*` (batch/seq/dtype/device) |

## 7. Failure behaviour

| Situation | Behaviour |
|---|---|
| search daemon crashes / is killed | next client call restarts it (or launchd does); it loads `serving/current` |
| indexer daemon stopped by a signal or `daemon stop` | the reason is logged and stored on the run (`interrupted`); the next start cleans up |
| indexer daemon crashes mid-run | worker keeps running until the next daemon start, which kills it and marks the run `interrupted`; rerun is cheap |
| worker killed by a signal (e.g. out of memory) | run `failed`, error names the signal (`SIGKILL`) |
| a document that cannot be indexed and has not changed (password-protected or damaged PDF, no text at all, the name of another file) | an error of the run that finds it; update runs after that list it as "not tried again" and can end `succeeded`; a complete run tries it again (`outcome.json`, `left_out.json`, see Freshness) |
| one document too slow (`RAG_SEARCH_DOC_TIMEOUT`) | that document is an error, the run continues (`partial`), retried next run. The limit is docling's, per call: a whole document, or one run of up to 20 pages of a routed PDF; the document reader has its own limit per page (`RAG_SEARCH_VLM_PAGE_TIMEOUT`) |
| a conversion process hangs (a call into native code that never returns: an Apple Vision request, a render of a file a cloud app never delivers) | **stall watch** (`core/stallwatch.py`, 0.9.25): every conversion process says what it does in the event log (`work`, `stage`, `step`, `page`); one with a document open that writes nothing for `indexer.stall_timeout` (default 1 hour, never less than the document timeout plus 15 minutes; `RAG_SEARCH_STALL_TIMEOUT=0` = off) is killed, its document is an error ("stalled: no progress for 60 min on page 12 (document reader) ...", retried next run, pages already read are in the page cache) and the run goes on. Time is counted in observed poll intervals, so a laptop that slept has not stalled. With `jobs=1` (and in the Playground) documents are converted in the run's own process, which can only be ended: the run fails with that message. The indexer daemon is the second line: a run whose event log does not grow at all for twice the limit is stopped and marked `failed` |
| a conversion process ends abruptly (killed by the system for memory, a crash in native code, stopped by the stall watch) | Python marks the whole pool broken and fails every waiting document; `indexer._convert_in_pool` keeps the finished documents, puts the rest in a **new pool**, and leaves out only a document that was open in a process that died twice (`STRIKES`), named as the likely cause; at most 8 new pools per run. Before 0.9.25 every document still waiting was reported "worker failed" |
| a conversion process cannot exit (docling abandons its OCR / layout threads after a document timeout; one stuck in a native call, e.g. an Apple Vision request, is joined forever by the interpreter) | the pool is closed without waiting (`indexer._shutdown_pool`): every result is already collected, so a process still alive 20 s later is terminated and the run goes on to embedding. The worker and `python -m rag_search.cli` children (Playground runs) end with `worker.leave` (`os._exit` after flushing), so a stuck thread cannot keep a finished run "active" |
| worker crashes | run `failed`, nothing published, workspace stays consistent (documents are complete only when `index.meta.json` exists) |
| the folder a daemon was started from is deleted (e.g. `install.sh` run inside a release folder that is rebuilt later) | no effect since 0.9.13: long-lived processes start in the data folder. Before, every document of the next run failed (`FileNotFoundError` from `os.getcwd()`, "partially initialized module 'torch'") until the daemons were restarted |
| a source is an online-only file of a cloud-storage folder (Box, Google Drive, iCloud, OneDrive under `~/Library/CloudStorage`) | read normally: every rag-search process switches on "fetch online-only files when read" for itself (`paths.allow_cloud_files`, macOS `setiopolicy_np`), which launchd-started daemons do not have by default. Before 0.9.15 each such file failed at once with `[Errno 11] Resource deadlock avoided`. The cloud app downloads the file; if it cannot (app not running, offline) that document is an error with that explanation and is retried next run |
| second `start` during a run | returns the active run; `restart` kills and restarts |
| bad `config.json` | defaults are used; the error is shown by `doctor` and in `ping` |
| models still loading | `list`/`grep` work; `search` says `warming_up` and the client may retry |
| generation with another model | followed: its embedder is loaded aside and swapped in with it (an explicitly injected embedder refuses it) |
| a model switch that is not finished (some documents on the old model) | `publish` refuses to mix models; the old generation and the old model keep serving; `index new` finishes it |
| a model download is interrupted / a model fails its test | the cache resumes; `config.json` is not changed until the test passed |
| a search's `stages` drops both `bm25` and `dense`, or names an unknown stage | `spec.parse_stages` rejects it (`bad_request`) before the request reaches the engine or the search lock |
| a search asks for the `rerank` stage but the reranker is disabled in `config.json` | reported as `rerank_error` in the reply; the reranker is never lazily loaded to satisfy a per-request override (only a genuinely enabled reranker is ever loaded) |
| a search's `retrieval_pool`/`rerank_pool`/`rrf_k` override is unreasonably large | clamped to a fixed ceiling (`spec.RETRIEVAL_POOL_MAX`/`RERANK_POOL_MAX`/`RRF_K_MAX`) regardless of caller, so it can never make one search scan or rerank an unbounded number of candidates |
