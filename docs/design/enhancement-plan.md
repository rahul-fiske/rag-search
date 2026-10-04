# rag-search enhancement plan (consolidated)

Combines three pieces of analysis already on record in this project:
`flexible-docs-location-analysis.md` (multi-location sources, export/import,
collection lifecycle), `architecture-critical-review.md` (whole-codebase
critical review of indexing/search/models/storage), and
`containerisation-plan.md` / `containerisation-analysis.md`
(packaging/sharing track). This doc sequences all of it into phases. Nothing here
is itself a new analysis — see those docs for the full reasoning behind each item;
this is the "what order do we actually do this in" layer.

## Since then

- **0.8.1**: the design record moved into the repository (`docs/design/`, indexed by
  `docs/design/README.md`; `CLAUDE.md` and `CONTRIBUTING.md` point to it). Per-collection details
  were added: `rag-search collection info NAME` and the Collections tab's expanded row (state,
  counts, folders, sizes, dates, last run, documents needing attention), from the new
  `inventory.py` (`ARCHITECTURE.md` §5.5).
- **0.8.2**: deletion semantics changed at the owner's request -- **delete removes only the
  workspace** (converted Markdown + index), for every kind of collection (docs folder, location,
  import); source documents are never touched, and the access rule, description and a location's
  registration are kept (a collection whose documents remain is rebuilt by the next run, with its
  restrictions). The earlier refusal for a docs-folder collection whose folder still holds documents
  is gone. Add collection (location), import, export (with browser download) and delete are now
  dashboard actions on the Collections tab, so the Phase 3 decision below is revised: deletion is
  administrator-only (CLI + dashboard, typed-name confirmation), still never MCP. The Architecture
  tab gained SVG pictures of the system, the indexing pipeline and the search pipeline.

## Status (2026-10-02): all four phases implemented in rag-search 0.8.0

Released as 0.8.0 (internal and external zips), synced to the Mac, 478 tests passing (51 new, in
`tests/test_enhancements.py`), ruff clean. An independent review pass of the finished change
found ten issues (four of them derived-data-loss paths); all were addressed, with regression tests, before
release. Decisions taken while implementing, where the plan left a choice open or the
implementation differs from the plan's wording:

- **Phase 0 #1** reverses a deliberate earlier design choice: a test had pinned "a
  differently-cased spelling is not an alias". Names now resolve exactly first, then
  case-insensitively, and always to the real spelling (ambiguous spellings are still refused).
- **Phase 0 #9** narrows locking in the engine instead of the daemon. Each query snapshots
  (generation, embedder, reranker); only model calls are serialised.
- **Phase 1 #3**: the "imported" marker is a `collection.origin.json` file in the collection's
  workspace folder, not an entry in the location registry. The indexer sees it directly.
- **Phase 1 #4**: `RAG_SEARCH_DOCS` stays the docs root (fully backward compatible). Registered
  locations (`locations.json`) are added alongside it rather than replacing it.
- **Phase 1 #5**: `locations.read_source()` exists. Enforcement is a behavioural test rather than
  a static scan: indexing (with pruning and a full rebuild), export and delete run over read-only
  source folders, and the test checks every byte, mode and mtime afterwards. `convert-legacy`, the
  one command that wrote into sources, now keeps originals unless `--delete-originals` is passed.
- **Phase 2 name collisions**: refused by default; `--as NAME` imports under another name;
  `--replace` replaces only an earlier import (never a docs folder, location or generated
  collection). Import also waits while a model switch is still in progress.
- **Phase 2 #4**: `install.sh --import-only` skips the OCR engine and docling's conversion
  models. The docling package stays a dependency, since making it optional would change every
  existing install.
- **Phase 3**: deletion is CLI-only (`collection delete`, `location remove`). It is not an MCP
  tool and not a dashboard button. A test enforces that the adapter cannot reach it.
  *(Revised in 0.8.2: also a dashboard action, workspace-only -- see "Since then".)*
- **Beyond the plan, from the review pass**:
  - Indexing refuses to run while `locations.json` is unreadable.
  - A sub-folder that cannot be listed freezes its collection.
  - A completely empty location, or relocated docs folder, counts as not mounted.
  - A case-only folder rename on macOS no longer re-indexes, because pruning also compares by
    file identity.
  - Import stages into a fresh folder, validates manifest types and rolls back a failed replace.
  - Model loads are serialised, so overlapping loads cannot interfere with the torch.load relaxation.
  - Exports are created world-readable.
  - The external release build now excludes tool caches.

Still open:
- The lower-priority review findings in `architecture-critical-review.md`: BM25
  compound-token length bias, table-split header loss, OCR-force default, orphan-worker
  detection without `ps`, grep subprocess per call, silent >64 MB grep skip, CUDA memory
  estimate, non-Latin truncation logging, sha pre-filter, per-document conversion timeout,
  `service.PASS_ENV` drift.
- The containerization-vs-export decision below.

## Guiding principle from the architecture review

The review's overall verdict was **improve, don't rewrite** — the layering (api.py
as shared entry point, thin CLI/MCP/UI adapters, warm-model daemons, atomic
publish) is sound. The recurring failure pattern was "a new store/feature doesn't
reuse an existing shared helper" (cache, lock, atomic-write, env-wiring). That
pattern is the reason Phase 0 below exists and comes first: several of the new
features in this plan (the location registry, export/import) are exactly the kind
of new metadata store that caused this pattern before, so the shared helpers need
fixing/hardening *before* a fifth bespoke copy gets added.

## Phase 0 — Fix the foundations before building on them

Small, independent, high-value fixes from the architecture review that either (a)
are correctness bugs worth shipping on their own regardless of anything else, or
(b) would otherwise be duplicated a second time by Phase 1/2 work below. All are
targeted fixes or localized refactors, no rewrites, per the review's verdict.

1. **Case-fold `policy.resolve_scope`** (policy.py:83-98) — currently the only
   place in the codebase that compares collection names case-sensitively; it's on
   the live `search`/`grep` path. Fix before Phase 1 adds more locations/collections
   where case variation is more likely.
2. **Fix the same-stem single-file-indexing collision** (paths.py:316-323,
   worker.py:83-86) — silent data loss today (indexing `report.docx` can
   silently delete `report.pdf`'s index). Independent of everything else; ship
   first.
3. **Add file locking around `access.json`/`descriptions.json` read-modify-write**,
   and collapse the three atomic-JSON-write implementations onto one
   (`paths.write_json_atomic` + optional `mode`). Do this *before* Phase 1's
   collection-location registry is built, so the registry can reuse the fixed
   pattern instead of becoming a fourth bespoke store.
4. **Give `descriptions.py` a cache analogous to `AccessStore`**, and have
   `api.py` actually use `AccessStore` instead of raw `load_rules()` calls. Same
   reasoning as #3 — fix the pattern once, then reuse it for the new registry.
5. **Ship the deleted-source cleanup fix** (incremental `index new` reclaims
   orphaned per-doc index/markup artifacts, not just full `--wipe`) — detailed in
   `flexible-docs-location-analysis.md` §1. This is both a standalone correctness
   fix and a prerequisite for Phase 1's location-reachability handling, since the
   two share the "is this genuinely gone, or just not visible right now"
   distinction.
6. **Dedupe collection scope end-to-end** (search.py:269-270 and upstream) — stops
   RRF double-counting; independent, no dependency on anything else.
7. **Reranker tunables + offline-first model loading** (core/embedding.py) — wire
   `Reranker` up to `RAG_SEARCH_RERANK_BATCH`/`RAG_SEARCH_RERANK_MAX_LEN`, and add
   `HF_HUB_OFFLINE`/`local_files_only` when a model is already cached, routing load
   failures through `model_tasks.explain_download_error`. Both directly serve the
   network-blocked deployment this tool targets; independent of the rest of this
   plan but worth bundling into the same work session as the model-identity fix
   below since they're in the same file.
8. **Record the resolved model commit SHA**, not just the repo-id string, in
   `index.meta.json`/`cache_state` (models.py) — low cost now, and Phase 2's
   export/import model-match gate should check against this, not just the id.
9. **Narrow `search_daemon.py`'s global `_search_lock`** so concurrent queries
   aren't fully serialized — independent, no dependency on anything else.
10. **Scope the legacy-torch-load safety patch** to the single trusted call
    instead of leaving it globally disabled for the process — independent,
    security-adjacent, ship opportunistically alongside #7/#8 since it's the same
    file.
11. **Route `rag-search describe` through `api.describe_collection`**, and add the
    dashboard's missing `/api/describe` route — closes the CLI/MCP/UI parity gap;
    independent, low risk.

The remaining lower-priority architecture-review findings (BM25 compound-token
length bias, chunk-table-split header loss, OCR-force default, orphan-worker
detection without `ps`, etc. — full list in `architecture-critical-review.md`) are
not blocking for Phase 1/2 and can be picked up opportunistically or in a later
pass; they're not repeated here to keep this plan focused on what gates the new
feature work.

## Phase 1 — Flexible source locations

Full design detail in `flexible-docs-location-analysis.md` §1, §4, §6, §7.

1. Collection-location registry (name → source root), stored and locked the way
   Phase 0 #3/#4 fixed `access.json`/`descriptions.json` to be — not a new bespoke
   pattern.
2. Location-level reachability check ahead of per-file existence checks: an
   unreachable location is skipped for that run (no pruning, clear status), never
   treated as "every file under it was deleted." Builds directly on Phase 0 #5's
   deleted-vs-unreachable distinction.
3. `origin: generated | imported` marker per collection (collection-level, binary
   for v1) so the source scanner and publish/merge pruning skip imported
   collections entirely — needed before Phase 2 (import) can be safe to run
   against the existing indexing cycle, so it belongs in this phase even though
   it's only exercised once Phase 2 ships.
4. `RAG_SEARCH_DOCS` becomes one registry entry (back-compat shorthand).
5. Read-only source enforcement: a single `read_source()` helper, a static
   layering test (same style as the existing "only `mcp/` imports `mcp`" tests),
   and — for the containerization track — read-only (`:ro`) mounts as defense in
   depth.

## Phase 2 — Collection export / import

Full design detail in `flexible-docs-location-analysis.md` §3 (as revised
2026-10-02) and §4.

1. Export: single `.tgz` archive containing the merged index (`nodes.json`,
   `embeddings.npy`, `merge.manifest.json`), converted Markdown
   (`markup/<coll>/*.md`), and a manifest (embedding model id **+ resolved commit
   SHA from Phase 0 #8**, dimension, chunk size/overlap, chunker/tokenizer
   versions, document count, description, bundle schema version). `access.json`
   rules never travel with the export.
2. Import: hard model-match gate — refuse outright on any mismatch (id or,
   once Phase 0 #8 is in place everywhere, SHA), no partial/degraded BM25-only
   fallback. Tags the collection `origin: imported` (Phase 1 #3). Name-collision
   policy decided explicitly (refuse / rename-on-import / overwrite-with-confirm)
   rather than left implicit.
3. `rag-search index all` against an imported collection is a clear refusal
   ("imported; no source to rebuild — delete and re-import"), not a silent no-op.
4. Slimmer install profile for pure importers (no docling/OCR needed, only
   embedding + reranker for query-time search) — `install.sh` refinement from
   `flexible-docs-location-analysis.md` §7.

## Phase 3 — Collection deletion

Full design detail in `flexible-docs-location-analysis.md` §5. Sequenced after
Phase 1/2 because deletion needs to also sweep the location/origin registry
entries those phases introduce, not just today's `access.json`/`descriptions.json`.
Decide explicitly, before shipping: MCP exposure for delete should likely be more
conservative than `rag_describe_collection` was (confirm=true and/or admin/CLI-only)
given the asymmetric cost of a wrong delete vs. a wrong description.

## Relationship to containerization

Carried over from `flexible-docs-location-analysis.md`'s cross-cutting section:
export/import (Phase 2) is arguably a cleaner, more targeted answer to "share with
some collections but not others" than the whole-container-plus-mounted-volume
approach in the containerization docs — a collection export formalizes the
curated-share-folder idea as a first-class feature instead of a manual convention.
This is still an open decision, not resolved by this plan: does containerization
ship first and Phase 2 rides on top of it, or does Phase 2 shrink what
containerization needs to solve? Worth an explicit call before detailing the
containerization track further, rather than running both to completion
independently.

## Suggested sequencing summary

Phase 0 (foundations) → Phase 1 (locations) → Phase 2 (export/import) → Phase 3
(deletion), with the containerization-track decision made explicitly somewhere
before or alongside Phase 2. Phase 0 items are independent of each other and can
be parallelized across sessions/agents; Phases 1-3 have the dependencies noted
inline above and should go in order.
