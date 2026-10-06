# Design documents

The architecture and design record of rag-search lives in this repository, next to the code.
**Read the relevant documents before changing anything, and update them in the same change.**

| Document | What it is |
|---|---|
| [`../../ARCHITECTURE.md`](../../ARCHITECTURE.md) | The current architecture: processes, data folder, protocol, layering, indexing/search flows, models, playground, locations/export/import/deletion, tunables, failure behaviour. **The source of truth for how the code works today.** (Shipped in the package and shown on the dashboard's Help tab.) |
| [`../../README.md`](../../README.md) | User-facing behaviour: commands, MCP tools, dashboard, configuration. |
| [`../../CONTRIBUTING.md`](../../CONTRIBUTING.md) | Code map, conventions, how to test and release. |
| [`enhancement-plan.md`](enhancement-plan.md) | The phased plan (0.8.0): what was done, decisions taken, what is still open. |
| [`architecture-critical-review.md`](architecture-critical-review.md) | Critical review of 0.7.8 (37 findings, prioritised); the open items are tracked in the plan. |
| [`flexible-docs-location-analysis.md`](flexible-docs-location-analysis.md) | Analysis behind source locations, deleted/unreachable sources, export/import and deletion. |
| [`containerisation-analysis.md`](containerisation-analysis.md) | Single-machine assumptions, input for running rag-search in a container. |
| [`containerisation-plan.md`](containerisation-plan.md) | The agreed containerisation baseline (phase 1); not implemented yet. |
| [`document-conversion-analysis.md`](document-conversion-analysis.md) | Critical evaluation of page-level routing / quality gate / VLM fallback for scanned PDFs, tables and images; options A–D and a phased plan (C0–C3). Proposal, not implemented. |
| [`test-strategy.md`](test-strategy.md) | The three test tiers (portable / real / machine): what each needs, runs on and proves, the rules that keep them separate (A is hermetic), the corpus, measured coverage and cost, what the new tests found. Read it before adding or moving a test. |
| [`document-conversion-plan.md`](document-conversion-plan.md) | Implementation plan for the new conversion pipeline: modules, per-page tracking (branch, time, cost), dashboard views (mockup), Architecture-tab flow chart with tools, phases P0–P5. P0 (tracking) done in 0.8.3; P1 (measurement harness), P2 (routing, page cache, gate) and P3 (document VLM reader) and P4 (cell repair, tables across pages, low-confidence flag) built; P5 in progress; see its "Status after 0.9.1-0.9.9" for what real documents showed and the open items. |
| [`conversion-routing-plan.md`](conversion-routing-plan.md) | Proposal (after 0.9.22): layered profiler (3.1A document facts, 3.1B page-image facts, 3.1C probe hooks), four read runways (3.2a docling, 3.2b docling+OCR, 3.2c docling+VLM regions, 3.2d VLM page), gate v2 with escalation, a routing harness, phases R0-R7 and a critical analysis. Fallback policy and non-Mac models deferred to R7. Nothing built. |

These notes are internal: they are not packaged and are left out of the external release build.

## Working rule

1. Before a change: read `ARCHITECTURE.md` and the design notes that touch the area (the code
   map in `CONTRIBUTING.md` says which module does what).
2. Check the change against the decisions recorded here (and the open items in the plan); call out
   anything that contradicts one rather than silently diverging.
3. In the same change: update `ARCHITECTURE.md` / `README.md` (and copy them into
   `src/rag_search/ui/static/docs/`, or run `scripts/build_release.sh`, which does it -- a test
   checks the copies), and update the plan's status when an item is done or a decision changes.

## Superseded: the built-in docs folder

The analyses and plans above that mention `<home>/docs`, `RAG_SEARCH_DOCS`, the `default` collection or
"docs-folder collections" describe the first design. They are kept as the record of why locations were
introduced. The docs folder has since been removed: every collection comes from a registered location
(`locations.json`), a playground experiment registers its own source folders the same way, and an index
with no registered folder and no import is a leftover that a full run removes. `ARCHITECTURE.md` §5.5
is the current description.
