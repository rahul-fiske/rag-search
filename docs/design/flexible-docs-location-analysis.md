# Flexible docs location, collection export/import, lifecycle — critical analysis

> Status (0.8.0): implemented -- source locations, deleted/unreachable source handling, export /
> import and collection deletion. Decisions that differ from this analysis are listed in the
> status section of `enhancement-plan.md`; the current design is in `../../ARCHITECTURE.md` §5.5.
> 0.8.2: deleting a collection now removes only its workspace (Markdown + index) and keeps its
> access rule, description and location registration; see `enhancement-plan.md` "Since then".

Proposal under review (as stated 2026-10-02): allow configuring multiple source-docs
locations (one location = one collection); reorganize the index workspace so each
collection's data is private to rag-search and exportable/importable for sharing;
tag imported collections so they survive indexing cycles; support deleting a
collection's index/metadata without touching source docs; enforce source docs as
strictly read-only; and make `install.sh`/the uv environment self-sufficient for
either generating or importing collections. Two further considerations were added
2026-10-02 (see the end of §1 and the revised §3): precise source
changed/deleted/inaccessible semantics, and an export/import packaging + model-match
requirement.

This doc is the critique and recommendations pass. Each point below gets detailed
out in a later phase.

## 1. Multiple source doc locations, one location = one collection

Real need. Today `docs/` is a single root and a collection is just a first-level
subfolder of it (`$RAG_SEARCH_DOCS` relocates the whole root, not per-collection).
Pointing rag-search at an existing vault, a Downloads folder, and a network share at
once currently means symlinking or copying into that one root.

Not a config tweak: collection identity currently comes from a file's *position*
under one root (`paths.mirror_rel()` / `collection_of()`). This needs a registry —
collection name → source root path — stored the same way `access.json` /
`descriptions.json` already are (mtime-cached, atomic-written JSON).

`RAG_SEARCH_DOCS` becomes one registry entry (or a deprecated single-location
shorthand) rather than disappearing outright, for backward compatibility.

### Source lifecycle: changed / deleted / inaccessible (added 2026-10-02)

Three distinct cases were called out explicitly, and the current code treats them
with varying degrees of correctness — this needed grounding in what `core/indexer.py`
actually does today, not assumption:

**Changed.** Already handled correctly today, confirmed by reading the code: each
per-document index dir carries a `.sha256` sidecar; a changed source hashes
differently and is reconverted/re-embedded on the next `index new`. No gap here —
this is the "current capability" the proposal correctly credits.

**Deleted — not actually cleaned up today; this is a real, present-day gap, not
just a future risk.** Walking the exact mechanics: `_live_docs()` filters
`per_doc_dirs()` down to documents whose `src_path` still passes
`Path(src).exists()`; `merge_collection()` then rebuilds the merged `_all/` index
from only that live set. That correctly makes a deleted document disappear from
search results. But `merge_collection()` never deletes the dead document's own
per-doc index directory or its converted `markup/<coll>/*.md` file — it only
excludes it from the merge. Those per-doc artifacts are reclaimed **only** by
`_wipe()`, and `_wipe()` is wired to run **only** on `rag-search index all` /
`playground index --wipe` (`wipe=spec.get("mode") == "all"` in `core/worker.py`,
confirmed by reading `cli.py`'s argparse setup for the `index` subcommands and the
`playground index` command). Crucially, `_wipe()` operates over the *current scan's*
source list — i.e. files that still exist right now — and is a "delete everything in
scope, then force a clean rebuild" mechanism (hence `rebuild = rebuild or wipe` and
`force_md = force_md or wipe` immediately after it runs), not a "find and remove the
documents whose sources are gone" mechanism. The two are easy to conflate from the
name, but they're orthogonal: `--wipe` never looks at what's missing, only at what's
present.

Net effect, verified rather than assumed: deleting a file from `docs/` today leaves
its per-doc index directory, embeddings, and converted Markdown on disk forever —
invisible to search (fine) but never reclaimed (not fine; unbounded disk growth, and
nothing in `rag_list_collections`/`list_view` surfaces these as orphaned garbage to
clean up). **Recommendation:** the normal incremental `index new` pass (not just the
destructive full-rebuild path) should compute, per collection, the set of per-doc
index dirs present on disk whose source no longer appears in the current scan, and
actually delete that per-doc dir + its markup `.md` + its `.sha256` sidecar — exactly
the cleanup the proposal asks for, and a fix worth shipping regardless of whether
multi-location support ships, since the bug already exists in the single-root case.

**Inaccessible — must not be treated as deleted, and today's code cannot tell the
difference.** `_live_docs()`'s check is a bare `Path(src).exists()`; there is no
intermediate state. For today's single fixed `docs/` root this rarely bites in
practice (the root itself is rarely unmounted mid-run), but it becomes a real
correctness bug the moment source locations can be removable drives or network
shares — precisely the scenario multi-location support introduces. The fix needs a
location-level reachability check *before* any per-file existence check: if a
registered location's root itself fails to list/stat, that whole location is
"skipped: unreachable" for this run — no pruning, no merge changes, previous index
for that collection left exactly as-is, with a clear status surfaced (not silently
folded into "0 live docs" the way an empty/unmounted directory would otherwise look
to `_live_docs()` today). Only a *reachable* location gets to assert "this specific
file is gone, prune it." This is the same category of fix as the deletion cleanup
above: a location-reachability gate has to sit in front of the per-document
existence check that already exists, not replace it.

## 2. Index workspace per collection, private to rag-search

Already true today. `indexer_workspace/markup/<coll>/` and
`indexer_workspace/index/<coll>/_all/` are already collection-scoped directories,
untouched by anything outside rag-search's own daemons. No new work here — good to
know so effort goes to export/import, not restructuring something already in place.

## 3. Export / import of a collection index

The real new feature, and the trickiest. Revised 2026-10-02 with two explicit
requirements: additional metadata travels with the export, and import enforces an
embedding-model match; export is a single compact archive file, not a loose
directory.

**Packaging: a single archive file, not a directory tree.** Export produces one
`.tgz` (or equivalent single-file archive) containing the merged index
(`nodes.json`, `embeddings.npy`, `merge.manifest.json`), the converted Markdown
(`markup/<coll>/*.md` — required for `rag_grep`, which scans Markdown directly, not
the index), and a manifest file. Import consumes exactly that one file. BM25 does
not need separate packaging — it's rebuilt in memory from `nodes.json` at load time.

**Manifest metadata (exported alongside the index, checked on import).** At minimum:
embedding model id + revision, embedding dimension, chunk size/overlap,
chunker/tokenizer versions, document count, the collection's `description`, and a
rag-search version/schema marker for the bundle format itself (so a future
incompatible bundle format fails with a clear message instead of a crash deep in
import code).

**Model match is now a hard gate, not a soft fallback.** The original pass through
this analysis had floated "import but mark unusable for dense search (BM25-only)
until the model matches" as one option alongside outright refusal. That's now
superseded: the explicit instruction is that import checks the manifest's embedding
model against the importer's configured model and **allows the import only on a
match** — otherwise refuse outright, with a clear error naming both models, no
partial/degraded import path. This is simpler to implement and avoids a
silently-half-working imported collection (dense search quietly absent, discoverable
only by someone noticing search quality is off). `publish.py`'s existing
"refuse to mix embedding models within one generation" rule is the precedent to
extend here, same gate, same bluntness.

**What should NOT travel with an export**: `access.json` rules (meaningless outside
the exporter's own client-identity scheme). Worth an explicit audit of the manifest
and `nodes.json` for anything that leaks the exporter's local filesystem layout —
`document["source"]` is already collection-relative, not an absolute path, which is
good, but confirm nothing else rides along before shipping this.

## 4. Tag imported collections so the indexing cycle doesn't overwrite them

Correctly flagged in the proposal, and it's more than a tag — it's a real landmine
in the current code, not a hypothetical one. Today, `publish.py`'s merge step treats
"source file not found under docs/" as "document was deleted" and drops it from the
merged index on the next publish. An imported collection has no source file
anywhere, under any registered location, so without this fix the very next
`index new` / publish cycle would silently erase it. (This is the same
`_live_docs()` mechanism detailed in §1 — an imported collection's documents are
indistinguishable from "all sources deleted" unless explicitly exempted.)

Recommended fix: an explicit `origin: imported | generated` marker per collection
(stored in the collection registry from #1, or a sibling file). The source scanner
skips imported collections entirely; the publish/merge step skips source-presence
pruning for them.

Recommend keeping origin **collection-level and binary** for v1 — no mixing
generated and imported documents inside one collection. Multi-collection search
already lets a client query "my docs" and "their shared collection" together in one
request, so there's no benefit to merging origins inside a single collection, only
added edge-case surface (partial freshness, partial pruning, partial rebuild
semantics).

`rag-search index all` (full rebuild) targeting an imported collection should be a
clear refusal ("this collection was imported; there's no source to rebuild from —
delete and re-import"), not a silent empty result.

## 5. Delete a collection (index + metadata only, source docs untouched)

Mechanically straightforward given the already-collection-scoped directory layout.
Needs to sweep: `indexer_workspace/markup|index/<coll>/`, future `serving/`
generations (old ones age out via the existing "last 3 kept" rotation), the
`access.json` entry, the `descriptions.json` entry, and the new origin/location
registry entry from #1/#4. Confirmation flow should match the codebase's existing
pattern (`playground rm NAME --yes`, `rag_index_rebuild confirm=true`) rather than
inventing a new one.

Needs to interact correctly with the indexer daemon's single-run-at-a-time lock:
refuse or cancel-then-delete if a run targeting the collection is active.

Opinion, not settled: be more conservative about exposing collection *deletion* as
an MCP tool than `rag_describe_collection` was. A wrong description is low-stakes
and reversible; an LLM calling a delete tool by mistake is not. Worth a deliberate
decision (confirm=true AND possibly admin/cli-only) rather than defaulting to "same
openness as describe."

## 6. Source docs strictly read-only, never modified or deleted

Already true as an accidental property of the current code — nothing in
`core/indexer.py` or `publish.py` ever opens a source file in write mode. The
proposal's value is turning "happens to be true" into an enforced guarantee, which
matters more once rag-search is pointed at other people's folders (vaults, shared
drives) rather than only its own `docs/`. This guarantee needs to keep holding under
§1's new deletion-cleanup recommendation too — that cleanup deletes rag-search's own
derived artifacts (per-doc index dir, markup `.md`, sidecar), never anything under a
source root; worth calling out explicitly in the implementation so the two "delete"
operations (derived-artifact cleanup vs. a source file) are never reachable through
the same code path.

Recommendations: (a) a single `read_source()` helper every read path funnels
through, so there's one place to audit instead of an implicit convention; (b) a
static test in the style of the codebase's existing layering tests (e.g. "only
`mcp/` imports `mcp`") asserting no write-mode file open occurs under a registered
source root; (c) defense in depth via the OS — mounting source locations read-only
(`:ro`) inside a container gives this guarantee for free, and ties directly into
the containerization track already in progress (see
`containerisation-plan.md`).

## 7. Install sets up everything needed for either generating or importing

Mostly already true (`install.sh` already does the full `uv tool install` + model
download). Real refinement: a pure *importer* (never converts their own documents)
doesn't need docling or any OCR engine at all — only the embedding model and
reranker, for query-time search over imported vectors. A slimmer install profile
for that case is worth adding, not just restating current install behavior.

Model default: keep defaulting to `BAAI/bge-m3` as the common baseline, precisely
*because* it maximizes compatibility for shared/imported collections (see #3's
model-compatibility risk) — not a reason to casually change the default later.

## Cross-cutting

**This is 4-5 separable features, not one change**: a collection-location
registry, export, import, delete, and read-only enforcement — plus, as of the
2026-10-02 additions, a sixth: deletion/reachability cleanup for the existing
single-root case, which stands on its own and should not wait for #1. Recommend
sequencing export/import *ahead of* multi-location support (#1) rather than treating
#1 as a prerequisite — packaging a finished collection has nothing to do with where
its source lived, so export/import works fine against today's single-docs-root
model. The deletion-cleanup fix in §1 can and should ship independently and first,
since it's a correctness fix to existing behavior rather than new surface area.

**Name collisions on import**: two people can independently export a collection
called "manuals." Needs an explicit policy (refuse / rename-on-import / overwrite
with confirmation) decided up front, not left implicit.

**Relationship to the containerization track**: this export/import mechanism is
arguably a cleaner answer to "share with others, but not everything" than the
whole-container-with-mounted-volume approach discussed in
`containerisation-analysis.md` and `containerisation-plan.md` — a
collection export *is* the curated-share-folder idea from that discussion, just
formalized as a first-class rag-search feature instead of a manual file-copy
convention. Worth an explicit decision on how the two tracks relate (does
containerization ship first and this rides on top, or does export/import make a
"share via container" need smaller than originally scoped) before detailing either
further.
