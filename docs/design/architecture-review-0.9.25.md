# Review of 0.9.24: correctness, stalls, reporting, architecture (leads to 0.9.25)

A review of the code after the four lanes of step 3.2 were wired, asked for with three questions: is it correct,
can a run stall, and does the dashboard report what really happens. What was found is fixed in 0.9.25 unless it
is listed under "Open". `ARCHITECTURE.md` describes the result; this note records the findings and the assessment.

## What was read

Line by line: `core/conversion/` (`routed`, `router`, `gate`, `vlm`, `vlm_worker`, `repair`, `degenerate`, `residue`,
`scanfacts`, `tesseract`, `pagecache`, `runview`, `trace`), `core/indexer.py`, `core/worker.py`,
`core/indexer_daemon.py`, `core/daemon_base.py`, `jobs.py`, the view of `playground_runs.py`, the model loading of
`core/embedding.py`. Searched for one kind of fault (a wait without a time limit, a subprocess without one): the
whole of `src/`. Looked at in the running dashboard, with a real run: every tab. **Not** read line by line: the search
path (`core/search.py`, `search_daemon.py`, `bm25.py`), `publish.py`, `locations.py`, `bundle.py`, `cli.py`,
`ui/server.py`, `models.py`, `model_tasks.py`, `chunker.py`, `tables.py`, `validators.py`, `reconcile.py`, `layer.py`.

## Findings that were fixed

| # | Finding | Effect | Fix |
|---|---|---|---|
| 1 | Nothing watched a conversion process. A reader call into native code that never returns has no time limit, and `as_completed` then waits for that document for ever | a run that never ends and shows "working" | `core/stallwatch.py`: a process with a document open that writes nothing to the event log for `indexer.stall_timeout` (1 h) is killed, its document reported as stalled, the run goes on; the daemon stops a run whose log is silent for twice the limit |
| 2 | When one pool process ended abruptly (out of memory, a crash in a library) Python failed every document still waiting, and the run reported them all as "worker failed" | one bad document or one memory spike cost the rest of the run | `indexer._convert_in_pool`: finished documents kept, the rest in a new pool, only a document open in a process that died twice is left out |
| 3 | The document reader read runs of up to 20 pages per call; pages were stored, gated and reported only when the run returned | up to 20 minutes of silence, 20 pages arriving at once, a cancelled run lost the whole run of pages | one page per call |
| 4 | `plan` events were dropped by the line filter of the log reader (`runview._read_new`); the unit tests fed the parser directly | "Pages in active files" never showed the kinds of pages in a real run (every file "not profiled yet") | filter fixed; a test goes through the file |
| 5 | A run's documents were keyed by collection and file name | two files with one name in two folders were one row: the last full run showed 1,945 of 2,059 indexed and 69 of 88 failed; in the Playground a nested document stayed "working" | `doc` events carry `path`; lists and counts keyed by it |
| 6 | Loading a cached model still contacted huggingface.co (transformers' background check for converted weights): four requests naming the model at every daemon start and every embed phase | the claim "search never uses the network" was not true | cached models are loaded in offline mode; verified in the daemon log |
| 7 | A repair that could not run (no memory free, the reader crashed) was stored as an attempt | the page was never repaired on any later run | only a completed attempt is remembered |
| 8 | After a doubted OCR page and a failing reader, docling's OCR read the page a second time and the result was thrown away; a Tesseract first try was labelled docling's | wasted read, wrong label | the first reading is kept directly |
| 9 | A text page handed to the reader as an image (garbled layer, lane a or c) was judged by the dpi of a picture on it | false `low_resolution` | `routed.gate_profile` |
| 10 | The reused reader result of an escalated text page could lead to a `KeyError` when the cached entry was a pre-guard runaway | the document fell back to whole-document conversion | checked before the page is switched |
| 11 | `Worker.read` restarted its timeout for every line the child printed; the loop check ran only when the token count was a multiple of 64 | a page could outlive its limit; a loop could go unnoticed if tokens arrived in chunks | one deadline per page; check every 64 tokens or more |
| 12 | The stage shown for a working process was the last stage that reported "done" | "stage 3.1 Profile" during a ten-minute read | a reader call (`step` event) sets 3.2 Read / 3.4 Repair; the Workers card shows the call, the page and for how long |
| 13 | The lane switches existed only as environment variables | they could not be set from the Settings tab or per Playground experiment | `indexer.ocr_first`, `residue`, `escalate_digital`, `layer_fill`, `stall_timeout` are settings |
| 14 | `claude` and `launchctl` were run without a time limit; the "OCR mode" setting described forced OCR although page routing ignores it | a registration could hang; a misleading setting | limits; text corrected |

## Assessment of the architecture

**What holds up.** The split into two daemons and a killable worker; the event log as the single source of the
dashboard's live view (it made the stall watch a reader of a file, with no new channel); derived data only, sources
read-only; `index.meta.json` written last; the page cache keyed by page content, reader and settings (a re-run after
any failure costs only what was not done); every optional reader behind a child process or a time limit; the
registry of stages and tunables that the CLI, the dashboard and the documents all read.

**Weak points, in the order I would address them.**

1. **There is no ground truth.** Every gate threshold and the decision to keep OCR first off rest on comparing a
   cheap reader with the document reader, which is itself sometimes wrong. A verified set of 50 to 100 pages (text
   checked by a person, taken from the page classes that matter: statements, deeds, forms, Devanagari) would turn
   "differs from the reader" into "wrong", and is the precondition for switching any cheap lane on.
2. **One document reader per conversion process.** With two conversion processes (the default above 17 GB) there are
   two reader processes and, when repair re-reads, two repair models: memory doubles, and both share one GPU. The
   full run read a scanned page in about 56 s with two processes; a Playground run with one read the same kind of
   page in about 22 s (different documents, so an indication, not a measurement). One reader service per run with a
   queue would halve the memory and cannot be slower. Worth measuring before building: `jobs=1` against `jobs=2` on
   one scan-heavy folder.
3. **A change of conversion settings re-embeds everything.** The fingerprint holds the conversion profile, so a
   version bump or a lane switch converts every document again (cheap: the page cache) and then embeds every chunk
   again (1 h 20 min on the full set) although most Markdown is unchanged. An embedding cache keyed by model and
   chunk text would make such changes cost only what changed.
4. **`routed.py` carries too much state.** One class, 950 lines, six dictionaries keyed by page number (results,
   keys, tags, escalations, first tries, hand-overs), and the page's mode mutated when it is handed on. Half the
   faults in the table above (8, 9, 10) were of this kind. When the lane design has settled: one small object per
   page with an explicit path (`planned -> read by -> gated -> handed on -> final`), and lanes as readers with one
   interface.
5. **With one conversion process the run converts in its own process** (`jobs=1`, every Playground run). A stall can
   then only be answered by ending the run. Converting in a pool of one would make it the same as the general case.
6. **Progress is counted in documents.** The estimate of the time left is the average time per finished document
   times the documents left; one 300-page scan among 2,000 text files makes it meaningless. The `plan` events now
   give pages by kind per document as soon as it is profiled: an estimate in pages by lane is possible.
7. **The document timeout means something else since page routing**: it limits one docling call (a run of up to 20
   pages), not a document. Nothing limits a document as a whole; with the stall watch nothing has to, but the
   setting's name promises more than it does.
8. **Traces written before 0.9.24 have no lane.** The dashboard works the lanes out from the branches and says so;
   only a re-conversion records them.

## Not verified

A real hang in native code cannot be produced on demand: the stall path was exercised with a process that sleeps and
with processes that kill themselves (`tests/portable/test_stallwatch.py`, real processes), not with a hung Apple
Vision request. The daemon's second line is tested with an embedder that sleeps. No full production run was made
with 0.9.25.
