# Line-by-line check of 0.9.27: correctness, tunables, tests (leads to 0.9.28)

Asked for with three tasks: check every line for correctness, validate every tunable, prune redundant tests.
`ARCHITECTURE.md` section 6 describes the result for the tunables; this note records what was read, what was
found, and what was left.

## What was read

Line by line, the part the review of 0.9.24 had left out: `core/search.py`, `core/bm25.py`, `core/search_daemon.py`,
`publish.py`, `core/chunker.py`, `core/conversion/tables.py`, `validators.py`, `reconcile.py`, `layer.py`, `paths.py`,
`locations.py`, `bundle.py`, `grep.py`, `policy.py`, `config.py`, `core/docling_convert.py`, `core/embedding.py`,
`models.py`, `api.py`, the first 700 lines of `cli.py`, the request handling of `ui/server.py`. Checked by wider lint
rules (ruff B, PLE, PLW, RUF, ARG, PERF: nothing that is a defect) and by the tests, **not** line by line: the rest of
`cli.py` and `ui/server.py`, `model_tasks.py`, `core/playground.py`, `inventory.py`, `access.py`, `catalog.py`,
`descriptions.py`, `lifecycle.py`, `client.py`, `effective.py`, `ui/info.py`, `ui/markdown.py`, `mcp/server.py`, the
conversion tools (`bench`, `engines`, `metrics`, `estimate`, `costs`, `records`, `synth`, `routeharness`,
`applevision`, `pagemd`) and the JavaScript.

## Findings that were fixed

| # | Finding | Effect | Fix |
|---|---|---|---|
| 1 | A value of the wrong kind in `config.json` (`"jobs": "two"`, a section that is a number) reached the code as it was | a daemon that fails at start or in the middle of a run with a `TypeError` | `config._sane`: the value is left out, the default used, the file's error message names it |
| 2 | `update_config` read, changed and wrote the file without the lock | two writers at once (the dashboard and the CLI) could lose one change | under `file_lock` |
| 3 | No whole-number tunable had an upper limit; an overlap could equal or exceed the chunk size | a chunk size of 1 or an overlap as large as the chunk made the chunker crawl; a stall limit of 1 s stopped every run | `spec.LIMITS`, `spec.check_chunking`, and the chunker clamps the overlap to half a chunk |
| 4 | `service.PASS_ENV`, the variables a launchd service inherits, was a hand-written list without `RAG_SEARCH_VLM*`, `REPAIR*`, `TESSERACT*`, the lane switches and `STALL_TIMEOUT` | a variable set in the shell worked for `rag-search index` and was silently dropped by the installed services | derived from the registry of tunables and stages |
| 5 | `.md` and `.txt` sources were copied byte for byte | a UTF-16 or Windows-1252 text file became unreadable Markdown, or failed later at chunking | `docling_convert.read_text_file`: byte-order mark, UTF-8, UTF-16, then cp1252 |
| 6 | The separator-row pattern of the table code could backtrack on a long line of blanks and bars from a runaway reading | seconds per line | `tables._SepRow`: only short lines of the right characters reach the pattern |
| 7 | `RAG_SEARCH_TABLE_MODE=`, `PIPELINE=`, `PDF_BACKEND=` set to nothing were an error where every other variable took nothing as "default" | a run refused for an empty variable | empty means default |
| 8 | A request with text where a number belongs (`wait_s`, `context_lines`, `max_matches`) raised in the search daemon; a collection name that cannot exist raised in `api.search`/`api.grep` | an `internal` error, or a traceback in a caller, for a bad request | `bad_request` |
| 9 | `rag-search index status` listed a document by file name | two files with one name in two folders looked like one | the path |
| 10 | Seven variables the code reads were not in the README's table | undocumented switches | added; a test compares the code with the README |

## Found and left as it is

* **A text page's cache key is its size and its text, not its pictures.** Two pages with the same text and different
  large pictures share the entry of lane c (the pictures' reading). Rare (a template filled with different photos);
  changing the key would read about 400 picture pages of the present index again. To do with the next change of the
  page cache version.
* **Walking a source folder has no time limit of its own.** A network folder that hangs during discovery is ended
  only by the daemon's second line (twice the stall limit).
* **The scans for an unclosed `<table` or `<!--`** are quadratic in the worst case on a runaway reading; the runaway
  guard cuts such readings before, so it was not seen in practice.
* **`RAG_SEARCH_OCR` with a value that is not a mode means `auto`**, by an old decision (it predates the registry);
  the same value in `config.json` is refused.

## The tests

Coverage per test (`coverage` with one context per test function, tier A): 1,088 tests with a context, 308 of which
run only lines that one other test also runs. That is not redundancy by itself: nearly all of them assert a
different outcome of the same code (another input, another branch of a result). Read pair by pair, with the pairs
of similar text added, six tests asserted what another test asserts and were removed (two 'is a validated tunable' tests and the bad-number test, covered by `test_tunables`; the copy of the header and footer test in `test_mine_traces`; 'nothing registered' in `test_enhancements`; 'unknown action' in `test_playground_ui`).

## The logs of a full run with 0.9.27 (2,170 files, 20,197 pages, 94 minutes)

Read after the run: the job's event log (59,584 events), its output, the three daemon logs, and the 2,067 traces.
No stall (the longest silence was 21 s), no process lost, no request to the network, every page accounted for
(18,782 from the page cache, 301 read). The run was a full rebuild, so 86 of the 94 minutes are embedding
(18 chunks a second throughout). It ended `partial` because of 69 password-protected PDFs.

| # | Finding in the logs | Effect | Fix |
|---|---|---|---|
| 11 | "Ink" is the share of pixels darker than the ground. Three images with a dark ground and light content (9 to 28 % of the pixels lighter) had no ink, were taken for blank pages and never read | "no text", with the advice to install a reader that was installed | `profiler.ink_and_hash` also gives `light`; a page is blank only when both are under the limit. A file of blank pages now says "every page is blank" |
| 12 | 1,021 digital pages `low` by `coverage`, 348 documents "low confidence". 498 of them had no missing line and 90 % or more of both recalls; of 310 looked at token by token, the missing numbers were 1,766 single digits and 310 longer ones | a flag that marked complete pages, mostly register descriptions | see `ARCHITECTURE.md` 5.1.3: single digits are not counted, a page with no missing line and 90 % is intact. 621 of the 1,021 clear |
| 13 | Two PDFs that no backend could open were reported as "no text extracted (scanned PDF? ...)" | advice for scans on a damaged file | `docling_convert.damaged_pdf_reason`: "cannot be opened as a PDF ...: damaged or incomplete" |
| 14 | 17 files left out because another file in the folder has the same name with another extension were in the run's errors (88) and not among its documents (71 failed) | two counts that disagree; the files not in the list | a document event for each |
| 15 | The daemon log named a finished or failed document without its folder, the stage lines with it | one document under two names in one log | the path |
| 16 | The dashboard log held a traceback for every browser connection that was closed | noise | `UiServer.handle_error` ignores a connection that went away |

| 17 | A remembered failure was not converted again but was put into the run's `errors` again | every update run of a folder with a protected PDF ended `partial` (69 "errors" each time here), so a real new failure did not stand out | `known`: listed as "not tried again" outside the errors; damaged PDFs and name collisions are remembered too; a complete run tries and reports all of them (`ARCHITECTURE.md`, Freshness) |
| 18 | In the Documents list the counts on the status filters ignored the text, collection, branch and outcome filters, and "skipped" was three separate things | "Skipped (46)" above an empty list when a word was still in the search box | the counts follow the other filters; one filter **Skipped** = unchanged + not tried again + unsupported, each row with its reason |

Seen and left: four Word files that hold only pictures (pasted scans) have no text, because the pictures of an
Office file are not read by any reader; three files without a usable extension (a PDF named `...8.4`, a JPEG and a
PNG with none) are "unsupported", since the kind of a file is taken from its name; `a.pdf` and `a.jpg` in one
folder are one document name, so the second is left out (17 files); the job's output is mostly progress bars of the
model loader and docling's warnings about boxes outside the page.

## Not verified

Tier C needs the GPU; whether it ran for this version is said in the commit. The launchd service change was checked
by its unit test and by reading the plist it writes, not by installing the services again.
