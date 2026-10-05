# Test strategy: three tiers, chosen by what they need

Goal: the tests must run the same way in a cloud session, in Cowork and on a laptop, without setup or download
that the tests do not need, and the Mac is asked only for what only the Mac can do.

## The tiers

| Tier | Folder | Needs | Runs in | Proves | Setup cost |
|---|---|---|---|---|---|
| **A** portable | `tests/portable/` | Python, `numpy`, `pypdfium2`, `Pillow`, `mcp` (about 100 MB, a minute). No model, no docling, no torch, no network | cloud, Cowork, laptop, CI | all of the framework's own logic: scan, freshness, chunking, BM25, fusion, publish, daemons and worker as real processes (with a fake embedder), protocol, access rules, locations, export/import, CLI, dashboard, MCP, config, stages; the routing, gate, repair and cache logic with fake readers; the glue around docling and the embedding libraries against small fakes; every corpus file's scan and routing | none beyond `pip install` |
| **B** real | `tests/real/` | A + docling, torch, sentence-transformers, an OCR engine (tesseract, or Apple Vision on the Mac), two small models from Hugging Face (about 100 MB) and docling's own models (about 500 MB) | cloud (with Hugging Face reachable), Cowork, laptop | that the real libraries still do what the fakes assume: docling converts every kind of file in the corpus, OCR reads a scan, the real embedder and reranker load and rank, the whole stack indexes the corpus and answers | `scripts/cloud_setup.sh`: about a minute, then 600 MB of models once |
| **C** machine | `tests/machine/` | an Apple Silicon Mac, the default models, MLX | the Mac only | MLX, Apple Vision, the document reader (a vision model), the default models | the user's own environment |

Rules that keep the tiers honest:

1. **A never depends on anything B or C installs.** It must pass (1) in a minimal environment, (2) in the full
   environment where docling and torch *are* installed, and (3) with `docling`, `torch`, `sentence_transformers`,
   `transformers` and `huggingface_hub` blocked from import. (2) and (3) catch the test that works only because the
   machine lacks, or has, a tool. A also runs with Hugging Face in offline mode: a test that tries to download a
   model fails at once. `tests/guard.py` does the blocking when `tests/portable` is imported, and
   `tests/portable/test_hermetic.py` pins it; tier B switches it off while it runs.
2. **A fakes what it must, at the narrowest seam**: the embedder (`RAG_SEARCH_EMBEDDER`), the reranker, the document
   reader (`RAG_SEARCH_VLM_BACKEND`), the second reader, docling's document (a small fake with the attributes the code
   reads), `torch` and `sentence_transformers` (fakes in `sys.modules`), `launchctl`, the `claude` CLI. The daemons,
   the worker, sockets, files and the CLI are real.
3. **B tests the seams, not the framework again.** Orchestration is A's job; B checks conversion of real files by
   real docling, real model loading and the real index of the corpus. B is small and runs the corpus once per class.
4. **C is only what B cannot do.** It does not repeat B.
5. **One corpus** (`tests/data/`, see its README) serves A (scan, routing) and B (real conversion). Tests that need
   page text that varies, or geometry tied to a fake reader's boxes, keep their own small generators.

## What the three tiers cost and cover now

Measured on a 4-CPU Linux machine, CPU only.

| Tier | Tests | Time | Statements covered |
|---|---|---|---|
| A portable | 962 | about 3 min (4.3 min under coverage) | **92 %** of `src/rag_search` (85 % at the start of this work) |
| B real | 10 | about 2.5 min | 41 % alone; adds 318 statements to A, mostly `docling_convert`, `embedding`, `model_tasks`, `vlm_worker` |
| C machine | 5 | on the Mac | MLX, Apple Vision, the document reader, the default models: not measurable here |

B adds little coverage because A already reaches the orchestration; its value is that it is the only run in which
the real docling, OCR and embedding are asked to do something. What A still misses (8 %) is mostly the rest of
`cli.py` (formatting and rare branches), the parts of `embedding.py` and `vlm*.py` that run a real model
(`Qwen3Reranker.score`, the MLX backend), and the daemons' error paths that need a process to be killed at the right
moment.

## Work

| # | Item | Status |
|---|---|---|
| 1 | Corpus of every kind of input, with its expected result (`tests/data/`, 45 files, 800 KB) | done |
| 2 | B: whole-corpus real run on small models (tesseract OCR on Linux) | done |
| 3 | Make A hermetic: heavy imports blocked, offline guard, passes in minimal and full environments | done |
| 4 | Raise A's coverage where no model is needed (CLI, dashboard, bundle, services, tasks, runs, glue around docling and the embedding libraries, worker, edge cases) | done: 85 % to 92 %, 146 tests added |
| 5 | Cut B's and the cloud's avoidable overhead: CPU torch when reachable, one setup script, verified from scratch | done (`scripts/cloud_setup.sh`; the small torch needs `download-r2.pytorch.org` reachable) |
| 6 | C: only the Mac-only checks | done (5 tests; the docling check moved to B) |
| 7 | Trim redundant tests in A | done as far as the evidence goes: 17 duplicates removed (assertions folded in first), 4 profiler tests replaced by the corpus; the rest each pin a distinct behaviour, so a bigger cut would lose coverage |
| 8 | Notes: `CLAUDE.md`, `CONTRIBUTING.md`, this file | done |
| 9 | Verify every tier as the rules above say; coverage figures | done: see below |

## Verification (all green)

* A in the minimal environment (numpy, pypdfium2, Pillow, mcp only): 962 tests, 1 skipped (`.heic`, no pillow-heif).
* A in the full environment with the heavy imports blocked: 962 tests.
* B in an environment built from scratch by `scripts/cloud_setup.sh`: 10 tests.
* Not run: anything on the Mac. The new tests were written and run on Linux; the ones that touch platform behaviour
  (launchd, `desktop_config_path`, `default_home`, device choice) pretend the platform, but a first run on the Mac
  may find a difference, and C (`tests/machine/`) has not been run at all in this work.

## What the new tests found (three defects fixed, one setup trap)

Writing tests for the branches nobody exercised found three defects, each now pinned by a test, and one trap:

1. `rag-search playground settings NAME` printed defaults for an experiment that does not exist (exit 0), while
   `status`, `bench` and `search` refuse it. Now it refuses too (`core/playground.py`).
2. `rag-search ui --detach --port N`, when the background dashboard never answered, still printed "running in the
   background" and exited 0 after its ten-second wait. Now it says it did not come up and exits 1 (`ui/server.py`).
3. `Reranker.load` has a fallback for a `sentence-transformers` that rejects `activation_fn`, but `_load_model` turned
   that `TypeError` into "download of ... failed" for a model that was not cached yet, so the fallback was reachable
   only for cached models and the first load on such a machine failed with a misleading message. A `TypeError` now
   passes through (`core/embedding.py`).
4. (A trap, not a defect.) The first real-tool run showed that `models.max_seq` (1024, sized for bge-m3) is larger
   than the 512 tokens of the small models used by tier B, so a 4-page table chunk failed to embed. The default setup
   is fine; tier B sets `RAG_SEARCH_MAX_SEQ=512`. It is recorded because anyone choosing a small embedding model will
   meet it.

## Acceptance

* A passes in the minimal environment, in the full environment and with heavy imports blocked; no test downloads. **Met.**
* A's coverage is reported per module, and higher than before; B and C are reported separately and are not needed
  to reach A's number. **Met (92 %).**
* B passes in the cloud session with `scripts/cloud_setup.sh` and nothing else. **Met**, given Hugging Face is
  reachable.
* `CLAUDE.md` tells a coding agent, in one place, which tier to run where and what each costs. **Met.**
