# rag-search: single-machine assumptions (input for containerisation)

Source read: rag-search 0.7.8 (`src/rag_search`, about 12k lines), scripts, README, ARCHITECTURE, CONTRIBUTING. Live state on 2026-10-01: generation 15, 7 collections, 2,021 docs, ~104k chunks. The last `index new` (1,220 docs) took 59 min: convert 46 min, embed 14 min.

> Note (0.8.0): several assumptions below have since changed -- deleted/unreachable sources
> (item 3: an unreachable folder no longer empties a collection, and moved documents are
> re-pointed), search concurrency (item 6: no daemon-wide search lock any more), and collection
> export/import. See `../../ARCHITECTURE.md` for the current state.

## Objective
A private, local RAG over a docs folder. docling turns files into page-marked Markdown. Each chunk is ~512 tokens and searched with BM25 plus bge-m3 dense vectors, fused with RRF and reranked by bge-reranker-v2-m3. Answers cite file and page. Front-ends are the CLI, the MCP adapter (Claude and others) and a loopback web UI. They all reach two daemons over Unix sockets. Per-client collection access applies, but client names are declared, not authenticated.

## Hard-wired single-host assumptions
1. **IPC is Unix sockets** in `<home>/run` (0600), one JSON request per connection. There is no TCP. Clients **auto-spawn** daemons with `sys.executable -m ...`, so client, daemons and data must share one filesystem and one Python environment.
2. **flock locks** on files: `<kind>.alive`, `<kind>.start.lock` and `index.lock`. They are unreliable on network or virtiofs volumes shared between containers.
3. **Absolute source paths are baked in.** `index.meta.json.src_path` and node `metadata.src_path` hold full paths. `_live_docs()` drops any doc whose `src_path` no longer exists. If docs are mounted at a new path, every collection empties on the next merge (publish then refuses with PublishError). Fix this by mounting at the same path or making the path relative to the docs root.
4. **Publish relies on hard links** within one filesystem, a symlink swap and `os.replace`. Across filesystems it falls back to copying, which doubles the disk used.
5. **Everything is held in RAM** by the search daemon: float32 vectors (n×1024×4 ≈ 430 MB now), all chunk text, and BM25 postings that are rebuilt by re-tokenizing every chunk at each load. Search is brute-force `emb @ q` with no ANN index, so cost grows linearly. A reload loads changed collections alongside the live ones, so memory briefly doubles. The models add bge-m3 (fp16 on MPS ~1.1 GB, fp32 on CPU ~2.3 GB), the reranker (fp32 ~2.3 GB) and ~1.5 GB of overhead.
6. **Concurrency is serialised.** One `_search_lock` lets only one search run at a time, and the reranker dominates its latency. There is one indexing run globally and one embedding process. Conversion uses 1 worker on ≤17 GB RAM, else 2 (each 1–2 GB). Merging rewrites a whole collection's `_all` (one big `nodes.json` plus `.npy`) whenever any doc changes.
7. **Acceleration is Apple-specific.** It uses MPS on Apple Silicon and `ocrmac` (Apple Vision) for OCR. A Linux container on a Mac has no MPS and no ocrmac, so it runs on CPU only and needs another OCR engine. Because the OCR engine is part of `convert_profile`, changing it re-converts **all** documents. CUDA works on Linux hosts.
8. **Host integration is macOS/desktop-only.** Services run through launchd only. `register` writes the absolute adapter path into Claude Desktop/Code config. The MCP adapter is stdio and launched by the host app. The UI binds 127.0.0.1 and rejects non-loopback `Host` headers.
9. **Process handling.** It uses `start_new_session` and `killpg`. Orphan detection and RSS (on macOS) shell out to `ps`, which slim images lack. Each grep runs in a Python subprocess. A container needs an init process (e.g. tini) to reap children.
10. **Config and cache locations.** Home is `~/Library/Application Support/rag-search` (macOS) or `$XDG_DATA_HOME/rag-search`. The HF cache is `~/.cache/huggingface` (~5 GB of models), and the packages (torch, docling) are ~5 GB. `paths.configured_model()` always reads `default_home()/config.json`, ignoring `--home`. Python is pinned to 3.11–3.12.

## Data-quality issues seen in the last run
- 17 docs collided because two files shared a name and differed only in extension.
- 3 PDFs were password-protected.
- 89 docs had no text, mostly photos.
- 50 files had unsupported extensions (.xls, .zip, .json, .xml).
