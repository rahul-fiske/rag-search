"""The retrieval pipeline's tunables in one stdlib-only place.

`core/search.py` and `core/bm25.py` take their numbers from here, and the web dashboard
(`rag_search.ui`) reads the same values for its Architecture tab, so what the page shows is
what the code does.  Nothing here imports numpy or torch.

``TUNABLES`` (near the bottom) is the single registry of every production tunable that lives in
``config.json`` -- its validation, one-line description, longer impact note, "when does it take
effect" tier and CLI flag name.  `config.py` (defaults), `cli.py` (``config set`` and its help),
`core/search.py` / `core/indexer_daemon.py` / model construction (applying a value), the dashboard
Settings/Models tabs (label + description + info-icon impact text) and `core/playground.py`
(so a playground experiment validates the same way) all read from this one place rather than each
re-describing or re-validating the same knob.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .paths import DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE, DEFAULT_TOP_K

RRF_K = 60            # reciprocal-rank-fusion constant: score = sum(1 / (RRF_K + rank))
MAX_TOP_K = 25        # most passages one search may return
SNIPPET_CHARS = 1200  # passage text longer than this is cut in results
BM25_K1 = 1.5         # BM25 term-frequency saturation
BM25_B = 0.75         # BM25 document-length normalisation
RERANK_CAP = 60       # most candidates ever sent to the cross-encoder by the default formula
RERANK_BATCH = 8      # cross-encoder pairs per forward pass
RERANK_MAX_LEN = 1024 # longest query+passage pair (tokens) the reranker scores
EMBED_BATCH = 32      # default embedding batch (RAG_SEARCH_EMBED_BATCH)
EMBED_MAX_SEQ = 1024  # default max tokens per chunk fed to the embedder (RAG_SEARCH_MAX_SEQ)
CHUNKER_VERSION = "v2"    # bump when chunk boundaries change: every document is then re-chunked
TOKENIZER_VERSION = "v1"  # bump when BM25 tokenisation changes

# Debugging/tuning overrides (search.py): a caller may ask for a different pool size or RRF
# constant than the formulas below compute, but never past these ceilings -- they exist so a
# careless value typed into the dashboard can never make one search stall the cross-encoder or
# scan an unreasonable number of candidates.
STAGES = ("bm25", "dense", "rerank")
RETRIEVAL_POOL_MAX = 300   # hard ceiling for an overridden retrieval_pool
RERANK_POOL_MAX = 100      # hard ceiling for an overridden rerank_pool (above the default cap)
RRF_K_MIN, RRF_K_MAX = 1, 500


def retrieval_pool(top_k: int) -> int:
    """Candidates each retriever (BM25, vectors) contributes per collection."""
    return max(top_k * 4, 20)


def rerank_pool(top_k: int) -> int:
    """Fused candidates handed to the cross-encoder."""
    return min(max(top_k * 3, 15), RERANK_CAP)


def clamp_retrieval_pool(n: int, top_k: int) -> int:
    """An overridden retrieval_pool, kept within [top_k, RETRIEVAL_POOL_MAX]."""
    return max(top_k, min(int(n), RETRIEVAL_POOL_MAX))


def clamp_rerank_pool(n: int, top_k: int) -> int:
    """An overridden rerank_pool, kept within [top_k, RERANK_POOL_MAX]."""
    return max(top_k, min(int(n), RERANK_POOL_MAX))


def clamp_rrf_k(k: int) -> int:
    """An overridden RRF constant, kept within [RRF_K_MIN, RRF_K_MAX] (it's a divisor: never 0)."""
    return max(RRF_K_MIN, min(int(k), RRF_K_MAX))


def parse_stages(value: "str | list[str] | None") -> tuple[str, ...]:
    """'bm25,rerank' / ['bm25','rerank'] -> ('bm25', 'rerank'); None/'' -> all stages (today's
    default).  Rejects unknown stage names and a combination with no retriever at all."""
    if not value:
        return STAGES
    items = value.split(",") if isinstance(value, str) else list(value)
    out: list[str] = []
    for item in items:
        name = item.strip().lower()
        if not name:
            continue
        if name not in STAGES:
            raise ValueError(f"unknown search stage: {name!r} (expected one of {STAGES})")
        if name not in out:
            out.append(name)
    if not out:
        return STAGES
    if "bm25" not in out and "dense" not in out:
        raise ValueError("at least one of 'bm25' or 'dense' must stay on")
    return tuple(out)


# ── shared tunables registry ────────────────────────────────────────────────────────────────
# Valid values for the docling conversion knobs -- kept here (not in core/docling_convert.py, which
# imports docling) so the CLI, dashboard and playground can validate without that heavy import.
OCR_MODES = ("force", "smart", "auto", "off")
OCR_ENGINES = ("auto", "ocrmac", "rapidocr", "easyocr", "tesseract", "tesserocr")
TABLE_MODES = ("accurate", "fast")
PDF_BACKENDS = ("pypdfium2", "docling-parse", "default")
PIPELINE_MODES = ("standard", "vlm")
ROUTING_MODES = ("pages", "document")
VLM_MODES = ("auto", "off")
LANE_MODES = ("off", "auto")
LAYER_FILL_MODES = ("fill", "report", "off")
DTYPES = ("float16", "bfloat16", "float32")
DEVICES = ("cpu", "cuda", "mps")

# "when does a change take effect", used for grouping/labelling in the dashboard:
IMMEDIATE = "immediate"        # applies to the very next search (search.py reads config fresh)
NEXT_RUN = "next-run"          # applies to the next indexing run; no daemon restart
RESTART = "restart"            # needs `rag-search daemon restart` (or `reload` for models)


@dataclass(frozen=True)
class Tunable:
    section: str          # config.json top-level section: "search" | "indexer" | "models"
    key: str              # key within that section
    kind: str             # "int" | "choice" | "text" | "stages"
    default: Any          # the DEFAULTS sentinel (0 or "" means "use the built-in default")
    label: str            # short human label, e.g. "Chunk size"
    what: str             # one-line "what it means" -- always shown next to the control
    impact: str           # longer "what impact changing it can have" -- shown behind an info icon
    applies: str          # IMMEDIATE | NEXT_RUN | RESTART
    cli_flag: str         # e.g. "--chunk-size" (rag-search config set / playground config)
    choices: tuple[str, ...] = ()  # for kind == "choice"
    env: str = ""         # the environment variable this feeds at the point of use, if any
    default_label: str = ""  # human-readable resolved default shown when the field is blank/0,
                              # e.g. "5" or "top_k x 4 (min 20)" for a formula -- NOT necessarily
                              # a literal copy of `default`, which just holds the sentinel value
    choice_help: dict[str, str] = field(default_factory=dict)  # for kind == "choice": one line
                              # per option, shown as that <option>'s tooltip so every dropdown
                              # value is explained without adding a new UI element


TUNABLES: tuple[Tunable, ...] = (
    # -- search: consulted fresh on every search (core/search.py / search_daemon.py) -----------
    Tunable("search", "retrieval_pool", "int", 0, "Retrieval pool (BM25 / dense)",
            "How many candidates BM25 and the vector search each contribute per collection before "
            "they are fused. 0 = the built-in formula (top_k × 4, floor 20).",
            "Larger finds more borderline matches but costs more CPU/RAM per search and widens "
            f"what reaches reranking (hard ceiling {RETRIEVAL_POOL_MAX}). Too small can miss a "
            "relevant passage before reranking ever sees it.",
            IMMEDIATE, "--retrieval-pool", default_label="top_k × 4 (floor 20)"),
    Tunable("search", "rerank_pool", "int", 0, "Rerank pool",
            "How many of the fused candidates are actually sent to the cross-encoder reranker. "
            "0 = the built-in formula (top_k × 3, floor 15, cap 60).",
            f"Larger improves ranking quality but is the main cost driver of a search (hard ceiling "
            f"{RERANK_POOL_MAX}); with reranking off this pool still determines what a plain "
            "BM25+dense fusion result contains.",
            IMMEDIATE, "--rerank-pool", default_label="top_k × 3 (floor 15, cap 60)"),
    Tunable("search", "rrf_k", "int", 0, "RRF k",
            f"The reciprocal-rank-fusion constant used to combine BM25 and dense ranks. 0 = the "
            f"built-in default ({RRF_K}).",
            "Smaller values weight the very top ranks of each retriever much more heavily; larger "
            "values flatten the fusion so lower ranks still contribute meaningfully.",
            IMMEDIATE, "--rrf-k", default_label=f"{RRF_K}"),
    Tunable("search", "stages", "stages", "", "Stages",
            "Which retrieval stages run: any comma-separated combination of bm25, dense, rerank. "
            "Blank = all three (today's default).",
            "Turning off rerank is much faster but ranking quality drops; at least one of bm25/dense "
            "must always stay on. This is the production default -- any client can still ask for a "
            "different combination on a single search.",
            IMMEDIATE, "--stages", default_label="bm25, dense, rerank (all three)"),
    Tunable("search", "top_k", "int", 0, "Default results (top_k)",
            "How many passages a search returns when the caller does not ask for a specific "
            f"number. 0 = the built-in default ({DEFAULT_TOP_K}).",
            "Affects every client that does not pass its own top_k (e.g. the MCP tool's default "
            "call). A caller-supplied value always wins over this.",
            IMMEDIATE, "--top-k", default_label=f"{DEFAULT_TOP_K}"),
    # -- indexer: read fresh at the start of every indexing job (core/indexer_daemon.py) --------
    Tunable("indexer", "chunk_size", "int", 0, "Chunk size",
            "Target size of a chunk, in estimated tokens (about 4 characters or 0.75 words each), when "
            "splitting a converted document. 0 = the built-in default.",
            "Bigger chunks give the embedder more context per vector but blur distinct topics "
            "together and use more tokens per result; changing it only affects documents "
            "(re-)indexed after the change -- existing chunks are not resized.",
            NEXT_RUN, "--chunk-size", default_label=f"{DEFAULT_CHUNK_SIZE}"),
    Tunable("indexer", "chunk_overlap", "int", 0, "Chunk overlap",
            "Estimated tokens shared between consecutive chunks of the same document. 0 = the "
            "built-in default.",
            "More overlap reduces the chance a fact is split awkwardly across a chunk boundary, at "
            "the cost of redundant text (more chunks, more embedding time) per document.",
            NEXT_RUN, "--chunk-overlap", default_label=f"{DEFAULT_CHUNK_OVERLAP}"),
    Tunable("indexer", "ocr", "choice", "", "OCR mode",
            "With page routing (the default) only on or off matters: any value but off lets docling use OCR where a "
            "page needs it (embedded bitmaps on a text page, the whole page of a scan the document reader does not "
            "read); off never uses OCR. The four values apply as written only when a file is converted whole "
            "(routing = document, Office files, or when page routing fails for a document): force = OCR every page; "
            "smart = force only for pages whose own text layer looks unreliable; auto = OCR only embedded images; "
            "off = trust only the PDF's own text. Blank = force.",
            "With page routing a text page is never forced through OCR, whatever this says. For whole-document "
            "conversion it is the biggest lever on speed against text quality; changing it only affects "
            "documents converted after the change (use --force-md / rebuild to redo existing ones).",
            NEXT_RUN, "--ocr", OCR_MODES, "RAG_SEARCH_OCR", default_label="force", choice_help={
                "force": "OCR every page, ignoring the PDF's own text layer -- best for scans or "
                         "odd fonts, but the slowest option (today's default).",
                "smart": "OCR only pages whose own text layer looks unreliable; trust the rest.",
                "auto": "OCR only embedded images; trust the PDF's own text layer elsewhere.",
                "off": "Never OCR -- trust only the PDF's own text layer, however good it is.",
            }),
    Tunable("indexer", "ocr_engine", "choice", "", "OCR engine",
            "Which OCR engine docling uses: auto (docling picks an installed one), ocrmac (Apple "
            "Vision, macOS-only), rapidocr (ONNX-based, cross-platform), easyocr (PyTorch-based, "
            "cross-platform, broad language support), tesseract (the Tesseract CLI, run as a "
            "subprocess per page) or tesserocr (the same Tesseract/Leptonica engine via direct "
            "Python bindings, no subprocess). Blank = auto.",
            "ocrmac needs the optional `ocrmac` package and only runs on macOS, where it is "
            "typically much faster (hardware-accelerated) than the cross-platform engines; any "
            "engine that is not installed fails the conversion for every PDF that needs OCR.",
            NEXT_RUN, "--ocr-engine", OCR_ENGINES, "RAG_SEARCH_OCR_ENGINE", default_label="auto",
            choice_help={
                "auto": "Let docling pick whichever supported engine is installed.",
                "ocrmac": "Apple's Vision framework -- macOS only, hardware-accelerated, needs "
                          "the optional `ocrmac` package.",
                "rapidocr": "RapidOCR (ONNX-based, cross-platform); needs the optional `rapidocr` "
                            "package.",
                "easyocr": "EasyOCR (PyTorch-based, cross-platform, broad language support); "
                           "needs the optional `easyocr` package.",
                "tesseract": "Runs the standalone Tesseract command-line tool as a subprocess per "
                             "page; needs Tesseract installed and on PATH.",
                "tesserocr": "Talks to the same Tesseract/Leptonica engine through direct Python "
                             "bindings instead of a subprocess; needs the optional `tesserocr` "
                             "package and its native libraries installed.",
            }),
    Tunable("indexer", "ocr_lang", "text", "", "OCR languages",
            "Comma-separated languages in the engine's own spelling (en-US for ocrmac, en for "
            "easyocr, eng for tesseract). Blank = the engine's default.",
            "Wrong or missing languages produce garbled OCR text for documents in that language; "
            "most engines slow down slightly per extra language requested.",
            NEXT_RUN, "--ocr-lang", (), "RAG_SEARCH_OCR_LANG", default_label="the engine's own default"),
    Tunable("indexer", "table_mode", "choice", "", "Table structure mode",
            "TableFormer accuracy mode: accurate uses the full model for the best structure "
            "recognition on complex tables; fast is a lighter, quicker pass that can misread "
            "complex table structure. Blank = accurate.",
            "fast can merge or split table cells incorrectly on complex tables; worth it mainly "
            "when documents have few or simple tables.",
            NEXT_RUN, "--table-mode", TABLE_MODES, "RAG_SEARCH_TABLE_MODE",
            default_label="accurate", choice_help={
                "accurate": "TableFormer's full model -- best structure recognition on complex "
                            "or difficult tables (today's default).",
                "fast": "A lighter TableFormer pass -- quicker, but can misread complex table "
                        "structure.",
            }),
    Tunable("indexer", "pdf_backend", "choice", "", "PDF backend",
            "How docling opens PDFs: pypdfium2 (recommended with OCR force), docling-parse "
            "(docling's own parser) or default (docling chooses). Blank = pypdfium2.",
            "Backends can differ in speed and in how they handle malformed PDFs; docling-parse is "
            "occasionally needed for a PDF pypdfium2 cannot open at all.",
            NEXT_RUN, "--pdf-backend", PDF_BACKENDS, "RAG_SEARCH_PDF_BACKEND",
            default_label="pypdfium2", choice_help={
                "pypdfium2": "Renders pages and lets OCR/table models read them uniformly -- the "
                             "default, and the combination this project's own defaults are tuned "
                             "for (force-OCR + pypdfium2).",
                "docling-parse": "docling's own parser -- reads pre-segmented text cells straight "
                                 "from the PDF's structure; occasionally opens a PDF pypdfium2 "
                                 "cannot.",
                "default": "Whatever docling itself defaults to for the version installed.",
            }),
    Tunable("indexer", "pipeline", "choice", "", "Conversion pipeline",
            "standard = docling's normal layout/OCR pipeline; vlm = docling's own vision-language "
            "pipeline reads whole pages (experimental; turns page routing, the document reader, the "
            "gate and repair OFF). Blank = standard.",
            "vlm is docling's experimental pipeline, not the document reader of the Models tab: it "
            "disables page routing, so scanned pages never reach the reader model, the gate or the "
            "repair step, and a page it cannot read is only rescued by Apple Vision plain text. It is "
            "slow on CPU and needs a VLM-capable docling install. Leave it on standard unless you are "
            "deliberately comparing docling's vision pipeline.",
            NEXT_RUN, "--pipeline", PIPELINE_MODES, "RAG_SEARCH_PIPELINE",
            default_label="standard", choice_help={
                "standard": "docling's normal layout, OCR and table-structure pipeline (today's "
                            "default).",
                "vlm": "docling's own vision-language pipeline reads whole pages instead -- "
                       "experimental and slower. Turns OFF page routing, so the document reader "
                       "(Models tab), the gate and repair are not used.",
            }),
    Tunable("indexer", "routing", "choice", "", "PDF routing",
            "pages = each page of a PDF or image file goes down the lane that suits it (a text layer is read by "
            "docling without forced OCR, a scan by the document reader or, for a clean one with OCR first on, by "
            "OCR; pictures on a text page by the document reader); document = the whole file is "
            "converted in one docling call, as before routing existed. Blank = pages.",
            "pages avoids the OCR damage forced OCR does to pages that already have clean text, and "
            "records what each page took in the conversion trace; if routing fails for a document it "
            "is converted as a whole instead. document is the escape hatch if routing misbehaves on "
            "your documents. With pages, the OCR setting only decides whether OCR is on or off.",
            NEXT_RUN, "--routing", ROUTING_MODES, "RAG_SEARCH_ROUTING",
            default_label="pages", choice_help={
                "pages": "Per-page routing: text-layer pages without forced OCR, scanned pages by the document reader "
                         "(docling OCR when it cannot run), a page cache so nothing is read twice (the default).",
                "document": "One docling call for the whole file, with the OCR setting applied to "
                            "every page (the behaviour before routing).",
            }),
    Tunable("indexer", "vlm", "choice", "", "Document reader (VLM)",
            "auto = scanned pages, large pictures in PDFs and image files are read by a vision-language "
            "model when one is installed and downloaded (Models tab); off = docling OCR reads them. "
            "Blank = auto.",
            "The model runs in its own process on Apple Silicon (MLX) and is far better than OCR at "
            "tables and handwriting-free scans; with auto, a missing model, too little free memory or a "
            "failed page simply falls back to docling OCR for that page and the trace says why. "
            "Needs the document-reader runtime (installed with rag-search on Apple Silicon) and a downloaded model; nothing is downloaded by itself.",
            NEXT_RUN, "--doc-reader", VLM_MODES, "RAG_SEARCH_VLM",
            default_label="auto", choice_help={
                "auto": "Use the document reader whenever it can run; fall back to OCR page by page "
                        "(the default).",
                "off": "Never start the document reader: scanned pages are read by docling OCR, "
                       "image files by docling.",
            }),
    Tunable("indexer", "repair", "choice", "", "Repair of table cells",
            "auto = a table cell on a scanned page that breaks the table's arithmetic (a running "
            "balance, a total) is cut out, read again and replaced when two independent readers and "
            "the arithmetic agree; off = pages are kept as the document reader read them. Blank = auto.",
            "Needs the document reader (above) and, for the independent second reading, Apple Vision "
            "(ocrmac, installed with rag-search on a Mac). A cell that cannot be confirmed is never changed: the "
            "page is flagged low-confidence instead. Every attempt is in the conversion trace.",
            NEXT_RUN, "--repair", VLM_MODES, "RAG_SEARCH_REPAIR",
            default_label="auto", choice_help={
                "auto": "Repair suspect cells when the readers can run (the default).",
                "off": "Never re-read cells: suspect pages are only flagged.",
            }),
    Tunable("indexer", "ocr_first", "choice", "", "Lane b: OCR first for clean scans",
            "auto = a scanned page whose image says clean print (resolution, contrast, sharpness, skew, speckle, no "
            "ruled table) is read by an OCR engine first -- docling's, or Tesseract for a skewed page and for image "
            "files -- and goes on to the document reader only when the gate doubts the text; off = the document "
            "reader reads every scan. Blank = off.",
            "OCR takes about a second a page where the document reader takes most of a minute, but OCR can lose "
            "part of a page without any check noticing. On 38 real scans the router sent 3 pages to OCR (none "
            "wrong), and the OCR gate alone would have kept 17 pages that were worse than the reader's text: the "
            "router is strict on purpose, so expect few pages to take this lane. Needs the document reader. "
            "Switching it on converts the documents again (the page cache is reused).",
            NEXT_RUN, "--ocr-first", LANE_MODES, "RAG_SEARCH_OCR_FIRST",
            default_label="off", choice_help={
                "off": "The document reader reads every scanned page (the default).",
                "auto": "Clean scans are read by OCR first and checked; a doubted page goes on to the document reader.",
            }),
    Tunable("indexer", "residue", "choice", "", "Lane c: regions the text layer does not explain",
            "auto = a page with a text layer is also searched for regions of ink that the text layer does not hold "
            "(a stamp, a signature, a drawing, at least 4 % of the page) and the document reader reads them; off = "
            "only large embedded pictures are read (always). Blank = off.",
            "About one text page in ten has such a region (measured on 99 pages), and each costs a reader call; "
            "whether those regions hold text worth finding is not measured yet. A region the reader cannot read "
            "leaves the page low-confidence. Needs the document reader. Switching it on converts the documents again.",
            NEXT_RUN, "--residue", LANE_MODES, "RAG_SEARCH_RESIDUE",
            default_label="off", choice_help={
                "off": "Only large embedded pictures of a text page are read by the document reader (the default).",
                "auto": "Regions of ink outside the text layer are read by the document reader too.",
            }),
    Tunable("indexer", "escalate_digital", "choice", "", "Text pages that lost text go to the reader",
            "auto = a page with a text layer whose result still has lost or garbled text after the text-layer fill "
            "is read as an image by the document reader; off = it is kept and flagged low-confidence. Blank = off.",
            "On the first full run about 370 of 16,842 text pages were flagged for coverage or script; the reader "
            "would take hours for them. When the reader cannot take over, the text-layer result is kept. "
            "Needs the document reader. Switching it on converts the documents again.",
            NEXT_RUN, "--escalate-digital", LANE_MODES, "RAG_SEARCH_ESCALATE_DIGITAL",
            default_label="off", choice_help={
                "off": "Such a page is kept and flagged (the default).",
                "auto": "Such a page is read again as an image by the document reader.",
            }),
    Tunable("indexer", "layer_fill", "choice", "", "Text-layer fill",
            "fill = a text page is compared with the PDF's own text layer and the lines the conversion left out "
            "(table cells, notes, labels) are appended; report = compared and recorded only; off = not compared. "
            "Blank = fill.",
            "The fill is deterministic and takes milliseconds; on the first full run 16 % of text pages had lost "
            "text that it restores. report is for measuring without changing any page.",
            NEXT_RUN, "--layer-fill", LAYER_FILL_MODES, "RAG_SEARCH_LAYER_FILL",
            default_label="fill", choice_help={
                "fill": "Missing lines are appended from the text layer (the default).",
                "report": "The comparison is recorded in the page trace; no page is changed.",
                "off": "Pages are not compared with their text layer.",
            }),
    Tunable("indexer", "stall_timeout", "int", 0, "Stall limit (seconds)",
            "A conversion process that reports nothing for this long while it has a document open is stopped; the "
            "document is reported as failed (stalled) and the run goes on with the others. 0 in this config means "
            "\"use the built-in default (3600s = 1 hour)\"; never less than the per-document timeout plus 15 minutes.",
            "A reader call into native code that never returns has no time limit of its own and would hold a run "
            "for ever. Too short stops a legitimately slow page; the pages already read are kept either way. "
            "Set the environment variable RAG_SEARCH_STALL_TIMEOUT=0 to switch the watch off.",
            NEXT_RUN, "--stall-timeout", (), "RAG_SEARCH_STALL_TIMEOUT", default_label="3600s (1 hour)"),
    Tunable("indexer", "doc_timeout", "int", 0, "Per-document timeout (seconds)",
            "Longest one document may take to convert before it is reported as failed and skipped "
            "(retried next run). 0 = no limit; 0 in this config means \"use the built-in default "
            "(2700s = 45 min)\".",
            "Too short abandons legitimately slow (large/scanned) documents as errors; too long "
            "lets one stuck document hold up a whole indexing run.",
            NEXT_RUN, "--doc-timeout", (), "RAG_SEARCH_DOC_TIMEOUT", default_label="2700s (45 min)"),
    Tunable("indexer", "docling_batch", "int", 0, "Docling page batch",
            "Pages per batch for docling's layout/table/OCR models. 0 = the built-in default (8).",
            "Larger batches can be faster but use more memory per worker; never changes the "
            "resulting text, only speed and memory.",
            NEXT_RUN, "--docling-batch", (), "RAG_SEARCH_DOCLING_BATCH", default_label="8"),
    # -- models: runtime knobs; need a model reload (search) or affect the next indexing run ----
    Tunable("models", "embed_batch", "int", 0, "Embedding batch size",
            "How many chunks the embedding model encodes per forward pass. 0 = the built-in "
            "default (32).",
            "Larger batches are faster on a GPU/MPS with enough memory but can run out of memory; "
            "smaller batches are slower but safer on constrained machines.",
            RESTART, "--embed-batch", (), "RAG_SEARCH_EMBED_BATCH", default_label="32"),
    Tunable("models", "max_seq", "int", 0, "Max tokens per chunk",
            "The longest chunk (in tokens) the embedder will encode; longer chunks are truncated. "
            "0 = the built-in default (1024).",
            "Raising it lets very long chunks keep their full text embedded, at the cost of more "
            "memory and slower embedding per chunk.",
            RESTART, "--max-seq", (), "RAG_SEARCH_MAX_SEQ", default_label="1024"),
    Tunable("models", "dtype", "choice", "", "Weight precision",
            "Force the embedding/reranker weight precision: float16 (more precision, a narrower "
            "exponent range), bfloat16 (float32's exponent range with less precision -- steadier "
            "on more devices) or float32 (full precision). Blank = chosen automatically for the "
            "device.",
            "float32 uses roughly double the memory of float16/bfloat16 but avoids the rare "
            "NaN/inf outputs some models produce in half precision on some devices.",
            RESTART, "--dtype", DTYPES, "RAG_SEARCH_DTYPE",
            default_label="auto-selected for the device", choice_help={
                "float16": "Half precision, 10-bit mantissa -- more precision than bfloat16, but "
                           "a narrower exponent range that can overflow to NaN/inf on some "
                           "models or devices.",
                "bfloat16": "Half precision with float32's exponent range (fewer mantissa bits) "
                            "-- more numerically stable than float16, at some cost in precision; "
                            "well supported on recent CPUs/GPUs and Apple Silicon.",
                "float32": "Full precision -- roughly double the memory of the half types, no "
                           "overflow risk.",
            }),
    Tunable("models", "rerank_batch", "int", 0, "Rerank batch size",
            "How many query/passage pairs the cross-encoder scores per forward pass. 0 = the "
            "built-in default (8).",
            "Larger batches are faster but use more memory; this is the main memory cost of "
            "reranking, separate from the rerank pool size above.",
            RESTART, "--rerank-batch", (), "RAG_SEARCH_RERANK_BATCH", default_label="8"),
    Tunable("models", "rerank_max_len", "int", 0, "Rerank max tokens",
            "The longest query+passage pair (in tokens) the reranker will score; longer pairs are "
            "truncated. 0 = the built-in default (1024).",
            "Raising it lets long passages be reranked without truncation, at the cost of more "
            "memory and slower reranking per pair.",
            RESTART, "--rerank-max-len", (), "RAG_SEARCH_RERANK_MAX_LEN", default_label="1024"),
    Tunable("models", "device", "choice", "", "Compute device",
            "Force cpu (no acceleration), cuda (NVIDIA GPU) or mps (Apple Silicon GPU) for both "
            "the embedder and reranker. Blank = detected automatically.",
            "Forcing a device that is not actually available fails model loading outright; forcing "
            "cpu on a machine with a usable GPU/MPS makes indexing and search much slower.",
            RESTART, "--device", DEVICES, "RAG_SEARCH_DEVICE",
            default_label="auto-detected", choice_help={
                "cpu": "No hardware acceleration -- slowest, but always available.",
                "cuda": "NVIDIA GPU acceleration via CUDA; needs an NVIDIA GPU and a "
                        "CUDA-enabled PyTorch install.",
                "mps": "Apple GPU acceleration via Metal Performance Shaders; macOS on Apple "
                       "Silicon (M-series) only.",
            }),
    Tunable("models", "quantize", "choice", "", "Quantization (CPU only)",
            "Run the embedder and the reranker with 8-bit weights (int8) on the CPU: about twice as fast and a quarter "
            "of the memory, with a small change in the scores. Only used when the models run on the CPU. Blank = "
            "chosen for this computer.",
            "Switching it changes the vectors of new documents very slightly (cosine similarity above 0.99 to the "
            "full-precision ones), so the existing index stays valid and nothing is re-embedded; the ranking of the "
            "search results can differ in the last places.",
            RESTART, "--quantize", ("off", "int8"), "RAG_SEARCH_QUANTIZE",
            default_label="chosen for this computer", choice_help={
                "off": "Full precision on the CPU: the most exact, the slowest.",
                "int8": "8-bit weights on the CPU (PyTorch dynamic quantization): faster and smaller.",
            }),
)

TUNABLES_BY_KEY: dict[str, Tunable] = {t.key: t for t in TUNABLES}
TUNABLES_BY_SECTION: dict[str, tuple[Tunable, ...]] = {
    section: tuple(t for t in TUNABLES if t.section == section) for section in ("search", "indexer", "models")
}


# The range a whole-number tunable may be set to (0 always means "the built-in default").  Outside it the value is
# refused where it is typed -- the CLI, the Settings tab, a Playground experiment -- instead of failing a run or a
# search later: a chunk of 3 tokens, a reranker window shorter than its own prompt, a batch that cannot fit.
LIMITS: dict[str, tuple[int, int]] = {
    "retrieval_pool": (1, RETRIEVAL_POOL_MAX), "rerank_pool": (1, RERANK_POOL_MAX), "rrf_k": (RRF_K_MIN, RRF_K_MAX),
    "top_k": (1, MAX_TOP_K),
    "chunk_size": (32, 8192), "chunk_overlap": (1, 4096),
    "stall_timeout": (60, 7 * 86400), "doc_timeout": (30, 7 * 86400), "docling_batch": (1, 64),
    "embed_batch": (1, 1024), "max_seq": (32, 32768), "rerank_batch": (1, 256), "rerank_max_len": (64, 32768),
}


def validate_tunable(t: Tunable, value: Any) -> Any:
    """Normalise/validate one raw value (typically CLI-argument or JSON-body text) against *t*.

    Blank/None always means "unset -- fall back to the built-in default", regardless of kind.
    Raises ValueError with a message naming the tunable's ``label`` on anything else invalid.
    """
    if value is None or value == "":
        return "" if t.kind in ("choice", "text") else 0
    if t.kind == "int":
        try:
            n = int(value)
        except (TypeError, ValueError):
            raise ValueError(f"{t.label} must be a whole number") from None
        if n < 0:
            raise ValueError(f"{t.label} must be zero or a positive integer")
        lo, hi = LIMITS.get(t.key, (0, 0))
        if n and hi and not lo <= n <= hi:
            raise ValueError(f"{t.label} must be between {lo} and {hi} (or 0 for the built-in default), got {n}")
        return n
    if t.kind == "choice":
        v = str(value).strip().lower()
        if t.choices and v not in t.choices:
            raise ValueError(f"{t.label} must be one of {', '.join(t.choices)} (or blank), got {v!r}")
        return v
    if t.kind == "stages":
        parse_stages(value)  # validate only; stored as given (comma-separated string)
        return value
    return str(value)  # "text": free-form (e.g. ocr_lang)


def validate_section(section: str, values: dict[str, Any]) -> dict[str, Any]:
    """Validate every key of ``values`` against the tunables registered for *section*.

    Unknown keys are rejected (a typo in a flag name should not be silently ignored).  Returns a
    new dict of normalised values, suitable for `config.update_config(paths, section, ...)`.
    """
    known = TUNABLES_BY_SECTION.get(section, ())
    by_key = {t.key: t for t in known}
    out: dict[str, Any] = {}
    for key, value in values.items():
        t = by_key.get(key)
        if t is None:
            raise ValueError(f"unknown {section} setting: {key!r}")
        out[key] = validate_tunable(t, value)
    if section == "indexer" and (out.get("chunk_size") or out.get("chunk_overlap")):
        check_chunking(out.get("chunk_size") or DEFAULT_CHUNK_SIZE, out.get("chunk_overlap") or DEFAULT_CHUNK_OVERLAP)
    return out


def check_chunking(size: int, overlap: int) -> None:
    """A chunk's overlap must leave room for new text: at most half the chunk.  (Checked with the values given
    together; the chunker itself also never carries more than half a chunk over.)"""
    if overlap * 2 > size:
        raise ValueError(f"Chunk overlap ({overlap}) must be at most half the chunk size ({size})")
