"""The indexing pipeline as one numbered list of stages, and the search pipeline likewise (stdlib only).

Everything that talks about "step 3.4" -- the Architecture diagram, the Indexing tab, the Settings tab,
the conversion trace, the progress events of a run, the Playground -- takes the numbers, names and the
settings each stage owns from here, so the pictures cannot drift from the code and a setting can be
traced to the stage that reads it.

Indexing (one run)::

    1  Discover        which files, in which collections          (run)
    2  Fingerprint     unchanged since last time? skip            (document)
    3  Convert         file -> page-marked Markdown               (document)
         3.1 Profile     what is on each page                      CPU
         3.2 Read        the reader each page needs                (lanes a: text layer  b: OCR  c: text layer + pictures  d: document reader)
         3.3 Gate        does the result look right
         3.4 Repair      re-read suspect table cells               GPU
         3.5 Reconcile   tables that run on across pages
    4  Chunk           Markdown -> passages                       (document)
    5  Embed           passages -> vectors                        (run, model loaded once)
    6  Write           nodes.json, embeddings, index.meta.json    (document)
    7  Merge           per-collection index for search            (collection)
    8  Publish         swap the serving generation                (collection)

Search (one query): S1 Access, S2 Keyword, S3 Vectors, S4 Fuse, S5 Rerank, S6 Top k.

Each :class:`Stage` lists the settings it *owns* (``section.key`` of ``config.json``, the same keys the
Settings tab edits) and the environment-only variables it reads.  ``tests/test_stages.py`` checks that
every tunable belongs to exactly one stage and that the code really reads what is listed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

RUN, DOCUMENT, COLLECTION, QUERY = "run", "document", "collection", "query"
CPU, GPU, BOTH = "cpu", "gpu", "cpu+gpu"


@dataclass(frozen=True)
class Stage:
    id: str                              # "3.4": referenced everywhere
    key: str                             # short machine name: events, trace, API
    name: str                            # "Repair"
    scope: str                           # RUN | DOCUMENT | COLLECTION | QUERY
    where: str                           # CPU | GPU | BOTH | ""
    what: str                            # one line, shown on the diagram and in the Settings tab
    settings: tuple[str, ...] = ()       # "section.key" this stage owns (a tunable, or a config key)
    env_only: tuple[str, ...] = ()       # environment variables with no config.json key
    also_used_by: dict[str, tuple[str, ...]] = field(default_factory=dict)   # setting -> other stage ids
    optional: bool = False               # the stage can be switched off / may not run
    constants: tuple[tuple[str, Any], ...] = ()   # fixed numbers worth showing (not configurable)

    @property
    def parent(self) -> str:
        return self.id.rsplit(".", 1)[0] if "." in self.id else ""

    @property
    def title(self) -> str:
        return f"{self.id} · {self.name}"


# 3.2 has four lanes; they are one stage with one set of settings (which lane reads a page is a decision of
# the router, per page), listed separately so the diagram and the trace can name them.
INDEXING: tuple[Stage, ...] = (
    Stage("1", "discover", "Discover", RUN, CPU,
          "list the files of every collection (registered folders, imports); formats the "
          "pipeline cannot read are skipped and counted"),
    Stage("2", "fingerprint", "Fingerprint", DOCUMENT, CPU,
          "SHA-256 of the file plus the chunk, model and conversion settings; unchanged documents are skipped "
          "in seconds"),
    Stage("3", "convert", "Convert", DOCUMENT, BOTH,
          "turn the file into page-marked Markdown, page by page, through the steps below; several documents are "
          "converted side by side when the machine has the memory for it; a conversion process that goes silent is "
          "stopped and its document reported, so one stuck document never holds the run",
          settings=("indexer.jobs", "indexer.stall_timeout")),
    Stage("3.1", "profile", "Profile", DOCUMENT, CPU,
          "look at every page once: text layer, scan or photo, pictures, ink, resolution, script"),
    Stage("3.2", "read", "Read", DOCUMENT, BOTH,
          "each page goes down the lane it needs: 3.2a docling on the text layer (and Office files), 3.2b OCR for a "
          "clean scan (docling's OCR, or Tesseract for a skewed page and image files), 3.2c docling on the text layer "
          "plus the document reader on its pictures and regions, 3.2d the document reader (a vision model) for every "
          "other scan, photo and image; a page a cheaper lane doubts goes on to 3.2d",
          settings=("indexer.routing", "indexer.ocr", "indexer.ocr_engine", "indexer.ocr_lang",
                    "indexer.table_mode", "indexer.pdf_backend", "indexer.pipeline", "indexer.vlm",
                    "indexer.ocr_first", "indexer.residue", "indexer.escalate_digital", "indexer.layer_fill",
                    "models.reader", "models.memory_limit_gb", "indexer.docling_batch",
                    "indexer.doc_timeout"),
          env_only=("RAG_SEARCH_THREADS", "RAG_SEARCH_VLM_PAGE_TIMEOUT", "RAG_SEARCH_VLM_FREE_GB",
                    "RAG_SEARCH_VLM_BACKEND", "RAG_SEARCH_TESSERACT", "RAG_SEARCH_TESSERACT_LANG")),
    Stage("3.3", "gate", "Gate", DOCUMENT, CPU,
          "deterministic checks on every page: coverage, script, tables, resolution, running balances and totals, runaway output; "
          "for a page read by OCR also the amount and plausibility of the text; for a page whose pictures the reader was to read, "
          "that it read them all",
          constants=(("min characters on a scanned page with ink", 20), ("share of a text layer that must survive", 0.5),
                     ("single-letter word share that marks OCR noise", 0.4), ("minimum resolution (dpi)", 150))),
    Stage("3.4", "repair", "Repair", DOCUMENT, GPU,
          "a table cell that breaks the arithmetic is cut out, read again and replaced only when a second, "
          "independent reader and the arithmetic agree",
          settings=("indexer.repair", "models.repair"), env_only=("RAG_SEARCH_REPAIR_SECOND",), optional=True,
          constants=(("cells repaired per page at most", 10),)),
    Stage("3.5", "reconcile", "Reconcile", DOCUMENT, CPU,
          "a table that runs across a page break is joined (the continuation gets the header) and checked across "
          "the break"),
    Stage("4", "chunk", "Chunk", DOCUMENT, CPU,
          "split the Markdown into passages that never cross a page; tables and code stay whole where they fit",
          settings=("indexer.chunk_size", "indexer.chunk_overlap")),
    Stage("5", "embed", "Embed", RUN, GPU,
          "turn every new passage into a vector; the model is loaded once per run, after all documents are converted",
          settings=("models.embedding", "models.embed_batch", "models.max_seq", "models.dtype", "models.device"),
          also_used_by={"models.dtype": ("S3", "S5"), "models.device": ("S3", "S5"), "models.embedding": ("S3",),
                        "models.max_seq": ("S3",), "models.embed_batch": ("S3",)}),
    Stage("6", "write", "Write", DOCUMENT, CPU,
          "nodes.json and embeddings.npy, then index.meta.json last, so a half-written document is never taken "
          "for a finished one"),
    Stage("7", "merge", "Merge", COLLECTION, CPU,
          "concatenate the documents' vectors into the collection's index for search (nothing is re-embedded)"),
    Stage("8", "publish", "Publish", COLLECTION, CPU,
          "hard-link the new generation, switch the \"current\" symlink, tell the search daemon to reload; the "
          "last three generations are kept", settings=("indexer.auto_publish",), constants=(("generations kept", 3),)),
)

SEARCH: tuple[Stage, ...] = (
    Stage("S1", "access", "Access", QUERY, CPU, "which collections this client may search; an unknown name is "
                                                  "refused like a typo"),
    Stage("S2", "keyword", "Keyword", QUERY, CPU, "BM25 over the chunk text: exact terms, identifiers, versions; "
                                                    "the pool size below applies to both retrievers",
          settings=("search.retrieval_pool",), also_used_by={"search.retrieval_pool": ("S3",)},
          constants=(("k1", 1.5), ("b", 0.75))),
    Stage("S3", "vectors", "Vectors", QUERY, GPU, "cosine similarity of the query vector with every passage vector; the "
                                                    "models are loaded when the search daemon starts",
          settings=("search.prewarm",)),
    Stage("S4", "fuse", "Fuse", QUERY, CPU, "reciprocal rank fusion of the two lists into one pool",
          settings=("search.rrf_k", "search.stages")),
    Stage("S5", "rerank", "Rerank", QUERY, GPU, "a cross-encoder reads query and passage together and re-orders the "
                                                  "best candidates",
          settings=("models.reranker", "search.rerank_pool", "models.rerank_batch", "models.rerank_max_len"),
          optional=True),
    Stage("S6", "top_k", "Top k", QUERY, CPU, "the best passages with page, heading, snippet and score",
          settings=("search.top_k",)),
)

ALL: tuple[Stage, ...] = INDEXING + SEARCH
BY_ID: dict[str, Stage] = {s.id: s for s in ALL}
BY_KEY: dict[str, Stage] = {s.key: s for s in ALL}

# What a change of each kind invalidates, in stage terms (the "When is a document re-indexed?" table).
FINGERPRINT_INPUTS: tuple[tuple[str, str], ...] = (
    ("the source file (SHA-256)", "3"),
    ("conversion settings: OCR mode, engine, languages, table mode, pipeline, PDF backend, routing, "
     "and the document reader or repair being switched off", "3"),
    ("chunk size, chunk overlap, chunker version, index format", "4"),
    ("the embedding model", "5"),
)

# Settings that are `config.json` keys but not tunables of spec.py (a model choice, a switch, a limit).
MODEL_KEYS = ("models.embedding", "models.reranker", "models.reader", "models.repair", "models.memory_limit_gb")
CONFIG_KEYS = MODEL_KEYS + ("indexer.jobs", "indexer.auto_publish", "search.prewarm")


def id_of(key: str) -> str:
    """The stage number of a stage key ("profile" -> "3.1"); the key itself when it is not a stage (never raises)."""
    st = BY_KEY.get(key)
    return st.id if st else key


def named(key: str) -> str:
    """"3.1 Profile" for a stage key; the key with underscores as spaces when it is not one."""
    st = BY_KEY.get(key)
    return f"{st.id} {st.name}" if st else key.replace("_", " ")


def indexing_ids() -> list[str]:
    return [s.id for s in INDEXING]


def children(stage_id: str) -> list[Stage]:
    return [s for s in ALL if s.parent == stage_id]


def owner_of(setting: str) -> Stage | None:
    """The stage that owns ``section.key`` (None if no stage does)."""
    for s in ALL:
        if setting in s.settings:
            return s
    return None


def owners_of(setting: str) -> list[Stage]:
    return [s for s in ALL if setting in s.settings]


def label(stage_id: str) -> str:
    s = BY_ID.get(stage_id)
    return s.title if s else stage_id


def describe() -> list[dict[str, Any]]:
    """The registry as JSON for the dashboard."""
    out = []
    for s in ALL:
        out.append({"id": s.id, "key": s.key, "name": s.name, "scope": s.scope, "where": s.where,
                    "what": s.what, "settings": list(s.settings), "env_only": list(s.env_only),
                    "also_used_by": {k: list(v) for k, v in s.also_used_by.items()},
                    "optional": s.optional, "parent": s.parent,
                    "constants": [{"label": a, "value": b} for a, b in s.constants]})
    return out
