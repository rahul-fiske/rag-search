"""Filesystem layout, path mirroring and small shared helpers (stdlib only).

Everything rag-search writes lives under one *home* directory:

    <home>/locations.json         the registered source folders: collection name -> folder anywhere
                                  (written only by `rag-search location add/remove`); the documents
                                  themselves stay where they are and are only ever read
    <home>/indexer_workspace/     private, persistent state of the indexer
        markup/                   Markdown converted from sources
        index/                    per-document and per-collection (_all) indexes
    <home>/serving/               what the search daemon reads
        gen-N/                    immutable published generations (index/, markup/, catalog)
        current -> gen-N          symlink, switched atomically on publish
    <home>/run/                   sockets, locks, pid files, daemon logs      (mode 0700)
    <home>/jobs/                  indexing job state, event streams and logs
    <home>/config.json            optional settings (see config.py)
    <home>/access.json            which clients may use which collections (see policy.py;
                                  written only by `rag-search access`)

Home is ``$RAG_SEARCH_HOME`` or, by default, ``~/Library/Application Support/rag-search``
on macOS and ``$XDG_DATA_HOME/rag-search`` (``~/.local/share/rag-search``) elsewhere.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import re
import sys
import tempfile
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_MODEL = "BAAI/bge-m3"
DEFAULT_RERANK_MODEL = "BAAI/bge-reranker-v2-m3"
DEFAULT_CHUNK_SIZE = 512
DEFAULT_CHUNK_OVERLAP = 64
DEFAULT_TOP_K = 5

# Formats docling (or the plain-text passthrough) can turn into Markdown.
SUPPORTED_EXTENSIONS = frozenset({
    ".pdf", ".docx", ".pptx", ".xlsx", ".html", ".htm", ".csv", ".adoc",
    ".md", ".txt",
    ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp",
    ".heic", ".heif",                       # read by the document reader (needs pillow-heif)
})
PASSTHROUGH_EXTENSIONS = frozenset({".md", ".txt"})
IMAGE_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp", ".heic", ".heif"})

PAGE_MARKER = "<!-- page {page} -->"
META_FILE = "index.meta.json"
NODES_FILE = "nodes.json"
EMB_FILE = "embeddings.npy"
MERGE_MANIFEST = "merge.manifest.json"
ALL_DIR = "_all"
INDEX_FORMAT = 1


def default_home() -> Path:
    env = os.environ.get("RAG_SEARCH_HOME")
    if env:
        return Path(env).expanduser().absolute()
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "rag-search"
    xdg = os.environ.get("XDG_DATA_HOME")
    base = Path(xdg).expanduser() if xdg else Path.home() / ".local" / "share"
    return base / "rag-search"


KINDS = ("search", "indexer")


@dataclass(frozen=True)
class Paths:
    home: Path
    workspace: Path
    index: Path          # workspace index (written by the indexer only)
    markup: Path         # workspace markup (written by the indexer only)
    serving: Path
    run: Path
    jobs: Path

    # -- per-daemon files --------------------------------------------------
    def socket(self, kind: str) -> Path:
        candidate = self.run / f"{kind}.sock"
        if len(str(candidate).encode()) <= 100:  # AF_UNIX paths are limited to ~104 bytes on macOS
            return candidate
        # fixed location (not $TMPDIR, which differs between GUI apps, launchd and shells)
        digest = hashlib.sha1(str(self.home).encode()).hexdigest()[:10]
        return Path("/tmp") / f"rag-search-{os.getuid()}" / f"{digest}-{kind}.sock"

    def alive_lock(self, kind: str) -> Path:
        return self.run / f"{kind}.alive"

    def start_lock(self, kind: str) -> Path:
        return self.run / f"{kind}.start.lock"

    def pid_file(self, kind: str) -> Path:
        return self.run / f"{kind}.pid"

    def log_file(self, kind: str) -> Path:
        return self.run / f"{kind}.log"

    @property
    def index_lock(self) -> Path:
        return self.run / "index.lock"

    @property
    def config_file(self) -> Path:
        return self.home / "config.json"

    @property
    def access_file(self) -> Path:
        return self.home / "access.json"

    @property
    def descriptions_file(self) -> Path:
        return self.home / "descriptions.json"

    @property
    def locations_file(self) -> Path:
        return self.home / "locations.json"

    # -- serving generations ----------------------------------------------
    @property
    def current_link(self) -> Path:
        return self.serving / "current"

    def gen_dir(self, n: int) -> Path:
        return self.serving / f"gen-{n:06d}"

    def current_gen(self) -> Path | None:
        """The live generation directory, or None if nothing has been published."""
        link = self.current_link
        if not link.is_symlink() and not link.exists():
            return None
        target = link.resolve()
        return target if target.is_dir() else None

    def live_index(self) -> Path | None:
        g = self.current_gen()
        return g / "index" if g else None

    def live_markup(self) -> Path | None:
        g = self.current_gen()
        return g / "markup" if g else None


def get_paths(home: Path | str | None = None) -> Paths:
    h = Path(home).expanduser().absolute() if home else default_home()
    ws = h / "indexer_workspace"
    return Paths(
        home=h,
        workspace=ws,
        index=ws / "index",
        markup=ws / "markup",
        serving=h / "serving",
        run=h / "run",
        jobs=h / "jobs",
    )


def ensure_dirs(p: Paths) -> None:
    for d in (p.home, p.index, p.markup, p.serving, p.jobs):
        d.mkdir(parents=True, exist_ok=True)
    p.run.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(p.run, 0o700)
    except OSError:
        pass


# ── settings from the environment ────────────────────────────────────────────

def env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, "") or default)
    except ValueError:
        return default


def env_flag(name: str, default: bool = False) -> bool:
    v = os.environ.get(name)
    if v is None or v == "":
        return default
    return v.strip().lower() in {"1", "true", "yes", "on"}


def configured_model(kind: str) -> str:
    """The model chosen in ``config.json`` ("models": {"embedding": ..., "reranker": ...}), or "".

    Read from the file each time (it is tiny), so a change made by `rag-search models set` or
    the dashboard reaches every process without a restart.
    """
    try:
        data = json.loads((default_home() / "config.json").read_text(encoding="utf-8"))
        value = (data.get("models") or {}).get(kind)
    except (OSError, ValueError, AttributeError):
        return ""
    return value.strip() if isinstance(value, str) else ""


def model_name() -> str:
    """Embedding model: $RAG_SEARCH_MODEL, else config.json, else the default."""
    return os.environ.get("RAG_SEARCH_MODEL") or configured_model("embedding") or DEFAULT_MODEL


def rerank_model_name() -> str:
    """Reranker: $RAG_SEARCH_RERANK_MODEL, else config.json, else the default."""
    return (os.environ.get("RAG_SEARCH_RERANK_MODEL") or configured_model("reranker")
            or DEFAULT_RERANK_MODEL)


def parse_collections(value: str | list[str] | None) -> list[str]:
    """'a, b' -> ['a', 'b'].  Rejects anything that is not a plain directory name.

    Repeats are dropped (first spelling kept): a collection named twice must not be searched
    twice -- each pass would add its rank contributions to the fused score again."""
    if not value:
        return []
    items = value.split(",") if isinstance(value, str) else list(value)
    out: list[str] = []
    for item in items:
        name = str(item).strip()
        if not name:
            continue
        if not is_plain_name(name):
            raise ValueError(f"invalid collection name: {name!r}")
        if name not in out:
            out.append(name)
    return out


def is_plain_name(name: str) -> bool:
    """A single directory-name token: not empty, not '.'/'..', no separators or NUL, not hidden."""
    return bool(name) and name not in {".", ".."} and not name.startswith(".") \
        and not any(c in name for c in "/\\\0")


# ── playground (sandbox experiments, structurally separate from production) ─────

PLAYGROUND_DIR = "playground"
_EXPERIMENT_RE_MSG = "letters, digits, '-' and '_' only"


def validate_experiment_name(name: str) -> str:
    """A playground experiment name: a plain directory-safe token, nothing that could escape
    <home>/playground/ (no '.', '..', separators, or empty string)."""
    name = (name or "").strip()
    if not name or name in {".", ".."} or any(c in name for c in "/\\\0"):
        raise ValueError(f"invalid experiment name: {name!r} ({_EXPERIMENT_RE_MSG})")
    if not all(c.isalnum() or c in "-_" for c in name):
        raise ValueError(f"invalid experiment name: {name!r} ({_EXPERIMENT_RE_MSG})")
    return name


def playground_root(base: Paths) -> Path:
    """<home>/playground/ -- everything under here is sandbox data, never read by production
    daemons, publish, or the search/indexer sockets."""
    return base.home / PLAYGROUND_DIR


def get_playground_paths(base: Paths, name: str) -> Paths:
    """A self-contained Paths rooted at <home>/playground/<name>/, built the same way
    get_paths() builds the real one, but never touching *base*'s serving/run/config -- a
    playground experiment has its own sources/, locations.json, workspace/, config.json, all under its own home."""
    name = validate_experiment_name(name)
    h = playground_root(base) / name
    ws = h / "workspace"
    return Paths(home=h, workspace=ws, index=ws / "index", markup=ws / "markup",
                 serving=h / "serving", run=h / "run", jobs=h / "jobs")


def list_playground_names(base: Paths) -> list[str]:
    root = playground_root(base)
    if not root.is_dir():
        return []
    return sorted(p.name for p in root.iterdir() if p.is_dir() and not p.name.startswith("."))


# ── small file helpers ───────────────────────────────────────────────────────

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def write_json_atomic(path: Path, data: Any, *, indent: int | None = 2,
                      mode: int | None = None, newline: bool = False) -> None:
    """Write JSON to *path* atomically (temp file in the same folder + ``os.replace``).

    *mode* sets the file's permissions before it becomes visible (e.g. ``0o600`` for
    ``access.json``); *newline* adds a trailing newline (nicer for hand-edited files).
    The one implementation every small JSON store in rag-search uses."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix="." + path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=indent, ensure_ascii=False)
            if newline:
                f.write("\n")
        if mode is not None:
            os.chmod(tmp, mode)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


@contextlib.contextmanager
def file_lock(path: Path, timeout: float = 10.0) -> Iterator[None]:
    """Exclusive advisory lock for a read-modify-write of *path* (``.<name>.lock`` beside it).

    The write itself is atomic (``write_json_atomic``), but two writers that both read the old
    content and then each write back their own change would lose one of them; holding this
    lock across the whole read-modify-write prevents that.  Waits up to *timeout* seconds."""
    import fcntl

    path.parent.mkdir(parents=True, exist_ok=True)
    lock = path.with_name("." + path.name + ".lock")
    fd = os.open(lock, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        deadline = time.monotonic() + timeout
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError:
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"{path} is being changed by another process; try again")
                time.sleep(0.05)
        yield
    finally:
        os.close(fd)                     # closing releases the lock


class IndexBusyError(RuntimeError):
    """Another indexing run (or a collection import/delete) holds the index lock."""


@contextlib.contextmanager
def index_lock(paths: Paths) -> Iterator[None]:
    """Only one writer of the indexer workspace at a time, across processes: an indexing run,
    a collection import or a collection deletion.  Raises IndexBusyError instead of waiting."""
    import fcntl

    paths.run.mkdir(parents=True, exist_ok=True)
    fd = os.open(paths.index_lock, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise IndexBusyError("another indexing run is already in progress") from exc
        yield
    finally:
        os.close(fd)  # closing releases the lock


class CachedFile:
    """A value parsed from a small file, re-parsed only when the file changes (mtime + size).

    ``loader(path) -> (value, error)``; a missing file is handed to the loader too (it decides
    what "absent" means).  Long-lived processes (daemons, the MCP adapter, the dashboard) keep
    one per file instead of re-reading it on every request."""

    def __init__(self, path: Path, loader: Callable[[Path], tuple[Any, str]]):
        self.path = path
        self._loader = loader
        self._sig: tuple | None = None
        self._loaded = False
        self._value: Any = None
        self.error = ""
        self._lock = threading.Lock()

    def get(self) -> Any:
        try:
            st = self.path.stat()
            sig: tuple | None = (st.st_mtime_ns, st.st_size, st.st_ino)
        except OSError:
            sig = None
        with self._lock:
            if not self._loaded or sig != self._sig:
                self._value, self.error = self._loader(self.path)
                self._sig, self._loaded = sig, True
            return self._value


# ── path mirroring ───────────────────────────────────────────────────────────

@dataclass(frozen=True)
class SourceRoots:
    """Where documents come from: the registered locations (a folder anywhere = one collection, see
    locations.py).  Accepted wherever the indexer needs to name a document (``mirror_rel`` & co.), and
    plain data (str paths) so it can travel to the conversion worker processes."""

    locations: tuple[tuple[str, str], ...] = ()      # (collection, root folder)

    @classmethod
    def of(cls, value: SourceRoots | dict) -> SourceRoots:
        if isinstance(value, SourceRoots):
            return value
        if isinstance(value, dict):
            return cls(tuple((str(n), str(r)) for n, r in value.get("locations", ())))
        raise TypeError(f"source roots expected, got {type(value).__name__}")

    def to_dict(self) -> dict[str, Any]:
        return {"locations": [list(x) for x in self.locations]}

    def location_names(self) -> list[str]:
        return [n for n, _ in self.locations]

    def root_of(self, collection: str) -> Path | None:
        """The folder holding *collection*'s sources, or None when no location of that name is registered
        (an imported collection, or one whose folder was never registered)."""
        for n, r in self.locations:
            if n == collection:
                return Path(r)
        return None

    def rel(self, src: Path) -> Path:
        """*src* as ``<collection>/<path inside the location>``.  Raises ValueError when it is not inside
        a registered location, or is a location's own folder."""
        s = Path(src).resolve()
        # longest root first, so a location is never mistaken for a parent folder
        for name, root in sorted(self.locations, key=lambda x: -len(x[1])):
            try:
                inner = s.relative_to(Path(root).resolve())
            except ValueError:
                continue
            if not inner.parts:
                raise ValueError(f"{src} is a location's own folder, not a document")
            return Path(name, *inner.parts)
        raise ValueError(f"{src} is not inside a registered location")


def mirror_rel(src: Path, roots: SourceRoots | dict) -> Path:
    """Path of *src* as ``<collection>/<path inside the location>``: the same names under the
    workspace's ``markup/`` and ``index/``.  Raises ValueError when *src* is in no registered location."""
    return SourceRoots.of(roots).rel(src)


def index_dir_for(src: Path, roots: SourceRoots | dict, index_root: Path) -> Path:
    rel = mirror_rel(src, roots)
    return index_root / rel.parent / rel.stem


def markup_path_for(src: Path, roots: SourceRoots | dict, markup_root: Path) -> Path:
    rel = mirror_rel(src, roots)
    return markup_root / rel.parent / (rel.stem + ".md")


def collection_of(src: Path, roots: SourceRoots | dict) -> str:
    return mirror_rel(src, roots).parts[0]


def is_within(child: Path, parent: Path) -> bool:
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def detached_start(home: Path, env: dict[str, str] | None = None) -> dict[str, Any]:
    """``cwd`` and ``env`` for starting a long-lived child (daemon, dashboard, worker, background task).

    A child inherits the caller's working directory, and a daemon started from a folder that is
    deleted later (an unpacked release folder that is rebuilt, a temporary directory) then fails in
    odd places: ``os.getcwd()`` raises, so ``os.path.abspath``, starting a process pool and even
    ``import torch`` break.  The child therefore starts in the data folder, which exists as long as
    rag-search does.  Relative ``PYTHONPATH`` entries (a development run with ``PYTHONPATH=src``) are
    made absolute first, so the child still finds the same code."""
    env = dict(os.environ if env is None else env)
    if env.get("PYTHONPATH"):
        env["PYTHONPATH"] = os.pathsep.join(
            os.path.abspath(e) if e else e for e in env["PYTHONPATH"].split(os.pathsep))
    home = Path(home)
    return {"cwd": str(home if home.is_dir() else Path(home.anchor or "/")), "env": env}


def pasted_path(text: str | Path | None) -> str:
    """A path as a person pastes it, turned into the path itself.

    A path copied from a terminal or from Finder often arrives wrapped in quotes
    (``'/Users/me/My Drive/x.pdf'``) or with shell escapes (``/Users/me/My\\ Drive``): that is
    shell syntax, not part of the name.  One matching pair of quotes is removed; backslash escapes
    are removed only when the text as given does not exist and the unescaped one does, so a real
    backslash in a name is never lost.  Spaces, ``@``, ``-`` and the like are ordinary characters."""
    raw = str(text or "").strip()
    if len(raw) >= 2 and raw[0] == raw[-1] and raw[0] in "'\"":
        raw = raw[1:-1].strip()
    if "\\" in raw and not os.path.exists(os.path.expanduser(raw)):
        plain = re.sub(r"\\(.)", r"\1", raw)
        if os.path.exists(os.path.expanduser(plain)):
            raw = plain
    return raw


def name_from_folder(folder: str | Path) -> str:
    """A collection name made from a folder's own name: letters, digits, ``-`` and ``_`` are kept,
    anything else becomes ``_`` (``"My Drive"`` -> ``My_Drive``); "" when nothing usable is left."""
    base = Path(str(folder)).name
    return re.sub(r"[^A-Za-z0-9_-]+", "_", base).strip("_-")


_CLOUD_FILES: dict[str, bool] = {}


def allow_cloud_files() -> bool:
    """Let this process read files that a cloud-storage app keeps online-only (macOS; a no-op elsewhere).

    Box, Google Drive, iCloud and OneDrive keep a folder under ``~/Library/CloudStorage`` whose files
    may be placeholders ("dataless"): the app fetches the content when a program reads the file.
    Whether a process may trigger that fetch is a per-process setting.  A program started from a
    terminal or an app has it on; one started by launchd -- the daemons installed by ``rag-search
    service install`` -- has it off, and then every read of such a file fails at once with
    ``[Errno 11] Resource deadlock avoided``.  This switches it on for the process; children inherit
    it.  Reading is all that happens: the app downloads the file, nothing in the folder is changed.
    Returns True when the setting is (now) on."""
    if "on" in _CLOUD_FILES:
        return _CLOUD_FILES["on"]
    ok = False
    if sys.platform == "darwin":
        try:
            import ctypes

            libc = ctypes.CDLL(None, use_errno=True)
            # IOPOL_TYPE_VFS_MATERIALIZE_DATALESS_FILES = 3, IOPOL_SCOPE_PROCESS = 0, ..._ON = 2
            ok = libc.setiopolicy_np(3, 0, 2) == 0
        except (OSError, AttributeError, ValueError):
            ok = False
    _CLOUD_FILES["on"] = ok
    return ok

