"""One indexing run in its own process:  ``python -m rag_search.core.worker <job-id>``.

The indexer daemon spawns this in a fresh process group so a cancel/restart can kill
the worker together with its converter children at any moment.  The worker reads its
spec from ``jobs/<id>.json`` and writes ONLY to ``indexer_workspace/`` and to its event
log ``jobs/<id>.events.jsonl`` (one JSON object per line).  The daemon owns the job
record and does the publishing afterwards.

Exit codes: 0 finished (see the 'result' event for per-document errors),
1 failed, 3 another indexing run holds the index lock.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

from ..jobs import events_file, job_file
from ..paths import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    Paths,
    allow_cloud_files,
    get_paths,
    read_json,
)

EXIT_OK, EXIT_FAILED, EXIT_BUSY = 0, 1, 3
PROGRESS_MIN_INTERVAL = 0.3


class EventWriter:
    def __init__(self, path: Path):
        self.path = path
        self.fh = open(path, "a", encoding="utf-8", buffering=1)  # line buffered
        self._last = 0.0
        self._last_phase = ""

    def emit(self, event: str, **fields: Any) -> None:
        self.fh.write(json.dumps({"ts": round(time.time(), 3), "event": event, **fields},
                                 ensure_ascii=False) + "\n")

    def progress(self, ev: dict[str, Any]) -> None:
        """Throttle chatty embed-progress events; always pass phase changes and totals."""
        if "doc" in ev:          # one finished document: never throttled
            self.emit("doc", **ev["doc"])
            return
        if "stage" in ev:        # one document entering / leaving a pipeline stage
            self.emit("stage", **{"pid": os.getpid(), **ev["stage"]})
            return
        now = time.monotonic()
        phase = ev.get("phase", "")
        final = ev.get("total") and ev.get("done") == ev.get("total")
        if phase == self._last_phase and not final and now - self._last < PROGRESS_MIN_INTERVAL:
            return
        self._last, self._last_phase = now, phase
        self.emit("progress", **ev)

    def close(self) -> None:
        self.fh.close()


def resolve_sources(paths: Paths, spec: dict[str, Any]) -> tuple[Any, list[Path], list[dict[str, str]]]:
    """(source roots, sources, unsupported) for a spec.  `path` may be a registered location's name or a
    file or folder inside one (see ``locations.plan_scan``).

    `unsupported` is every file seen but left out of `sources` because its extension isn't one
    indexing reads -- `{"src", "extension"}` per file -- so a document with the wrong extension
    is reported rather than just silently missing from every status (see `SUPPORTED_EXTENSIONS`).
    """
    plan = plan_for(paths, spec)
    return plan.roots, plan.sources, plan.unsupported


def plan_for(paths: Paths, spec: dict[str, Any]):
    from ..locations import LocationError, plan_scan

    try:
        return plan_scan(paths, str(spec.get("path") or "").strip())
    except LocationError as exc:
        raise ValueError(str(exc)) from None


def run_spec(paths: Paths, spec: dict[str, Any], events: EventWriter,
             embedder: Any = None) -> dict[str, Any]:
    from .indexer import run_plan

    plan = plan_for(paths, spec)
    events.emit("start", files=len(plan.sources), mode=spec.get("mode", "new"),
                unreachable=plan.unreachable)
    return run_plan(
        paths, plan, jobs=int(spec.get("jobs") or 1),
        rebuild=bool(spec.get("rebuild")), wipe=spec.get("mode") == "all",
        force_md=bool(spec.get("force_md")),
        chunk_size=int(spec.get("chunk_size") or DEFAULT_CHUNK_SIZE),
        chunk_overlap=int(spec.get("chunk_overlap") or DEFAULT_CHUNK_OVERLAP),
        embedder=embedder, progress=events.progress, stage_log=getattr(events, "path", None),
    )


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        print("usage: python -m rag_search.core.worker <job-id>", file=sys.stderr)
        return 2
    logging.basicConfig(level=logging.INFO, stream=sys.stderr,
                        format="[worker] %(levelname)s %(message)s")
    from .embedding import prepare_environment
    from .indexer import IndexBusyError

    allow_cloud_files()
    paths = get_paths()
    rec = read_json(job_file(paths, argv[0])) or {}
    events = EventWriter(events_file(paths, argv[0]))
    try:
        prepare_environment()
        summary = run_spec(paths, rec.get("spec", {}), events)
        events.emit("result", summary=summary)
        return EXIT_OK
    except IndexBusyError as exc:
        events.emit("error", error=str(exc), busy=True)
        return EXIT_BUSY
    except Exception as exc:  # noqa: BLE001
        logging.exception("indexing failed")
        events.emit("error", error=f"{type(exc).__name__}: {exc}")
        return EXIT_FAILED
    finally:
        events.close()


def leave(code: int) -> None:
    """End this process now.  A normal interpreter exit joins every non-daemon thread, and a
    docling thread abandoned after a document timeout may never return (see
    ``indexer._shutdown_pool``): the run would be finished and its process would stay forever,
    holding the daemon's "run active".  Everything is written and closed by now, so nothing is lost."""
    logging.shutdown()
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.flush()
        except (OSError, ValueError):
            pass
    os._exit(code)


if __name__ == "__main__":
    leave(main())
