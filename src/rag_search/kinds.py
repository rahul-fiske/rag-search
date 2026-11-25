"""The kinds of file a run tells apart (stdlib only): one registry, like ``stages.py`` for the stages.

3.0 Sniff gives every file one of these routes, and every kind takes the same way through a run -- 2 Fingerprint, 3.0 Sniff,
its own steps of 3 Convert, then 4 Chunk, 5 Embed, 6 Write, 7 Merge, 8 Publish.  Only its own steps differ: a document of
pages goes lane by lane (3.1-3.5), a photograph is indexed from its details (3.6, 3.7, 3.9, 3.10), a video from its header
(3.11, 3.7), a Google Maps Timeline as a day-by-day log (3.8, 3.7).  A kind is recorded in a conversion trace as a page
``branch`` of its own (pages have the branches of their lanes instead); ``understanding.py`` names what was understood of it,
the run view counts it as a track, and the dashboard draws one lane per kind.  A new kind (frames of a video, sound) is a new
entry here plus its steps in ``stages.py`` and its converter in ``core/indexer.py`` (``CONVERT``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Kind:
    key: str                       # the route 3.0 Sniff gives (also the track of the run view)
    branch: str                    # the page branch its trace records ("" for pages: they have their lanes' branches)
    label: str                     # one, in words
    plural: str                    # many, in words
    steps: tuple[str, ...]         # its own steps of 3 Convert (stage ids of stages.py), in order
    what: str                      # one line: how it is indexed (the note of its trace record)


KINDS: tuple[Kind, ...] = (
    Kind("pages", "", "document", "documents", ("3.1", "3.2", "3.3", "3.4", "3.5"),
         "a document of pages: each page read down the lane it needs"),
    Kind("photograph", "photo", "photograph", "photographs", ("3.6", "3.7", "3.9", "3.10"),
         "a photograph: indexed from its details (3.6, 3.7), not read by the document reader"),
    Kind("video", "video", "video", "videos", ("3.11", "3.7"),
         "a video: indexed from its details (3.11, 3.7), not its frames or sound"),
    Kind("timeline", "timeline", "Timeline", "Timelines", ("3.8", "3.7"),
         "a Google Maps Timeline export: a day-by-day log written by 3.8, places by 3.7"),
)

BY_KEY = {k.key: k for k in KINDS}
BY_BRANCH = {k.branch: k for k in KINDS if k.branch}
TRACKS = tuple(k.key for k in KINDS)
WHOLE_BRANCHES = frozenset(BY_BRANCH)          # files without pages: counted as documents, never as pages
PAGES = "pages"


def route_of_branches(branches: Any) -> str:
    """The kind a conversion recorded (the page branches of its summary): ``photograph`` for ``{"photo"}``, ``pages`` for any
    mix of lane branches."""
    keys = set(branches or ())
    if len(keys) == 1:
        k = BY_BRANCH.get(next(iter(keys)))
        if k is not None:
            return k.key
    return PAGES


def describe() -> list[dict[str, Any]]:
    """The registry for ``/api/pipeline`` and the dashboard."""
    return [{"key": k.key, "branch": k.branch, "label": k.label, "plural": k.plural, "steps": list(k.steps), "what": k.what}
            for k in KINDS]
