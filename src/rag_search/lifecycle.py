"""Deleting a collection's index (stdlib only; administrator: CLI and dashboard -- never MCP).

Deleting removes what rag-search built for the collection in its workspace, and nothing else:

* its converted Markdown (``indexer_workspace/markup/<name>/``) and
* its index (``indexer_workspace/index/<name>/``),

after which the caller publishes, so it disappears from search at once (published generations
that still hold it age out with the normal rotation).  It works the same for every collection --
a registered location, an import or an index whose folder is no longer registered.

Source documents are never touched.  Everything else is kept too: the collection's access rule
(so a rebuilt collection keeps its restrictions), its description and a location's registration.
A collection whose location is still registered is therefore built again by the next indexing
run -- deleting is how to drop a broken or unwanted index and start over.  To stop indexing a
folder for good, unregister the location (``rag-search location remove`` = unregister + delete).
"""

from __future__ import annotations

import shutil
from typing import Any

from . import locations
from .catalog import collection_names, live_catalog
from .paths import (
    IndexBusyError,
    Paths,
    index_lock,
    is_plain_name,
    is_within,
)


class LifecycleError(ValueError):
    """A deletion the user can fix (unknown name, an indexing run active...)."""


def _find(paths: Paths, name: str) -> str:
    name = (name or "").strip()
    if not is_plain_name(name):
        raise LifecycleError(f"{name!r} is not a collection name")
    known = list(dict.fromkeys([*locations.workspace_collections(paths), *locations.names(paths),
                                *collection_names(live_catalog(paths))]))
    hit = name if name in known else next((k for k in known if k.casefold() == name.casefold()), "")
    if not hit:
        raise LifecycleError(f"no collection named {name!r} (known: {', '.join(known) or 'none'})")
    return hit


def describe_kind(paths: Paths, name: str) -> str:
    if name in locations.load(paths)[0]:
        return "location"
    if locations.is_imported(paths, name):
        return "imported"
    return "unregistered"


def delete_collection(paths: Paths, name: str, *, unregister: bool = False) -> dict[str, Any]:
    """Delete collection *name*'s Markdown and index from the workspace (never its sources).
    *unregister* also removes a location's registration (``location remove``).  The caller
    publishes afterwards (``api.collection_delete``).  Returns what was removed."""
    coll = _find(paths, name)
    kind = describe_kind(paths, coll)
    removed: list[str] = []
    try:
        with index_lock(paths):
            for root in (paths.index, paths.markup):
                target = root / coll
                # only ever inside rag-search's own workspace, never a source folder
                if not is_within(target, paths.workspace) or target.resolve() == root.resolve():
                    raise LifecycleError(f"refusing to delete {target}: not a workspace folder")
                if target.exists():
                    shutil.rmtree(target)
                    removed.append(str(target))
            unregistered = (locations.remove_entry(paths, coll)
                            if unregister and kind == "location" else "")
    except IndexBusyError:
        raise LifecycleError("an indexing run is in progress; cancel it or wait for it to finish "
                             "(rag-search index status), then delete again") from None
    remains = kind != "imported" and not unregistered and _sources_remain(paths, coll, kind)
    note = ("its documents are still in place, so the next indexing run builds it again"
            if remains else "")
    return {"collection": coll, "kind": kind, "removed": removed,
            "location_unregistered": unregistered, "sources_remain": remains, "note": note}


def _sources_remain(paths: Paths, coll: str, kind: str) -> bool:
    """Will the next indexing run build this collection again?  Only a registered location is read."""
    return kind == "location"            # still registered: indexed again when readable
