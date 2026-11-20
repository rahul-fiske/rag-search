"""Maps: the place data indexing uses to describe a position, and how it is downloaded.

Three datasets, each downloaded only when asked (``rag-search maps`` or the Models tab), never while indexing:

    geonames             towns and villages of 1,000+ people, worldwide, one file (``reference.py``): the nearest town
    overture-divisions   Overture Maps areas (district, state, locality ...): the area a position is inside
    overture-places      Overture Maps places of interest (restaurants, fuel, temples ...): what is around it

The two Overture datasets are kept per *cell* of the map, a square of ``CELL_DEG`` degrees (about 55 km), so only the
parts of the world that matter are on disk.  Which cells matter is read from the indexed documents themselves: every
position a converted document mentions is written as ``geo:LAT,LON`` (photos, Google Timeline), and ``plan`` lists the
cells those positions fall in that are not downloaded yet.  Both are stored in SQLite files with a spatial index under
``<home>/reference/overture/`` and pinned to one Overture release; ``update`` moves every cell to the newest release.

Overture is read straight from its public bucket with pyarrow (row groups outside the cells are not transferred);
pyarrow and shapely come with docling's dependencies and are imported only here.  ``refresh`` makes every document
with positions be converted again by the next indexing run, so new map data reaches documents indexed before it.
"""

from __future__ import annotations

import contextlib
import math
import os
import re
import sqlite3
import time
from collections import Counter
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from . import reference
from .paths import META_FILE, Paths

CELL_DEG = 0.5
BUCKET = "overturemaps-us-west-2/release"
REGION = "us-west-2"
KINDS = ("divisions", "places")
DATASETS = ("geonames", "overture-divisions", "overture-places")
TITLES = {"geonames": "GeoNames towns and villages",
          "overture-divisions": "Overture Maps areas (divisions)",
          "overture-places": "Overture Maps places of interest"}
USED_FOR = {"geonames": "the town a position is near",
            "overture-divisions": "the locality, district, state and country a position is in",
            "overture-places": "the restaurants, shops, temples ... near a position"}
LICENSES = {"geonames": "CC BY 4.0, geonames.org",
            "overture-divisions": "ODbL (built from OpenStreetMap), overturemaps.org",
            "overture-places": "CDLA Permissive 2.0 / Apache 2.0 (per record), overturemaps.org"}
EST_MB_PER_CELL = 1.0              # rough: a town cell is a fraction of this, a big city several times it
SIMPLIFY_DEG = 0.0005              # area outlines are kept to about 50 m: enough to say which side of a border
SKIP_SUBTYPES = {"country"}        # a country's outline is large; its name comes from the country code
GEO_RE = re.compile(r"geo:(-?\d{1,2}(?:\.\d+)?),(-?\d{1,3}(?:\.\d+)?)")
MAX_SCAN_BYTES = 64 * 1024 * 1024


class MapsError(RuntimeError):
    """The request cannot be done; the message says why."""


# ── cells ────────────────────────────────────────────────────────────────────

def cell_of(lat: float, lon: float) -> str:
    return f"{math.floor(lat / CELL_DEG) * CELL_DEG:.1f},{math.floor(lon / CELL_DEG) * CELL_DEG:.1f}"


def cell_box(cell: str) -> tuple[float, float, float, float]:
    """(west, south, east, north) of a cell."""
    s, w = (float(x) for x in cell.split(","))
    return w, s, w + CELL_DEG, s + CELL_DEG


def cells_in_box(west: float, south: float, east: float, north: float) -> set[str]:
    if not (-180 <= west < east <= 180 and -90 <= south < north <= 90):
        raise MapsError("a box is WEST,SOUTH,EAST,NORTH in degrees, west < east and south < north")
    out = set()
    lat = math.floor(south / CELL_DEG) * CELL_DEG
    while lat < north:
        lon = math.floor(west / CELL_DEG) * CELL_DEG
        while lon < east:
            out.add(cell_of(lat + 1e-9, lon + 1e-9))
            lon += CELL_DEG
        lat += CELL_DEG
    if len(out) > 20_000:
        raise MapsError(f"that box is {len(out)} cells; download a smaller area (a state or a country)")
    return out


def clusters(cells: Iterable[str]) -> list[list[str]]:
    """Cells that touch (sides or corners) are fetched together: one read of their box costs about what one cell does."""
    todo = set(cells)
    out = []
    while todo:
        first = todo.pop()
        group, stack = [first], [first]
        while stack:
            s, w = (round(float(x) / CELL_DEG) for x in stack.pop().split(","))
            for ds in (-1, 0, 1):
                for dw in (-1, 0, 1):
                    n = f"{(s + ds) * CELL_DEG:.1f},{(w + dw) * CELL_DEG:.1f}"
                    if n in todo:
                        todo.remove(n)
                        group.append(n)
                        stack.append(n)
        out.append(sorted(group))
    return sorted(out, key=lambda g: g[0])


def group_box(group: Iterable[str]) -> tuple[float, float, float, float]:
    boxes = [cell_box(c) for c in group]
    return min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes)


# ── where the data is ────────────────────────────────────────────────────────

def positions(paths: Paths) -> Counter:
    """How many times each position (to about 100 m) appears in the converted documents."""
    found: Counter = Counter()
    root = paths.markup
    if not root.is_dir():
        return found
    for md in root.rglob("*.md"):
        with contextlib.suppress(OSError):
            if md.stat().st_size > MAX_SCAN_BYTES:
                continue
            text = md.read_text(encoding="utf-8", errors="replace")
            if "geo:" not in text:
                continue
            for m in GEO_RE.finditer(text):
                lat, lon = float(m.group(1)), float(m.group(2))
                if -90 <= lat <= 90 and -180 <= lon <= 180:
                    found[(round(lat, 3), round(lon, 3))] += 1
    return found


def documents_with_positions(paths: Paths) -> list[Path]:
    out = []
    if paths.markup.is_dir():
        for md in paths.markup.rglob("*.md"):
            with contextlib.suppress(OSError):
                if md.stat().st_size <= MAX_SCAN_BYTES and GEO_RE.search(md.read_text(encoding="utf-8", errors="replace")):
                    out.append(md)
    return out


def plan(paths: Paths) -> dict[str, Any]:
    """The cells the indexed documents need, which of them are downloaded, and a rough size for the rest."""
    pos = positions(paths)
    need = Counter()
    for (lat, lon), n in pos.items():
        need[cell_of(lat, lon)] += n
    have = {k: set(installed_cells(paths, k)) for k in KINDS}
    missing = sorted(c for c in need if any(c not in have[k] for k in KINDS))
    return {"positions": len(pos), "cells_needed": len(need), "cells_missing": len(missing), "missing": missing,
            "fetches": len(clusters(missing)), "estimate_mb": round(len(missing) * EST_MB_PER_CELL, 1),
            "cell_km": round(CELL_DEG * 111)}


# ── local store ──────────────────────────────────────────────────────────────

def store_dir(paths: Paths) -> Path:
    return reference.root(paths) / "overture"


def db_path(paths: Paths, kind: str) -> Path:
    return store_dir(paths) / f"{kind}.sqlite"


SCHEMA = {
    "places": ("create table if not exists place(id integer primary key, oid text, name text, category text, "
               "confidence real, lat real, lon real, cell text)",
               "create virtual table if not exists place_rt using rtree(id, minx, maxx, miny, maxy)",
               "create index if not exists place_cell on place(cell)"),
    "divisions": ("create table if not exists area(id integer primary key, oid text unique, name text, subtype text, "
                  "country text, wkb blob)",
                  "create virtual table if not exists area_rt using rtree(id, minx, maxx, miny, maxy)"),
}


def connect(paths: Paths, kind: str, *, create: bool = False) -> sqlite3.Connection | None:
    p = db_path(paths, kind)
    if not p.exists() and not create:
        return None
    p.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(p, timeout=30)
    if create:
        con.execute("create table if not exists meta(key text primary key, value text)")
        con.execute("create table if not exists cell(cell text primary key, rows integer, release text, at text)")
        for sql in SCHEMA[kind]:
            con.execute(sql)
    return con


def meta(paths: Paths, kind: str) -> dict[str, str]:
    con = connect(paths, kind)
    if con is None:
        return {}
    with contextlib.closing(con):
        try:
            return dict(con.execute("select key, value from meta"))
        except sqlite3.Error:
            return {}


def installed_cells(paths: Paths, kind: str) -> dict[str, str]:
    con = connect(paths, kind)
    if con is None:
        return {}
    with contextlib.closing(con):
        try:
            return dict(con.execute("select cell, release from cell"))
        except sqlite3.Error:
            return {}


def ref(paths: Paths, kind: str) -> str:
    rel = meta(paths, kind).get("release", "")
    return f"overture-{kind}@{rel}" if rel else ""


def status(paths: Paths) -> list[dict[str, Any]]:
    """The three datasets as rows of one table (the Models tab and `rag-search maps`)."""
    g = next(d for d in reference.status(paths) if d["name"] == "geonames")
    rows = [{"name": "geonames", "title": TITLES["geonames"], "used_for": USED_FOR["geonames"],
             "license": LICENSES["geonames"], "scope": "worldwide", "installed": g["installed"],
             "release": g["fetched"], "cells": None, "bytes": g["bytes"], "size_hint": g["size"],
             "folder": g["folder"]}]
    for kind in KINDS:
        name = f"overture-{kind}"
        cells = installed_cells(paths, kind)
        p = db_path(paths, kind)
        rows.append({"name": name, "title": TITLES[name], "used_for": USED_FOR[name], "license": LICENSES[name],
                     "scope": "by cell", "installed": bool(cells), "release": meta(paths, kind).get("release", ""),
                     "cells": len(cells), "bytes": p.stat().st_size if p.exists() else 0,
                     "size_hint": f"about {EST_MB_PER_CELL:g} MB per cell", "folder": str(store_dir(paths))})
    return rows


# ── download ─────────────────────────────────────────────────────────────────

Reader = Callable[[str, tuple[float, float, float, float], str], list[dict[str, Any]]]


def _fs() -> Any:
    try:
        import pyarrow.fs as pafs
    except ImportError as exc:                       # pragma: no cover - comes with docling's dependencies
        raise MapsError("pyarrow is needed to download Overture Maps (reinstall rag-search)") from exc
    return pafs.S3FileSystem(anonymous=True, region=REGION, connect_timeout=30, request_timeout=120)


def latest_release() -> str:
    import pyarrow.fs as pafs

    try:
        infos = _fs().get_file_info(pafs.FileSelector(BUCKET))
    except OSError as exc:
        raise MapsError(f"cannot reach Overture Maps ({exc})") from exc
    rels = sorted(i.base_name for i in infos if i.type == pafs.FileType.Directory and re.match(r"\d{4}-\d\d-\d\d", i.base_name))
    if not rels:
        raise MapsError("Overture Maps lists no release")
    return rels[-1]


def overture_rows(kind: str, box: tuple[float, float, float, float], release: str) -> list[dict[str, Any]]:
    """The records of one Overture type whose bounding box meets *box* (west, south, east, north)."""
    import pyarrow.compute as pc
    import pyarrow.dataset as ds

    theme, typ = ("places", "place") if kind == "places" else ("divisions", "division_area")
    try:
        d = ds.dataset(f"{BUCKET}/{release}/theme={theme}/type={typ}/", filesystem=_fs(), format="parquet")
    except OSError as exc:
        raise MapsError(f"cannot read Overture release {release} ({exc})") from exc
    w, s, e, n = box
    flt = ((pc.field("bbox", "xmin") <= e) & (pc.field("bbox", "xmax") >= w)
           & (pc.field("bbox", "ymin") <= n) & (pc.field("bbox", "ymax") >= s))
    names = set(d.schema.names)
    if kind == "places":
        cat = "basic_category" if "basic_category" in names else "categories"
        cols = [c for c in ("id", "names", cat, "confidence", "bbox") if c in names]
        tab = d.to_table(columns=cols, filter=flt).to_pylist()
        out = []
        for r in tab:
            name = (r.get("names") or {}).get("primary")
            b = r.get("bbox") or {}
            if not name or b.get("xmin") is None:
                continue
            c = r.get(cat)
            out.append({"oid": r["id"], "name": name, "category": (c or {}).get("primary") if isinstance(c, dict) else c,
                        "confidence": r.get("confidence"), "lat": b["ymin"], "lon": b["xmin"]})
        return out
    cols = [c for c in ("id", "names", "subtype", "country", "geometry", "bbox") if c in names]
    out = []
    for r in d.to_table(columns=cols, filter=flt).to_pylist():
        if not (r.get("names") or {}).get("primary"):
            continue
        out.append({"oid": r["id"], "name": r["names"]["primary"], "subtype": r.get("subtype") or "",
                    "country": r.get("country") or "", "wkb": r["geometry"]})
    return out


def _store_places(con: sqlite3.Connection, rows: list[dict[str, Any]], cells: set[str]) -> Counter:
    counts: Counter = Counter()
    con.execute(f"delete from place_rt where id in (select id from place where cell in ({','.join('?' * len(cells))}))", sorted(cells))
    con.execute(f"delete from place where cell in ({','.join('?' * len(cells))})", sorted(cells))
    for r in rows:
        c = cell_of(r["lat"], r["lon"])
        if c not in cells:
            continue
        cur = con.execute("insert into place(oid, name, category, confidence, lat, lon, cell) values (?,?,?,?,?,?,?)",
                          (r["oid"], r["name"], r["category"], r["confidence"], r["lat"], r["lon"], c))
        con.execute("insert into place_rt values (?,?,?,?,?)", (cur.lastrowid, r["lon"], r["lon"], r["lat"], r["lat"]))
        counts[c] += 1
    return counts


def _store_divisions(con: sqlite3.Connection, rows: list[dict[str, Any]]) -> int:
    import shapely

    added = 0
    for r in rows:
        if r.get("subtype") in SKIP_SUBTYPES or con.execute("select 1 from area where oid = ?", (r["oid"],)).fetchone():
            continue
        geom = shapely.from_wkb(r["wkb"])
        if geom is None or geom.is_empty:
            continue
        geom = shapely.simplify(geom, SIMPLIFY_DEG, preserve_topology=True)
        x0, y0, x1, y1 = geom.bounds
        cur = con.execute("insert into area(oid, name, subtype, country, wkb) values (?,?,?,?,?)",
                          (r["oid"], r["name"], r["subtype"], r["country"], shapely.to_wkb(geom)))
        con.execute("insert into area_rt values (?,?,?,?,?)", (cur.lastrowid, x0, x1, y0, y1))
        added += 1
    return added


def download_overture(paths: Paths, cells: Iterable[str], *, release: str = "", reader: Reader | None = None,
                      progress: Callable[[int, int, str], None] | None = None,
                      log: Callable[[str], None] | None = None) -> dict[str, Any]:
    """Fetch both Overture datasets for *cells*.  The store keeps one release: when *release* differs from the one on
    disk, every cell already there is fetched again with it (``update``)."""
    reader = reader or overture_rows
    say = log or (lambda _m: None)
    wanted = set(cells)
    on_disk = {k: meta(paths, k).get("release", "") for k in KINDS}
    release = release or (on_disk["places"] or on_disk["divisions"]) or latest_release()
    for k in KINDS:
        if on_disk[k] and on_disk[k] != release:
            wanted |= set(installed_cells(paths, k))
    if not wanted:
        return {"release": release, "cells": 0, "places": 0, "areas": 0}
    groups = clusters(wanted)
    say(f"Overture {release}: {len(wanted)} cell(s) in {len(groups)} fetch(es)")
    places = areas = 0
    stamp = time.strftime("%Y-%m-%d %H:%M")
    for k in KINDS:
        if on_disk[k] and on_disk[k] != release:           # a new release replaces the old one wholesale
            with contextlib.suppress(OSError):
                db_path(paths, k).unlink()
    for i, group in enumerate(groups):
        box = group_box(group)
        gset = set(group) & wanted
        for kind in KINDS:
            if progress:
                progress(i * 2 + (kind == "places"), len(groups) * 2, f"{kind} for {len(gset)} cell(s) around {group[0]}")
            rows = reader(kind, box, release)
            con = connect(paths, kind, create=True)
            with contextlib.closing(con), con:
                con.execute("insert or replace into meta values ('release', ?)", (release,))
                if kind == "places":
                    counts = _store_places(con, rows, gset)
                    places += sum(counts.values())
                else:
                    areas += _store_divisions(con, rows)
                    counts = Counter()
                con.executemany("insert or replace into cell values (?,?,?,?)",
                                [(c, counts.get(c, 0), release, stamp) for c in gset])
        say(f"  {i + 1}/{len(groups)}: {len(gset)} cell(s) around {group[0]} done")
    if progress:
        progress(len(groups) * 2, len(groups) * 2, "done")
    return {"release": release, "cells": len(wanted), "places": places, "areas": areas}


def remove(paths: Paths, name: str) -> bool:
    if name == "geonames":
        target = reference.folder(paths, "geonames")
        if not target.exists():
            return False
        import shutil

        shutil.rmtree(target, ignore_errors=True)
        return True
    if name in ("overture", "overture-places", "overture-divisions"):
        kinds = KINDS if name == "overture" else (name.split("-", 1)[1],)
        gone = False
        for k in kinds:
            with contextlib.suppress(OSError):
                db_path(paths, k).unlink()
                gone = True
        return gone
    raise MapsError(f"unknown map dataset {name!r}; choose from geonames, overture")


# ── refresh ──────────────────────────────────────────────────────────────────

def refresh(paths: Paths) -> int:
    """Make the next indexing run convert every document with positions again (its Markdown and its index are marked
    out of date; the published index keeps serving until then).  Returns how many were marked."""
    n = 0
    for md in documents_with_positions(paths):
        rel = md.relative_to(paths.markup).with_suffix("")
        for stale in (md.with_name(md.name + ".sha256"), paths.index / rel / META_FILE):
            with contextlib.suppress(OSError):
                os.unlink(stale)
        n += 1
    return n
