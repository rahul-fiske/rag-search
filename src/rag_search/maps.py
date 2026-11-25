"""Maps: the place data indexing uses to describe a position, and how it is downloaded.

Three datasets, each downloaded only when asked (``rag-search maps`` or the Models tab), never while indexing:

    geonames             towns and villages of 1,000+ people, worldwide, one file (``reference.py``): the nearest town
    overture-divisions   Overture Maps areas (district, state, locality ...): the area a position is inside
    overture-places      Overture Maps places of interest (restaurants, fuel, temples ...): what is around it

The two Overture datasets are kept per *cell* of the map, a square of ``CELL_DEG`` degrees (0.05, about 6 km), so only
the parts of the world that matter are on disk.  Which cells matter is read from the indexed documents themselves:
every position a converted document mentions is written as ``geo:LAT,LON`` (photos, Google Timeline), and ``plan``
lists the cells those positions fall in that are not downloaded yet.  Cells are *fetched* in larger groups (``FETCH_DEG``,
0.5 degree squares that touch are read together: a read costs about the same whatever its size) and only the needed
cells are kept.  Both are stored in SQLite files under ``<home>/reference/overture/``, pinned to one Overture release
(``update`` moves every cell to the newest one).  Places are stored compactly: no Overture id, the category as a small
number, positions as integers (about 1 m), and one ``(cell, lat)`` index instead of a spatial index.

Each converted document with positions gets a *stamp* (``<doc>.md.geo.json``): for every cell it touches, which map
data described it (the GeoNames download, the Overture release and download time of that cell).  ``refresh`` compares
the stamps with what is installed now and marks only the documents whose cells changed.

Overture is read straight from its public bucket with pyarrow (row groups outside the cells are not transferred);
pyarrow and shapely come with docling's dependencies and are imported only here.  ``refresh`` makes the documents whose
map data changed be described again by the next indexing run: a photo in place (its Markdown is kept and only its Photo
details section is rewritten; the picture is not read again), anything else (a timeline) converted again.
"""

from __future__ import annotations

import contextlib
import json
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
from .paths import META_FILE, Paths, read_json, write_json_atomic

PER_DEG = 20                       # store cells per degree of latitude or longitude
CELL_DEG = 1 / PER_DEG             # a store cell: 0.05 degree, about 6 km
FETCH_DEG = 0.5                    # cells are read from Overture in groups inside squares of this size
FETCH_CELLS = round(FETCH_DEG * PER_DEG)
GRID = "0.05"                      # stored in each database: the cell size it was made with
E5 = 100_000                       # positions are stored as integers: degrees x E5 (about 1 m)
MAX_BOX_CELLS = 250_000
MARGIN_M = 150                     # what a lookup reads around a position: a position needs every cell this reaches into
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
EST_MB_PER_CELL = 0.15             # measured: 528 cells around Pune, Mumbai, Delhi, Bengaluru ... are 58 MB of places and 15 MB of areas
SIMPLIFY_DEG = 0.0005              # area outlines are kept to about 50 m: enough to say which side of a border
SKIP_SUBTYPES = {"country"}        # a country's outline is large; its name comes from the country code
GEO_RE = re.compile(r"geo:(-?\d{1,2}(?:\.\d+)?),(-?\d{1,3}(?:\.\d+)?)")
MAX_SCAN_BYTES = 64 * 1024 * 1024
STAMP_SUFFIX = ".geo.json"         # <doc>.md.geo.json
PHOTO_HEADING = "## Photo details"


class MapsError(RuntimeError):
    """The request cannot be done; the message says why."""


# ── cells ────────────────────────────────────────────────────────────────────

def cell_index(lat: float, lon: float) -> tuple[int, int]:
    return math.floor(lat * PER_DEG), math.floor(lon * PER_DEG)


def label(i: int, j: int) -> str:
    return f"{i / PER_DEG:.2f},{j / PER_DEG:.2f}"


def parse(cell: str) -> tuple[int, int]:
    s, w = cell.split(",")
    return round(float(s) * PER_DEG), round(float(w) * PER_DEG)


def cell_of(lat: float, lon: float) -> str:
    """The cell of a position, named by its south-west corner: ``19.05,78.30``."""
    return label(*cell_index(lat, lon))


def cells_around(lat: float, lon: float) -> set[str]:
    """The cells a lookup at this position reads: its own and, near an edge, the neighbours the ``MARGIN_M`` around it
    reaches into (a place across the edge is as near as one on this side)."""
    dlat = MARGIN_M / 111_000
    dlon = MARGIN_M / (111_000 * max(0.01, math.cos(math.radians(lat))))
    i0, j0 = cell_index(max(-90.0, lat - dlat), max(-180.0, lon - dlon))
    i1, j1 = cell_index(min(89.999999, lat + dlat), min(179.999999, lon + dlon))
    return {label(i, j) for i in range(i0, i1 + 1) for j in range(j0, j1 + 1)}


def cell_no(i: int, j: int) -> int:
    """The number a cell has in the places table (positive, fits in four bytes)."""
    return (i + 90 * PER_DEG) * 10_000 + (j + 180 * PER_DEG)


def cell_box(cell: str) -> tuple[float, float, float, float]:
    """(west, south, east, north) of a cell."""
    i, j = parse(cell)
    return j / PER_DEG, i / PER_DEG, (j + 1) / PER_DEG, (i + 1) / PER_DEG


def cells_in_box(west: float, south: float, east: float, north: float) -> set[str]:
    if not (-180 <= west < east <= 180 and -90 <= south < north <= 90):
        raise MapsError("a box is WEST,SOUTH,EAST,NORTH in degrees, west < east and south < north")
    i0, i1 = math.floor(south * PER_DEG), math.ceil(north * PER_DEG)
    j0, j1 = math.floor(west * PER_DEG), math.ceil(east * PER_DEG)
    if (i1 - i0) * (j1 - j0) > MAX_BOX_CELLS:
        raise MapsError(f"that box is {(i1 - i0) * (j1 - j0):,} cells of about 6 km; download a smaller area "
                        f"(a state, not a country)")
    return {label(i, j) for i in range(i0, i1) for j in range(j0, j1)}


def fetch_cell(cell: str) -> tuple[int, int]:
    """The 0.5 degree square a cell is read in."""
    i, j = parse(cell)
    return i // FETCH_CELLS, j // FETCH_CELLS


def clusters(cells: Iterable[str]) -> list[list[str]]:
    """Cells are fetched together with the cells of the 0.5 degree squares that touch theirs (sides or corners): one
    read of the group costs about what one square does.  Returns the groups of cells."""
    by_square: dict[tuple[int, int], list[str]] = {}
    for c in cells:
        by_square.setdefault(fetch_cell(c), []).append(c)
    todo = set(by_square)
    out = []
    while todo:
        first = todo.pop()
        squares, stack = [first], [first]
        while stack:
            a, b = stack.pop()
            for da in (-1, 0, 1):
                for db in (-1, 0, 1):
                    n = (a + da, b + db)
                    if n in todo:
                        todo.remove(n)
                        squares.append(n)
                        stack.append(n)
        out.append(sorted(c for sq in squares for c in by_square[sq]))
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
        for c in cells_around(lat, lon):
            need[c] += n
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
    "places": ("create table if not exists place(cell integer, lat integer, lon integer, name text, cat integer, "
               "conf integer)",
               "create index if not exists place_cell on place(cell, lat)",
               "create table if not exists cat(id integer primary key, name text unique)"),
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
        con.execute("insert or ignore into meta values ('grid', ?)", (GRID,))
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
    """The cells downloaded for *kind* and the Overture release of each (a store made with the old 0.5 degree cells
    counts as empty until ``migrate`` converts it)."""
    con = connect(paths, kind)
    if con is None:
        return {}
    with contextlib.closing(con):
        try:
            if (dict(con.execute("select key, value from meta")).get("grid") or "") != GRID:
                return {}
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
                     "size_hint": f"about {EST_MB_PER_CELL:g} MB per cell", "folder": str(store_dir(paths)),
                     "needs_convert": is_old_grid(paths, kind)})
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
            out.append({"name": name, "category": (c or {}).get("primary") if isinstance(c, dict) else c,
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


def _cat_ids(con: sqlite3.Connection) -> dict[str, int]:
    return {name: i for i, name in con.execute("select id, name from cat")}


def _cat_id(con: sqlite3.Connection, ids: dict[str, int], name: str | None) -> int:
    """The small number of a category name (0: none), added to the ``cat`` table the first time it is seen."""
    if not name:
        return 0
    if name not in ids:
        ids[name] = con.execute("insert into cat(name) values (?)", (name,)).lastrowid or 0
    return ids[name]


def _delete_cells(con: sqlite3.Connection, cells: Iterable[str]) -> None:
    nos = [cell_no(*parse(c)) for c in cells]
    for k in range(0, len(nos), 500):
        part = nos[k:k + 500]
        con.execute(f"delete from place where cell in ({','.join('?' * len(part))})", part)


def _store_places(con: sqlite3.Connection, rows: list[dict[str, Any]], cells: set[str]) -> Counter:
    """Replace the places of *cells* with those of *rows* that lie in them (a read covers more than was asked for)."""
    counts: Counter = Counter()
    _delete_cells(con, cells)
    ids = _cat_ids(con)
    batch = []
    for r in rows:
        i, j = cell_index(r["lat"], r["lon"])
        c = label(i, j)
        if c not in cells:
            continue
        conf = r.get("confidence")
        batch.append((cell_no(i, j), round(r["lat"] * E5), round(r["lon"] * E5), r["name"],
                      _cat_id(con, ids, r.get("category")), None if conf is None else round(conf * 100)))
        counts[c] += 1
    con.executemany("insert into place(cell, lat, lon, name, cat, conf) values (?,?,?,?,?,?)", batch)
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
    say(f"Overture {release}: {len(wanted)} cell(s) of about 6 km in {len(groups)} fetch(es)")
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


# ── conversion of a store made with 0.5 degree cells ─────────────────────────

def is_old_grid(paths: Paths, kind: str) -> bool:
    return db_path(paths, kind).exists() and meta(paths, kind).get("grid", "") != GRID


def _parent_label(cell: str) -> str:
    i, j = fetch_cell(cell)
    return f"{i * FETCH_DEG:.1f},{j * FETCH_DEG:.1f}"


def migrate(paths: Paths, log: Callable[[str], None] | None = None) -> dict[str, Any] | None:
    """Convert a store made with 0.5 degree cells (4.0.0.dev3 and dev4) to the 0.05 degree cells and the compact places
    table, keeping only the cells the indexed documents need now: nothing is downloaded, and what was downloaded for
    the rest of each big cell is dropped (a later position there fetches it again).  None when there is nothing old."""
    old = [k for k in KINDS if is_old_grid(paths, k)]
    if not old:
        return None
    say = log or (lambda _m: None)
    need = {c for la, lo in positions(paths) for c in cells_around(la, lo)}
    res: dict[str, Any] = {"cells": 0, "places": 0}
    for kind in old:
        path = db_path(paths, kind)
        before = path.stat().st_size
        src = sqlite3.connect(path, timeout=30)
        try:
            old_cells = {c: (rel, at) for c, rel, at in src.execute("select cell, release, at from cell")}
            release = dict(src.execute("select key, value from meta")).get("release", "")
            keep = {c: old_cells[_parent_label(c)] for c in need if _parent_label(c) in old_cells}
            say(f"converting {kind}: {len(keep)} of the {len(need)} cells your documents need were downloaded")
            if kind == "divisions":
                with src:
                    src.execute("delete from cell")
                    src.executemany("insert into cell values (?,?,?,?)", [(c, 0, rel, at) for c, (rel, at) in keep.items()])
                    src.execute("insert or replace into meta values ('grid', ?)", (GRID,))
                src.close()
                src = None
                with contextlib.closing(sqlite3.connect(path)) as con:
                    con.execute("vacuum")
                res["cells"] = max(res["cells"], len(keep))
            else:
                tmp = path.with_name(path.name + ".new")
                with contextlib.suppress(OSError):
                    tmp.unlink()
                new = sqlite3.connect(tmp)
                new.execute("create table meta(key text primary key, value text)")
                new.execute("create table cell(cell text primary key, rows integer, release text, at text)")
                for sql in SCHEMA["places"]:
                    new.execute(sql)
                ids: dict[str, int] = {}
                counts: Counter = Counter()
                batch: list[tuple] = []
                for name, cat, conf, lat, lon in src.execute("select name, category, confidence, lat, lon from place"):
                    i, j = cell_index(lat, lon)
                    c = label(i, j)
                    if c not in keep:
                        continue
                    batch.append((cell_no(i, j), round(lat * E5), round(lon * E5), name, _cat_id(new, ids, cat),
                                  None if conf is None else round(conf * 100)))
                    counts[c] += 1
                    if len(batch) >= 50_000:
                        new.executemany("insert into place values (?,?,?,?,?,?)", batch)
                        batch = []
                new.executemany("insert into place values (?,?,?,?,?,?)", batch)
                new.executemany("insert into cell values (?,?,?,?)", [(c, counts[c], rel, at) for c, (rel, at) in keep.items()])
                new.executemany("insert into meta values (?,?)", [("release", release), ("grid", GRID)])
                new.commit()
                new.execute("vacuum")
                new.close()
                src.close()
                src = None
                os.replace(tmp, path)
                res["cells"] = len(keep)
                res["places"] = sum(counts.values())
        finally:
            if src is not None:
                src.close()
        say(f"  {kind}: {before / 2**20:.0f} MB -> {path.stat().st_size / 2**20:.0f} MB")
        res[f"{kind}_mb"] = round(path.stat().st_size / 2**20, 1)
        res[f"{kind}_was_mb"] = round(before / 2**20, 1)
    return res


# ── stamps and refresh ───────────────────────────────────────────────────────

def stamp_path(md: Path) -> Path:
    return md.with_name(md.name + STAMP_SUFFIX)


def cells_in_text(text: str) -> set[str]:
    out: set[str] = set()
    for m in GEO_RE.finditer(text):
        lat, lon = float(m.group(1)), float(m.group(2))
        if -90 <= lat <= 90 and -180 <= lon <= 180:
            out |= cells_around(lat, lon)
    return out


def _cell_rows(paths: Paths, kind: str) -> dict[str, str]:
    con = connect(paths, kind)
    if con is None:
        return {}
    with contextlib.closing(con):
        try:
            return {c: f"{r}/{at}" for c, r, at in con.execute("select cell, release, at from cell")}
        except sqlite3.Error:
            return {}


def map_state(paths: Paths) -> dict[str, Any]:
    """What is installed now, in the form the stamps are compared with."""
    return {"geonames": reference.ref(paths, "geonames"), "divisions": _cell_rows(paths, "divisions"),
            "places": _cell_rows(paths, "places")}


def cell_stamp(cell: str, state: dict[str, Any]) -> str:
    return f"{state['geonames']}|{state['divisions'].get(cell, '')}|{state['places'].get(cell, '')}"


def write_stamp(paths: Paths, md: Path, text: str, state: dict[str, Any] | None = None) -> None:
    """Record which map data described the positions of *md* (called when its Markdown was made or its photo section
    renewed)."""
    state = state or map_state(paths)
    cells = cells_in_text(text)
    with contextlib.suppress(OSError):
        stamp_path(md).write_text(json.dumps({"cells": {c: cell_stamp(c, state) for c in sorted(cells)}}),
                                  encoding="utf-8")


def is_current(md: Path, text: str, state: dict[str, Any]) -> bool:
    try:
        saved = json.loads(stamp_path(md).read_text(encoding="utf-8"))["cells"]
    except (OSError, ValueError, KeyError, TypeError):
        return False                       # converted before stamps existed, or the stamp is unreadable
    return all(saved.get(c) == cell_stamp(c, state) for c in cells_in_text(text))


def mark_stale(meta_file: Path, why: str) -> None:
    """Mark a document's index record out of date (``stale``): the next run converts or describes it again and embeds it,
    while its last complete version keeps being merged and served (``indexer.is_fresh`` and the quick check see the mark)."""
    if not meta_file.exists():
        return
    meta = read_json(meta_file)
    if isinstance(meta, dict) and meta.get("stale") != why:
        meta["stale"] = why
        write_json_atomic(meta_file, meta)


def refresh(paths: Paths, *, everything: bool = False) -> dict[str, int]:
    """Mark the documents whose map data changed (every document with positions, with *everything*) so the next
    indexing run describes them again.  A photo keeps its Markdown (only its index is marked out of date: the run
    rewrites its Photo details section and embeds it again, without reading the picture); any other document has its
    Markdown marked out of date too and is converted again.  The published index keeps serving until then."""
    res = {"checked": 0, "current": 0, "refreshed": 0, "in_place": 0, "reconverted": 0}
    if not paths.markup.is_dir():
        return res
    state = map_state(paths)
    for md in paths.markup.rglob("*.md"):
        try:
            if md.stat().st_size > MAX_SCAN_BYTES:
                continue
            text = md.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if not GEO_RE.search(text):
            continue
        res["checked"] += 1
        if not everything and is_current(md, text, state):
            res["current"] += 1
            continue
        rel = md.relative_to(paths.markup).with_suffix("")
        # the record is marked, not removed: the document stays complete (and searchable) until the run has embedded it again
        mark_stale(paths.index / rel / META_FILE, "places")
        if PHOTO_HEADING in text:
            res["in_place"] += 1
        else:
            with contextlib.suppress(OSError):
                os.unlink(md.with_name(md.name + ".sha256"))
            res["reconverted"] += 1
        res["refreshed"] += 1
    return res
