"""What is known about a position: the area it is in, the town it is near, and the places around it.

``describe(lat, lon, paths)`` asks the map data that is installed (``maps.py``) and returns lines for a document's
Markdown, plus the reference data they came from::

    Area: Aundh, Pune, Maharashtra, India                        Overture divisions (smallest first)
    Town: near Pune, Maharashtra, India (about 6 km NW of it)    GeoNames (the nearest town of 1,000+ people)
    Nearby: Quick Bite (restaurant, 6 m); Green Gate Resort (resort, 29 m)      Overture places within 100 m

Nothing is chosen for the reader: the places around a position are candidates with their category and distance (a
phone's GPS is off by tens of metres, and in a busy street ten shops are that close), and the one searching weighs them
against what else is known.  A dataset that is not installed, or a cell not downloaded, adds no line.  Tables are
opened once per process and reopened when a download replaces them.
"""

from __future__ import annotations

import contextlib
import math
import re
import sqlite3
from dataclasses import dataclass
from typing import Any

from .. import maps, reference
from ..paths import Paths

MAX_KM = 50.0                    # a town further than this is not "near"
AT_KM = 1.0                      # closer than this: "at", not "near"
NEARBY_M = 100.0
NEARBY_MAX = 5
FOOD_M = float(maps.MARGIN_M)    # the nearest place to eat is named even when it is a little further than NEARBY_M
FOOD_RE = re.compile(r"restaurant|food|cafe|coffee|tea|bakery|dhaba|diner|eatery|canteen|sweet|dessert|ice_cream|"
                     r"pizza|burger|bar\b|pub\b|juice|snack", re.I)
EARTH_KM = 6371.0
_COMPASS = ("N", "NE", "E", "SE", "S", "SW", "W", "NW")
AREA_ORDER = ("microhood", "neighborhood", "macrohood", "locality", "localadmin", "county", "macrocounty", "region",
              "macroregion", "dependency")


@dataclass(frozen=True)
class Place:
    name: str
    admin1: str
    country: str
    km: float
    direction: str               # where the position lies from the place ("" when at it)

    def text(self) -> str:
        where = ", ".join(x for x in (self.name, self.admin1, self.country) if x)
        if self.km < AT_KM:
            return f"at {where}"
        return f"near {where} (about {self.km:.0f} km {self.direction} of it)"


def km_between(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    a = (math.sin(math.radians(lat2 - lat1) / 2) ** 2
         + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(math.radians(lon2 - lon1) / 2) ** 2)
    return 2 * EARTH_KM * math.asin(min(1.0, math.sqrt(a)))


def _bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    lat1, lon1, lat2, lon2 = (math.radians(x) for x in (lat1, lon1, lat2, lon2))
    dl = lon2 - lon1
    y = math.sin(dl) * math.cos(lat2)
    x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dl)
    return (math.degrees(math.atan2(y, x)) + 360) % 360


# ── GeoNames: the nearest town ───────────────────────────────────────────────

_TOWNS: dict[str, Any] = {}


def _towns(paths: Paths) -> dict[str, Any] | None:
    key = reference.ref(paths, "geonames")
    if not key:
        return None
    if _TOWNS.get("key") == key:
        return _TOWNS["data"]
    import numpy as np

    cities = reference.path(paths, "geonames", "cities1000.txt")
    if cities is None:
        return None
    admin: dict[str, str] = {}
    p = reference.path(paths, "geonames", "admin1CodesASCII.txt")
    if p:
        for line in p.read_text(encoding="utf-8").splitlines():
            parts = line.split("\t")
            if len(parts) >= 2:
                admin[parts[0]] = parts[1]
    names, a1, cc, lat, lon = [], [], [], [], []
    with open(cities, encoding="utf-8") as fh:
        for line in fh:
            f = line.rstrip("\n").split("\t")
            if len(f) < 11:
                continue
            try:
                la, lo = float(f[4]), float(f[5])
            except ValueError:
                continue
            names.append(f[1])
            cc.append(f[8])
            a1.append(admin.get(f"{f[8]}.{f[10]}", ""))
            lat.append(la)
            lon.append(lo)
    data = {"names": names, "admin1": a1, "cc": cc, "lat": np.radians(np.array(lat)), "lon": np.radians(np.array(lon))}
    _TOWNS.update(key=key, data=data)
    return data


_COUNTRIES: dict[str, Any] = {}


def country_name(code: str, paths: Paths) -> str:
    """A country's name from its ISO code (GeoNames' country list, when installed), else the code."""
    key = reference.ref(paths, "geonames")
    if _COUNTRIES.get("key") != key:
        names: dict[str, str] = {}
        p = reference.path(paths, "geonames", "countryInfo.txt") if key else None
        if p:
            for line in p.read_text(encoding="utf-8").splitlines():
                parts = line.split("\t")
                if len(parts) > 4 and not line.startswith("#"):
                    names[parts[0]] = parts[4]
        _COUNTRIES.update(key=key, names=names)
    return _COUNTRIES["names"].get(code, code)


def near(lat: float, lon: float, paths: Paths) -> Place | None:
    """The nearest GeoNames town within ``MAX_KM``, or None (not installed, nothing near)."""
    data = _towns(paths)
    if not data or not data["names"] or not (-90 <= lat <= 90 and -180 <= lon <= 180):
        return None
    import numpy as np

    la, lo = math.radians(lat), math.radians(lon)
    h = np.sin((data["lat"] - la) / 2) ** 2 + math.cos(la) * np.cos(data["lat"]) * np.sin((data["lon"] - lo) / 2) ** 2
    i = int(np.argmin(h))
    km = 2 * EARTH_KM * math.asin(min(1.0, math.sqrt(float(h[i]))))
    if km > MAX_KM:
        return None
    tlat, tlon = math.degrees(float(data["lat"][i])), math.degrees(float(data["lon"][i]))
    d = _COMPASS[int(((_bearing(tlat, tlon, lat, lon) + 22.5) % 360) // 45)]
    return Place(data["names"][i], data["admin1"][i], country_name(data["cc"][i], paths), km, d)


# ── Overture: the area and the places around ─────────────────────────────────

_DB: dict[str, tuple[float, sqlite3.Connection]] = {}
_SHAPES: dict[int, Any] = {}
_CATS: dict[str, dict[int, str]] = {}


def _db(paths: Paths, kind: str) -> sqlite3.Connection | None:
    p = maps.db_path(paths, kind)
    try:
        mtime = p.stat().st_mtime
    except OSError:
        return None
    key = str(p)
    if key in _DB and _DB[key][0] == mtime:
        return _DB[key][1]
    if key in _DB:
        with contextlib.suppress(sqlite3.Error):
            _DB[key][1].close()
        if kind == "divisions":
            _SHAPES.clear()
    try:
        con = sqlite3.connect(f"file:{p}?mode=ro", uri=True, check_same_thread=False)
        if kind == "places":
            _CATS[key] = {i: name for i, name in con.execute("select id, name from cat")}
    except sqlite3.Error:
        return None                      # a store made with the old cells, until it is converted
    _DB[key] = (mtime, con)
    return con


def covered(lat: float, lon: float, paths: Paths, kind: str) -> bool:
    """Is everything a lookup at this position reads downloaded for *kind* (its cell and, near an edge, the neighbours)?"""
    con = _db(paths, kind)
    if con is None:
        return False
    want = sorted(maps.cells_around(lat, lon))
    try:
        have, = con.execute(f"select count(*) from cell where cell in ({','.join('?' * len(want))})", want).fetchone()
    except sqlite3.Error:
        return False
    return have == len(want)


LOCAL = {"microhood", "neighborhood", "macrohood", "locality", "localadmin"}


def area_parts(lat: float, lon: float, paths: Paths) -> list[tuple[str, str]]:
    """(subtype, name) of the Overture areas the position is in, smallest first, ending with ("country", name)."""
    con = _db(paths, "divisions")
    if con is None:
        return []
    import shapely

    pt = shapely.Point(lon, lat)
    found: dict[str, str] = {}
    country = ""
    try:
        rows = con.execute("select a.id, a.name, a.subtype, a.country, a.wkb from area a join area_rt r on a.id = r.id "
                           "where r.minx <= ? and r.maxx >= ? and r.miny <= ? and r.maxy >= ?", (lon, lon, lat, lat)).fetchall()
    except sqlite3.Error:
        return []
    for aid, name, subtype, cc, wkb in rows:
        shape = _SHAPES.get(aid)
        if shape is None:
            if len(_SHAPES) > 5000:
                _SHAPES.clear()
            shape = _SHAPES[aid] = shapely.from_wkb(wkb)
        if shape.contains(pt):
            found.setdefault(subtype, name)
            country = country or cc
    out: list[tuple[str, str]] = []
    for s in AREA_ORDER:
        if s in found and all(found[s] != n for _, n in out):
            out.append((s, found[s]))
    if country:
        out.append(("country", country_name(country, paths)))
    return out


def area(lat: float, lon: float, paths: Paths) -> list[str]:
    """The names of the Overture areas the position is in, smallest first, ending with the country."""
    return [n for _, n in area_parts(lat, lon, paths)]


def nearby(lat: float, lon: float, paths: Paths) -> list[tuple[float, str, str]]:
    """(metres, name, category) of the Overture places within ``NEARBY_M`` (at most ``NEARBY_MAX``, nearest first),
    plus the nearest place to eat within ``FOOD_M`` when none of those is one."""
    con = _db(paths, "places")
    if con is None:
        return []
    cats = _CATS.get(str(maps.db_path(paths, "places")), {})
    dlat = FOOD_M / 111_000
    nos = [maps.cell_no(*maps.parse(c)) for c in maps.cells_around(lat, lon)]
    try:
        rows = con.execute(f"select name, cat, lat, lon from place where cell in ({','.join('?' * len(nos))}) "
                           "and lat between ? and ?",
                           (*nos, round((lat - dlat) * maps.E5), round((lat + dlat) * maps.E5))).fetchall()
    except sqlite3.Error:
        return []
    hits = []
    for name, cat, la, lo in rows:
        d = km_between(lat, lon, la / maps.E5, lo / maps.E5) * 1000
        if d <= FOOD_M:
            hits.append((d, name, cats.get(cat, "")))
    hits.sort(key=lambda h: (round(h[0]), h[1], h[2]))           # places at one spot (a building) in a fixed order
    out = [h for h in hits if h[0] <= NEARBY_M][:NEARBY_MAX]
    if not any(FOOD_RE.search(h[2]) for h in out):
        food = next((h for h in hits if FOOD_RE.search(h[2])), None)
        if food:
            out.append(food)
    return out


def describe(lat: float, lon: float, paths: Paths) -> tuple[list[str], list[str]]:
    """Lines about the position (without the leading "- ") and the reference data they used."""
    lines: list[str] = []
    refs: list[str] = []
    names = area(lat, lon, paths)
    if names:
        lines.append("Area: " + ", ".join(names))
        refs.append(maps.ref(paths, "divisions"))
    town = near(lat, lon, paths)
    if town:
        lines.append(f"Town: {town.text()}")
        refs.append(reference.ref(paths, "geonames"))
    if covered(lat, lon, paths, "places"):
        around = nearby(lat, lon, paths)
        lines.append("Nearby: " + ("; ".join(f"{n} ({c.replace('_', ' ') or 'place'}, {m:.0f} m)" for m, n, c in around)
                                   if around else f"no places known within {NEARBY_M:.0f} m"))
        refs.append(maps.ref(paths, "places"))
    return lines, [r for r in refs if r]


def short(lat: float, lon: float, paths: Paths) -> str:
    """A few words for where a journey starts or ends: the locality and the area above it when Overture knows the
    locality, else the nearest town, else the areas it does know."""
    parts = area_parts(lat, lon, paths)
    if parts and parts[0][0] in LOCAL:
        return ", ".join(n for _, n in parts[:2])
    town = near(lat, lon, paths)
    if town:
        return f"{'at' if town.km < AT_KM else 'near'} {town.name}"
    return ", ".join(n for _, n in parts[:2])
