"""The place a GPS position is near, from the GeoNames reference dataset (see ``reference.py``).

``near(lat, lon, paths)`` returns the closest town or village of 1,000 people or more with its state and country,
or None when the dataset is not installed or nothing is within ``MAX_KM``.  The table is read once per process.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from .. import reference
from ..paths import Paths

MAX_KM = 50.0
AT_KM = 1.0                      # closer than this: "at", not "near"
EARTH_KM = 6371.0
_COMPASS = ("N", "NE", "E", "SE", "S", "SW", "W", "NW")


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


_CACHE: dict[str, Any] = {}


def _load(paths: Paths) -> dict[str, Any] | None:
    key = reference.ref(paths, "places")
    if not key:
        return None
    if _CACHE.get("key") == key:
        return _CACHE["data"]
    import numpy as np

    cities = reference.path(paths, "places", "cities1000.txt")
    if cities is None:
        return None
    admin: dict[str, str] = {}
    p = reference.path(paths, "places", "admin1CodesASCII.txt")
    if p:
        for line in p.read_text(encoding="utf-8").splitlines():
            parts = line.split("\t")
            if len(parts) >= 2:
                admin[parts[0]] = parts[1]
    countries: dict[str, str] = {}
    p = reference.path(paths, "places", "countryInfo.txt")
    if p:
        for line in p.read_text(encoding="utf-8").splitlines():
            parts = line.split("\t")
            if len(parts) > 4 and not line.startswith("#"):
                countries[parts[0]] = parts[4]
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
    data = {"names": names, "admin1": a1, "cc": cc, "countries": countries,
            "lat": np.radians(np.array(lat)), "lon": np.radians(np.array(lon))}
    _CACHE.update(key=key, data=data)
    return data


def _bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    dl = lon2 - lon1
    y = math.sin(dl) * math.cos(lat2)
    x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dl)
    return (math.degrees(math.atan2(y, x)) + 360) % 360


def near(lat: float, lon: float, paths: Paths) -> Place | None:
    data = _load(paths)
    if not data or not data["names"] or not (-90 <= lat <= 90 and -180 <= lon <= 180):
        return None
    import numpy as np

    la, lo = math.radians(lat), math.radians(lon)
    dlat, dlon = data["lat"] - la, data["lon"] - lo
    h = np.sin(dlat / 2) ** 2 + math.cos(la) * np.cos(data["lat"]) * np.sin(dlon / 2) ** 2
    i = int(np.argmin(h))
    km = 2 * EARTH_KM * math.asin(min(1.0, math.sqrt(float(h[i]))))
    if km > MAX_KM:
        return None
    # the direction of the photograph as seen from the place
    d = _COMPASS[int(((_bearing(float(data["lat"][i]), float(data["lon"][i]), la, lo) + 22.5) % 360) // 45)]
    cc = data["cc"][i]
    return Place(data["names"][i], data["admin1"][i], data["countries"].get(cc, cc), km, d)
