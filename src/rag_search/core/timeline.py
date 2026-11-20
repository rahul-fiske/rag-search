"""A Google Maps Timeline export, read as a day-by-day log of visits and journeys.

A ``.json`` file is looked at before it is converted: when it is one of the three shapes Google has exported, it is
written as Markdown here instead of as generic JSON (``docling_convert.json_to_markdown``):

    phone      the Timeline export from a phone (2024 on): a list of entries with ``startTime`` / ``endTime`` and a
               ``visit``, ``activity`` or ``timelinePath``; positions as ``geo:LAT,LON``
    segments   the same from the newer settings page: ``{"semanticSegments": [...]}``; positions as ``"18.5°, 73.8°"``
    takeout    the older Google Takeout file: ``{"timelineObjects": [...]}`` with ``placeVisit`` / ``activitySegment``;
               positions as integers (degrees x 10^7), sometimes with the place's name and address

Each visit and journey becomes one line with its times (and the offset they were recorded in), its positions as
``geo:LAT,LON``, the Google place ID as it is (an agent that can ask Google Maps resolves it; rag-search does not), and
what the installed map data says about the place (``geo.describe``).  Grouping journeys into trips, meal times and the
like is left to whoever reads it.  Raw GPS traces (``timelinePath``) are counted, not listed.
"""

from __future__ import annotations

import json
import os
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from ..paths import Paths
from . import geo

SNIFF_BYTES = 256 * 1024
_NUM = re.compile(r"-?\d+(?:\.\d+)?")


def sniff(head: str) -> str:
    """The Timeline shape the start of a file looks like, or ""."""
    h = head.lstrip("﻿ \t\r\n")
    if '"semanticSegments"' in h:
        return "segments"
    if '"timelineObjects"' in h:
        return "takeout"
    if h.startswith("[") and '"startTime"' in h and any(k in h for k in ('"visit"', '"activity"', '"timelinePath"')):
        return "phone"
    return ""


def _geo_str(v: Any) -> tuple[float, float] | None:
    """``geo:18.5,73.8``, ``18.5°, 73.8°`` or ``{"latLng": ...}`` as (lat, lon)."""
    if isinstance(v, dict):
        v = v.get("latLng") or v.get("placeLocation") or ""
    nums = _NUM.findall(str(v or ""))
    if len(nums) < 2:
        return None
    lat, lon = float(nums[0]), float(nums[1])
    return (lat, lon) if -90 <= lat <= 90 and -180 <= lon <= 180 else None


def _e7(loc: dict[str, Any] | None) -> tuple[float, float] | None:
    if not loc or loc.get("latitudeE7") is None or loc.get("longitudeE7") is None:
        return None
    return int(loc["latitudeE7"]) / 1e7, int(loc["longitudeE7"]) / 1e7


def _num(v: Any) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def entries(data: Any, shape: str) -> tuple[list[dict[str, Any]], int]:
    """Visits and journeys in one form, and how many raw trace segments were left out."""
    out: list[dict[str, Any]] = []
    traces = 0
    if shape in ("phone", "segments"):
        items = data if shape == "phone" else data.get("semanticSegments", [])
        for it in items:
            if not isinstance(it, dict):
                continue
            start, end = it.get("startTime", ""), it.get("endTime", "")
            if "visit" in it:
                v, top = it["visit"], (it["visit"].get("topCandidate") or {})
                out.append({"kind": "visit", "start": start, "end": end, "at": _geo_str(top.get("placeLocation")),
                            "place_id": top.get("placeID") or top.get("placeId") or "",
                            "type": top.get("semanticType") or "", "probability": _num(v.get("probability"))})
            elif "activity" in it:
                a, top = it["activity"], (it["activity"].get("topCandidate") or {})
                out.append({"kind": "travel", "start": start, "end": end, "from": _geo_str(a.get("start")),
                            "to": _geo_str(a.get("end")), "mode": top.get("type") or "",
                            "metres": _num(a.get("distanceMeters"))})
            elif "timelinePath" in it:
                traces += 1
    else:
        for it in data.get("timelineObjects", []):
            if "placeVisit" in it:
                pv = it["placeVisit"]
                loc, dur = pv.get("location") or {}, pv.get("duration") or {}
                out.append({"kind": "visit", "start": dur.get("startTimestamp", ""), "end": dur.get("endTimestamp", ""),
                            "at": _e7(loc), "place_id": loc.get("placeId") or "", "type": loc.get("semanticType") or "",
                            "probability": _num(pv.get("visitConfidence")), "name": loc.get("name") or "",
                            "address": loc.get("address") or ""})
            elif "activitySegment" in it:
                seg, dur = it["activitySegment"], it["activitySegment"].get("duration") or {}
                out.append({"kind": "travel", "start": dur.get("startTimestamp", ""), "end": dur.get("endTimestamp", ""),
                            "from": _e7(seg.get("startLocation")), "to": _e7(seg.get("endLocation")),
                            "mode": seg.get("activityType") or "", "metres": _num(seg.get("distance"))})
    return out, traces


def _when(s: str) -> datetime | None:
    try:
        return datetime.fromisoformat(str(s).replace("Z", "+00:00"))
    except ValueError:
        return None


def _offset(d: datetime) -> str:
    off = d.utcoffset()
    if off is None:
        return ""
    mins = int(off.total_seconds() // 60)
    return f"{'+' if mins >= 0 else '-'}{abs(mins) // 60:02d}:{abs(mins) % 60:02d}"


def _mode(m: str) -> str:
    return m.replace("_", " ").lower().replace("in passenger vehicle", "by car").strip() or "moving"


def _pos(p: tuple[float, float]) -> str:
    return f"geo:{p[0]:.6f},{p[1]:.6f}"


def to_markdown(data: Any, shape: str, paths: Paths, *, source: str = "") -> str:
    items, traces = entries(data, shape)
    memo: dict[tuple[float, float], tuple[list[str], str]] = {}

    def about(p: tuple[float, float]) -> tuple[list[str], str]:
        k = (round(p[0], 4), round(p[1], 4))
        if k not in memo:
            memo[k] = (geo.describe(p[0], p[1], paths)[0], geo.short(p[0], p[1], paths))
        return memo[k]

    days: dict[str, list[str]] = defaultdict(list)
    first = last = None
    visits = journeys = 0
    for e in items:
        s, t = _when(e["start"]), _when(e["end"])
        if s is None:
            continue
        first, last = min(first or s, s), max(last or s, s)
        mins = round((t - s).total_seconds() / 60) if t else None
        span = f"{s:%H:%M}-{t:%H:%M}" if t else f"{s:%H:%M}"
        span += f" {_offset(s)}" if _offset(s) else ""
        span += f" ({mins} min)" if mins is not None else ""
        key = s.strftime("%Y-%m-%d %A")
        if e["kind"] == "visit":
            visits += 1
            bits = [f"- VISIT {span}"]
            if e.get("name"):
                bits.append(f"{e['name']}" + (f", {e['address']}" if e.get("address") else ""))
            if e.get("at"):
                bits.append(_pos(e["at"]))
            if e.get("place_id"):
                bits.append(f"Google place ID {e['place_id']}")
            if e.get("type") and e["type"].lower() not in ("unknown", "type_unknown"):
                bits.append(f"type: {e['type'].replace('_', ' ').lower()}")
            if e.get("probability") is not None:
                bits.append(f"probability {e['probability']:.2f}")
            lines = [" · ".join(bits)]
            if e.get("at"):
                lines += [f"  - {x}" for x in about(e["at"])[0]]
            days[key].extend(lines)
        else:
            journeys += 1
            dist = f" {e['metres'] / 1000:.1f} km" if e.get("metres") else ""
            ends = []
            for word, p in (("from", e.get("from")), ("to", e.get("to"))):
                if p:
                    near = about(p)[1]
                    ends.append(f"{word} {_pos(p)}" + (f" ({near})" if near else ""))
            days[key].append(f"- TRAVEL {span} · {_mode(e['mode'])}{dist}" + (" · " + " ".join(ends) if ends else ""))
    head = [f"# Google Maps Timeline{f' ({source})' if source else ''}", "",
            f"{visits} visits and {journeys} journeys"
            + (f", {first:%d %b %Y} to {last:%d %b %Y}" if first else "")
            + (f"; {traces} raw GPS traces not listed" if traces else "") + ".",
            "Positions are geo:LAT,LON. Areas, towns and nearby places come from the map data installed when this was "
            "indexed and are candidates, not certainties; a Google place ID can be looked up in Google Maps."]
    body = []
    for day in sorted(days):
        body += ["", f"## {day}", ""] + days[day]
    return "\n".join(head + body) + "\n"


def convert(src: Path, md_path: Path, paths: Paths) -> bool:
    """Write the Markdown of *src* when it is a Google Timeline export; False (nothing written) when it is not."""
    with open(src, encoding="utf-8", errors="replace") as fh:
        shape = sniff(fh.read(SNIFF_BYTES))
    if not shape:
        return False
    try:
        data = json.loads(src.read_text(encoding="utf-8-sig", errors="replace"))
    except ValueError:
        return False
    if (shape == "phone") != isinstance(data, list):
        return False
    md = to_markdown(data, shape, paths, source=shape)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = md_path.with_name(md_path.name + ".partial")
    tmp.write_text("<!-- page 1 -->\n\n" + md, encoding="utf-8")
    os.replace(tmp, md_path)
    return True
