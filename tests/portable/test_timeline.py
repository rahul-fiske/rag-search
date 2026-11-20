"""Google Maps Timeline exports (three shapes) read as a day-by-day log, and indexed as such."""

from __future__ import annotations

import json
import os
from unittest import mock

from tests.helpers import FakeEmbedder, TempHome
from tests.portable.test_maps import fake_reader
from tests.portable.test_photo import fake_fetch

from rag_search import maps, reference
from rag_search.core import indexer, timeline

PHONE = [
    {"startTime": "2024-03-14T12:52:00.000+05:30", "endTime": "2024-03-14T13:40:00.000+05:30",
     "visit": {"probability": "0.91", "topCandidate": {"placeID": "ChIJabc", "semanticType": "Unknown",
                                                        "placeLocation": "geo:19.090600,78.341200"}}},
    {"startTime": "2024-03-14T13:40:00.000+05:30", "endTime": "2024-03-14T16:10:00.000+05:30",
     "activity": {"start": "geo:19.090600,78.341200", "end": "geo:17.380000,78.480000", "distanceMeters": "205000",
                  "topCandidate": {"type": "in passenger vehicle"}}},
    {"startTime": "2024-03-14T16:10:00.000+05:30", "endTime": "2024-03-14T17:10:00.000+05:30",
     "timelinePath": [{"point": "geo:17.38,78.48", "durationMinutesOffsetFromStartTime": "3"}]},
]
SEGMENTS = {"semanticSegments": [
    {"startTime": "2025-01-02T09:00:00.000+05:30", "endTime": "2025-01-02T10:00:00.000+05:30",
     "visit": {"probability": 0.8, "topCandidate": {"placeId": "ChIJxyz", "semanticType": "HOME",
                                                    "placeLocation": {"latLng": "18.5662790°, 73.8162730°"}}}},
    {"startTime": "2025-01-02T10:00:00.000+05:30", "endTime": "2025-01-02T10:30:00.000+05:30",
     "activity": {"start": {"latLng": "18.5662790°, 73.8162730°"}, "end": {"latLng": "18.6026480°, 73.7159440°"},
                  "distanceMeters": 12000.0, "topCandidate": {"type": "IN_PASSENGER_VEHICLE"}}}]}
TAKEOUT = {"timelineObjects": [
    {"placeVisit": {"location": {"latitudeE7": 190906000, "longitudeE7": 783412000, "placeId": "ChIJold",
                                 "name": "Hotel Sai Krishna", "address": "NH44, Nirmal"},
                    "duration": {"startTimestamp": "2019-06-01T07:20:00Z", "endTimestamp": "2019-06-01T08:00:00Z"},
                    "visitConfidence": 87}},
    {"activitySegment": {"startLocation": {"latitudeE7": 190906000, "longitudeE7": 783412000},
                         "endLocation": {"latitudeE7": 211500000, "longitudeE7": 790900000},
                         "duration": {"startTimestamp": "2019-06-01T08:00:00Z", "endTimestamp": "2019-06-01T12:00:00Z"},
                         "distance": 250000, "activityType": "IN_PASSENGER_VEHICLE"}}]}


class SniffTests(TempHome):
    def test_shapes(self):
        self.assertEqual(timeline.sniff(json.dumps(PHONE)), "phone")
        self.assertEqual(timeline.sniff(json.dumps(SEGMENTS)), "segments")
        self.assertEqual(timeline.sniff("﻿" + json.dumps(TAKEOUT)), "takeout")
        self.assertEqual(timeline.sniff('[{"id": 1, "startTime": "x"}]'), "")
        self.assertEqual(timeline.sniff('{"orders": []}'), "")

    def test_not_a_timeline_writes_nothing(self):
        src = self.tmp / "orders.json"
        src.write_text('[{"startTime": "x", "visit": 1}', encoding="utf-8")     # looks like one, is not JSON
        self.assertFalse(timeline.convert(src, self.tmp / "o.md", self.paths))
        self.assertFalse((self.tmp / "o.md").exists())


class MarkdownTests(TempHome):
    def setUp(self):
        super().setUp()
        reference.download(self.paths, "geonames", fetch=fake_fetch)
        maps.download_overture(self.paths, ["19.0,78.0"], release="r1", reader=fake_reader())

    def convert(self, data, name="Timeline.json"):
        src = self.tmp / name
        src.write_text(json.dumps(data), encoding="utf-8")
        md = self.tmp / "out.md"
        self.assertTrue(timeline.convert(src, md, self.paths))
        return md.read_text(encoding="utf-8")

    def test_phone_export(self):
        text = self.convert(PHONE)
        self.assertIn("1 visits and 1 journeys, 14 Mar 2024 to 14 Mar 2024; 1 raw GPS traces not listed", text)
        self.assertIn("## 2024-03-14 Thursday", text)
        self.assertIn("- VISIT 12:52-13:40 +05:30 (48 min) · geo:19.090600,78.341200 · Google place ID ChIJabc "
                      "· probability 0.91", text)
        self.assertIn("  - Area: Nirmal Town, Nirmal, Telangana, India", text)
        self.assertIn("  - Town: at Nirmal, Telangana, India", text)
        self.assertIn("  - Nearby: Hotel Sai Krishna (restaurant, 0 m)", text)
        self.assertIn("- TRAVEL 13:40-16:10 +05:30 (150 min) · by car 205.0 km · from geo:19.090600,78.341200 "
                      "(Nirmal Town, Nirmal) to geo:17.380000,78.480000 (at Hyderabad)", text)
        self.assertNotIn("type: unknown", text)

    def test_settings_export(self):
        text = self.convert(SEGMENTS)
        self.assertIn("- VISIT 09:00-10:00 +05:30 (60 min) · geo:18.566279,73.816273 · Google place ID ChIJxyz "
                      "· type: home · probability 0.80", text)
        self.assertIn("· by car 12.0 km · from geo:18.566279,73.816273", text)

    def test_takeout_export_keeps_googles_name_and_utc(self):
        text = self.convert(TAKEOUT)
        self.assertIn("## 2019-06-01 Saturday", text)
        self.assertIn("- VISIT 07:20-08:00 +00:00 (40 min) · Hotel Sai Krishna, NH44, Nirmal · geo:19.090600,78.341200 "
                      "· Google place ID ChIJold", text)
        self.assertIn("to geo:21.150000,79.090000 (at Nagpur)", text)


class IndexedTimelineTests(TempHome):
    def setUp(self):
        super().setUp()
        self.use_fake_backends()
        os.environ["RAG_SEARCH_ROUTING"] = "document"

    def test_a_timeline_is_indexed_and_other_json_is_not_treated_as_one(self):
        (self.sdir / "maps").mkdir()
        (self.sdir / "maps" / "Timeline.json").write_text(json.dumps(PHONE), encoding="utf-8")
        (self.sdir / "maps" / "orders.json").write_text(json.dumps([{"id": "A7", "total": 12}]), encoding="utf-8")
        srcs = indexer.scan_sources(self.sdir, indexer.exclude_dirs(self.paths))
        with mock.patch("rag_search.core.docling_convert.convert_file",
                        wraps=__import__("rag_search.core.docling_convert", fromlist=["x"]).convert_file) as conv:
            indexer.run_index(self.paths, srcs, self.roots(), jobs=1, embedder=FakeEmbedder(), progress=lambda e: None)
        converted = [c.args[0].name for c in conv.call_args_list]
        self.assertEqual(converted, ["orders.json"])                          # the timeline never reached it
        text = next(self.paths.markup.rglob("Timeline.md")).read_text(encoding="utf-8")
        self.assertIn("- VISIT 12:52-13:40 +05:30", text)
        self.assertIn("geo:19.090600,78.341200", text)
        self.assertEqual(maps.plan(self.paths)["cells_needed"], 2)             # Nirmal and Hyderabad
        md = next(self.paths.markup.rglob("Timeline.md"))
        self.assertTrue(maps.stamp_path(md).exists())
        self.assertEqual(maps.refresh(self.paths)["refreshed"], 0)
        maps.download_overture(self.paths, ["19.0,78.0"], release="r1", reader=fake_reader())
        self.assertEqual(maps.refresh(self.paths)["reconverted"], 1)
        indexer.run_index(self.paths, srcs, self.roots(), jobs=1, embedder=FakeEmbedder(), progress=lambda e: None)
        self.assertIn("  - Nearby: Hotel Sai Krishna", md.read_text(encoding="utf-8"))
        self.assertEqual(maps.refresh(self.paths)["refreshed"], 0)
