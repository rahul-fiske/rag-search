"""Map data: cells, the Overture store (filled by a stand-in for the bucket), lookups, the plan built from the indexed
documents, refresh, the task and the dashboard route.  No network."""

from __future__ import annotations

import os
from unittest import mock

import shapely

from tests.helpers import TempHome
from tests.portable.test_photo import fake_fetch
from tests.portable.test_ui import Dash

from rag_search import maps, model_tasks, reference
from rag_search.core import geo
from rag_search.paths import META_FILE

# Nirmal (Telangana): a district polygon, a locality inside it, India (dropped: countries come from the code)
DISTRICT = shapely.box(78.0, 18.8, 78.8, 19.6)
LOCALITY = shapely.box(78.30, 19.05, 78.40, 19.15)
AREAS = [
    {"oid": "d1", "name": "Nirmal", "subtype": "county", "country": "IN", "wkb": shapely.to_wkb(DISTRICT)},
    {"oid": "l1", "name": "Nirmal Town", "subtype": "locality", "country": "IN", "wkb": shapely.to_wkb(LOCALITY)},
    {"oid": "r1", "name": "Telangana", "subtype": "region", "country": "IN", "wkb": shapely.to_wkb(shapely.box(77, 16, 81, 20))},
    {"oid": "c1", "name": "India", "subtype": "country", "country": "IN", "wkb": shapely.to_wkb(shapely.box(68, 6, 97, 36))},
]
PLACES = [
    {"oid": "p1", "name": "Hotel Sai Krishna", "category": "restaurant", "confidence": 0.9, "lat": 19.0906, "lon": 78.3412},
    {"oid": "p2", "name": "HP Petrol Pump", "category": "gas_station", "confidence": 0.8, "lat": 19.0910, "lon": 78.3416},
    {"oid": "p3", "name": "Far Dhaba", "category": "restaurant", "confidence": 0.8, "lat": 19.1000, "lon": 78.3412},
    {"oid": "p4", "name": "Outside the cells", "category": "shop", "confidence": 0.5, "lat": 25.0, "lon": 78.3},
]


def fake_reader(calls=None, release_rows=None):
    def read(kind, box, release):
        if calls is not None:
            calls.append((kind, box, release))
        rows = (release_rows or {}).get(release, {})
        return rows.get(kind, PLACES if kind == "places" else AREAS)
    return read


class CellTests(TempHome):
    def test_cells_boxes_and_clusters(self):
        self.assertEqual(maps.cell_of(19.09, 78.34), "19.0,78.0")
        self.assertEqual(maps.cell_of(-0.1, -0.1), "-0.5,-0.5")
        self.assertEqual(maps.cell_box("19.0,78.0"), (78.0, 19.0, 78.5, 19.5))
        self.assertEqual(maps.cells_in_box(78.0, 19.0, 79.0, 19.5), {"19.0,78.0", "19.0,78.5"})
        groups = maps.clusters({"19.0,78.0", "19.5,78.5", "41.5,12.0"})
        self.assertEqual(sorted(len(g) for g in groups), [1, 2])          # corners touch: one fetch
        self.assertEqual(maps.group_box(["19.0,78.0", "19.5,78.5"]), (78.0, 19.0, 79.0, 20.0))
        with self.assertRaises(maps.MapsError):
            maps.cells_in_box(10, 10, 5, 20)
        with self.assertRaises(maps.MapsError):
            maps.cells_in_box(-180, -90, 180, 90)


class StoreTests(TempHome):
    def download(self, cells=("19.0,78.0",), release="2026-11-18.0", **kw):
        return maps.download_overture(self.paths, cells, release=release, reader=kw.pop("reader", fake_reader()), **kw)

    def test_download_keeps_only_the_cells_and_describes_a_position(self):
        steps = []
        res = self.download(progress=lambda d, t, what: steps.append((d, t)))
        self.assertEqual((res["cells"], res["places"], res["areas"]), (1, 3, 3))      # country outline and p4 dropped
        self.assertEqual(steps[-1], (2, 2))
        self.assertEqual(maps.installed_cells(self.paths, "places"), {"19.0,78.0": "2026-11-18.0"})
        self.assertEqual(maps.ref(self.paths, "places"), "overture-places@2026-11-18.0")
        self.assertEqual(geo.area(19.09, 78.34, self.paths), ["Nirmal Town", "Nirmal", "Telangana", "IN"])
        around = geo.nearby(19.0906, 78.3412, self.paths)
        self.assertEqual([n for _m, n, _c in around], ["Hotel Sai Krishna", "HP Petrol Pump"])
        lines, refs = geo.describe(19.0906, 78.3412, self.paths)
        self.assertEqual(lines[0], "Area: Nirmal Town, Nirmal, Telangana, IN")
        self.assertTrue(lines[-1].startswith("Nearby: Hotel Sai Krishna (restaurant, 0 m); HP Petrol Pump (gas station, "))
        self.assertEqual(refs, ["overture-divisions@2026-11-18.0", "overture-places@2026-11-18.0"])
        reference.download(self.paths, "geonames", fetch=fake_fetch)      # now the country has its name and a town
        lines, refs = geo.describe(19.0906, 78.3412, self.paths)
        self.assertEqual(lines[0], "Area: Nirmal Town, Nirmal, Telangana, India")
        self.assertTrue(lines[1].startswith("Town: at Nirmal, Telangana, India"))
        self.assertEqual(len(refs), 3)

    def test_the_nearest_place_to_eat_is_named_a_little_further_away(self):
        self.download()
        around = geo.nearby(19.0950, 78.3412, self.paths)                 # ~500 m from p1, ~550 m from p3: nothing
        self.assertEqual(around, [])
        around = geo.nearby(19.0990, 78.3412, self.paths)                 # p3 is 110 m away: further than 100, a dhaba
        self.assertEqual([n for _m, n, _c in around], ["Far Dhaba"])

    def test_outside_the_downloaded_cells_there_is_no_nearby_line(self):
        self.download()
        lines, _ = geo.describe(25.0, 78.3, self.paths)
        self.assertFalse(any(x.startswith("Nearby") for x in lines))
        lines, _ = geo.describe(19.4, 78.45, self.paths)                  # inside the cell, nothing around
        self.assertIn("Nearby: no places known within 100 m", lines)

    def test_a_new_release_refetches_every_cell_and_remove_deletes(self):
        calls = []
        self.download(cells=("19.0,78.0",), reader=fake_reader(calls))
        self.download(cells=("19.0,78.5",), reader=fake_reader(calls))   # same release: only the new cell
        self.assertEqual(len(calls), 4)
        newer = {"2026-12-16.0": {"places": PLACES[:1]}}
        res = self.download(cells=(), release="2026-12-16.0", reader=fake_reader(calls, newer))
        self.assertEqual(res["cells"], 2)
        self.assertEqual(set(maps.installed_cells(self.paths, "places").values()), {"2026-12-16.0"})
        self.assertEqual(geo.nearby(19.0906, 78.3412, self.paths)[0][1], "Hotel Sai Krishna")
        self.assertEqual(len(geo.nearby(19.0910, 78.3416, self.paths)), 1)           # the petrol pump is gone
        self.assertTrue(maps.remove(self.paths, "overture"))
        self.assertEqual(maps.installed_cells(self.paths, "places"), {})
        self.assertEqual(geo.area(19.09, 78.34, self.paths), [])
        self.assertFalse(maps.remove(self.paths, "geonames"))
        with self.assertRaises(maps.MapsError):
            maps.remove(self.paths, "faces")

    def test_status_lists_three_equal_rows(self):
        self.download()
        rows = {r["name"]: r for r in maps.status(self.paths)}
        self.assertEqual(list(rows), ["geonames", "overture-divisions", "overture-places"])
        self.assertFalse(rows["geonames"]["installed"])
        self.assertTrue(rows["overture-places"]["installed"])
        self.assertEqual(rows["overture-places"]["cells"], 1)
        self.assertGreater(rows["overture-places"]["bytes"], 0)

    def test_old_geonames_folder_is_renamed(self):
        reference.download(self.paths, "geonames", fetch=fake_fetch)
        os.replace(reference.folder(self.paths, "geonames"), reference.root(self.paths) / "places")
        self.assertTrue(reference.installed(self.paths, "geonames"))
        self.assertFalse((reference.root(self.paths) / "places").exists())


class PlanTests(TempHome):
    def write_md(self, rel, text):
        p = self.paths.markup / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
        p.with_name(p.name + ".sha256").write_text("x\nprofile\n", encoding="utf-8")
        idx = self.paths.index / rel[:-3]
        idx.mkdir(parents=True, exist_ok=True)
        (idx / META_FILE).write_text("{}", encoding="utf-8")
        return p

    def test_plan_counts_positions_and_missing_cells(self):
        self.write_md("photos/a.md", "## Photo details\n\n- Position: 19.09 N (geo:19.09056,78.34111)\n")
        self.write_md("trips/t.md", "VISIT geo:19.091000,78.341200 then geo:41.9,12.5 and geo:91.0,0.0\n")
        self.write_md("docs/plain.md", "no positions here\n")
        pl = maps.plan(self.paths)
        self.assertEqual((pl["positions"], pl["cells_needed"], pl["cells_missing"]), (2, 2, 2))   # a and t: one place
        self.assertEqual(pl["missing"], ["19.0,78.0", "41.5,12.5"])
        self.assertEqual(pl["fetches"], 2)
        maps.download_overture(self.paths, ["19.0,78.0"], release="r1", reader=fake_reader())
        self.assertEqual(maps.plan(self.paths)["missing"], ["41.5,12.5"])

    def test_refresh_marks_only_documents_whose_cells_changed(self):
        photo = self.write_md("photos/a.md", "## Photo details\n\n- Position: 19.09 N (geo:19.09056,78.34111)\n")
        trip = self.write_md("trips/t.md", "VISIT geo:41.900000,12.500000\n")
        plain = self.write_md("docs/plain.md", "no positions here\n")
        res = maps.refresh(self.paths)                       # no stamps yet: both are described again, once
        self.assertEqual((res["checked"], res["refreshed"], res["in_place"], res["reconverted"]), (2, 2, 1, 1))
        self.assertTrue(photo.with_name("a.md.sha256").exists())             # a photo keeps its Markdown
        self.assertFalse((self.paths.index / "photos/a" / META_FILE).exists())
        self.assertFalse(trip.with_name("t.md.sha256").exists())            # a timeline is converted again
        self.assertTrue(plain.with_name("plain.md.sha256").exists())
        self.assertTrue((self.paths.index / "docs/plain" / META_FILE).exists())
        for md in (photo, trip):                                             # the next run stamps them
            maps.write_stamp(self.paths, md, md.read_text(encoding="utf-8"))
        self.assertEqual(maps.refresh(self.paths)["refreshed"], 0)
        maps.download_overture(self.paths, ["19.0,78.0"], release="r1", reader=fake_reader())
        res = maps.refresh(self.paths)                       # only the photo's cell changed
        self.assertEqual((res["refreshed"], res["in_place"], res["current"]), (1, 1, 1))
        reference.download(self.paths, "geonames", fetch=fake_fetch)
        self.assertEqual(maps.refresh(self.paths)["refreshed"], 2)            # GeoNames describes every cell
        self.assertEqual(maps.refresh(self.paths, everything=True)["refreshed"], 2)


class TaskAndRouteTests(TempHome):
    def test_task_downloads_a_box_and_geonames(self):
        with mock.patch.object(maps, "overture_rows", fake_reader()), \
                mock.patch.object(maps, "latest_release", return_value="2026-11-18.0"):
            rec = model_tasks.run(self.paths, {"op": "maps", "action": "download", "dataset": "overture",
                                               "bbox": [78.0, 19.0, 78.5, 19.5]})
        self.assertEqual(rec["status"], "succeeded", rec.get("error"))
        self.assertEqual(rec["result"]["cells"], 1)
        with mock.patch.object(reference, "_fetch", fake_fetch):
            rec = model_tasks.run(self.paths, {"op": "maps", "action": "download", "dataset": "geonames"})
        self.assertEqual(rec["status"], "succeeded", rec.get("error"))
        rec = model_tasks.run(self.paths, {"op": "maps", "action": "download", "dataset": "overture"})
        self.assertIn("no cells", rec["result"]["note"])
        rec = model_tasks.run(self.paths, {"op": "maps", "action": "nonsense", "dataset": "overture"})
        self.assertEqual(rec["status"], "failed")

    def test_dashboard_route(self):
        dash = Dash(self.paths)
        try:
            status, js, _, _ = dash.req("GET", "/api/maps")
            self.assertEqual(status, 200)
            self.assertEqual([d["name"] for d in js["datasets"]], ["geonames", "overture-divisions", "overture-places"])
            self.assertEqual(js["plan"]["positions"], 0)
            status, js, _, _ = dash.req("POST", "/api/maps/download", {"dataset": "faces"})
            self.assertEqual(status, 400)
            status, js, _, _ = dash.req("POST", "/api/maps/download", {"dataset": "overture", "bbox": "1,2"})
            self.assertEqual(status, 400)
        finally:
            dash.close()

    def test_read_only_dashboard_cannot_download(self):
        dash = Dash(self.paths, read_only=True)
        try:
            status, _, _, _ = dash.req("POST", "/api/maps/download", {"dataset": "geonames"})
            self.assertEqual(status, 403)
        finally:
            dash.close()
