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
    def test_cells_boxes_and_fetch_groups(self):
        self.assertEqual(maps.cell_of(19.09, 78.34), "19.05,78.30")
        self.assertEqual(maps.cell_of(19.10, 78.34), "19.10,78.30")                # the corner belongs to the cell north-east of it
        self.assertEqual(maps.cell_of(-0.01, -0.01), "-0.05,-0.05")
        self.assertEqual(maps.cell_box("19.05,78.30"), (78.30, 19.05, 78.35, 19.10))
        self.assertEqual(len(maps.cells_in_box(78.0, 19.0, 78.5, 19.5)), 100)      # a 0.5 degree square: 10 x 10 cells
        self.assertEqual(maps.cell_no(*maps.parse("19.05,78.30")), (381 + 1800) * 10_000 + (1566 + 3600))
        # cells of squares that touch are fetched together, others apart; the box is only as big as the cells
        near = ["19.05,78.30", "19.55,78.80", "19.05,78.55"]                       # three touching 0.5 squares (corner, side)
        far = ["41.90,12.50"]
        groups = maps.clusters(near + far)
        self.assertEqual(sorted(len(g) for g in groups), [1, 3])
        self.assertEqual(maps.group_box(near), (78.30, 19.05, 78.85, 19.60))
        with self.assertRaises(maps.MapsError):
            maps.cells_in_box(10, 10, 5, 20)
        with self.assertRaises(maps.MapsError):
            maps.cells_in_box(68, 6, 97, 36)                                       # India: far too many cells


class StoreTests(TempHome):
    def download(self, cells=("19.05,78.30", "19.10,78.30"), release="2026-11-18.0", **kw):
        return maps.download_overture(self.paths, cells, release=release, reader=kw.pop("reader", fake_reader()), **kw)

    def test_download_keeps_only_the_cells_and_describes_a_position(self):
        steps = []
        res = self.download(progress=lambda d, t, what: steps.append((d, t)))
        self.assertEqual((res["cells"], res["places"], res["areas"]), (2, 3, 3))      # country outline and p4 dropped
        self.assertEqual(steps[-1], (2, 2))
        self.assertEqual(maps.installed_cells(self.paths, "places"),
                         {"19.05,78.30": "2026-11-18.0", "19.10,78.30": "2026-11-18.0"})
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
        around = geo.nearby(19.0990, 78.3412, self.paths)                 # p3 is 110 m away (in the next cell north)
        self.assertEqual([n for _m, n, _c in around], ["Far Dhaba"])      # further than 100 m, but the nearest dhaba

    def test_outside_the_downloaded_cells_there_is_no_nearby_line(self):
        self.download()
        lines, _ = geo.describe(25.0, 78.3, self.paths)
        self.assertFalse(any(x.startswith("Nearby") for x in lines))
        lines, _ = geo.describe(19.1400, 78.3100, self.paths)             # a downloaded cell with nothing around says so
        self.assertIn("Nearby: no places known within 100 m", lines)
        lines, _ = geo.describe(19.3000, 78.3100, self.paths)             # a cell that was not downloaded says nothing
        self.assertFalse(any(x.startswith("Nearby") for x in lines))

    def test_a_new_release_refetches_every_cell_and_remove_deletes(self):
        calls = []
        self.download(cells=("19.05,78.30",), reader=fake_reader(calls))
        self.download(cells=("19.05,78.50",), reader=fake_reader(calls))   # same release: only the new cell
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
        self.assertEqual(rows["overture-places"]["cells"], 2)
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
        self.write_md("trips/t.md", "VISIT geo:19.091000,78.341200 then geo:41.93,12.53 and geo:91.0,0.0\n")
        self.write_md("docs/plain.md", "no positions here\n")
        pl = maps.plan(self.paths)
        self.assertEqual((pl["positions"], pl["cells_needed"], pl["cells_missing"]), (2, 2, 2))   # a and t: one place
        self.assertEqual(pl["missing"], ["19.05,78.30", "41.90,12.50"])
        self.assertEqual((pl["fetches"], pl["cell_km"]), (2, 6))
        maps.download_overture(self.paths, ["19.05,78.30"], release="r1", reader=fake_reader())
        self.assertEqual(maps.plan(self.paths)["missing"], ["41.90,12.50"])

    def test_a_position_near_an_edge_needs_the_neighbouring_cell_too(self):
        # 19.0996 is 40 m south of the 19.10 edge: a place 60 m away across the edge belongs to the cell above
        self.assertEqual(maps.cells_around(19.0996, 78.3412), {"19.05,78.30", "19.10,78.30"})
        self.assertEqual(maps.cells_around(19.0906, 78.3412), {"19.05,78.30"})
        self.assertEqual(len(maps.cells_around(19.0999, 78.3499)), 4)                  # a corner
        self.write_md("photos/e.md", "- Position: (geo:19.099600,78.341200)\n")
        pl = maps.plan(self.paths)
        self.assertEqual((pl["cells_needed"], pl["missing"]), (2, ["19.05,78.30", "19.10,78.30"]))
        across = [{"name": "Across the edge", "category": "restaurant", "confidence": 0.9, "lat": 19.1002, "lon": 78.3412}]
        only_south = lambda kind, box, release: across if kind == "places" else AREAS    # noqa: E731
        maps.download_overture(self.paths, ["19.05,78.30"], release="r1", reader=only_south)
        self.assertFalse(geo.covered(19.0996, 78.3412, self.paths, "places"))            # the neighbour is missing
        maps.download_overture(self.paths, ["19.10,78.30"], release="r1", reader=only_south)
        self.assertTrue(geo.covered(19.0996, 78.3412, self.paths, "places"))
        self.assertEqual([(n, round(m)) for m, n, _c in geo.nearby(19.0996, 78.3412, self.paths)], [("Across the edge", 67)])
        self.assertEqual(maps.plan(self.paths)["cells_missing"], 0)

    def test_refresh_marks_only_documents_whose_cells_changed(self):
        photo = self.write_md("photos/a.md", "## Photo details\n\n- Position: 19.09 N (geo:19.09056,78.34111)\n")
        trip = self.write_md("trips/t.md", "VISIT geo:41.930000,12.530000\n")
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
        maps.download_overture(self.paths, ["19.05,78.30"], release="r1", reader=fake_reader())
        res = maps.refresh(self.paths)                       # only the photo's cell changed
        self.assertEqual((res["refreshed"], res["in_place"], res["current"]), (1, 1, 1))
        reference.download(self.paths, "geonames", fetch=fake_fetch)
        self.assertEqual(maps.refresh(self.paths)["refreshed"], 2)            # GeoNames describes every cell
        self.assertEqual(maps.refresh(self.paths, everything=True)["refreshed"], 2)


class MigrationTests(PlanTests):
    """A store made with 4.0.0.dev3's 0.5 degree cells becomes small cells without a download."""

    def old_store(self):
        import sqlite3
        self.old_places = [("Hotel Sai Krishna", "restaurant", 0.9, 19.0906, 78.3412),
                           ("HP Petrol Pump", "gas_station", 0.8, 19.0910, 78.3416),
                           ("Far away in the same big cell", "shop", 0.5, 19.4000, 78.1000),
                           ("Dhaba outside a needed cell", "restaurant", 0.7, 19.2000, 78.4000)]
        maps.store_dir(self.paths).mkdir(parents=True, exist_ok=True)
        con = sqlite3.connect(maps.db_path(self.paths, "places"))
        con.executescript("""create table meta(key text primary key, value text);
            create table cell(cell text primary key, rows integer, release text, at text);
            create table place(id integer primary key, oid text, name text, category text, confidence real,
                               lat real, lon real, cell text);
            create virtual table place_rt using rtree(id, minx, maxx, miny, maxy);""")
        for k, (name, cat, conf, la, lo) in enumerate(self.old_places):
            con.execute("insert into place values (?,?,?,?,?,?,?,?)", (k, f"id{k}", name, cat, conf, la, lo, "19.0,78.0"))
        con.execute("insert into cell values ('19.0,78.0', 4, 'r0', '2026-11-20 10:00')")
        con.execute("insert into meta values ('release', 'r0')")
        con.commit()
        con.close()
        con = sqlite3.connect(maps.db_path(self.paths, "divisions"))
        con.executescript("""create table meta(key text primary key, value text);
            create table cell(cell text primary key, rows integer, release text, at text);
            create table area(id integer primary key, oid text unique, name text, subtype text, country text, wkb blob);
            create virtual table area_rt using rtree(id, minx, maxx, miny, maxy);""")
        for k, a in enumerate(AREAS[:3]):
            con.execute("insert into area values (?,?,?,?,?,?)", (k, a["oid"], a["name"], a["subtype"], a["country"], a["wkb"]))
            x0, y0, x1, y1 = shapely.from_wkb(a["wkb"]).bounds
            con.execute("insert into area_rt values (?,?,?,?,?)", (k, x0, x1, y0, y1))
        con.execute("insert into cell values ('19.0,78.0', 0, 'r0', '2026-11-20 10:00')")
        con.execute("insert into meta values ('release', 'r0')")
        con.commit()
        con.close()

    def test_old_store_counts_as_empty_and_is_converted_keeping_only_needed_cells(self):
        self.write_md("photos/a.md", "- Position: (geo:19.09056,78.34111)\n")        # needs 19.05,78.30
        self.write_md("trips/t.md", "VISIT geo:19.140000,78.310000\n")                # needs 19.10,78.30 (nothing in it)
        self.write_md("far/f.md", "VISIT geo:41.930000,12.530000\n")                  # never downloaded
        self.old_store()
        self.assertTrue(maps.is_old_grid(self.paths, "places"))
        self.assertEqual(maps.installed_cells(self.paths, "places"), {})
        self.assertTrue(maps.status(self.paths)[2]["needs_convert"])
        res = maps.migrate(self.paths)
        self.assertEqual((res["cells"], res["places"]), (2, 2))                        # the two hotel/pump rows; the rest dropped
        self.assertEqual(maps.installed_cells(self.paths, "places"),
                         {"19.05,78.30": "r0", "19.10,78.30": "r0"})
        self.assertEqual(maps.installed_cells(self.paths, "divisions"), {"19.05,78.30": "r0", "19.10,78.30": "r0"})
        self.assertFalse(maps.is_old_grid(self.paths, "places"))
        self.assertIsNone(maps.migrate(self.paths))                                    # once
        self.assertEqual(maps.plan(self.paths)["missing"], ["41.90,12.50"])
        around = geo.nearby(19.0906, 78.3412, self.paths)
        self.assertEqual([(n, c) for _m, n, c in around], [("Hotel Sai Krishna", "restaurant"), ("HP Petrol Pump", "gas_station")])
        self.assertEqual(geo.area(19.09, 78.34, self.paths), ["Nirmal Town", "Nirmal", "Telangana", "IN"])
        self.assertEqual(geo.nearby(19.4000, 78.1000, self.paths), [])                 # its cell was not needed: not kept

    def test_new_store_has_the_compact_columns(self):
        import sqlite3
        maps.download_overture(self.paths, ["19.05,78.30"], release="r1", reader=fake_reader())
        con = sqlite3.connect(maps.db_path(self.paths, "places"))
        self.assertEqual([r[1] for r in con.execute("pragma table_info(place)")], ["cell", "lat", "lon", "name", "cat", "conf"])
        self.assertEqual(con.execute("select count(*) from place").fetchone()[0], 2)
        self.assertEqual(con.execute("select lat, lon, conf from place where name = 'Hotel Sai Krishna'").fetchone(),
                         (1909060, 7834120, 90))
        self.assertEqual(sorted(r[0] for r in con.execute("select name from cat")), ["gas_station", "restaurant"])
        self.assertEqual(dict(con.execute("select key, value from meta"))["grid"], "0.05")


class TaskAndRouteTests(TempHome):
    def test_task_downloads_a_box_and_geonames(self):
        with mock.patch.object(maps, "overture_rows", fake_reader()), \
                mock.patch.object(maps, "latest_release", return_value="2026-11-18.0"):
            rec = model_tasks.run(self.paths, {"op": "maps", "action": "download", "dataset": "overture",
                                               "bbox": [78.30, 19.05, 78.35, 19.10]})
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
