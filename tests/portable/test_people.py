"""People in photographs: faces with fingerprints, names and groups kept apart from the index, who each face is worked out
from your labels, and a search that names people keeping to their photographs.  No face model is loaded: the faces and
their fingerprints are made up (two people, each a direction in 128 dimensions)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest import mock

import numpy as np

from tests.helpers import FakeEmbedder, TempHome
from tests.portable.test_search_daemon import DaemonCase

from rag_search import client, protocol
from rag_search.core import faces, people
from rag_search.core.conversion import runview


def unit(v):
    v = np.asarray(v, dtype=np.float32)
    return v / np.linalg.norm(v)


RNG = np.random.default_rng(7)
A, B, C = (unit(RNG.normal(size=128)) for _ in range(3))     # three people's "true" faces: nearly orthogonal


def like(base, noise=0.25):
    return unit(base + noise * unit(RNG.normal(size=128)))


def face(vec, px=120, score=0.93):
    return {"x": 0.1, "y": 0.1, "w": 0.2, "h": 0.2, "score": score, "px": px, "vec": vec, "thumb": b"jpg"}


class PeopleCase(TempHome):
    def setUp(self):
        super().setUp()
        self.st = faces.Store(self.paths)
        self.addCleanup(self.st.close)
        people._CACHE.clear()

    def photo(self, sha, *vecs, src=None, **kw):
        self.st.add_photo(sha, Path(src or f"/p/{sha}.jpg"), [face(v, **kw) for v in vecs])
        return [r[0] for r in self.st.con.execute("SELECT id FROM faces WHERE sha = ? ORDER BY id", (sha,))]


class WhoTests(PeopleCase):
    def test_a_face_takes_the_person_of_the_closest_labelled_face_when_it_is_clearly_theirs(self):
        (a1,) = self.photo("s1", like(A))
        (b1,) = self.photo("s2", like(B))
        a2, b2 = self.photo("s3", like(A), like(B))
        (small,) = self.photo("s4", like(A), px=20)
        (c1,) = self.photo("s5", like(C))
        avni, sam = self.st.person_id("Avni", create=True), self.st.person_id("Samidha", create=True)
        self.st.label([a1], avni)
        self.st.label([b1], sam)
        w = people.work_out(self.st)
        self.assertEqual(w.person[a1], (avni, "you", 1.0))
        self.assertEqual((w.person[a2][:2], w.person[b2][:2]), ((avni, "auto"), (sam, "auto")))
        self.assertNotIn(small, w.person)                     # too small to name automatically
        self.assertNotIn(c1, w.person)                        # somebody else
        self.assertEqual(w.photos_of(avni), {"s1", "s3"})
        self.assertEqual(w.people_in("s3"), ["Avni", "Samidha"])
        self.assertIn(c1, w.unknown)
        self.assertNotIn(small, w.unknown)

    def test_no_means_not_that_person_and_a_weaker_match_is_only_a_suggestion(self):
        (a1,) = self.photo("s1", like(A))
        (a2,) = self.photo("s2", like(A))
        (half,) = self.photo("s3", unit(A * 0.36 + C * 0.93))  # somewhat like A
        avni = self.st.person_id("Avni", create=True)
        self.st.label([a1], avni)
        self.st.reject(a2, avni)
        w = people.work_out(self.st)
        self.assertNotIn(a2, w.person)
        self.assertEqual(w.suggest[half][0], avni)
        self.assertNotIn(half, w.person)

    def test_look_alike_faces_nobody_is_matched_to_are_grouped(self):
        ids = self.photo("s1", like(A, .1), like(B, .1)) + self.photo("s2", like(A, .1), like(B, .1)) + self.photo("s3", like(C, .1))
        w = people.work_out(self.st)
        groups = sorted(w.clusters(), key=lambda g: g[0])
        self.assertEqual(groups, [[ids[0], ids[2]], [ids[1], ids[3]], [ids[4]]])

    def test_who_is_worked_out_again_only_when_the_store_changes(self):
        (a1,) = self.photo("s1", like(A))
        first = people.who(self.paths)
        self.assertIs(people.who(self.paths), first)
        self.st.label([a1], self.st.person_id("Avni", create=True))
        self.assertIsNot(people.who(self.paths), first)


class NamesTests(PeopleCase):
    def test_renaming_is_one_row_and_a_taken_name_merges_on_request(self):
        (a1,) = self.photo("s1", like(A))
        (a2,) = self.photo("s2", like(A))
        x, y = self.st.person_id("Avni", create=True), self.st.person_id("Avni G", create=True)
        self.st.label([a1], x)
        self.st.label([a2], y)
        g = self.st.group_id("family", create=True)
        self.st.set_members(g, [y])
        self.assertEqual(self.st.rename(x, "Avni Fiske"), {"renamed": x})
        self.assertEqual(self.st.rename(y, "avni fiske"), {"merge_needed": True, "into": x})
        self.assertEqual(self.st.rename(y, "Avni Fiske", merge=True), {"merged_into": x})
        self.assertEqual(self.st.labels()[0], {a1: x, a2: x})
        self.assertEqual(self.st.groups()[0]["members"], [x])
        self.assertEqual([p["name"] for p in self.st.people()], ["Avni Fiske"])

    def test_a_word_names_one_person_or_group_only(self):
        me = self.st.person_id("Rahul", create=True)
        self.assertEqual(self.st.set_aliases(me, ["me", " papa ", "me"]), ["me", "papa"])
        sam = self.st.person_id("Samidha", create=True)
        with self.assertRaises(faces.StoreError):
            self.st.set_aliases(sam, ["Papa"])
        with self.assertRaises(faces.StoreError):
            self.st.group_id("rahul", create=True)
        with self.assertRaises(faces.StoreError):
            self.st.person_id("a, b", create=True)

    def test_a_face_is_one_person(self):
        (a1,) = self.photo("s1", like(A))
        x, y = self.st.person_id("X", create=True), self.st.person_id("Y", create=True)
        self.st.label([a1], x)
        self.st.label([a1], y)
        self.assertEqual(self.st.labels()[0], {a1: y})


class ResolveTests(PeopleCase):
    def setUp(self):
        super().setUp()
        f = {}
        f["me1"], = self.photo("only-me", like(A))
        f["me2"], f["sam2"] = self.photo("me-sam", like(A), like(B))
        f["avni3"], = self.photo("avni", like(C))
        f["me4"], f["avni4"] = self.photo("me-avni", like(A), like(C))
        self.me, self.sam, self.avni = (self.st.person_id(n, create=True) for n in ("Rahul", "Samidha", "Avni"))
        self.st.set_aliases(self.me, ["me"])
        for k, p in (("me1", self.me), ("sam2", self.sam), ("avni3", self.avni)):
            self.st.label([f[k]], p)
        fam = self.st.group_id("family", create=True)
        self.st.set_members(fam, [self.me, self.sam, self.avni])

    def test_names_become_the_photographs_that_have_all_of_them_and_the_rest_is_searched(self):
        r = people.resolve(self.paths, "photos of me and Samidha")
        self.assertEqual((r["people"], r["photos"], r["rest"]), (["Rahul", "Samidha"], {"/p/me-sam.jpg"}, ""))
        r = people.resolve(self.paths, "Avni's eating ice cream in the jungle")
        self.assertEqual((r["photos"], r["rest"]), ({"/p/avni.jpg", "/p/me-avni.jpg"}, "eating ice cream in the jungle"))

    def test_show_me_asks_and_a_group_needs_another_member(self):
        r = people.resolve(self.paths, "show me photos of Avni")
        self.assertEqual(r["people"], ["Avni"])
        r = people.resolve(self.paths, "me and my family at the beach")
        self.assertEqual((r["groups"], r["photos"], r["rest"]), (["family"], {"/p/me-sam.jpg", "/p/me-avni.jpg"}, "at the beach"))
        self.assertIsNone(people.resolve(self.paths, "tigers in Ranthambore"))


class FacesStepTests(TempHome):
    """3.9 Faces is a step of the run like the others: its own stage event, in the photograph's way before Describe, and a
    catch-up for photographs indexed before that converts and embeds nothing again."""

    def setUp(self):
        super().setUp()
        from tests.portable.test_photo import jpeg

        (self.sdir / "c").mkdir()
        self.src = jpeg(self.sdir / "c" / "p.jpg")
        self.events = []

    def run_index_(self):
        from rag_search.core import indexer

        srcs = indexer.scan_sources(self.sdir, indexer.exclude_dirs(self.paths))
        return indexer.run_index(self.paths, srcs, self.roots(), jobs=1, embedder=FakeEmbedder(),
                                 progress=lambda e: self.events.append(e))

    def test_without_the_models_no_face_is_looked_for_and_the_step_is_left_out(self):
        self.assertIn("not downloaded", faces.available(self.paths))
        self.assertEqual(faces.for_photo(self.paths, Path("/nowhere.jpg"), "x"), (None, "unavailable"))
        self.assertFalse(faces.store_file(self.paths).exists())

    def test_a_photograph_gets_its_face_count_and_is_looked_at_once(self):
        from rag_search.core import indexer, photo

        calls = []
        stage = []

        def fake(paths, src, sha, *, store=None):
            calls.append(sha)
            return 2, "found"
        with mock.patch.object(faces, "available", return_value=""), mock.patch.object(faces, "for_photo", fake), \
                mock.patch.object(indexer, "stage_event", lambda *a, **k: stage.append((a[2], a[3], k))), \
                mock.patch.object(photo, "visible_text", return_value=""):
            self.run_index_()
        self.assertEqual(len(calls), 1)
        text = next(self.paths.markup.rglob("p.md")).read_text(encoding="utf-8")
        self.assertIn(f"- {faces.LINE} 2", text)
        step = [k for st, status, k in stage if st == "faces"]
        self.assertEqual([(k["how"], k["faces"], k["earlier"]) for k in step], [("found", 2, False)])
        order = [st for st, _s, _k in stage if st in ("locate", "faces", "describe", "chunk")]
        self.assertLess(order.index("faces"), order.index("describe"))

    def test_a_photograph_indexed_before_is_looked_at_without_being_converted_again(self):
        from rag_search.core import indexer, photo

        with mock.patch.object(faces, "available", return_value="the models are not here"), \
                mock.patch.object(photo, "visible_text", return_value=""):
            self.run_index_()
        md = next(self.paths.markup.rglob("p.md"))
        before = (md.read_text(encoding="utf-8"), md.stat().st_mtime_ns, next(self.paths.index.rglob("nodes.json")).stat().st_mtime_ns)
        stage = []
        with mock.patch.object(faces, "available", return_value=""), \
                mock.patch.object(faces, "for_photo", return_value=(3, "found")) as fp, \
                mock.patch.object(indexer, "stage_event", lambda *a, **k: stage.append((a[2], k))), \
                mock.patch.object(photo, "visible_text", side_effect=AssertionError("not sniffed again")):
            s = self.run_index_()
        fp.assert_called_once()
        self.assertEqual(s["indexed"], 0)                                       # nothing converted or embedded
        self.assertEqual(before, (md.read_text(encoding="utf-8"), md.stat().st_mtime_ns,
                                  next(self.paths.index.rglob("nodes.json")).stat().st_mtime_ns))
        self.assertEqual([k for st, k in stage if st == "faces"],
                         [{"seconds": mock.ANY, "how": "found", "earlier": True, "faces": 3}])

    def test_the_run_view_counts_the_step_like_the_others(self):
        st = runview._fresh()
        feed = lambda **kw: runview._feed(st, json.dumps(kw))
        feed(event="stage", stage="faces", status="done", pid=1, ts=1, seconds=0.2, how="found", faces=3, earlier=False)
        feed(event="stage", stage="faces", status="done", pid=1, ts=2, seconds=0.1, how="known", faces=0, earlier=True)
        v = runview._steps_view(st)["faces"]
        self.assertEqual((v["n"], v["faces"], v["how"], v["scope"]),
                         (2, 3, {"found": 1, "known": 1}, {"in this run": 1, "indexed earlier": 1}))


class OnePhotographTests(PeopleCase):
    def test_a_photograph_is_looked_at_once(self):
        self.st.add_photo("s", Path("/a.jpg"), [face(like(A))])
        self.st.add_photo("s", Path("/a.jpg"), [face(like(A)), face(like(B))])
        self.st.seen_at("s", Path("/moved/a.jpg"))
        self.assertEqual((self.st.photo_faces("s"), self.st.counts()["faces"], self.st.sources(["s"])),
                         (1, 1, {"s": "/moved/a.jpg"}))


class SearchTests(DaemonCase):
    def setUp(self):
        super().setUp()
        self.write_doc("pics/beach.md", "# Beach\n\nA beach with ice cream stalls. " * 3)
        self.write_doc("pics/forest.md", "# Forest\n\nA jungle path with ice cream. " * 3)
        self.write_doc("pics/office.md", "# Office\n\nDesks and chairs. " * 3)
        self.index(embedder=FakeEmbedder())
        self.publish()
        nodes = {}
        for f in self.paths.index.rglob("nodes.json"):
            for n in json.loads(f.read_text())["nodes"]:
                nodes[n["metadata"]["file_name"]] = n["metadata"]["src_path"]
        self.src = {k.split(".")[0]: v for k, v in nodes.items()}
        people._CACHE.clear()
        with faces.Store(self.paths) as st:
            st.add_photo("f", Path(self.src["forest"]), [face(like(C))])
            st.add_photo("o", Path(self.src["office"]), [face(like(C))])
            st.label([1], st.person_id("Avni", create=True))

    def search(self, q):
        r = client.roundtrip(self.paths, "search", protocol.make_request("search", "cli", query=q, top_k=5), 10)
        self.assertTrue(r["ok"], r)
        return r["result"]

    def test_a_query_that_names_someone_keeps_to_their_photographs(self):
        self.start_daemon()
        res = self.search("Avni ice cream")
        self.assertEqual(res["people"], {"people": ["Avni"], "groups": [], "photos": 2, "searched_for": "ice cream"})
        self.assertEqual({r["source"] for r in res["results"]}, {"forest.md", "office.md"} & {r["source"] for r in res["results"]})
        self.assertTrue(all(r["source"] in ("forest.md", "office.md") for r in res["results"]))
        self.assertTrue(all(r["people"] == ["Avni"] for r in res["results"]))
        listing = self.search("photos of Avni")
        self.assertEqual(sorted(r["source"] for r in listing["results"]), ["forest.md", "office.md"])
        plain = self.search("beach")
        self.assertNotIn("people", plain)
        self.assertTrue(any(r["source"] == "beach.md" for r in plain["results"]))


class DashboardTests(PeopleCase):
    def test_the_people_tab_names_faces_merges_people_and_makes_groups(self):
        from tests.portable.test_ui import Dash

        a1, a2 = self.photo("s1", like(A), like(B))
        dash = Dash(self.paths)
        self.addCleanup(dash.close)
        st, js, _p, _h = dash.req("GET", "/api/people")
        self.assertEqual((st, js["counts"]["faces"], js["people"]), (200, 2, []))
        self.assertTrue(js["missing_models"])
        st, js, _p, _h = dash.req("GET", "/api/people/clusters")
        self.assertEqual(js["total"], 2)
        st, js, _p, _h = dash.req("POST", "/api/people/name", {"name": "Avni", "faces": [a1]})
        avni = js["person"]
        st, js, _p, _h = dash.req("POST", "/api/people/name", {"name": "Avni G", "faces": [a2]})
        other = js["person"]
        st, js, _p, _h = dash.req("POST", "/api/people/rename", {"person": other, "name": "avni"})
        self.assertEqual((st, js["merge_needed"]), (409, True))
        st, js, _p, _h = dash.req("POST", "/api/people/rename", {"person": other, "name": "Avni", "merge": True})
        self.assertEqual(js["merged_into"], avni)
        st, js, _p, _h = dash.req("POST", "/api/people/group/create", {"name": "family"})
        dash.req("POST", "/api/people/group/members", {"group": js["group"], "people": [avni]})
        st, js, _p, _h = dash.req("GET", f"/api/people/person?id={avni}")
        self.assertEqual((js["name"], js["labelled_total"], js["photos"]), ("Avni", 2, 1))
        st, js, _p, _h = dash.req("GET", "/api/people")
        self.assertEqual(js["people"][0]["groups"], ["family"])
        st, _js, payload, hdrs = dash.req("GET", f"/api/people/face?id={a1}")
        self.assertEqual((st, payload, hdrs["Content-Type"]), (200, b"jpg", "image/jpeg"))
        st, js, _p, _h = dash.req("POST", "/api/people/group/create", {"name": "Avni"})
        self.assertEqual(st, 400)


if __name__ == "__main__":
    import unittest

    unittest.main()
