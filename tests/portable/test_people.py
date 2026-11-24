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
        (a1,) = self.photo("s1", A)
        (a2,) = self.photo("s2", like(A))
        (half,) = self.photo("s3", unit(0.38 * A + 0.925 * unit(C - (C @ A) * A)))  # 0.38 like A: over OpenCV's 0.363, under a match
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

    def test_a_chain_of_look_alikes_is_split_into_groups_of_a_size_that_can_be_looked_at(self):
        ids = []
        for k in range(18):                                    # each face 20 degrees from the next: one chain at 0.45
            v = np.zeros(128, dtype=np.float32)
            v[0], v[1] = np.cos(np.radians(20 * k)), np.sin(np.radians(20 * k))
            ids += self.photo(f"c{k}", v)
        with mock.patch.object(people, "MAX_GROUP", 5):
            groups = people.work_out(self.st).clusters()
        self.assertEqual(sorted(f for g in groups for f in g), sorted(ids))
        self.assertTrue(all(len(g) <= 5 for g in groups), [len(g) for g in groups])
        self.assertGreater(len(groups), 1)

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
        # the Markdown said no count, the store holds 3: the photograph is made again from what is known and embedded in this run
        self.assertEqual(s["indexed"], 1)
        self.assertEqual(md.read_text(encoding="utf-8"), before[0] + f"- {faces.LINE} 3\n")
        self.assertEqual([k for st, k in stage if st == "faces"],
                         [{"seconds": mock.ANY, "how": "found", "earlier": True, "faces": 3}])      # looked at once, not twice

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


class PhotoViewerTests(PeopleCase):
    def test_a_face_shows_its_whole_photograph_with_every_face_boxed_and_named(self):
        from PIL import Image

        from tests.portable.test_ui import Dash

        folder = self.source("trip")
        pic = folder / "IMG_1.jpg"
        Image.new("RGB", (400, 300), "white").save(pic)
        a1, b1 = self.photo("s1", A, B, src=str(pic))
        outside = self.sdir.parent / "elsewhere.jpg"
        Image.new("RGB", (40, 30)).save(outside)
        (o1,) = self.photo("s2", C, src=str(outside))
        avni = self.st.person_id("Avni", create=True)
        self.st.label([a1], avni)
        people._CACHE.clear()
        dash = Dash(self.paths)
        self.addCleanup(dash.close)
        st, js, _p, _h = dash.req("GET", f"/api/people/photo/faces?face={b1}")
        self.assertEqual((st, js["name"], js["available"]), (200, "IMG_1.jpg", True))
        self.assertEqual([(f["id"], f["name"], f["how"]) for f in js["faces"]], [(a1, "Avni", "you"), (b1, "", "")])
        self.assertTrue(all(0 <= f["x"] <= 1 and f["w"] > 0 for f in js["faces"]))
        self.assertNotIn("src", js)                            # the server's paths stay on the server
        st, _j, raw, _h = dash.req("GET", f"/api/people/photo?face={a1}")
        self.assertEqual(st, 200)
        self.assertTrue(raw.startswith(b"\xff\xd8"))
        st, js, _p, _h = dash.req("GET", f"/api/people/photo/faces?face={o1}")      # not in a registered folder: no picture
        self.assertEqual((st, js["available"]), (200, False))
        st, _b, _p, _h = dash.req("GET", f"/api/people/photo?face={o1}")
        self.assertEqual(st, 404)
        st, _b, _p, _h = dash.req("GET", "/api/people/photo/faces?face=99999")
        self.assertEqual(st, 404)


    def test_a_face_the_model_missed_can_be_added_by_drawing_a_box(self):
        from PIL import Image

        from tests.portable.test_ui import Dash

        folder = self.source("trip")
        pic = folder / "IMG_2.jpg"
        Image.new("RGB", (400, 300), "white").save(pic)
        (a1,) = self.photo("s1", A, src=str(pic))
        dash = Dash(self.paths)
        self.addCleanup(dash.close)
        made = {"x": 0.5, "y": 0.2, "w": 0.2, "h": 0.3, "score": 1.0, "px": 80, "vec": B, "thumb": b"jpg"}
        with mock.patch.object(faces, "available", return_value=""), \
                mock.patch.object(faces, "face_in_box", return_value=made) as box:
            st, js, _p, _h = dash.req("POST", "/api/people/face/add", {"face": a1, "x": 0.5, "y": 0.2, "w": 0.2, "h": 0.3})
            self.assertEqual(st, 200)
            self.assertEqual(box.call_args[0][1], pic.resolve())
            st, bad, _p, _h = dash.req("POST", "/api/people/face/add", {"face": a1, "x": 0.9, "y": 0.2, "w": 0.4, "h": 0.3})
            self.assertEqual(st, 400)                                          # runs off the photograph
            st, bad, _p, _h = dash.req("POST", "/api/people/face/add", {"face": 99999, "x": 0.1, "y": 0.1, "w": 0.2, "h": 0.2})
            self.assertEqual(st, 400)
        st, shown, _p, _h = dash.req("GET", f"/api/people/photo/faces?face={a1}")
        self.assertEqual([f["id"] for f in shown["faces"]], [a1, js["face"]])
        self.assertEqual(self.st.photo_faces("s1"), 2)
        st, _j, _p, _h = dash.req("POST", "/api/people/name", {"name": "Avni", "faces": [js["face"]]})
        self.assertEqual(st, 200)


class FinderRulesTests(PeopleCase):
    def test_boxes_of_one_face_found_at_two_sizes_overlap_enough_to_merge(self):
        self.assertGreater(faces._iou((0.1, 0.1, 0.2, 0.2), (0.11, 0.12, 0.2, 0.19)), faces.MERGE_IOU)
        self.assertEqual(faces._iou((0.1, 0.1, 0.1, 0.1), (0.5, 0.5, 0.1, 0.1)), 0.0)

    def test_a_photograph_looked_at_by_either_finder_is_known(self):
        self.photo("old", A)                                    # stored with the finder of this version
        self.st.con.execute("INSERT INTO photos (sha, src, detector, faces, ts) VALUES ('older', '/p/o.jpg', ?, 0, 1)",
                            (faces.DETECTOR,))                        # one looked at by the first version (one size)
        self.st.con.execute("INSERT INTO photos (sha, src, detector, faces, ts) VALUES ('other', '/p/x.jpg', 'some-other-model', 0, 1)")
        self.st.con.commit()
        self.assertEqual((self.st.photo_faces("old"), self.st.photo_faces("older"), self.st.photo_faces("other")), (1, 0, None))

    def test_a_box_the_finder_is_unsure_of_is_never_suggested_or_asked_about(self):
        (a1,) = self.photo("s1", A)
        (weak,) = self.photo("s2", A, score=0.75)
        self.st.label([a1], self.st.person_id("Avni", create=True))
        w = people.work_out(self.st)
        self.assertNotIn(weak, w.suggest)
        self.assertNotIn(weak, w.person)
        self.assertNotIn(weak, w.unknown)


class SplitSectionTests(TempHome):
    def test_a_description_written_after_a_blank_line_still_belongs_to_the_photo_details(self):
        from rag_search.core import describe

        text = ("<!-- page 1 -->\n\n## Photo details\n\n- Taken: 2012-01-31 06:32:06 (Tuesday)\n- Size: 2816 x 2112 pixels\n\n"
                "- Looks like (model-written): A man holds a child.\n- Described by: Qwen\n- Faces detected: 2\n")
        section, rest = describe.section_of(text)
        self.assertIn("Faces detected: 2", section)
        self.assertEqual(rest.strip(), "")
        self.assertTrue(describe.is_photograph(text))
        page = ("<!-- page 1 -->\n\n## Photo details\n\n- Taken: 2020-01-01 10:00:00 (Wednesday)\n\n- a bullet of the page\n\n"
                + "words of a page " * 40)
        self.assertFalse(describe.is_photograph(page))            # a list in a photographed page is still the page


class FacesLineFollowsTheStoreTests(TempHome):
    def test_a_markdown_that_says_another_count_than_the_store_is_made_again(self):
        from rag_search.core import indexer

        md = self.tmp / "p.md"
        head = "<!-- page 1 -->\n\n## Photo details\n\n- Taken: 2020-01-01 10:00:00 (Wednesday)\n"
        md.write_text(head + "- Faces detected: 2\n- Described by: x\n", encoding="utf-8")
        self.assertFalse(indexer._faces_line_differs(md, 2))                 # already the store's count
        self.assertFalse(indexer._faces_line_differs(md, None))              # no answer from the store: nothing to do
        self.assertTrue(indexer._faces_line_differs(md, 4))                  # Apple Vision said 2, the store holds 4
        md.write_text(head, encoding="utf-8")
        self.assertTrue(indexer._faces_line_differs(md, 1))                  # no line yet
        self.assertFalse(indexer._faces_line_differs(md, 0))                 # no faces and no line: as it should be
        md.write_text("<!-- page 1 -->\n\n- Looks like (model-written): x\n- Faces detected: 1\n", encoding="utf-8")   # heading lost
        self.assertTrue(indexer._faces_line_differs(md, 3))
        self.assertFalse(indexer._faces_line_differs(md, 1))

    def test_a_photograph_is_what_its_conversion_recorded(self):
        from rag_search.core import describe, indexer, photo

        idx = self.tmp / "i"
        idx.mkdir()
        self.assertIsNone(photo.recorded_photo(idx))
        text = "<!-- page 1 -->\n\n- Looks like (model-written): " + "a long description " * 30 + "\n"      # no heading, a lot of text
        self.assertFalse(describe.is_photograph(text))                       # what the Markdown alone says
        (idx / "index.meta.json").write_text('{"conversion": {"branches": {"photo": 1}}}')
        self.assertTrue(photo.recorded_photo(idx))
        self.assertTrue(indexer._is_photo_doc(idx, text))                    # the record wins
        (idx / "index.meta.json").write_text('{"conversion": {"branches": {"digital": 2}}}')
        self.assertFalse(indexer._is_photo_doc(idx, text))


class CoverageTests(PeopleCase):
    def test_every_step_is_counted_against_what_each_photograph_should_have(self):
        import json as _json

        from rag_search.core import coverage, describe, pictures

        os_env = {"RAG_SEARCH_PICTURE_MODEL": "google/siglip2-base-patch16-224"}
        made_by = describe.current_id()
        rows = {"s1": {"conv": {"branches": {"photo": 1}}, "pic": True, "desc": True, "faces": True},
                "s2": {"conv": {"branches": {"photo": 1}}, "pic": False, "desc": False, "faces": False},
                "s3": {"conv": {"branches": {"digital": 3}}}, "s4": {"conv": {"branches": {"video": 1}}}}
        describe.remember(self.paths, "s1", made_by, {"description": "a", "tags": [], "people": 1})
        for sha, r in rows.items():
            d = self.paths.index / "c" / sha
            d.mkdir(parents=True)
            (d / "index.meta.json").write_text(_json.dumps({"src_sha256": sha, "conversion": r["conv"]}))
            if r.get("pic"):
                (d / pictures.FILE).write_bytes(b"x")
                (d / pictures.META).write_text(_json.dumps({"model": "an-older-picture-model", "src_sha256": sha}))
        self.photo("s1", A)
        with mock.patch.dict("os.environ", os_env), mock.patch.object(faces, "available", return_value=""):
            rep = coverage.report(self.paths)
        c = rep["collections"]["c"]
        self.assertEqual((c["files"], c["photographs"], c["videos"], c["documents"]), (4, 2, 1, 1))
        steps = c["steps"]
        self.assertEqual((steps["described"]["done"], steps["described"]["expected"]), (1, 2))
        self.assertEqual((steps["faces"]["done"], steps["faces"]["expected"], steps["faces"]["stale"]), (1, 2, 0))
        self.assertEqual((steps["pictures"]["done"], steps["pictures"]["expected"], steps["pictures"]["stale"]), (1, 2, 1))
        self.assertEqual(steps["faces"]["examples"], ["c/s2"])
        self.assertEqual(coverage.missing(rep), {"described": 1, "faces": 1, "pictures": 1, "details": 1})     # and the video has none
        with mock.patch.dict("os.environ", {"RAG_SEARCH_PICTURE_MODEL": "off"}), mock.patch.object(faces, "available", return_value="no models"):
            rep = coverage.report(self.paths)
        self.assertEqual(rep["total"]["steps"]["pictures"]["expected"], 0)           # a step that is off expects nothing
        self.assertEqual(rep["total"]["steps"]["faces"]["expected"], 0)
        self.assertIn("faces", rep["notes"])


class GuessesAreGuessesTests(PeopleCase):
    @staticmethod
    def at(sim):
        """A face whose similarity to A is *sim*."""
        o = B - (B @ A) * A
        return unit(sim * A + float(np.sqrt(1 - sim * sim)) * unit(o))

    def test_a_tiny_face_or_a_weak_similarity_is_never_suggested(self):
        (a1,) = self.photo("s1", A)
        (weak,) = self.photo("s2", self.at(0.30), score=0.82)        # under OpenCV's line for "the same person"
        (guess,) = self.photo("s3", self.at(0.45), score=0.82)       # over it, but the finder is not sure enough to name it
        (tiny,) = self.photo("s4", self.at(0.45), px=20, score=0.82)
        (unsure,) = self.photo("s5", self.at(0.45), score=0.7)
        self.st.label([a1], self.st.person_id("Avni", create=True))
        w = people.work_out(self.st)
        self.assertEqual(sorted(w.suggest), [guess])
        for f in (weak, tiny, unsure):
            self.assertNotIn(f, w.suggest)
        self.assertGreaterEqual(people.SUGGEST, 0.363)

    def test_the_viewer_data_says_how_sure_each_name_is(self):
        (a1,) = self.photo("s1", A, src="/p/s1.jpg")
        (a2,) = self.photo("s1b", like(A, 0.1), src="/p/s1b.jpg")
        avni = self.st.person_id("Avni", create=True)
        self.st.label([a1], avni)
        people._CACHE.clear()
        shown = people.photo_of_face(self.paths, a2)["faces"]
        self.assertEqual([(f["how"], f["name"]) for f in shown], [("auto", "Avni")])
        self.assertGreater(shown[0]["similarity"], 0.9)
        self.assertEqual(people.photo_of_face(self.paths, a1)["faces"][0]["similarity"], 1.0)


class BatchTests(PeopleCase):
    def test_many_ticks_and_crosses_are_saved_together(self):
        from tests.portable.test_ui import Dash

        a1, a2, a3 = self.photo("s1", A, A, A)
        b1, = self.photo("s2", B)
        avni = self.st.person_id("Avni", create=True)
        self.st.label([a3], avni)
        dash = Dash(self.paths)
        self.addCleanup(dash.close)
        ops = [{"op": "confirm", "person": avni, "faces": [a1]}, {"op": "reject", "person": avni, "faces": [b1]},
               {"op": "unlabel", "faces": [a3]}]
        st, js, _p, _h = dash.req("POST", "/api/people/batch", {"ops": ops})
        self.assertEqual((st, js["applied"]), (200, 3))
        yes, no = self.st.labels()
        self.assertEqual((yes, no), ({a1: avni}, {(b1, avni)}))
        st, js, _p, _h = dash.req("POST", "/api/people/batch", {"ops": [{"op": "explode", "faces": [1]}]})
        self.assertEqual(st, 400)
        st, js, _p, _h = dash.req("POST", "/api/people/batch", {"ops": [{"op": "confirm", "person": "x", "faces": [1]}]})
        self.assertEqual(st, 400)
        st, js, _p, _h = dash.req("POST", "/api/people/batch", {"ops": "nope"})
        self.assertEqual(st, 400)


class RescanTests(PeopleCase):
    def test_a_rescan_adds_only_the_faces_the_first_finder_missed_and_keeps_the_ones_it_found(self):
        from PIL import Image

        folder = self.source("trip")
        pic = folder / "IMG_9.jpg"
        Image.new("RGB", (40, 30)).save(pic)
        (kept,) = self.photo("s1", A, src=str(pic))
        with self.st.con:
            self.st.con.execute("UPDATE photos SET detector = ? WHERE sha = 's1'", (faces.DETECTOR,))     # looked at by the first finder
        avni = self.st.person_id("Avni", create=True)
        self.st.label([kept], avni)
        again = {"x": 0.1, "y": 0.1, "w": 0.2, "h": 0.2, "score": 0.93, "px": 120, "vec": B, "thumb": b"t"}      # the same face
        missed = {"x": 0.6, "y": 0.2, "w": 0.2, "h": 0.2, "score": 0.9, "px": 90, "vec": B, "thumb": b"t"}
        weak = {"x": 0.3, "y": 0.6, "w": 0.1, "h": 0.1, "score": 0.72, "px": 50, "vec": B, "thumb": b"t"}
        fake = mock.Mock()
        fake.faces.return_value = [again, missed, weak]
        with mock.patch.object(faces, "available", return_value=""), mock.patch.object(faces, "finder", return_value=fake):
            res = faces.rescan(self.paths)
            again_res = faces.rescan(self.paths)                       # nothing is left to look at
        self.assertEqual(res, {"looked": 1, "added": 2, "sure_added": 1, "missing": 0, "failed": 0})
        self.assertEqual(again_res["looked"], 0)
        rows = self.st.con.execute("SELECT id, score FROM faces WHERE sha = 's1' ORDER BY id").fetchall()
        self.assertEqual(rows[0][0], kept)                              # the face that was found keeps its id and its name
        self.assertEqual(self.st.labels()[0], {kept: avni})
        self.assertEqual((len(rows), self.st.sure_faces("s1")), (3, 2))
        self.assertEqual(self.st.con.execute("SELECT detector, faces FROM photos WHERE sha = 's1'").fetchone(), (faces.DETECTOR_MS, 3))


class ResidueTests(PeopleCase):
    def setUp(self):
        super().setUp()
        from PIL import Image

        from rag_search.core import describe, sniffcache

        self.folder = self.source("trip")
        self.alive = self.folder / "IMG_1.jpg"
        Image.new("RGB", (40, 30)).save(self.alive)
        self.renamed = self.folder / "renamed" / "IMG_3.jpg"
        self.renamed.parent.mkdir()
        Image.new("RGB", (40, 30)).save(self.renamed)
        for sha, src in (("s1", self.alive), ("s2", self.folder / "deleted" / "IMG_2.jpg"), ("s3", self.folder / "old name" / "IMG_3.jpg")):
            self.photo(sha, A, src=str(src))
        self.ids = {s: [r[0] for r in self.st.con.execute("SELECT id FROM faces WHERE sha = ?", (s,))][0] for s in ("s1", "s2", "s3")}
        self.avni = self.st.person_id("Avni", create=True)
        self.st.label([self.ids["s1"], self.ids["s2"]], self.avni)
        for sha, path in (("s1", self.alive), ("s3", self.renamed)):         # the index knows these two
            d = self.paths.index / "trip" / sha
            d.mkdir(parents=True)
            (d / "index.meta.json").write_text(json.dumps({"src_sha256": sha, "src_path": str(path)}))
        made_by = describe.current_id()
        for sha in ("s1", "s2"):
            describe.remember(self.paths, sha, made_by, {"description": "x", "tags": [], "people": 0})
        describe.remember_failure(self.paths, "s2", made_by, "no")
        for sha in ("s1", "s2"):
            sniffcache.put(self.paths.workspace / sniffcache.FILE, sha, "text")

    def counts(self):
        import sqlite3

        con = sqlite3.connect(self.paths.workspace / "describe_cache.sqlite")
        d = sorted(r[0] for r in con.execute("SELECT sha FROM d"))
        con.close()
        return (sorted(r[0] for r in self.st.con.execute("SELECT sha FROM photos")), d)

    def test_a_photograph_that_left_every_folder_is_forgotten_everywhere_and_a_renamed_folder_is_followed(self):
        from rag_search.core import residue

        dry = residue.purge(self.paths, dry_run=True, force=True)
        self.assertEqual((dry["photographs"], dry["faces"], dry["labelled"], dry["descriptions"], dry["text_checks"], dry["repaired"]),
                         (1, 1, 1, 1, 1, 1))
        self.assertEqual(self.counts()[0], ["s1", "s2", "s3"])                 # a dry run changes nothing
        res = residue.purge(self.paths, force=True)
        self.assertEqual((res["photographs"], res["faces"], res["labelled"]), (1, 1, 1))
        self.assertEqual(self.counts(), (["s1", "s3"], ["s1"]))
        self.assertEqual(self.st.labels()[0], {self.ids["s1"]: self.avni})        # the names of the others are untouched
        self.assertIsNone(self.st.thumb(self.ids["s2"]))
        self.assertEqual(self.st.con.execute("SELECT COUNT(*) FROM vectors").fetchone()[0], 2)
        self.assertEqual(self.st.sources(["s3"]), {"s3": str(self.renamed)})      # the renamed folder is followed

    def test_nothing_is_forgotten_when_the_workspace_cannot_be_read_or_a_great_part_would_go(self):
        import shutil

        from rag_search.core import residue

        shutil.rmtree(self.paths.index)
        res = residue.purge(self.paths, force=True)
        self.assertEqual((res["photographs"], list(res["skipped"])), (0, ["all"]))
        self.assertEqual(self.counts()[0], ["s1", "s2", "s3"])
        d = self.paths.index / "trip" / "s1"                                    # one photograph left in the index
        d.mkdir(parents=True)
        (d / "index.meta.json").write_text(json.dumps({"src_sha256": "s1", "src_path": str(self.alive)}))
        for i in range(120):                                                    # and many in the store that nobody has
            self.photo(f"x{i}", B, src=f"/gone/{i}.jpg")
        res = residue.purge(self.paths)
        self.assertEqual(res["photographs"], 0)
        self.assertIn("people", res["skipped"])                                 # more than a fifth: stopped
        self.assertEqual(residue.purge(self.paths, force=True)["photographs"], 122)

    def test_an_indexing_run_forgets_what_is_left_of_deleted_photographs(self):
        from rag_search.core import indexer, residue

        with mock.patch.object(residue, "purge", return_value={"photographs": 1}) as purge:
            s = indexer.run_index(self.paths, [], self.roots(), prune=["trip"], jobs=1, embedder=FakeEmbedder(), progress=lambda e: None)
            self.assertEqual((purge.call_count, s["residue"]), (1, {"photographs": 1}))
            s = indexer.run_index(self.paths, [], self.roots(), jobs=1, embedder=FakeEmbedder(), progress=lambda e: None)
            self.assertEqual(purge.call_count, 2)                                      # after every run, not only a whole-collection one
