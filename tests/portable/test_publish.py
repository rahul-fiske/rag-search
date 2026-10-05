import json
import os
import unittest

from tests.helpers import TempHome
from rag_search import catalog, publish
from rag_search.paths import EMB_FILE, META_FILE, NODES_FILE, read_json

A = "# A\n\n<!-- page 1 -->\n\nalpha text about widgets"
B = "# B\n\n<!-- page 1 -->\n\nbeta text about gadgets"


class PublishTests(TempHome):
    def test_nothing_to_publish(self):
        r = self.publish()
        self.assertEqual((r["changed"], r["generation"]), (False, None))
        self.assertIsNone(self.paths.current_gen())

    def test_first_publish_layout_and_catalog(self):
        self.write_doc("manuals/net/g.md", A)
        self.index()
        r = self.publish()
        self.assertEqual((r["changed"], r["generation"], r["documents"]), (True, 1, 1))
        gen = self.paths.current_gen()
        self.assertEqual(gen.name, "gen-000001")
        self.assertTrue((gen / "index/manuals/_all" / NODES_FILE).exists())
        self.assertTrue((gen / "index/manuals/_all" / EMB_FILE).exists())
        self.assertTrue((gen / "markup/manuals/net/g.md").exists())
        cat = read_json(gen / "catalog.json")
        self.assertEqual(cat["collections"][0]["documents"][0]["name"], "net/g")
        self.assertEqual(cat["collections"][0]["documents"][0]["source"], "g.md")
        self.assertFalse(list(self.paths.serving.glob(".tmp-*")))

    def test_publish_is_idempotent(self):
        self.write_doc("c/a.md", A)
        self.index()
        self.publish()
        r = self.publish()
        self.assertEqual((r["changed"], r["generation"]), (False, 1))
        self.assertTrue(self.publish(force=True)["changed"])

    def test_workspace_rebuild_never_alters_published_generation(self):
        p = self.write_doc("c/a.md", A)
        self.index()
        self.publish()
        gen1 = self.paths.current_gen()
        before = (gen1 / "index/c/_all" / NODES_FILE).read_bytes()
        p.write_text(A + "\n\nGamma extra paragraph.")
        self.index()  # rewrites workspace _all via atomic replace
        self.assertEqual((gen1 / "index/c/_all" / NODES_FILE).read_bytes(), before)
        self.publish()
        self.assertNotEqual((self.paths.current_gen() / "index/c/_all" / NODES_FILE).read_bytes(),
                            before)

    def test_generations_gc_keeps_three_and_never_live(self):
        p = self.write_doc("c/a.md", A)
        for i in range(5):
            p.write_text(A + f"\n\nrevision {i}")
            self.index()
            self.publish()
        gens = publish.list_generations(self.paths)
        self.assertEqual(gens, [3, 4, 5])
        self.assertEqual(publish.current_generation(self.paths), 5)

    def test_rollback(self):
        p = self.write_doc("c/a.md", A)
        self.index()
        self.publish()
        p.write_text(A + "\n\nsecond revision")
        self.index()
        self.publish()
        self.assertEqual(publish.rollback(self.paths)["generation"], 1)
        self.assertEqual(publish.current_generation(self.paths), 1)
        with self.assertRaises(publish.PublishError):
            publish.rollback(self.paths)

    def test_removing_everything_publishes_an_empty_generation(self):
        a = self.write_doc("c/a.md", A)
        self.index()
        self.publish()
        a.unlink()
        self.index()
        r = self.publish()
        self.assertTrue(r["changed"])
        self.assertEqual(read_json(self.paths.current_gen() / "catalog.json")["collections"], [])

    def test_mixed_models_refused(self):
        self.write_doc("x/a.md", A)
        self.write_doc("y/b.md", B)
        self.index()
        meta = self.paths.index / "y" / "b" / META_FILE
        data = json.loads(meta.read_text())
        data["model"] = "other/model"
        meta.write_text(json.dumps(data))
        # _all manifest sha is unaffected; publish must notice differing models across collections
        with self.assertRaises(publish.PublishError):
            self.publish()

    def test_copy_fallback_when_link_fails(self):
        self.write_doc("c/a.md", A)
        self.index()
        real = os.link

        def broken(*a, **k):
            raise OSError("cross-device")
        os.link = broken
        try:
            self.assertTrue(self.publish()["changed"])
        finally:
            os.link = real
        gen = self.paths.current_gen()
        self.assertTrue((gen / "index/c/_all" / EMB_FILE).exists())

    def test_catalog_list_view_respects_client_access(self):
        self.write_doc("manuals/a.md", A)
        self.write_doc("personal/id.md", B)
        self.index()
        self.publish()
        from rag_search import policy
        rules = policy.load_rules(self.paths)[0]
        seen = lambda client: [c["collection"] for c in  # noqa: E731
                               catalog.list_view(self.paths, rules, client)["collections"]]
        self.assertEqual(seen("agent"), ["manuals", "personal"])   # open by default
        policy.save_rules(self.paths, {"personal": ["claude"]})
        rules = policy.load_rules(self.paths)[0]
        self.assertEqual(seen("claude"), ["manuals", "personal"])
        self.assertEqual(seen("agent"), ["manuals"])
        view = catalog.list_view(self.paths, rules, "agent")
        self.assertEqual(view["generation"], 1)
        self.assertEqual(view["totals"]["collections"], 1)      # totals only count what it sees
        self.assertNotIn("private", view["collections"][0])



class PublishGuardTests(TempHome):
    def test_interrupted_rebuild_does_not_silently_drop_a_collection(self):
        import shutil
        from rag_search import publish as pub
        self.write_doc("kitchen/bread.md", "# B\n\n<!-- page 1 -->\n\nsourdough starter")
        self.index()
        self.publish()
        # a cancelled `index all` leaves the workspace without the merged index
        shutil.rmtree(self.paths.index / "kitchen" / "_all")
        with self.assertRaises(pub.PublishError) as cm:
            self.publish()
        self.assertIn("kitchen", str(cm.exception))
        self.assertEqual(pub.current_generation(self.paths), 1)  # still serving the old one
        self.assertEqual(self.publish(allow_drop=True)["generation"], 2)  # explicit override

    def test_incomplete_documents_are_reported(self):
        self.write_doc("kitchen/bread.md", "# B\n\n<!-- page 1 -->\n\nsourdough starter")
        self.write_doc("kitchen/rye.md", "# R\n\n<!-- page 1 -->\n\nrye flour")
        self.index()
        (self.paths.index / "kitchen" / "rye" / "index.meta.json").unlink()  # unfinished
        r = self.publish()
        self.assertEqual(r["incomplete"], ["kitchen/rye"])


if __name__ == "__main__":
    unittest.main()
