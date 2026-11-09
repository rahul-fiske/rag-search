"""What one collection costs: on disk (chunks, vectors, working copy, Markdown) and in the search daemon's memory."""

from __future__ import annotations

import unittest

from tests.helpers import TempHome

from rag_search import inventory
from rag_search.core import search
from rag_search.core.bm25 import BM25, tokenize

DOC = "# Title\n\n" + "\n\n".join(f"Paragraph {i} about topic {i % 7} with some words to index. " * 6 for i in range(40))


class CostTests(TempHome):
    def setUp(self):
        super().setUp()
        self.write_doc("coll/a.md", DOC)
        self.write_doc("coll/sub/b.md", DOC + "\n\nextra")
        self.index()
        self.publish()

    def test_the_disk_cost_is_what_each_part_is_for(self):
        k = inventory.collection_info(self.paths, "coll")["cost"]
        merged = self.paths.index / "coll" / "_all"
        self.assertEqual(k["chunks"], (merged / "nodes.json").stat().st_size)
        self.assertEqual(k["vectors"], (merged / "embeddings.npy").stat().st_size)
        self.assertEqual(k["search_index"], sum(f.stat().st_size for f in merged.iterdir()))
        self.assertGreaterEqual(k["search_index"], k["chunks"] + k["vectors"])
        self.assertLess(k["search_index"] - k["chunks"] - k["vectors"], 4096)         # only the small manifest besides
        self.assertGreater(k["working_copy"], 0)
        md = sum(f.stat().st_size for f in (self.paths.markup / "coll").rglob("*.md"))
        self.assertEqual(k["markdown"], md)
        self.assertEqual(k["total"], k["search_index"] + k["working_copy"] + k["markdown"])

    def test_the_catalogue_sizes_agree_with_the_inventory(self):
        from rag_search.catalog import list_view
        from rag_search.policy import Rules

        c = list_view(self.paths, Rules(), "cli")["collections"][0]
        k = inventory.collection_info(self.paths, "coll")["cost"]
        self.assertEqual((c["chunk_bytes"], c["vector_bytes"]), (k["chunks"], k["vectors"]))

    def test_the_memory_of_a_collection_adds_up(self):
        info = self.engine().memory_info()["collections"]["coll"]
        self.assertGreater(info["passages_bytes"], info["text_bytes"] * 0.5)
        self.assertGreater(info["keyword_bytes"], 0)
        self.assertEqual(info["ram_bytes"], info["embeddings_bytes"] + info["passages_bytes"] + info["keyword_bytes"])
        total = self.engine().memory_info()
        self.assertEqual(total["indexes_bytes"], sum(c["ram_bytes"] for c in total["collections"].values()))

    def test_the_estimate_is_worked_out_once(self):
        import numpy as np

        nodes = [{"id": str(i), "text": f"alpha beta {i}", "metadata": {"page_label": "1"}} for i in range(50)]
        d = search.IndexData(nodes, np.zeros((50, 8), dtype=np.float32), BM25([tokenize(n["text"]) for n in nodes]))
        first = search.index_memory(d)
        self.assertIs(search.index_memory(d), first)
        self.assertEqual(first["vectors"], 50 * 8 * 4)

    def test_bm25_memory_grows_with_the_postings(self):
        small = BM25([tokenize("one two three")] * 3)
        big = BM25([tokenize(f"word{i} shared common terms here") for i in range(300)])
        self.assertLess(small.memory_bytes(), big.memory_bytes())


if __name__ == "__main__":
    unittest.main()
