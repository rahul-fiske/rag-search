import os
import sys
import unittest

from tests.helpers import FakeEmbedder, TempHome
from rag_search.core import indexer


class ExtraIndexerTests(TempHome):
    def sources(self):
        return indexer.scan_sources(self.paths.docs, indexer.exclude_dirs(self.paths))

    def test_parallel_pool_path(self):
        for i in range(3):
            self.write_doc(f"c/d{i}.md", f"# T\n\n<!-- page 1 -->\n\ndocument number {i} text")
        s = indexer.run_index(self.paths, self.sources(), self.paths.docs, jobs=2,
                              embedder=FakeEmbedder())
        self.assertEqual((s["indexed"], s["errors"]), (3, []))

    def _stage_events(self, log):
        import json
        return [json.loads(x) for x in log.read_text().splitlines() if x.strip()]

    def test_every_file_reports_each_pipeline_stage_serial_and_parallel(self):
        for jobs in (1, 2):
            for i in range(3):
                self.write_doc(f"c/d{jobs}{i}.md", f"# T\n\n<!-- page 1 -->\n\ndocument {jobs} {i} text")
            log = self.tmp / f"events{jobs}.jsonl"
            got = []
            s = indexer.run_index(self.paths, self.sources(), self.paths.docs, jobs=jobs,
                                  embedder=FakeEmbedder(), stage_log=log,
                                  progress=lambda ev: got.append(ev) if "stage" in ev else None)
            self.assertEqual(s["errors"], [])
            events = self._stage_events(log) + [dict(e["stage"], event="stage") for e in got]
            for i in range(3):
                name = f"c/d{jobs}{i}.md"
                seq = [(e["stage"], e["status"]) for e in sorted(
                    (e for e in events if e["file"] == name), key=lambda e: e["ts"] if "ts" in e else 1e18)]
                # stages 2 Fingerprint, 3 Convert (with 3.1 Profile), 4 Chunk, 6 Write (nodes.json), 5 Embed, 6 Write (vectors)
                self.assertEqual(seq, [("fingerprint", "done"), ("convert", "start"), ("profile", "done"),
                                       ("convert", "done"), ("chunk", "done"), ("write", "done"),
                                       ("embed", "start"), ("embed", "done"), ("write", "done")], (jobs, name, seq))

    def test_stage_events_are_written_by_prepare_document_without_a_log(self):
        indexer.stage_event(None, "c/a.md", "convert", "start")             # no file: silently nothing
        indexer.stage_event("/nonexistent-dir/x.jsonl", "c/a.md", "convert", "start")   # unwritable: no error

    def test_force_md_reconverts_even_when_the_index_is_fresh(self):
        src = self.write_doc("c/a.md", "# T\n\n<!-- page 1 -->\n\noriginal text")
        self.assertEqual(self.index()["indexed"], 1)
        md = self.paths.markup / "c" / "a.md"
        md.write_text("# T\n\n<!-- page 1 -->\n\nstale text")
        self.assertEqual(self.index()["skipped_fresh"], 1)  # plain run trusts the index
        s = self.index(force_md=True)
        self.assertEqual((s["indexed"], s["skipped_fresh"]), (1, 0))
        self.assertEqual(md.read_text(), src.read_text())

    def test_missing_docling_is_a_per_file_error_not_a_crash(self):
        self.write_doc("c/x.pdf", "%PDF-1.4 not really")
        self.write_doc("c/ok.md", "# T\n\n<!-- page 1 -->\n\nfine")
        s = indexer.run_index(self.paths, self.sources(), self.paths.docs, jobs=1,
                              embedder=FakeEmbedder())
        self.assertEqual(s["indexed"], 1)
        self.assertEqual(len(s["errors"]), 1)
        self.assertIn("x.pdf", s["errors"][0]["src"])

    def test_docling_python_subprocess_failure_is_reported(self):
        os.environ["RAG_SEARCH_DOCLING_PYTHON"] = sys.executable
        self.write_doc("c/x.pdf", "%PDF-1.4 not really")
        s = indexer.run_index(self.paths, self.sources(), self.paths.docs, jobs=1,
                              embedder=FakeEmbedder())
        self.assertEqual(s["indexed"], 0)
        self.assertIn("docling subprocess failed", s["errors"][0]["message"])

    def test_bad_docling_python_path(self):
        os.environ["RAG_SEARCH_DOCLING_PYTHON"] = "/nonexistent/python"
        self.write_doc("c/x.pdf", "x")
        s = indexer.run_index(self.paths, self.sources(), self.paths.docs, jobs=1,
                              embedder=FakeEmbedder())
        self.assertIn("not found", s["errors"][0]["message"])

    def test_source_outside_docs_root(self):
        other = self.tmp / "other.md"
        other.write_text("hello")
        s = indexer.run_index(self.paths, [other], self.paths.docs, jobs=1,
                              embedder=FakeEmbedder())
        self.assertIn("outside docs_root", s["errors"][0]["message"])

    def test_only_one_indexer_at_a_time(self):
        with indexer.index_lock(self.paths):
            with self.assertRaises(indexer.IndexBusyError):
                indexer.run_index(self.paths, [], self.paths.docs, jobs=1,
                                  embedder=FakeEmbedder())
        # lock is released afterwards
        indexer.run_index(self.paths, [], self.paths.docs, jobs=1, embedder=FakeEmbedder())

    def test_model_change_makes_index_stale(self):
        self.write_doc("c/a.md", "# T\n\n<!-- page 1 -->\n\nhello there")
        s1 = indexer.run_index(self.paths, self.sources(), self.paths.docs, jobs=1,
                               embedder=FakeEmbedder())
        os.environ["RAG_SEARCH_MODEL"] = "some/other-model"
        s2 = indexer.run_index(self.paths, self.sources(), self.paths.docs, jobs=1,
                               embedder=FakeEmbedder())
        self.assertEqual((s1["indexed"], s2["indexed"]), (1, 1))


if __name__ == "__main__":
    unittest.main()


class PoolShutdownTests(unittest.TestCase):
    """A conversion process that cannot exit (a thread abandoned after a document timeout is stuck
    in a native call) must not keep the run from reaching embedding."""

    def test_a_process_that_cannot_exit_is_terminated_after_the_grace_period(self):
        import concurrent.futures as cf
        import multiprocessing as mp
        import time

        from tests.helpers import leave_a_stuck_thread

        pool = cf.ProcessPoolExecutor(max_workers=1, mp_context=mp.get_context("spawn"))
        self.assertEqual(pool.submit(leave_a_stuck_thread).result(timeout=60), "done")
        t0 = time.monotonic()
        self.assertEqual(indexer._shutdown_pool(pool, grace=1.0), 1)
        self.assertLess(time.monotonic() - t0, 15)

    def test_a_healthy_pool_closes_without_killing_anything(self):
        import concurrent.futures as cf
        import multiprocessing as mp

        pool = cf.ProcessPoolExecutor(max_workers=1, mp_context=mp.get_context("spawn"))
        self.assertEqual(pool.submit(abs, -3).result(timeout=60), 3)
        self.assertEqual(indexer._shutdown_pool(pool, grace=30.0), 0)
