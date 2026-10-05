"""Edge cases of the small modules: chunking of text that has no break, the location registry's refusals,
path helpers, inventory states.  Tier A, files in a temporary folder; the corpus supplies real documents."""

from __future__ import annotations

import json
import os
import unittest
from pathlib import Path
from unittest import mock

from tests import corpus
from tests.helpers import TempHome

from rag_search import access, api, catalog, inventory, locations, paths as paths_mod
from rag_search.core import chunker


class ChunkerEdgeTests(unittest.TestCase):
    def test_a_paragraph_without_any_break_is_split_by_words_and_loses_none(self):
        words = [f"w{i}" for i in range(3000)]
        chunks = chunker.split_text(" ".join(words), 100, 10)
        self.assertGreater(len(chunks), 10)
        self.assertTrue(all(chunker.est_tokens(c) <= 140 for c in chunks), max(map(chunker.est_tokens, chunks)))
        self.assertEqual(set(" ".join(chunks).split()), set(words))

    def test_one_long_sentence_among_short_ones_is_split_on_its_own(self):
        text = "Short one. " + "long " * 400 + "end. Another short one."
        chunks = chunker.split_text(text, 60, 0)
        self.assertGreater(len(chunks), 3)
        self.assertIn("Another short one.", chunks[-1])

    def test_a_table_row_longer_than_a_chunk_is_split_and_the_header_repeats_for_the_rest(self):
        header = "| Item | Note |\n|---|---|\n"
        rows = "| a | short |\n| b | " + "word " * 300 + " |\n| c | short |\n"
        chunks = chunker.split_text(header + rows, 60, 0)
        self.assertGreater(len(chunks), 3)
        self.assertTrue(any(c.startswith("| Item | Note |") for c in chunks[1:]))
        self.assertIn("| c | short |", chunks[-1])

    def test_the_overlap_carries_the_end_of_one_chunk_into_the_next(self):
        paras = [f"Paragraph number {i} has a few words in it." for i in range(30)]
        chunks = chunker.split_text("\n\n".join(paras), 40, 20)
        for a, b in zip(chunks, chunks[1:]):
            self.assertTrue(set(a.split("\n\n")) & set(b.split("\n\n")), "no shared block between neighbours")
        self.assertEqual(chunker.est_tokens(""), 0)


class PathHelperTests(TempHome):
    def test_an_integer_setting_that_is_not_a_number_falls_back_to_its_default(self):
        os.environ["RAG_SEARCH_X_TEST"] = "lots"
        self.addCleanup(os.environ.pop, "RAG_SEARCH_X_TEST", None)
        self.assertEqual(paths_mod.env_int("RAG_SEARCH_X_TEST", 7), 7)
        os.environ["RAG_SEARCH_X_TEST"] = "12"
        self.assertEqual(paths_mod.env_int("RAG_SEARCH_X_TEST", 7), 12)

    def test_experiment_names_are_checked_and_the_default_home_is_the_macs_application_support(self):
        with self.assertRaises(ValueError):
            paths_mod.validate_experiment_name("../escape")
        with mock.patch.object(paths_mod.sys, "platform", "darwin"):
            os.environ.pop("RAG_SEARCH_HOME")
            self.assertEqual(paths_mod.default_home().parts[-3:], ("Library", "Application Support", "rag-search"))

    def test_a_failed_atomic_write_leaves_no_temporary_file_and_the_old_content(self):
        target = self.tmp / "x.json"
        paths_mod.write_json_atomic(target, {"a": 1})
        with mock.patch.object(paths_mod.os, "replace", side_effect=OSError("disk full")), self.assertRaises(OSError):
            paths_mod.write_json_atomic(target, {"a": 2})
        self.assertEqual(json.loads(target.read_text()), {"a": 1})
        self.assertEqual(sorted(p.name for p in self.tmp.glob("x.json*")), ["x.json"])

    def test_source_roots_tell_the_collection_a_file_belongs_to(self):
        loc = self.tmp / "vault"
        loc.mkdir()
        other = self.tmp / "notes"
        roots = paths_mod.SourceRoots((("vault", str(loc)), ("notes", str(other))))
        self.assertEqual(roots.location_names(), ["vault", "notes"])
        self.assertEqual(roots.root_of("vault"), loc)
        self.assertIsNone(roots.root_of("team"))
        self.assertEqual(roots.rel(loc / "p" / "a.md").parts, ("vault", "p", "a.md"))
        with self.assertRaises(ValueError):
            roots.rel(loc)                                      # a location's own folder is not a document
        with self.assertRaises(ValueError):
            roots.rel(self.tmp / "top.md")                      # in no location: there is no default collection
        self.assertIs(paths_mod.SourceRoots.of(roots), roots)
        self.assertEqual(paths_mod.SourceRoots.of(roots.to_dict()), roots)
        with self.assertRaises(TypeError):
            paths_mod.SourceRoots.of(self.tmp)

    def test_asking_the_os_to_fetch_cloud_files_is_harmless_where_it_cannot(self):
        with mock.patch.object(paths_mod.sys, "platform", "darwin"), mock.patch("ctypes.CDLL", side_effect=OSError("no libc")):
            paths_mod.allow_cloud_files()                       # must not raise


class LocationRegistryTests(TempHome):
    def folder(self, name="shared"):
        f = self.tmp / name
        corpus.copy("text/notes.md", f / "notes.md")
        return f

    def test_a_damaged_registry_is_read_as_empty_with_the_reason_and_blocks_changes(self):
        f = self.paths.locations_file
        for content, text in (("{ nope", "Expecting"), ('{"locations": []}', "expected"),
                              ('{"locations": {"a/b": "/x"}}', "bad entry"), ('{"locations": {"ok": ""}}', "bad entry")):
            with self.subTest(content):
                f.write_text(content)
                locs, err = locations._load_file(f)
                self.assertEqual(locs, {})
                self.assertIn(text, err)
        f.write_text("{ nope")
        with self.assertRaises(locations.LocationError) as cm:
            locations.add(self.paths, "x", str(self.folder()))
        self.assertIn("cannot change locations", str(cm.exception))
        with self.assertRaises(locations.LocationError):
            locations.remove_entry(self.paths, "x")
        f.unlink()
        f.mkdir()                                                # not a file at all
        self.assertIn("locations.json", locations._load_file(f)[1])

    def test_what_cannot_be_registered_says_why(self):
        shared = self.folder()
        locations.add(self.paths, "shared", str(shared))
        cases = [(("", str(self.tmp / "none")), None),
                 (("x", ""), "give the folder"),
                 (("shared", str(self.folder("other"))), "already registered"),
                 (("again", str(shared / "sub")), None),
                 (("a/b", str(self.folder("e"))), "not a usable collection name"),
                 (("home", str(self.paths.home)), "overlaps")]
        (shared / "sub").mkdir()
        for (name, folder), text in cases:
            with self.subTest(name or "unnamed", folder=Path(folder).name), self.assertRaises(locations.LocationError) as cm:
                locations.add(self.paths, name, folder)
            if text:
                self.assertIn(text, str(cm.exception))

    def test_an_overlap_with_another_location_and_a_relative_folder(self):
        outer = self.folder("outer")
        locations.add(self.paths, "outer", str(outer))
        inner = outer / "inner"
        inner.mkdir()
        with self.assertRaises(locations.LocationError) as cm:
            locations.add(self.paths, "inner", str(inner))
        self.assertIn("overlaps the folder of", str(cm.exception))
        cwd = Path.cwd()
        os.chdir(self.tmp)
        self.addCleanup(os.chdir, cwd)
        (self.tmp / "shared-relative").mkdir()
        res = locations.add(self.paths, "rel", "shared-relative")
        self.assertEqual(Path(res["folder"]), (self.tmp / "shared-relative").resolve())
        self.assertEqual(locations.remove_entry(self.paths, "REL"), "rel")
        self.assertEqual(locations.remove_entry(self.paths, "rel"), "")

    def test_targets_are_resolved_against_the_locations_and_refused_when_unreachable(self):
        shared = self.folder()
        locations.add(self.paths, "shared", str(shared))
        self.assertEqual(locations.resolve_target(self.paths, ""), (None, ""))
        self.assertEqual(locations.resolve_target(self.paths, "SHARED")[1], "shared")           # case-insensitive
        self.assertEqual(locations.resolve_target(self.paths, "shared/notes.md")[1], "")
        with self.assertRaises(locations.LocationError) as cm:
            locations.resolve_target(self.paths, "shared/missing.md")
        self.assertIn("not found", str(cm.exception))
        with self.assertRaises(locations.LocationError) as cm:
            locations.resolve_target(self.paths, str(self.tmp))
        self.assertIn("must be inside a registered location", str(cm.exception))
        with self.assertRaises(locations.LocationError) as cm:
            locations.resolve_target(self.paths, "elsewhere/x")
        self.assertIn("neither a registered location", str(cm.exception))
        with mock.patch.object(locations, "reachable", return_value=False):
            for target in ("shared", str(shared / "notes.md")):
                with self.assertRaises(locations.LocationError) as cm:
                    locations.resolve_target(self.paths, target)
                self.assertIn("not reachable", str(cm.exception))

    def test_nothing_registered_is_refused_with_one_message(self):
        with self.assertRaises(locations.LocationError) as cm:
            locations.resolve_target(self.paths, "anything")
        self.assertEqual(str(cm.exception), locations.NO_LOCATIONS)
        with self.assertRaises(locations.LocationError) as cm:
            locations.plan_scan(self.paths)
        self.assertEqual(str(cm.exception), locations.NO_LOCATIONS)

    def test_the_dashboards_source_listing_always_answers_and_a_source_file_can_be_read(self):
        with mock.patch.object(locations, "load", side_effect=RuntimeError("boom")):
            res = locations.sources(self.paths)
        self.assertEqual((res["error"], res["locations"]), ("boom", []))
        f = self.tmp / "a.bin"
        f.write_bytes(b"\x00\x01")
        self.assertEqual(locations.read_source(f), b"\x00\x01")


class InventoryStateTests(TempHome):
    def test_a_collection_is_a_location_or_an_import_and_nothing_else(self):
        corpus.copy("text/notes.md", self.sdir / "team" / "notes.md")
        corpus.copy("text/readme.txt", self.sdir / "keep" / "readme.txt")
        self.index()
        self.publish()
        locations.remove_entry(self.paths, "team")           # the index is left behind
        rows = {r["collection"]: r["kind"] for r in access.overview(self.paths)["collections"]}
        self.assertEqual(rows, {"keep": "location"})
        self.assertEqual(sorted(catalog.known_names(self.paths)), ["keep"])
        with self.assertRaises(inventory.InventoryError):
            inventory.collection_info(self.paths, "team")

    def test_a_collection_with_documents_but_nothing_indexed_yet_says_what_to_run(self):
        corpus.copy("text/notes.md", self.sdir / "team" / "notes.md")
        self.register_tree()
        info = inventory.collection_info(self.paths, "team")
        self.assertEqual(info["state"], "not_indexed")
        self.assertIn("rag-search index new", info["state_detail"])


class ApiRefusalTests(TempHome):
    def test_requests_that_cannot_be_answered_say_so_in_the_protocol(self):
        self.assertEqual(api.conversion_documents(self.paths, job_id="20260101-000000-none")["code"], "bad_request")
        self.assertEqual(api.conversion_documents(self.paths)["documents"]["total"], 0)
        self.assertEqual(api.location_remove(self.paths, "nope")["code"], "bad_request")
        self.assertEqual(api.collection_import(self.paths, str(self.tmp / "nope.rag.tgz"))["code"], "bad_request")
        gen = api.index_follow(self.paths)
        self.assertEqual(next(gen)["code"], "unavailable")

    def test_publishing_after_a_run_reports_a_failure_instead_of_raising(self):
        from rag_search import publish

        with mock.patch.object(publish, "publish", side_effect=RuntimeError("disk full")):
            out = api.publish_and_reload(self.paths)
        self.assertEqual(out["publish"], {"changed": False, "error": "RuntimeError: disk full"})

    def test_a_page_that_cannot_be_drawn_is_a_clean_error(self):
        corpus.copy("pdf/damaged.pdf", self.sdir / "c" / "damaged.pdf")
        self.register_tree()
        meta = self.paths.index / "c" / "damaged" / "index.meta.json"
        meta.parent.mkdir(parents=True)
        meta.write_text(json.dumps({"src_path": str(self.sdir / "c" / "damaged.pdf")}))
        r = api.conversion_page_image(self.paths, "c", "damaged", 1)
        self.assertFalse(r["ok"])
        self.assertIn("cannot render", r["error"])


if __name__ == "__main__":
    unittest.main()
