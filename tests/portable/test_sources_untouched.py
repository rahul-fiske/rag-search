"""No rag-search operation changes or deletes a source document.

Every kind of collection (a registered folder; an imported collection, which has no source folder) is made read-only and photographed (content, mode, size, mtime, the folder listing); then every
operation that creates, rebuilds, prunes, exports, imports, deletes or unregisters anything runs, through the
same API calls the CLI and the dashboard use; afterwards the photograph must be identical.  For a non-root
user any write attempt would fail loudly on its own; the comparison also covers running as root.

There is no exception: the one command that used to write into a source folder (``convert-legacy``) was
removed in 0.9.18, and ``test_no_command_or_module_can_convert_a_source_in_place`` keeps it that way.
"""

from __future__ import annotations

import unittest
from pathlib import Path

from tests.portable.test_enhancements import DOC, NOTE, Base, _snapshot
from rag_search import api, lifecycle, locations
from rag_search.core import playground


def _lock(*roots: Path, on: bool) -> None:
    for root in roots:
        if not root.exists():                          # the test's folder is already gone
            continue
        for p in [*root.rglob("*"), root]:
            p.chmod((0o555 if p.is_dir() else 0o444) if on else (0o755 if p.is_dir() else 0o644))


class NothingDeletesASourceTests(Base):
    def setUp(self):
        super().setUp()
        self.policies = self.make_location("policies", {"auth.md": DOC, "sub/roles.md": "# Roles\n\nAuditors read ledgers.\n"})
        self.vault = self.make_location("vault", {"note.md": NOTE, "deep/er/x.md": DOC})
        self.garden = self.make_location("garden", {"g.md": "# G\n\nTomatoes and basil.\n"})
        self.roots = (self.policies, self.vault, self.garden)
        self.addCleanup(_lock, *self.roots, on=False)

    def photograph(self):
        return _snapshot(*self.roots), sorted(str(p) for r in self.roots for p in r.rglob("*"))

    def test_indexing_pruning_exporting_importing_and_every_delete_leave_the_sources_alone(self):
        self.plan_index()
        self.publish()
        _lock(*self.roots, on=True)
        before = self.photograph()

        self.plan_index(wipe=True)                                  # full rebuild
        self.plan_index(force_md=True)                              # re-convert everything
        self.plan_index("vault")                                    # one collection
        self.plan_index(str(self.vault / "note.md"))                # one file
        out = self.tmp / "exports"
        for name in ("policies", "vault"):
            self.assertTrue(api.collection_export(self.paths, name, str(out))["ok"])
        self.assertEqual(self.photograph(), before)

        # unregister a location and delete its index (`location remove`, the dashboard's Remove)
        arch = next(out.glob("vault*.rag.tgz"))
        res = api.location_remove(self.paths, "vault")
        self.assertTrue(res["ok"], res)
        self.assertNotIn("vault", locations.names(self.paths))
        self.assertEqual(self.photograph(), before)

        # an imported collection has no source folder: importing and deleting it touches nothing
        self.assertTrue(api.collection_import(self.paths, str(arch))["ok"])
        self.assertTrue(api.collection_delete(self.paths, "vault")["ok"])
        self.assertEqual(self.photograph(), before)

        # delete a registered collection's index without unregistering it: the folder stays, the
        # registration stays and the next run builds it again
        self.assertTrue(api.collection_delete(self.paths, "policies")["ok"])
        self.assertTrue((self.policies / "auth.md").is_file())
        self.assertEqual(self.photograph(), before)
        self.assertGreaterEqual(self.plan_index()["indexed"], 2)
        self.assertEqual(self.photograph(), before)

    def test_an_index_nobody_owns_is_cleaned_without_touching_any_source(self):
        self.plan_index()
        _lock(*self.roots, on=True)
        before = self.photograph()
        locations.remove_entry(self.paths, "garden")                # registration gone, its index remains
        self.plan_index()                                           # a full run removes the orphan index
        self.assertFalse((self.paths.index / "garden").exists())
        self.assertEqual(self.photograph(), before)

    def test_a_source_file_that_vanishes_is_pruned_but_never_recreated_or_touched(self):
        self.plan_index()
        gone = self.vault / "deep" / "er" / "x.md"
        gone.unlink()
        _lock(*self.roots, on=True)
        before = self.photograph()
        s = self.plan_index()
        self.assertTrue(s["removed"])
        self.assertFalse(gone.exists())
        self.assertEqual(self.photograph(), before)

    def test_deleting_with_a_hostile_name_cannot_reach_outside_the_workspace(self):
        self.plan_index()
        outside = self.tmp / "outside"
        outside.mkdir()
        (outside / "keep.txt").write_text("keep")
        for bad in ("../outside", "..", ".", "/", str(outside), "policies/../../outside", ""):
            self.assertFalse(api.collection_delete(self.paths, bad)["ok"], bad)
            self.assertFalse(api.location_remove(self.paths, bad)["ok"], bad)
        self.assertEqual((outside / "keep.txt").read_text(), "keep")
        self.assertTrue((self.policies / "auth.md").is_file())

    def test_a_playground_experiment_reads_its_source_where_it_is_and_removing_it_keeps_it(self):
        _lock(*self.roots, on=True)
        before = self.photograph()
        playground.create_experiment(self.paths, "trial", sources=[str(self.vault)])
        self.assertEqual(self.photograph(), before)
        playground.remove_experiment(self.paths, "trial")
        self.assertEqual(self.photograph(), before)
        self.assertTrue((self.vault / "note.md").is_file())

    def test_the_delete_helpers_refuse_what_is_not_theirs(self):
        with self.assertRaises(lifecycle.LifecycleError):
            lifecycle.delete_collection(self.paths, "nothere")
        with self.assertRaises(playground.PlaygroundError):
            playground.remove_experiment(self.paths, "nothere")

    def test_no_command_or_module_can_convert_a_source_in_place(self):
        import importlib.util

        from tests.portable.test_cli_api import run

        self.assertIsNone(importlib.util.find_spec("rag_search.core.legacy_convert"))
        rc, out, err = run("convert-legacy", str(self.vault))
        self.assertEqual(rc, 2)                                 # an unknown command
        src = Path(api.__file__).parent
        for f in src.rglob("*.py"):
            text = f.read_text(encoding="utf-8")
            self.assertNotIn("soffice", text, f)
            self.assertNotIn("libreoffice", text.lower(), f)


if __name__ == "__main__":
    unittest.main()
