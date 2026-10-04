"""0.8.0: source locations, deleted/unreachable sources, collection export/import/delete, and
the foundation fixes from the architecture review (locking, caching, name resolution, search
concurrency, model loading)."""

from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import stat
import tarfile
import threading
import unittest
from pathlib import Path
from unittest import mock

from tests.helpers import FakeEmbedder, TempHome
from tests.portable.test_cli_api import run
from tests.portable.test_ui import UiBase
from rag_search import (access, api, bundle, catalog, descriptions, lifecycle, locations, models,
                        policy, publish)
from rag_search.core import indexer
from rag_search.paths import (ALL_DIR, META_FILE, ensure_dirs, file_lock, get_paths, mirror_rel,
                              read_json)

DOC = ("# Title\n\n<!-- page 1 -->\n\nPublic key authentication protects every account. "
       "Role based access control limits each user.\n")
NOTE = "# Note\n\nThe vault holds gardening notes about tomatoes and basil.\n"


class Base(TempHome):
    def plan_index(self, raw: str = "", **kw):
        plan = locations.plan_scan(self.paths, raw)
        return indexer.run_plan(self.paths, plan, jobs=1, embedder=FakeEmbedder(), **kw)

    def make_location(self, name: str = "vault", files: dict[str, str] | None = None) -> Path:
        folder = self.tmp / "elsewhere" / name
        folder.mkdir(parents=True, exist_ok=True)
        for rel, text in (files or {"note.md": NOTE}).items():
            p = folder / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(text, encoding="utf-8")
        locations.add(self.paths, name, str(folder))
        return folder

    def search(self, query: str, colls=None):
        e = self.engine()
        return e.search(query, 5, colls)


# ── Phase 0: indexing correctness ──────────────────────────────────────────

class DeletedAndCollidingSourcesTests(Base):
    def test_deleted_source_loses_its_markup_and_index(self):
        self.write_doc("security/auth.md", DOC)
        gone = self.write_doc("security/old.md", "# Old\n\nObsolete policy text.\n")
        s = self.plan_index()
        self.assertEqual(s["indexed"], 2)
        idx = self.paths.index / "security" / "old"
        md = self.paths.markup / "security" / "old.md"
        self.assertTrue(idx.is_dir() and md.is_file())
        gone.unlink()
        s = self.plan_index()
        self.assertEqual(s["removed"], ["security/old"])
        self.assertFalse(idx.exists())
        self.assertFalse(md.exists())
        self.assertFalse(md.with_name("old.md.sha256").exists())
        merged = read_json(self.paths.index / "security" / ALL_DIR / "merge.manifest.json")
        self.assertEqual(merged["docs"], ["auth"])

    def test_a_whole_deleted_collection_disappears(self):
        self.write_doc("security/auth.md", DOC)
        self.write_doc("hr/leave.md", "# Leave\n\nSixteen weeks.\n")
        self.plan_index()
        self.publish()
        shutil.rmtree(self.paths.docs / "hr")
        self.plan_index()
        self.assertFalse((self.paths.index / "hr").exists())
        self.assertFalse((self.paths.markup / "hr").exists())
        self.publish()          # docs/hr is gone, so publishing may drop it
        self.assertEqual(catalog.collection_names(catalog.live_catalog(self.paths)), ["security"])

    def test_a_scoped_run_prunes_only_what_it_fully_read(self):
        self.write_doc("security/auth.md", DOC)
        other = self.write_doc("hr/leave.md", "# Leave\n\nSixteen weeks.\n")
        self.plan_index()
        other.unlink()
        s = self.plan_index("security")         # hr was not read: nothing concluded about it
        self.assertEqual(s["removed"], [])
        self.assertTrue((self.paths.index / "hr" / "leave").is_dir())
        s = self.plan_index("hr")
        self.assertEqual(s["removed"], ["hr/leave"])

    def test_pruning_a_document_keeps_a_nested_namesake(self):
        # a.md and a folder a/ with b.md: b's index lives inside a's index folder
        a = self.write_doc("c/a.md", DOC)
        self.write_doc("c/a/b.md", "# B\n\nNested document about turbines.\n")
        self.plan_index()
        a.unlink()
        s = self.plan_index()
        self.assertEqual(s["removed"], ["c/a"])
        self.assertTrue((self.paths.index / "c" / "a" / "b" / META_FILE).is_file())
        self.assertEqual(read_json(self.paths.index / "c" / ALL_DIR / "merge.manifest.json")["docs"],
                         ["a/b"])

    def test_a_single_file_run_cannot_replace_another_documents_index(self):
        pdf_like = self.write_doc("r/report.md", DOC)
        self.plan_index()
        before = read_json(self.paths.index / "r" / "report" / META_FILE)
        twin = self.write_doc("r/report.txt", "Completely different text about ferries.")
        s = self.plan_index(str(twin))
        self.assertEqual(s["indexed"], 0)
        self.assertIn("already indexed", s["errors"][0]["message"])
        after = read_json(self.paths.index / "r" / "report" / META_FILE)
        self.assertEqual(before["src_path"], after["src_path"])
        self.assertEqual(after["src_path"], str(pdf_like))
        # a full run keeps the document that owned the name, and reports the other
        s = self.plan_index()
        self.assertEqual(read_json(self.paths.index / "r" / "report" / META_FILE)["src_path"],
                         str(pdf_like))
        self.assertTrue(any("report.txt" in e["src"] for e in s["errors"]))

    def test_a_moved_docs_folder_is_not_mistaken_for_deleted_documents(self):
        self.write_doc("security/auth.md", DOC)
        self.plan_index()
        new_docs = self.tmp / "moved-docs"
        shutil.move(str(self.paths.docs), str(new_docs))
        os.environ["RAG_SEARCH_DOCS"] = str(new_docs)
        self.paths = get_paths()
        s = self.plan_index()
        self.assertEqual((s["indexed"], s["skipped_fresh"], s["removed"]), (0, 1, []))
        meta = read_json(self.paths.index / "security" / "auth" / META_FILE)
        self.assertEqual(meta["src_path"], str(new_docs / "security" / "auth.md"))
        self.assertEqual(read_json(self.paths.index / "security" / ALL_DIR / "merge.manifest.json")
                         ["docs"], ["auth"])

    def test_an_unmounted_docs_folder_freezes_everything(self):
        outside = self.tmp / "volume" / "docs"
        outside.mkdir(parents=True)
        (outside / "security").mkdir()
        (outside / "security" / "auth.md").write_text(DOC)
        os.environ["RAG_SEARCH_DOCS"] = str(outside)
        self.paths = get_paths()
        self.plan_index()
        self.publish()
        shutil.rmtree(self.tmp / "volume")              # the drive is unplugged
        ensure_dirs(self.paths)                          # must not create a stand-in folder
        self.assertFalse(outside.exists())
        s = self.plan_index()
        self.assertEqual(s["removed"], [])
        self.assertTrue(s["unreachable"])
        self.assertIn("security", s["frozen"])
        self.assertTrue((self.paths.index / "security" / ALL_DIR / "nodes.json").is_file())
        self.publish()
        self.assertEqual(catalog.collection_names(catalog.live_catalog(self.paths)), ["security"])


# ── Phase 0: names, metadata stores, search ───────────────────────────────

class NamesAndStoresTests(Base):
    def test_concurrent_access_changes_are_never_lost(self):
        errors = []

        def restrict(i):
            try:
                access.restrict(self.paths, f"coll{i}", ["claude"])
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=restrict, args=(i,)) for i in range(12)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(errors, [])
        rules, _ = policy.load_rules(self.paths)
        self.assertEqual(len(rules.by_name), 12)
        self.assertEqual(stat.S_IMODE(self.paths.access_file.stat().st_mode), 0o600)

    def test_file_lock_times_out_while_held(self):
        with file_lock(self.paths.access_file):
            def other():
                with self.assertRaises(TimeoutError):
                    with file_lock(self.paths.access_file, timeout=0.2):
                        pass
            t = threading.Thread(target=other)
            t.start()
            t.join()

    def test_cached_stores_notice_changes(self):
        self.assertEqual(policy.current_rules(self.paths).by_name, {})
        policy.save_rules(self.paths, {"hr": ["claude"]})
        self.assertEqual(policy.current_rules(self.paths).by_name, {"hr": ["claude"]})
        descriptions.set_description(self.paths, "hr", "People matters")
        self.assertEqual(descriptions.cached_descriptions(self.paths), {"hr": "People matters"})
        descriptions.set_description(self.paths, "HR", "Leave and pay")
        self.assertEqual(descriptions.cached_descriptions(self.paths), {"hr": "Leave and pay"})

    def test_one_name_resolver_for_every_store(self):
        self.write_doc("Manuals/a.md", DOC)
        self.assertEqual(catalog.canonical_name(self.paths, "manuals"), "Manuals")
        self.assertEqual(descriptions.set_description(self.paths, "MANUALS", "x")["collection"],
                         "Manuals")
        self.assertEqual(access.restrict(self.paths, "manuals", ["claude"])["collection"], "Manuals")
        with self.assertRaises(ValueError):
            catalog.canonical_name(self.paths, "../etc")

    def test_search_counts_a_repeated_collection_once(self):
        self.write_doc("security/auth.md", DOC)
        self.write_doc("hr/leave.md", "# Leave\n\nAccess to leave records needs a manager role.\n")
        self.index()
        self.publish()
        e = self.engine()
        once = e.search("role based access", 5, ["security", "hr"])
        twice = e.search("role based access", 5, ["security", "hr", "security"])
        self.assertEqual([(r["collection"], r["score"]) for r in once["results"]],
                         [(r["collection"], r["score"]) for r in twice["results"]])
        self.assertEqual(twice["timing"]["collections"], 2)

    def test_concurrent_searches_get_consistent_results(self):
        self.write_doc("security/auth.md", DOC)
        self.index()
        self.publish()
        e = self.engine()
        expected = e.search("public key authentication", 3)["results"]
        out, errors = [], []

        def go():
            try:
                for _ in range(20):
                    out.append(e.search("public key authentication", 3)["results"])
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=go) for _ in range(6)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(errors, [])
        self.assertTrue(all(r == expected for r in out))

    def test_cli_describe_goes_through_the_api_and_honours_client(self):
        self.write_doc("personal/diary.md", DOC)
        self.index()
        self.publish()
        policy.save_rules(self.paths, {"personal": ["claude"]})
        rc, out, err = run("describe", "--client", "agent", "personal", "my diary")
        self.assertNotEqual(rc, 0)
        self.assertIn("unknown collection", err)
        self.assertEqual(descriptions.get_description(self.paths, "personal"), "")
        rc, out, err = run("describe", "personal", "my diary")
        self.assertEqual(rc, 0, err)
        self.assertEqual(descriptions.get_description(self.paths, "personal"), "my diary")
        # the administrator may describe a collection that is not published yet
        self.write_doc("drafts/x.md", DOC)
        rc, out, err = run("describe", "drafts", "work in progress")
        self.assertEqual(rc, 0, err)


# ── Phase 0: models ───────────────────────────────────────────────────────

class ModelLoadingTests(unittest.TestCase):
    def setUp(self):
        from rag_search.core import embedding
        self.embedding = embedding
        saved = {k: os.environ.pop(k, None) for k in ("RAG_SEARCH_RERANK_BATCH",
                                                       "RAG_SEARCH_RERANK_MAX_LEN", "HF_HUB_CACHE")}
        self.addCleanup(lambda: [os.environ.pop(k, None) or (v is not None and os.environ.update({k: v}))
                                 for k, v in saved.items()])

    def test_the_default_reranker_honours_its_tunables(self):
        os.environ["RAG_SEARCH_RERANK_BATCH"] = "3"
        os.environ["RAG_SEARCH_RERANK_MAX_LEN"] = "512"
        r = self.embedding.Reranker("BAAI/bge-reranker-v2-m3")
        self.assertEqual((r.batch_size, r.max_length), (3, 512))
        os.environ.pop("RAG_SEARCH_RERANK_BATCH")
        os.environ.pop("RAG_SEARCH_RERANK_MAX_LEN")
        r = self.embedding.Reranker("BAAI/bge-reranker-v2-m3")
        self.assertEqual((r.batch_size, r.max_length), (8, 1024))

    def test_a_cached_model_loads_without_the_network(self):
        calls = []

        def factory(**kw):
            calls.append(kw)
            return "model"

        with mock.patch.object(models, "cache_state", return_value={"cached": True}):
            self.assertEqual(self.embedding._load_model("x/y", factory), "model")
        self.assertEqual(calls, [{"local_files_only": True}])

    def test_falls_back_when_the_library_lacks_the_option(self):
        calls = []

        def factory(**kw):
            calls.append(kw)
            if kw:
                raise TypeError("unexpected keyword argument 'local_files_only'")
            return "model"

        with mock.patch.object(models, "cache_state", return_value={"cached": True}):
            self.assertEqual(self.embedding._load_model("x/y", factory), "model")
        self.assertEqual(calls, [{"local_files_only": True}, {}])

    def test_a_failed_download_is_explained(self):
        class ConnectionError(Exception):  # noqa: A001 - mimics requests' name
            pass

        def factory(**kw):
            raise ConnectionError("Max retries exceeded")

        with mock.patch.object(models, "cache_state", return_value={"cached": False}):
            with self.assertRaises(RuntimeError) as cm:
                self.embedding._load_model("x/y", factory)
        self.assertIn("cannot reach huggingface.co", str(cm.exception))

    def test_the_torch_load_relaxation_is_undone_after_the_load(self):
        import sys
        import types

        mod = types.ModuleType("transformers.fake_scope_test")

        def refuse():
            raise ValueError("unsafe")

        mod.check_torch_load_is_safe = refuse
        with mock.patch.dict(sys.modules, {"transformers.fake_scope_test": mod}), \
                mock.patch.object(self.embedding, "_torch_version", return_value=(2, 2)):
            with self.embedding.trusted_load("BAAI/bge-m3") as relaxed:
                self.assertTrue(relaxed)
                mod.check_torch_load_is_safe()           # relaxed while loading
            with self.assertRaises(ValueError):
                mod.check_torch_load_is_safe()           # and checked again afterwards

    def test_cached_revision_reads_refs_main(self):
        import tempfile
        d = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, d, True)
        os.environ["HF_HUB_CACHE"] = str(d)
        self.assertEqual(models.cached_revision("BAAI/bge-m3"), "")
        ref = d / "models--BAAI--bge-m3" / "refs"
        ref.mkdir(parents=True)
        (ref / "main").write_text("5617a9f61b028005a4858fdac845db406aefb181\n")
        self.assertEqual(models.cached_revision("BAAI/bge-m3"),
                         "5617a9f61b028005a4858fdac845db406aefb181")


# ── Phase 1: source locations ─────────────────────────────────────────────

class LocationTests(Base):
    def test_validation(self):
        ok = self.tmp / "ok"
        ok.mkdir()
        bad = [("default", str(ok)), ("x", str(self.tmp / "missing")),
               ("x", str(self.paths.docs)), ("x", str(self.paths.home)), ("../x", str(ok))]
        for name, folder in bad:
            with self.assertRaises(locations.LocationError, msg=(name, folder)):
                locations.add(self.paths, name, folder)
        self.write_doc("notes/a.md", DOC)
        with self.assertRaises(locations.LocationError):      # name taken by a docs folder
            locations.add(self.paths, "Notes", str(ok))
        locations.add(self.paths, "vault", str(ok))
        (ok / "inner").mkdir()
        with self.assertRaises(locations.LocationError):      # overlaps another location
            locations.add(self.paths, "inner", str(ok / "inner"))
        with self.assertRaises(locations.LocationError):      # already registered
            locations.add(self.paths, "VAULT", str(self.tmp))

    def test_a_location_is_indexed_and_searched_as_one_collection(self):
        folder = self.make_location("vault", {"note.md": NOTE, "sub/deep.md": DOC})
        self.write_doc("security/auth.md", DOC)
        s = self.plan_index()
        self.assertEqual(s["indexed"], 3)
        self.assertEqual(sorted(s["covered"]), ["default", "security", "vault"])
        self.publish()
        cat = catalog.live_catalog(self.paths)
        self.assertEqual(sorted(catalog.collection_names(cat)), ["security", "vault"])
        hits = self.search("tomatoes basil gardening", ["vault"])["results"]
        self.assertEqual(hits[0]["collection"], "vault")
        self.assertEqual(mirror_rel(folder / "sub" / "deep.md", locations.source_roots(self.paths)),
                         Path("vault/sub/deep.md"))
        # a path that starts with the location's name indexes inside it
        target, coll = locations.resolve_target(self.paths, "vault/sub")
        self.assertEqual((target, coll), (folder.resolve() / "sub", ""))
        self.assertEqual(locations.resolve_target(self.paths, "vault"), (folder.resolve(), "vault"))

    def test_an_unreachable_location_keeps_its_index_and_stays_searchable(self):
        folder = self.make_location("vault", {"note.md": NOTE, "keep.md": DOC})
        self.plan_index()
        self.publish()
        hidden = folder.with_name("vault-unplugged")
        folder.rename(hidden)                         # drive unplugged / share down
        s = self.plan_index()
        self.assertEqual(s["unreachable"], ["vault"])
        self.assertEqual(s["removed"], [])
        self.assertTrue((self.paths.index / "vault" / "note" / META_FILE).is_file())
        self.assertTrue((self.paths.markup / "vault" / "note.md").is_file())
        self.publish()
        self.assertIn("vault", catalog.collection_names(catalog.live_catalog(self.paths)))
        with self.assertRaises(locations.LocationError):
            locations.resolve_target(self.paths, "vault")
        hidden.rename(folder)                         # back again: business as usual
        (folder / "note.md").unlink()
        s = self.plan_index()
        self.assertEqual(s["removed"], ["vault/note"])

    def test_a_docs_folder_with_a_locations_name_is_ignored(self):
        self.make_location("vault")
        # created after registration: the location wins, the folder is reported and left alone
        self.write_doc("vault/sneaky.md", DOC)
        plan = locations.plan_scan(self.paths)
        self.assertEqual(plan.shadowed, ["vault"])
        self.assertFalse(any("sneaky" in str(s) for s in plan.sources))

    def test_cli_location_commands(self):
        folder = self.tmp / "notes-elsewhere"
        folder.mkdir()
        (folder / "n.md").write_text(NOTE)
        rc, out, err = run("location", "add", "garden", str(folder))
        self.assertEqual(rc, 0, err)
        rc, out, err = run("location", "list")
        self.assertIn("garden", out)
        self.assertIn("ok", out)
        rc, out, err = run("location", "remove", "garden")      # not a terminal: needs --yes
        self.assertNotEqual(rc, 0)
        rc, out, err = run("location", "remove", "garden", "--yes")
        self.assertEqual(rc, 0, err)
        self.assertEqual(locations.names(self.paths), [])
        self.assertTrue((folder / "n.md").exists())


# ── sources are never written ─────────────────────────────────────────────

def _snapshot(*roots: Path) -> dict[str, tuple]:
    out = {}
    for root in roots:
        for p in sorted(root.rglob("*")):
            st = p.lstat()
            out[str(p)] = (st.st_mode, st.st_size, st.st_mtime_ns,
                           hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else "")
    return out


class SourcesAreReadOnlyTests(Base):
    def test_nothing_rag_search_does_changes_a_source_folder(self):
        self.write_doc("security/auth.md", DOC)
        gone = self.write_doc("security/old.md", "# Old\n\nObsolete.\n")
        folder = self.make_location("vault", {"note.md": NOTE, "sub/x.md": DOC})
        self.plan_index()
        self.publish()
        gone.unlink()
        roots = (self.paths.docs, folder)
        # read-only (for a non-root user any write attempt fails loudly), and compared byte for
        # byte, mode and mtime afterwards (which also covers running the tests as root)
        for root in roots:
            for p in [*root.rglob("*"), root]:
                p.chmod(0o555 if p.is_dir() else 0o444)
        try:
            before = _snapshot(*roots)
            s = self.plan_index()                               # prunes old.md's index
            self.assertEqual(s["removed"], ["security/old"])
            self.assertEqual(s["errors"], [])
            self.plan_index(wipe=True)                          # full rebuild
            self.publish()
            bundle.export_collection(self.paths, "vault", self.tmp / "exports")
            lifecycle.delete_collection(self.paths, "vault")     # unregisters, deletes the index
            self.assertEqual(_snapshot(*roots), before)
        finally:
            for root in roots:
                for p in [*root.rglob("*"), root]:
                    p.chmod(0o755 if p.is_dir() else 0o644)


# ── Phase 2: export / import ──────────────────────────────────────────────

class BundleTests(Base):
    def _exported(self, name="security") -> Path:
        self.write_doc(f"{name}/auth.md", DOC)
        self.write_doc(f"{name}/sub/roles.md", "# Roles\n\nAuditors may read every ledger.\n")
        self.plan_index()
        self.publish()
        descriptions.set_description(self.paths, name, "Security policies")
        policy.save_rules(self.paths, {name: ["claude"]})
        res = bundle.export_collection(self.paths, name, self.tmp / "out")
        return Path(res["file"])

    def _other_home(self) -> None:
        os.environ["RAG_SEARCH_HOME"] = str(self.tmp / "home2")
        self.paths = get_paths()
        ensure_dirs(self.paths)

    def test_round_trip(self):
        f = self._exported()
        self.assertTrue(f.name.endswith(".rag.tgz"))
        with tarfile.open(f) as tf:
            names = tf.getnames()
            manifest = json.loads(tf.extractfile("manifest.json").read())
            nodes = json.loads(tf.extractfile("index/_all/nodes.json").read())
            meta = json.loads(tf.extractfile("index/docs/sub/roles/index.meta.json").read())
        self.assertIn("markup/sub/roles.md", names)
        self.assertEqual(manifest["model"], "BAAI/bge-m3")
        self.assertEqual((manifest["documents"], manifest["dim"]), (2, 64))
        self.assertEqual(manifest["description"], "Security policies")
        self.assertNotIn(str(self.tmp), json.dumps(nodes))       # no local paths travel
        self.assertNotIn(str(self.tmp), json.dumps(meta))
        self.assertEqual(meta["src_path"], "roles.md")

        self._other_home()
        r = api.collection_import(self.paths, str(f))
        self.assertTrue(r["ok"], r)
        self.assertEqual(r["result"]["access"], "everyone")     # rules did not travel
        self.assertTrue(r["publish"]["changed"])
        cat = catalog.live_catalog(self.paths)
        c = cat["collections"][0]
        self.assertEqual((c["collection"], c["origin"], len(c["documents"])),
                         ("security", "imported", 2))
        self.assertEqual(descriptions.get_description(self.paths, "security"), "Security policies")
        hits = self.search("auditors ledger")["results"]
        self.assertEqual(hits[0]["source"], "roles.md")
        g = api.local_grep(self.paths, "Auditors", None, 1, 5, "cli")
        self.assertEqual(g["result"]["matches"][0]["collection"], "security")

        # indexing never touches it, and refuses to target it
        self.write_doc("mine/a.md", DOC)
        s = self.plan_index()
        self.assertIn({"collection": "security", "skipped": True, "imported": True, "docs": 2,
                       "nodes": c["chunks"]}, s["collections"])
        self.assertTrue((self.paths.index / "security" / ALL_DIR / "nodes.json").is_file())
        with self.assertRaises(locations.LocationError) as cm:
            locations.resolve_target(self.paths, "security")
        self.assertIn("imported collection", str(cm.exception))
        self.plan_index(wipe=True)
        self.assertTrue((self.paths.index / "security" / ALL_DIR / "nodes.json").is_file())

    def test_model_must_match(self):
        f = self._exported()
        self._other_home()
        os.environ["RAG_SEARCH_MODEL"] = "Qwen/Qwen3-Embedding-0.6B"
        r = api.collection_import(self.paths, str(f))
        self.assertFalse(r["ok"])
        self.assertIn("cannot be imported", r["error"])
        self.assertEqual(locations.workspace_collections(self.paths), [])

    def test_weights_commit_must_match_when_both_sides_know_it(self):
        m = {"model": "BAAI/bge-m3", "model_revision": "a" * 40}
        with mock.patch.object(models, "cached_revision", return_value="b" * 40):
            with self.assertRaises(bundle.BundleError):
                bundle.check_model(m)
        with mock.patch.object(models, "cached_revision", return_value="a" * 40):
            self.assertEqual(bundle.check_model(m)["note"], "")
        with mock.patch.object(models, "cached_revision", return_value=""):
            self.assertIn("could not be compared", bundle.check_model(m)["note"])

    def test_names_are_never_silently_taken(self):
        f = self._exported()
        self._other_home()
        self.write_doc("security/local.md", DOC)                 # a docs folder of that name
        r = api.collection_import(self.paths, str(f))
        self.assertFalse(r["ok"])
        self.assertIn("--as", r["error"])
        r = api.collection_import(self.paths, str(f), as_name="their-security")
        self.assertTrue(r["ok"], r)
        nodes = read_json(self.paths.index / "their-security" / ALL_DIR / "nodes.json")["nodes"]
        self.assertEqual({n["metadata"]["collection"] for n in nodes}, {"their-security"})
        r = api.collection_import(self.paths, str(f), as_name="their-security")
        self.assertFalse(r["ok"])
        self.assertIn("--replace", r["error"])
        r = api.collection_import(self.paths, str(f), as_name="their-security", replace=True)
        self.assertTrue(r["ok"], r)
        self.assertTrue(r["result"]["replaced"])

    def test_damaged_or_hostile_archives_are_refused(self):
        f = self._exported()
        self._other_home()
        with tarfile.open(f) as tf:
            members = {m.name: tf.extractfile(m).read() for m in tf if m.isfile()}

        def write(name, files, extra=None):
            p = self.tmp / name
            with tarfile.open(p, "w:gz") as tf:
                for k, v in files.items():
                    info = tarfile.TarInfo(k)
                    info.size = len(v)
                    tf.addfile(info, io.BytesIO(v))
                if extra:
                    tf.addfile(extra)
            return p

        tampered = dict(members)
        tampered["markup/auth.md"] = b"# changed\n"
        r = api.collection_import(self.paths, str(write("t.tgz", tampered)))
        self.assertIn("damaged", r["error"])

        evil = dict(members)
        evil["../../escape.txt"] = b"x"
        r = api.collection_import(self.paths, str(write("e.tgz", evil)))
        self.assertIn("unsafe path", r["error"])
        self.assertFalse((self.tmp / "escape.txt").exists())

        link = tarfile.TarInfo("index/_all/link")
        link.type, link.linkname = tarfile.SYMTYPE, "/etc/passwd"
        r = api.collection_import(self.paths, str(write("l.tgz", members, link)))
        self.assertIn("only plain files", r["error"])

        r = api.collection_import(self.paths, str(self.tmp / "nope.tgz"))
        self.assertIn("no such file", r["error"])
        self.assertEqual(locations.workspace_collections(self.paths), [])

    def test_cli_export_import(self):
        self.write_doc("security/auth.md", DOC)
        self.plan_index()
        self.publish()
        rc, out, err = run("collection", "export", "security", "-o", str(self.tmp / "x.rag.tgz"))
        self.assertEqual(rc, 0, err)
        self.assertIn("exported 'security'", out)
        self._other_home()
        rc, out, err = run("collection", "import", str(self.tmp / "x.rag.tgz"))
        self.assertEqual(rc, 0, err)
        self.assertIn("imported 'security'", out)


# ── Phase 3: delete ───────────────────────────────────────────────────────

class DeleteTests(Base):
    def test_delete_an_imported_collection(self):
        self.write_doc("security/auth.md", DOC)
        self.plan_index()
        self.publish()
        f = bundle.export_collection(self.paths, "security", self.tmp)["file"]
        os.environ["RAG_SEARCH_HOME"] = str(self.tmp / "home2")
        self.paths = get_paths()
        api.collection_import(self.paths, f)
        access.restrict(self.paths, "security", ["claude"])
        r = api.collection_delete(self.paths, "SECURITY")
        self.assertTrue(r["ok"], r)
        self.assertEqual((r["result"]["kind"], r["result"]["sources_remain"]), ("imported", False))
        self.assertEqual(catalog.collection_names(catalog.live_catalog(self.paths)), [])
        self.assertFalse((self.paths.index / "security").exists())
        self.assertFalse((self.paths.markup / "security").exists())
        # only the workspace goes: the access rule stays (it applies again if it comes back)
        self.assertEqual(policy.load_rules(self.paths)[0].by_name, {"security": ["claude"]})

    def test_a_docs_collection_is_deleted_and_rebuilt_by_the_next_run(self):
        src = self.write_doc("security/auth.md", DOC)
        self.plan_index()
        self.publish()
        access.restrict(self.paths, "security", ["claude"])
        descriptions.set_description(self.paths, "security", "auth docs")
        r = api.collection_delete(self.paths, "security")
        self.assertTrue(r["ok"], r)
        self.assertTrue(r["result"]["sources_remain"])
        self.assertIn("builds it again", r["result"]["note"])
        self.assertFalse((self.paths.index / "security").exists())
        self.assertFalse((self.paths.markup / "security").exists())
        self.assertTrue(src.is_file())                          # the document is untouched
        self.assertEqual(catalog.collection_names(catalog.live_catalog(self.paths)), [])
        self.assertEqual(policy.load_rules(self.paths)[0].by_name, {"security": ["claude"]})
        self.assertEqual(descriptions.get_description(self.paths, "security"), "auth docs")
        s = self.plan_index()
        self.assertEqual(s["indexed"], 1)                       # built again, rule still applies
        self.publish()
        self.assertEqual(catalog.collection_names(catalog.live_catalog(self.paths)), ["security"])

    def test_deleting_a_location_keeps_its_registration_and_folder(self):
        folder = self.make_location("vault")
        self.plan_index()
        self.publish()
        r = api.collection_delete(self.paths, "vault")
        self.assertTrue(r["ok"], r)
        self.assertEqual(r["result"]["location_unregistered"], "")
        self.assertTrue((folder / "note.md").is_file())
        self.assertEqual(locations.names(self.paths), ["vault"])
        self.assertFalse((self.paths.index / "vault").exists())
        self.assertNotIn("vault", catalog.collection_names(catalog.live_catalog(self.paths)))
        # location remove = unregister + delete
        self.plan_index()
        r = api.location_remove(self.paths, "vault")
        self.assertTrue(r["ok"], r)
        self.assertEqual(r["result"]["location_unregistered"], "vault")
        self.assertEqual(locations.names(self.paths), [])
        self.assertTrue((folder / "note.md").is_file())

    def test_delete_waits_for_no_indexing_run(self):
        folder = self.make_location("vault")
        self.plan_index()
        from rag_search.paths import index_lock
        with index_lock(self.paths):
            r = api.collection_delete(self.paths, "vault")
        self.assertFalse(r["ok"])
        self.assertIn("indexing run is in progress", r["error"])
        self.assertTrue((folder / "note.md").is_file())
        self.assertIn("vault", locations.names(self.paths))

    def test_cli_delete_needs_confirmation(self):
        self.make_location("vault")
        self.plan_index()
        rc, _, err = run("collection", "delete", "vault")
        self.assertNotEqual(rc, 0)
        rc, out, err = run("collection", "delete", "vault", "--yes")
        self.assertEqual(rc, 0, err)
        self.assertIn("documents untouched", out)

    def test_the_mcp_adapter_cannot_reach_collection_management(self):
        import re
        from tests.helpers import SRC
        pat = re.compile(r"(import|from)\s+\.*(rag_search)?\.?(bundle|lifecycle|inventory)\b|"
                         r"collection_(delete|import|export)|location_(add|remove)")
        for f in (SRC / "rag_search" / "mcp").glob("*.py"):
            self.assertIsNone(pat.search(f.read_text(encoding="utf-8")), f.name)

    def test_delete_is_not_an_mcp_tool(self):
        from rag_search.mcp import server
        names = server.make_tools("claude")
        self.assertFalse([n for n in names if any(w in n for w in ("delete", "import", "export",
                                                                    "location"))])


class PublishGuardTests(Base):
    def test_imported_model_conflict_is_explained(self):
        self.write_doc("security/auth.md", DOC)
        self.plan_index()
        self.publish()
        f = bundle.export_collection(self.paths, "security", self.tmp)["file"]
        os.environ["RAG_SEARCH_HOME"] = str(self.tmp / "home2")
        self.paths = get_paths()
        api.collection_import(self.paths, f)
        # this installation later switches model and re-embeds its own documents
        self.write_doc("mine/a.md", DOC)
        indexer.run_plan(self.paths, locations.plan_scan(self.paths), jobs=1,
                         embedder=FakeEmbedder(), model="other/model")
        with self.assertRaises(publish.PublishError) as cm:
            publish.publish(self.paths)
        self.assertIn("Imported collections cannot be re-embedded", str(cm.exception))


class ReviewFollowUpTests(Base):
    """Cases found in review of the first cut: none of them may lose derived data."""

    def test_a_broken_locations_file_stops_indexing_instead_of_pruning(self):
        self.make_location("vault", {"note.md": NOTE})
        self.plan_index()
        self.paths.locations_file.write_text('{"locations": {"vault": "/x",}}')   # trailing comma
        with self.assertRaises(locations.LocationError):
            locations.plan_scan(self.paths)
        self.assertTrue((self.paths.index / "vault" / "note" / META_FILE).is_file())

    def test_an_unreadable_subfolder_freezes_its_collection(self):
        self.write_doc("security/auth.md", DOC)
        self.write_doc("security/deep/roles.md", "# Roles\n\nAuditors read ledgers.\n")
        self.plan_index()
        real_walk = os.walk

        def flaky_walk(top, onerror=None, followlinks=False):
            for dp, dirs, files in real_walk(top, followlinks=followlinks):
                if Path(dp).name == "deep":
                    if onerror:
                        e = PermissionError(13, "Permission denied")
                        e.filename = dp
                        onerror(e)
                    continue
                yield dp, dirs, files

        with mock.patch.object(locations.os, "walk", flaky_walk):
            s = self.plan_index()
        self.assertEqual(s["removed"], [])
        self.assertIn("security", s["frozen"])
        self.assertTrue((self.paths.index / "security" / "deep" / "roles" / META_FILE).is_file())

    def test_an_empty_location_folder_is_treated_as_not_mounted(self):
        folder = self.make_location("vault", {"note.md": NOTE})
        self.plan_index()
        (folder / "note.md").unlink()                    # mount point now empty
        s = self.plan_index()
        self.assertEqual(s["removed"], [])
        self.assertIn("vault", s["frozen"])
        self.assertTrue((self.paths.index / "vault" / "note" / META_FILE).is_file())

    def test_a_case_only_rename_on_a_case_insensitive_disk_keeps_the_index(self):
        # simulate a case-insensitive disk: index/Security is the same folder as index/security
        self.write_doc("security/auth.md", DOC)
        self.plan_index()
        (self.paths.docs / "security").rename(self.paths.docs / "Security")
        for root in (self.paths.index, self.paths.markup):
            if not (root / "Security").exists():         # already the same folder on a case-insensitive disk
                (root / "Security").symlink_to(root / "security")
        s = self.plan_index()
        self.assertEqual(s["removed"], [])
        self.assertEqual(s["indexed"], 0)
        self.assertTrue((self.paths.index / "security" / "auth" / META_FILE).is_file())
        self.assertTrue((self.paths.markup / "security" / "auth.md").is_file())

    def test_documents_named_docs_survive_export_and_import(self):
        self.write_doc("guide/docs.md", DOC)
        self.write_doc("guide/docs/inner.md", "# Inner\n\nTurbine maintenance.\n")
        self.write_doc("guide/top.md", NOTE)
        self.plan_index()
        self.publish()
        f = bundle.export_collection(self.paths, "guide", self.tmp)["file"]
        os.environ["RAG_SEARCH_HOME"] = str(self.tmp / "home2")
        self.paths = get_paths()
        r = api.collection_import(self.paths, f)
        self.assertTrue(r["ok"], r)
        for rel in ("docs", "docs/inner", "top"):
            self.assertTrue((self.paths.index / "guide" / rel / META_FILE).is_file(), rel)
        again = bundle.export_collection(self.paths, "guide", self.tmp / "again")
        self.assertEqual(again["documents"], 3)

    def test_import_waits_for_a_model_switch_to_finish(self):
        self.write_doc("security/auth.md", DOC)
        self.plan_index()
        self.publish()
        f = bundle.export_collection(self.paths, "security", self.tmp)["file"]
        os.environ["RAG_SEARCH_HOME"] = str(self.tmp / "home2")
        self.paths = get_paths()
        self.write_doc("mine/a.md", DOC)
        self.plan_index()
        self.publish()                                    # serving BAAI/bge-m3
        os.environ["RAG_SEARCH_MODEL"] = "other/model"     # switched, not yet re-embedded
        r = api.collection_import(self.paths, f)
        self.assertFalse(r["ok"])
        self.assertIn("in progress", r["error"])

    def test_a_malformed_manifest_is_a_clean_error(self):
        p = self.tmp / "bad.rag.tgz"
        body = json.dumps({"format": bundle.BUNDLE_FORMAT, "bundle_version": 1, "collection": "x",
                           "model": "BAAI/bge-m3", "files": ["not", "a", "dict"]}).encode()
        with tarfile.open(p, "w:gz") as tf:
            info = tarfile.TarInfo("manifest.json")
            info.size = len(body)
            tf.addfile(info, io.BytesIO(body))
        r = api.collection_import(self.paths, str(p))
        self.assertFalse(r["ok"])
        self.assertIn("unexpected form", r["error"])


class DashboardDescribeTests(UiBase):
    def test_describe_from_the_dashboard(self):
        st, js, _, _ = self.dash.req("POST", "/api/describe",
                                     {"collection": "HR", "description": "Leave and pay"})
        self.assertEqual(st, 200, js)
        self.assertEqual(js["result"]["collection"], "hr")
        self.assertEqual(descriptions.get_description(self.paths, "hr"), "Leave and pay")
        st, js, _, _ = self.dash.req("POST", "/api/describe", {"collection": "hr", "description": ""})
        self.assertEqual(st, 200, js)
        self.assertEqual(descriptions.get_description(self.paths, "hr"), "")
        for body in ({"collection": 5}, {"collection": "../x", "description": "y"},
                     {"collection": "hr", "description": "x" * 501}):
            st, _, _, _ = self.dash.req("POST", "/api/describe", body)
            self.assertEqual(st, 400, body)

    def test_overview_rows_say_where_documents_come_from(self):
        folder = self.tmp / "notes"
        folder.mkdir()
        locations.add(self.paths, "garden", str(folder))
        rows = {r["collection"]: r for r in access.overview(self.paths)["collections"]}
        self.assertEqual((rows["garden"]["kind"], rows["garden"]["folder"]),
                         ("location", str(folder.resolve())))
        self.assertEqual(rows["hr"]["kind"], "docs")


if __name__ == "__main__":
    unittest.main()

