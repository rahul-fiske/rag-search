"""Collection export/import: what is refused, and why.  An export is a file someone else made, so every
rule about what may be in it, and what must match, has a test: the archive is crafted from a real export of a
corpus collection and then damaged one way at a time.  Nothing is imported unless it is whole."""

from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path

from tests import corpus
from tests.helpers import TempHome

from rag_search import bundle
from rag_search.paths import index_lock


class ExportBase(TempHome):
    def setUp(self):
        super().setUp()
        corpus.copy("text/notes.md", self.paths.docs / "team" / "notes.md")
        corpus.copy("text/readme.txt", self.paths.docs / "team" / "readme.txt")
        self.index()
        self.good = Path(bundle.export_collection(self.paths, "team", self.tmp / "good.rag.tgz")["file"])

    def members(self, path: Path | None = None) -> dict[str, bytes]:
        out = {}
        with tarfile.open(path or self.good, "r:gz") as tf:
            for m in tf:
                if m.isfile():
                    out[m.name] = tf.extractfile(m).read()
        return out

    def craft(self, name: str, edit=None, *, refresh=(), extra=()) -> Path:
        """A copy of the good export with *edit(members, manifest)* applied; the checksums of the files named in
        *refresh* are recomputed so only the intended rule is broken.  *extra*: TarInfo members to append."""
        files = self.members()
        manifest = json.loads(files.pop("manifest.json"))
        if edit:
            edit(files, manifest)
        for f in refresh:
            manifest["files"][f] = bundle._sha(files[f])
        out = self.tmp / name
        with tarfile.open(out, "w:gz") as tf:
            bundle._add(tf, "manifest.json", json.dumps(manifest).encode())
            for n, data in files.items():
                bundle._add(tf, n, data)
            for info in extra:
                tf.addfile(info, io.BytesIO(b"x" * info.size) if info.isfile() else None)
        return out

    def refused(self, archive, text, **kw):
        with self.assertRaises(bundle.BundleError) as cm:
            bundle.import_collection(self.paths, archive, as_name=kw.pop("as_name", "copy"), **kw)
        self.assertIn(text, str(cm.exception))
        self.assertFalse((self.paths.index / "copy").exists(), "a refused import left something behind")


class ManifestTests(ExportBase):
    def test_a_file_that_is_not_an_export_says_what_it_is(self):
        with self.assertRaises(bundle.BundleError) as cm:
            bundle.read_manifest(self.tmp / "nowhere.rag.tgz")
        self.assertIn("no such file", str(cm.exception))
        text = self.tmp / "text.rag.tgz"
        text.write_text("not an archive")
        with self.assertRaises(bundle.BundleError) as cm:
            bundle.read_manifest(text)
        self.assertIn("not a readable collection export", str(cm.exception))
        bare = self.tmp / "bare.rag.tgz"
        with tarfile.open(bare, "w:gz") as tf:
            bundle._add(tf, "other.txt", b"x")
        with self.assertRaises(bundle.BundleError) as cm:
            bundle.read_manifest(bare)
        self.assertIn("no manifest.json", str(cm.exception))

    def test_a_manifest_is_checked_field_by_field(self):
        def manifest(**change):
            def edit(files, m):
                for k, v in change.items():
                    if v is None:
                        m.pop(k, None)
                    else:
                        m[k] = v
            return edit

        cases = [({"format": "something-else"}, "not a rag-search collection export"),
                 ({"bundle_version": 999}, "made by a newer rag-search"),
                 ({"collection": None}, "has no 'collection'"),
                 ({"model": None}, "has no 'model'"),
                 ({"files": ["a", "b"]}, "unexpected form"),
                 ({"collection": 7}, "damaged"),
                 ({"dim": "many"}, "non-numeric 'dim'")]
        for change, text in cases:
            with self.subTest(change):
                with self.assertRaises(bundle.BundleError) as cm:
                    bundle.read_manifest(self.craft("m.rag.tgz", manifest(**change)))
                self.assertIn(text, str(cm.exception))
        listed = json.dumps([1, 2])
        bad = self.tmp / "list.rag.tgz"
        with tarfile.open(bad, "w:gz") as tf:
            bundle._add(tf, "manifest.json", listed.encode())
        with self.assertRaises(bundle.BundleError):
            bundle.read_manifest(bad)


class HostileArchiveTests(ExportBase):
    def test_members_that_are_not_plain_files_in_the_right_place_are_refused(self):
        def link(name, target):
            i = tarfile.TarInfo(name)
            i.type, i.linkname = tarfile.SYMTYPE, target
            return i

        def reg(name, size=3):
            i = tarfile.TarInfo(name)
            i.size = size
            return i

        for member, text in ((link("index/_all/link", "/etc/passwd"), "only plain files"),
                             (reg("/etc/cron.d/x"), "unsafe path"),
                             (reg("index/../../escape"), "unsafe path"),
                             (reg("etc/passwd"), "not part of a collection export"),
                             (reg("index/.hidden"), "hidden files")):
            with self.subTest(member.name):
                self.refused(self.craft("h.rag.tgz", extra=[member]), text)

    def test_an_implausibly_large_member_is_refused_before_it_is_read(self):
        info = tarfile.TarInfo("index/_all/big.bin")
        info.size = bundle.MAX_MEMBER_BYTES + 1
        with self.assertRaises(bundle.BundleError) as cm:
            bundle._safe_member(info)
        self.assertIn("implausibly large", str(cm.exception))

    def test_a_file_the_manifest_does_not_list_is_refused(self):
        extra = tarfile.TarInfo("markup/team/surprise.md")
        extra.size = 3
        self.refused(self.craft("u.rag.tgz", extra=[extra]), "not listed in the manifest")

    def test_a_name_that_is_not_a_collection_name_is_refused(self):
        self.refused(self.good, "cannot be a collection name here", as_name="a/b")
        self.refused(self.good, "cannot be a collection name here", as_name="default")


class DamagedExportTests(ExportBase):
    def test_a_missing_or_altered_file_is_found_by_its_checksum(self):
        gone = self.craft("gone.rag.tgz", lambda files, m: files.pop("index/_all/nodes.json"))
        self.refused(gone, "incomplete: missing index/_all/nodes.json")

        def alter(files, m):
            files["index/_all/nodes.json"] = files["index/_all/nodes.json"] + b" "
        self.refused(self.craft("altered.rag.tgz", alter), "do not match their checksums")

    def test_chunks_and_vectors_that_do_not_belong_together_are_refused(self):
        nodes = "index/_all/nodes.json"

        def fewer(files, m):
            doc = json.loads(files[nodes])
            doc["nodes"] = doc["nodes"][:-1]
            files[nodes] = json.dumps(doc).encode()
        self.refused(self.craft("fewer.rag.tgz", fewer, refresh=[nodes]), "chunks but embeddings of shape")

        def not_a_list(files, m):
            files[nodes] = json.dumps({"nodes": "none"}).encode()
        self.refused(self.craft("nolist.rag.tgz", not_a_list, refresh=[nodes]), "no readable nodes.json")

        def wrong_dim(files, m):
            m["dim"] = 999
        self.refused(self.craft("dim.rag.tgz", wrong_dim), "dimensions")

    def test_an_embeddings_file_that_is_not_numpy_is_refused(self):
        emb = "index/_all/embeddings.npy"

        def garbage(files, m):
            files[emb] = b"this is not an npy file"
        self.refused(self.craft("g.rag.tgz", garbage, refresh=[emb]), "not a NumPy array file")

        def bad_header(files, m):
            files[emb] = b"\x93NUMPY\x01\x00" + (5).to_bytes(2, "little") + b"{oops"
        self.refused(self.craft("h.rag.tgz", bad_header, refresh=[emb]), "unreadable header")

    def test_npy_headers_of_both_versions_are_read_without_numpy(self):
        import numpy as np

        buf = io.BytesIO()
        np.save(buf, np.zeros((3, 5), dtype=np.float32))
        self.assertEqual(bundle.npy_shape(buf.getvalue()), (3, 5))
        v2 = io.BytesIO()
        np.lib.format.write_array(v2, np.zeros((2, 4), dtype=np.float32), version=(2, 0))
        self.assertEqual(bundle.npy_shape(v2.getvalue()), (2, 4))
        path = self.tmp / "a.npy"
        path.write_bytes(buf.getvalue())
        self.assertEqual(bundle.npy_shape(path), (3, 5))


class ExportRefusalTests(ExportBase):
    def test_what_cannot_be_exported_says_why(self):
        for name, text in (("nowhere", "no indexed collection named"), ("a/b", "is not a collection name")):
            with self.assertRaises(bundle.BundleError) as cm:
                bundle.export_collection(self.paths, name)
            self.assertIn(text, str(cm.exception))
        with index_lock(self.paths), self.assertRaises(bundle.BundleError) as cm:
            bundle.export_collection(self.paths, "team", self.tmp / "busy.rag.tgz")
        self.assertIn("an indexing run is in progress", str(cm.exception))
        (self.paths.index / "team" / "_all" / "merge.manifest.json").unlink()
        with self.assertRaises(bundle.BundleError) as cm:
            bundle.export_collection(self.paths, "team", self.tmp / "none.rag.tgz")
        self.assertIn("no complete index to export", str(cm.exception))

    def test_documents_built_with_different_models_are_not_exported_together(self):
        meta_file = self.paths.index / "team" / "notes" / "index.meta.json"
        meta = json.loads(meta_file.read_text())
        meta["model"] = "someone/other-model"
        meta_file.write_text(json.dumps(meta))
        with self.assertRaises(bundle.BundleError) as cm:
            bundle.export_collection(self.paths, "team", self.tmp / "mixed.rag.tgz")
        self.assertIn("mixes model values", str(cm.exception))

    def test_a_document_without_its_metadata_is_not_exported(self):
        (self.paths.index / "team" / "notes" / "index.meta.json").unlink()
        with self.assertRaises(bundle.BundleError) as cm:
            bundle.export_collection(self.paths, "team", self.tmp / "nometa.rag.tgz")
        self.assertIn("index.meta.json is missing", str(cm.exception))


if __name__ == "__main__":
    import unittest

    unittest.main()
