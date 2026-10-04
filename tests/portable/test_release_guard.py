import importlib.util
import tempfile
import unittest
import zipfile
from pathlib import Path

GUARD = Path(__file__).resolve().parents[2] / "scripts" / "release_guard.py"


def load():
    spec = importlib.util.spec_from_file_location("release_guard", GUARD)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@unittest.skipUnless(GUARD.exists(), "scripts/ is not part of this tree")
class ReleaseGuardTests(unittest.TestCase):
    def setUp(self):
        self.g = load()
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: __import__("shutil").rmtree(self.tmp, ignore_errors=True))

    def test_finds_words_in_names_contents_and_nested_archives(self):
        (self.tmp / "ok.txt").write_text("nothing here")
        self.assertEqual(self.g.scan([self.tmp], ["acme"]), [])
        (self.tmp / "notes.txt").write_text("made for ACME corp")
        self.assertEqual(len(self.g.scan([self.tmp], ["acme"])), 1)
        with zipfile.ZipFile(self.tmp / "pkg.whl", "w") as z:
            z.writestr("acme_plugin.py", "x = 1")
        hits = self.g.scan([self.tmp / "pkg.whl"], ["acme"])
        self.assertTrue(hits and "file name contains" in hits[0])

    def test_a_digest_in_a_wheel_record_is_not_a_hit(self):
        # base64 digests can spell any word by chance, so the digest column is not scanned
        with zipfile.ZipFile(self.tmp / "pkg.whl", "w") as z:
            z.writestr("pkg-1.dist-info/RECORD", "pkg/jobs.py,sha256=hS1W0Eq84b9SAcMeW2FXW,4132\n")
        self.assertEqual(self.g.scan([self.tmp / "pkg.whl"], ["acme"]), [])
        with zipfile.ZipFile(self.tmp / "bad.whl", "w") as z:
            z.writestr("pkg-1.dist-info/RECORD", "pkg/acme.py,sha256=abc,4132\n")
        self.assertEqual(len(self.g.scan([self.tmp / "bad.whl"], ["acme"])), 1)   # a real path still counts


if __name__ == "__main__":
    unittest.main()
