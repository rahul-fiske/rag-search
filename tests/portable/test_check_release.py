"""scripts/check_release.py: the version is read from the package, the tag must equal it, the files must be clean."""

from __future__ import annotations

import importlib.util
import io
import tarfile
import unittest
import zipfile
from pathlib import Path

from tests.helpers import TempHome

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("check_release", ROOT / "scripts" / "check_release.py")
cr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cr)

META = "Metadata-Version: 2.4\nName: rag-search\nVersion: {v}\nRequires-Dist: docling>=2\nRequires-Dist: torch>=2\nRequires-Dist: mcp<2,>=1\n"


def make_dist(d: Path, v: str, *, extra_wheel=(), extra_sdist=(), license=True, wheel_version=None):
    with zipfile.ZipFile(d / f"rag_search-{v}-py3-none-any.whl", "w") as z:
        z.writestr(f"rag_search-{v}.dist-info/METADATA", META.format(v=wheel_version or v))
        if license:
            z.writestr(f"rag_search-{v}.dist-info/licenses/LICENSE", "MIT")
        z.writestr("rag_search/__init__.py", "")
        for n in extra_wheel:
            z.writestr(n, "")
    with tarfile.open(d / f"rag_search-{v}.tar.gz", "w:gz") as t:
        for n in (f"rag_search-{v}/LICENSE" if license else f"rag_search-{v}/README.md", *extra_sdist):
            info = tarfile.TarInfo(n)
            info.size = 0
            t.addfile(info, io.BytesIO(b""))


class TagTests(unittest.TestCase):
    def test_the_version_is_read_from_the_package(self):
        self.assertRegex(cr.read_version(ROOT), r"^\d+\.\d+\.\d+")

    def test_a_tag_must_be_a_release_and_equal_the_version(self):
        self.assertEqual(cr.check_tag("v1.2.3", "1.2.3"), [])
        self.assertEqual(cr.check_tag("v1.2.3rc1", "1.2.3rc1"), [])
        self.assertTrue(cr.check_tag("v1.2.4", "1.2.3"))                    # another number
        self.assertTrue(cr.check_tag("1.2.3", "1.2.3"))                     # no v
        self.assertTrue(cr.check_tag("v1.2", "1.2"))
        self.assertTrue(cr.check_tag("v1.2.3-beta", "1.2.3-beta"))
        self.assertTrue(cr.check_tag("v1.1.0.dev0", "1.1.0.dev0"))          # never a development version


class DistTests(TempHome):
    def test_a_clean_release_passes(self):
        make_dist(self.tmp, "1.2.3")
        self.assertEqual(cr.check_dist(self.tmp, "1.2.3"), [])

    def test_the_wrong_version_or_extra_files_fail(self):
        make_dist(self.tmp, "1.2.2")
        self.assertTrue(cr.check_dist(self.tmp, "1.2.3"))
        (self.tmp / "other").mkdir()
        make_dist(self.tmp / "other", "1.2.3", wheel_version="1.2.0")
        self.assertIn("Version: 1.2.3", " ".join(cr.check_dist(self.tmp / "other", "1.2.3")))

    def test_private_files_and_a_missing_license_fail(self):
        make_dist(self.tmp, "1.2.3", extra_wheel=("rag_search/hosts_secret.py",), extra_sdist=("rag_search-1.2.3/INTERNAL_HOSTS.md",),
                  license=False)
        errors = " ".join(cr.check_dist(self.tmp, "1.2.3"))
        for part in ("no LICENSE", "hosts_secret.py", "INTERNAL_HOSTS.md"):
            self.assertIn(part, errors)

    def test_the_command_line_exit_codes(self):
        make_dist(self.tmp, "1.2.3")
        v = cr.read_version(ROOT)
        self.assertEqual(cr.main(["tag", f"v{v}", "--root", str(ROOT)]) == 0, "dev" not in v)
        self.assertEqual(cr.main(["tag", "v9.9.9", "--root", str(ROOT)]), 1)
        self.assertEqual(cr.main(["dist", str(self.tmp), "--root", str(ROOT)]), 1)    # the dist is 1.2.3, the package is not


if __name__ == "__main__":
    unittest.main()
