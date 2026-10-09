"""scripts/check_wheels.py: which packages have no wheel on a platform (the Intel Mac install problem)."""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("check_wheels", ROOT / "scripts" / "check_wheels.py")
cw = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cw)

INTEL, ARM, LINUX = "x86_64-apple-darwin", "aarch64-apple-darwin", "x86_64-unknown-linux-gnu"


class WheelTests(unittest.TestCase):
    def test_which_wheel_fits_which_machine(self):
        cases = [
            ("torch-2.2.2-cp312-none-macosx_10_9_x86_64.whl", {INTEL}),
            ("numpy-2.5.3-cp312-cp312-macosx_14_0_arm64.whl", {ARM}),
            ("cryptography-48.0.1-cp311-abi3-macosx_10_9_universal2.whl", {INTEL, ARM}),      # abi3 of an older Python fits
            ("docling_parse-7.22.2-cp312-cp312-macosx_14_0_arm64.whl", {ARM}),
            ("numpy-1.26.4-cp311-cp311-macosx_10_9_x86_64.whl", set()),                       # another Python
            ("typer-0.21.2-py3-none-any.whl", {INTEL, ARM, LINUX}),
            ("six-1.16.0-py2.py3-none-any.whl", {INTEL, ARM, LINUX}),
            ("torch-2.14.1-cp312-cp312-manylinux_2_28_x86_64.whl", {LINUX}),
            ("docling_parse-4.7.3.tar.gz", set()),                                            # a source archive is not a wheel
        ]
        for name, fits in cases:
            with self.subTest(name):
                self.assertEqual({p for p in (INTEL, ARM, LINUX) if cw.wheel_fits(name, p, "3.12")}, fits)

    def test_a_python_that_the_wheel_does_not_cover(self):
        self.assertFalse(cw.wheel_fits("numpy-2.5.3-cp312-cp312-macosx_14_0_arm64.whl", ARM, "3.11"))
        self.assertTrue(cw.wheel_fits("cryptography-48.0.1-cp311-abi3-macosx_10_9_universal2.whl", ARM, "3.11"))
        self.assertFalse(cw.wheel_fits("x-1-cp313-abi3-macosx_10_9_universal2.whl", ARM, "3.12"))

    def test_the_packages_without_a_wheel_are_found(self):
        files = {
            ("docling-parse", "7.22.2"): ["docling_parse-7.22.2-cp312-cp312-macosx_14_0_arm64.whl", "docling_parse-7.22.2.tar.gz"],
            ("docling-parse", "4.7.2"): ["docling_parse-4.7.2-cp312-cp312-macosx_10_9_x86_64.whl"],
            ("typer", "0.21.2"): ["typer-0.21.2-py3-none-any.whl"],
            ("antlr4-python3-runtime", "4.9.3"): ["antlr4-python3-runtime-4.9.3.tar.gz"],         # pure Python: an sdist is fine
        }
        fetch = lambda name, version: files[(name, version)]                                      # noqa: E731
        pins = [("docling-parse", "7.22.2"), ("typer", "0.21.2"), ("antlr4-python3-runtime", "4.9.3")]
        self.assertEqual(cw.missing(pins, INTEL, "3.12", fetch), [("docling-parse", "7.22.2")])
        self.assertEqual(cw.missing(pins, ARM, "3.12", fetch), [])
        self.assertEqual(cw.missing([("docling-parse", "4.7.2")], INTEL, "3.12", fetch), [])

    def test_the_intel_limits_are_in_the_project(self):
        import tomllib

        deps = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]["dependencies"]
        for limit in ("docling-parse<4.7.3", "cryptography<49"):
            self.assertTrue(any(d.startswith(limit) and "x86_64" in d and "darwin" in d for d in deps), limit)


if __name__ == "__main__":
    unittest.main()
