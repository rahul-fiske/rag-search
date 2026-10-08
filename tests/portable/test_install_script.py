"""scripts/install.sh: what it passes to uv and to `rag-search setup`, run with stand-in `uv` and `rag-search` programs."""

from __future__ import annotations

import shutil
import stat
import subprocess
import unittest
from pathlib import Path

from tests.helpers import TempHome

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "install.sh"

FAKE_UV = """#!/bin/sh
echo "uv $*" >> "$LOG"
if [ "$1" = "tool" ] && [ "$2" = "dir" ]; then echo "$BIN"; fi
exit 0
"""
FAKE_RAG = """#!/bin/sh
echo "rag-search $*" >> "$LOG"
if [ "$1" = "--version" ]; then echo "rag-search 1.2.3"; fi
exit 0
"""


def make_exe(path: Path, text: str) -> None:
    path.write_text(text)
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


@unittest.skipUnless(shutil.which("bash"), "needs bash")
class InstallerTests(TempHome):
    def setUp(self):
        super().setUp()
        self.rel = self.tmp / "release"
        self.rel.mkdir()
        shutil.copy(SCRIPT, self.rel / "install.sh")
        (self.rel / "rag_search-1.2.3-py3-none-any.whl").write_bytes(b"not a wheel: the stand-in uv does not read it")
        self.bin = self.tmp / "bin"
        self.bin.mkdir()
        make_exe(self.bin / "uv", FAKE_UV)
        make_exe(self.bin / "rag-search", FAKE_RAG)
        self.log = self.tmp / "calls.log"

    def run_install(self, *args: str) -> list[str]:
        env = {"PATH": f"{self.bin}:/usr/bin:/bin", "HOME": str(self.tmp), "LOG": str(self.log), "BIN": str(self.bin)}
        proc = subprocess.run(["bash", str(self.rel / "install.sh"), *args], capture_output=True, text=True, env=env,
                              timeout=60, cwd=self.rel)
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        return self.log.read_text().splitlines()

    def test_the_wheel_is_installed_with_no_extra_packages_and_setup_does_the_rest(self):
        calls = self.run_install()
        installs = [c for c in calls if c.startswith("uv tool install")]
        self.assertEqual(len(installs), 1)
        self.assertNotIn("--with", installs[0])                       # every package is a dependency of the wheel
        self.assertTrue(installs[0].endswith("rag_search-1.2.3-py3-none-any.whl"))
        self.assertIn("--python 3.12", installs[0])
        setups = [c for c in calls if c.startswith("rag-search") and " setup" in c]
        self.assertEqual(setups, ["rag-search setup"])                # one call: models, reader, Tesseract, daemons, Claude
        for old in ("models download", "doctor", "daemon start", "register"):
            self.assertFalse([c for c in calls if c.startswith("rag-search " + old)], old)

    def test_the_options_become_setup_options(self):
        calls = self.run_install("--home", "/data/rag", "--models", "qwen3-small", "--no-tesseract", "--service",
                                 "--tool-prefix", "x_", "--import-only", "--skip-models", "--no-register")
        setup = [c for c in calls if " setup" in c][0]
        self.assertTrue(setup.startswith("rag-search --home /data/rag setup"))
        for part in ("--models qwen3-small", "--skip-models", "--skip-docling", "--no-tesseract", "--service",
                     "--no-register", "--tool-prefix x_"):
            self.assertIn(part, setup)

    def test_no_mcp_is_no_registration(self):
        setup = [c for c in self.run_install("--no-mcp") if " setup" in c][0]
        self.assertIn("--no-register", setup)

    def test_the_script_is_valid_shell_and_lists_no_packages(self):
        self.assertEqual(subprocess.run(["bash", "-n", str(SCRIPT)]).returncode, 0)
        text = SCRIPT.read_text(encoding="utf-8")
        for package in ("ocrmac", "mlx-vlm", "pillow-heif", "mcp>=", "numpy<2", "transformers>="):
            self.assertNotIn(f'--with "{package}', text)
            self.assertNotIn(f"--with {package}", text)


if __name__ == "__main__":
    unittest.main()
