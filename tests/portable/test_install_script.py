"""scripts/install.sh: the Tesseract step, run on its own with stand-in binaries."""

from __future__ import annotations

import re
import shutil
import stat
import subprocess
import unittest
from pathlib import Path

from tests.helpers import TempHome

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "install.sh"


def tesseract_block() -> str:
    text = SCRIPT.read_text(encoding="utf-8")
    m = re.search(r"^# 4c\. Tesseract.*?(?=^say \"Health check\")", text, re.S | re.M)
    assert m, "the Tesseract step is missing from install.sh"
    return m.group(0)


@unittest.skipUnless(shutil.which("bash"), "needs bash")
class TesseractStepTests(TempHome):
    def run_block(self, *, fake_tesseract: str | None, extra_env: dict[str, str] | None = None) -> str:
        bindir = self.tmp / "bin"
        bindir.mkdir()
        if fake_tesseract is not None:
            exe = bindir / "tesseract"
            exe.write_text(fake_tesseract)
            exe.chmod(exe.stat().st_mode | stat.S_IXUSR)
        script = ('set -euo pipefail; say(){ printf "==> %s\\n" "$*"; }; IMPORT_ONLY=0; NO_TESSERACT='
                  + (extra_env or {}).get("NO_TESSERACT", "0") + "\n" + tesseract_block())
        # only the tools the step uses, and no tesseract, brew or curl from this machine: the step must not find a
        # real Tesseract or download language data into the system (a Linux test box has both)
        for tool in ("sed", "grep", "rm", "cat"):
            found = shutil.which(tool)
            if found and not (bindir / tool).exists():
                (bindir / tool).symlink_to(found)
        uname = bindir / "uname"                         # the Mac branch of the step, on any machine
        uname.write_text("#!/bin/sh\necho Darwin\n")
        uname.chmod(uname.stat().st_mode | stat.S_IXUSR)
        env = {"PATH": str(bindir), "HOME": str(self.tmp)}
        proc = subprocess.run([shutil.which("bash"), "-c", script], capture_output=True, text=True, env=env, timeout=60)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        return proc.stdout

    def fake(self, langs: str, data_dir: Path) -> str:
        return ('#!/bin/sh\nif [ "$1" = "--list-langs" ]; then\n'
                f'  echo \'List of available languages in "{data_dir}/" (2):\'\n'
                + "".join(f"  echo {x}\n" for x in langs.split()) + "fi\n")

    def test_nothing_is_done_when_the_languages_are_there(self):
        out = self.run_block(fake_tesseract=self.fake("eng hin mar osd", self.tmp))
        self.assertNotIn("note:", out)
        self.assertNotIn("adding", out)

    def test_missing_language_data_is_reported_when_the_folder_cannot_be_written(self):
        out = self.run_block(fake_tesseract=self.fake("eng", Path("/nonexistent/tessdata")))
        self.assertIn("no 'mar' data", out)
        self.assertIn("no 'hin' data", out)

    def test_a_missing_tesseract_without_homebrew_says_how_to_get_it(self):
        out = self.run_block(fake_tesseract=None)
        self.assertIn("Tesseract is not installed", out)

    def test_no_tesseract_skips_the_step(self):
        out = self.run_block(fake_tesseract=None, extra_env={"NO_TESSERACT": "1"})
        self.assertNotIn("Checking Tesseract", out)

    def test_the_options_and_the_reader_models_are_in_the_installer(self):
        text = SCRIPT.read_text(encoding="utf-8")
        self.assertIn("--no-tesseract", text)
        self.assertIn("models download --reader --repair", text)
        self.assertEqual(subprocess.run(["bash", "-n", str(SCRIPT)]).returncode, 0)


if __name__ == "__main__":
    unittest.main()
