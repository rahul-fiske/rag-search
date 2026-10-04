"""core/legacy_convert.py and `rag-search convert-legacy`: no real LibreOffice needed --
``subprocess.run`` is stubbed to behave like ``soffice --convert-to`` would."""

from __future__ import annotations

import json
import subprocess
import zipfile
from pathlib import Path
from unittest import mock

from tests.helpers import TempHome
from tests.portable.test_cli_api import run

from rag_search.core.legacy_convert import (
    LEGACY_FORMATS,
    convert_tree,
    find_soffice,
    is_lock_file,
    is_valid_ooxml,
    iter_legacy_files,
)

FAKE_SOFFICE = "/fake/soffice"


def _write_ooxml_zip(path: Path) -> None:
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("[Content_Types].xml", "<Types/>")


def _fake_run(fail: set[str] = frozenset()):
    """A `subprocess.run` stand-in: writes a valid OOXML file into --outdir, unless the
    source's name is in *fail* (then it reports a non-zero exit like a real failure)."""

    def _run(cmd, **kwargs):
        out_ext = cmd[cmd.index("--convert-to") + 1]
        outdir = Path(cmd[cmd.index("--outdir") + 1])
        src = Path(cmd[-1])
        if src.name in fail:
            return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="simulated failure\n")
        _write_ooxml_zip(outdir / (src.stem + "." + out_ext))
        return subprocess.CompletedProcess(cmd, 0, stdout="ok\n", stderr="")

    return _run


class HelperTests(TempHome):
    def test_is_lock_file(self):
        self.assertTrue(is_lock_file(Path("~$report.doc")))
        self.assertTrue(is_lock_file(Path(".~lock.report.doc#")))
        self.assertFalse(is_lock_file(Path("report.doc")))

    def test_is_valid_ooxml(self):
        good = self.tmp / "good.docx"
        _write_ooxml_zip(good)
        self.assertTrue(is_valid_ooxml(good))

        not_office_zip = self.tmp / "plain.docx"
        with zipfile.ZipFile(not_office_zip, "w") as zf:
            zf.writestr("hello.txt", "not an office file")
        self.assertFalse(is_valid_ooxml(not_office_zip))

        not_a_zip = self.tmp / "notzip.docx"
        not_a_zip.write_text("just text")
        self.assertFalse(is_valid_ooxml(not_a_zip))

        self.assertFalse(is_valid_ooxml(self.tmp / "missing.docx"))

    def test_iter_legacy_files_recursive_sorted_skips_locks(self):
        (self.tmp / "sub").mkdir()
        (self.tmp / "b.doc").write_text("x")
        (self.tmp / "sub" / "a.xls").write_text("x")
        (self.tmp / "c.docx").write_text("x")          # already modern: not legacy
        (self.tmp / "~$b.doc").write_text("x")          # lock file: excluded

        found = list(iter_legacy_files(self.tmp, {".doc", ".xls"}))
        self.assertEqual(sorted(p.name for p in found), ["a.xls", "b.doc"])

    def test_find_soffice_prefers_path_lookup(self):
        with mock.patch("shutil.which", return_value="/usr/bin/soffice"):
            self.assertEqual(find_soffice(), "/usr/bin/soffice")
        with mock.patch("shutil.which", return_value=None), \
             mock.patch("rag_search.core.legacy_convert._MAC_SOFFICE") as mac_path:
            mac_path.exists.return_value = False
            self.assertIsNone(find_soffice())


class ConvertTreeTests(TempHome):
    def setUp(self):
        super().setUp()
        self.root = self.tmp / "docs_root"
        self.root.mkdir()

    def test_unknown_extension_rejected(self):
        with self.assertRaises(ValueError):
            convert_tree(self.root, exts={".pdf"}, soffice=FAKE_SOFFICE)

    def test_dry_run_touches_nothing_and_writes_no_log(self):
        (self.root / "a.doc").write_text("legacy")
        log = self.root / "log.jsonl"
        with mock.patch("subprocess.run", side_effect=_fake_run()):
            summary = convert_tree(self.root, soffice=FAKE_SOFFICE, dry_run=True, log_path=log)
        self.assertEqual([r.src.name for r in summary.converted], ["a.doc"])
        self.assertTrue((self.root / "a.doc").exists())          # untouched
        self.assertFalse((self.root / "a.docx").exists())        # nothing written
        self.assertFalse(log.exists())

    def test_successful_conversion_deletes_original_and_logs(self):
        (self.root / "a.doc").write_text("legacy")
        log = self.root / "log.jsonl"
        with mock.patch("subprocess.run", side_effect=_fake_run()):
            summary = convert_tree(self.root, soffice=FAKE_SOFFICE, log_path=log)
        self.assertEqual(len(summary.converted), 1)
        self.assertTrue(summary.converted[0].deleted)
        self.assertFalse((self.root / "a.doc").exists())
        self.assertTrue(is_valid_ooxml(self.root / "a.docx"))
        entries = [json.loads(line) for line in log.read_text().splitlines()]
        self.assertEqual(len(entries), 1)
        self.assertTrue(entries[0]["ok"])
        self.assertTrue(entries[0]["deleted"])

    def test_keep_originals_flag(self):
        (self.root / "a.doc").write_text("legacy")
        with mock.patch("subprocess.run", side_effect=_fake_run()):
            summary = convert_tree(self.root, soffice=FAKE_SOFFICE, keep_originals=True)
        self.assertEqual(len(summary.converted), 1)
        self.assertFalse(summary.converted[0].deleted)
        self.assertTrue((self.root / "a.doc").exists())           # kept
        self.assertTrue((self.root / "a.docx").exists())

    def test_failure_never_deletes_original(self):
        (self.root / "bad.doc").write_text("legacy")
        with mock.patch("subprocess.run", side_effect=_fake_run(fail={"bad.doc"})):
            summary = convert_tree(self.root, soffice=FAKE_SOFFICE)
        self.assertEqual(len(summary.failed), 1)
        self.assertEqual(summary.failed[0].error, "simulated failure")
        self.assertTrue((self.root / "bad.doc").exists())
        self.assertFalse((self.root / "bad.docx").exists())

    def test_timeout_reported_as_failure_not_exception(self):
        (self.root / "slow.doc").write_text("legacy")

        def _timeout(cmd, **kwargs):
            raise subprocess.TimeoutExpired(cmd, kwargs.get("timeout", 1))

        with mock.patch("subprocess.run", side_effect=_timeout):
            summary = convert_tree(self.root, soffice=FAKE_SOFFICE, timeout=1)
        self.assertEqual(len(summary.failed), 1)
        self.assertIn("timed out", summary.failed[0].error)
        self.assertTrue((self.root / "slow.doc").exists())

    def test_existing_destination_skipped_unless_forced(self):
        (self.root / "a.doc").write_text("legacy")
        _write_ooxml_zip(self.root / "a.docx")
        with mock.patch("subprocess.run", side_effect=_fake_run()) as m:
            summary = convert_tree(self.root, soffice=FAKE_SOFFICE)
        self.assertEqual(len(summary.skipped), 1)
        m.assert_not_called()
        self.assertTrue((self.root / "a.doc").exists())            # skip = no touch at all

        with mock.patch("subprocess.run", side_effect=_fake_run()):
            summary = convert_tree(self.root, soffice=FAKE_SOFFICE, force=True)
        self.assertEqual(len(summary.converted), 1)
        self.assertFalse((self.root / "a.doc").exists())           # forced through, then deleted

    def test_all_legacy_formats_map_to_expected_target(self):
        self.assertEqual(LEGACY_FORMATS, {".doc": "docx", ".xls": "xlsx",
                                          ".ppt": "pptx", ".rtf": "docx"})
        # distinct stems: .doc and .rtf both target .docx, and same-stem files would
        # otherwise collide on the second one seeing the first's output already there
        for ext in LEGACY_FORMATS:
            (self.root / f"file_{ext.lstrip('.')}{ext}").write_text("legacy")
        with mock.patch("subprocess.run", side_effect=_fake_run()):
            summary = convert_tree(self.root, soffice=FAKE_SOFFICE)
        self.assertEqual(len(summary.converted), len(LEGACY_FORMATS))
        for ext, target in LEGACY_FORMATS.items():
            self.assertTrue((self.root / f"file_{ext.lstrip('.')}.{target}").exists(), target)


class ConvertLegacyCliTests(TempHome):
    def test_dry_run_defaults_to_docs_folder(self):
        self.write_doc("legacy/report.doc", "legacy")
        with mock.patch("rag_search.core.legacy_convert.find_soffice",
                        return_value=FAKE_SOFFICE):
            rc, out, err = run("convert-legacy", "--dry-run")
        self.assertEqual(rc, 0, err)
        self.assertIn("report.doc", out)
        self.assertTrue((self.paths.docs / "legacy" / "report.doc").exists())

    def test_missing_libreoffice_reports_unavailable(self):
        with mock.patch("rag_search.core.legacy_convert.find_soffice", return_value=None):
            rc, out, err = run("convert-legacy")
        self.assertEqual(rc, 3)
        self.assertIn("brew install", err)

    def test_json_output_and_real_conversion_keeps_the_original_by_default(self):
        # sources are read-only to rag-search: the original stays unless deletion is asked for
        self.write_doc("a.doc", "legacy")
        with mock.patch("rag_search.core.legacy_convert.find_soffice",
                        return_value=FAKE_SOFFICE), \
             mock.patch("subprocess.run", side_effect=_fake_run()):
            rc, out, err = run("convert-legacy", "--json")
        self.assertEqual(rc, 0, err)
        payload = json.loads(out)
        self.assertEqual(len(payload["converted"]), 1)
        self.assertFalse(payload["converted"][0]["deleted"])
        self.assertTrue((self.paths.docs / "a.doc").exists())
        self.assertTrue((self.paths.docs / "a.docx").exists())

    def test_delete_originals_is_explicit(self):
        self.write_doc("a.doc", "legacy")
        with mock.patch("rag_search.core.legacy_convert.find_soffice",
                        return_value=FAKE_SOFFICE), \
             mock.patch("subprocess.run", side_effect=_fake_run()):
            rc, out, err = run("convert-legacy", "--json", "--delete-originals")
        self.assertEqual(rc, 0, err)
        self.assertTrue(json.loads(out)["converted"][0]["deleted"])
        self.assertFalse((self.paths.docs / "a.doc").exists())
        self.assertTrue((self.paths.docs / "a.docx").exists())

    def test_unsupported_extension_is_a_usage_error(self):
        rc, out, err = run("convert-legacy", "--ext", "pdf")
        self.assertEqual(rc, 2)
        self.assertIn("unsupported extension", err)

    def test_not_a_directory_is_a_usage_error(self):
        rc, out, err = run("convert-legacy", str(self.paths.docs / "nope"))
        self.assertEqual(rc, 2)
        self.assertIn("not a directory", err)
