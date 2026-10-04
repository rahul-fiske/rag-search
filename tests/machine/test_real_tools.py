"""The real tools and models, through `scripts/sanity_check.py` (the same checks, as unit tests).

Run them with the Python of the installed tool, on the Mac:

    PYTHONPATH=src "$(uv tool dir)/rag-search/bin/python" -m unittest discover -s tests/machine -t .

Each test is one check of the script: it generates its own documents (no personal data), downloads
nothing, and reports a missing model or package as a failure with the script's own explanation.
Not an Apple Silicon Mac: every test is skipped.
"""

import importlib.util
import os
import platform
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "sanity_check.py"
APPLE_SILICON = sys.platform == "darwin" and platform.machine() == "arm64"


def load_script():
    spec = importlib.util.spec_from_file_location("sanity_check", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@unittest.skipUnless(APPLE_SILICON, "needs an Apple Silicon Mac with the real tools and models")
class RealToolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        # child processes (the Playground run, `models verify`) must run the same code as this test
        src = str(SCRIPT.parent.parent / "src")
        os.environ["PYTHONPATH"] = os.pathsep.join([src, *filter(None, [os.environ.get("PYTHONPATH")])])
        cls.sc = load_script()
        cls._tmp = tempfile.TemporaryDirectory(prefix="rag-machine-")
        cls.tmp = Path(cls._tmp.name)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def check(self, name, fn):
        r = self.sc.timed(name, fn)
        if r["status"] == "SKIP":
            self.skipTest(r["detail"])
        self.assertEqual(r["status"], "PASS", r["detail"])

    def test_1_environment_packages_gpu_and_models(self):
        for name, fn in (("machine", self.sc.check_machine), ("packages", self.sc.check_packages),
                         ("GPU", self.sc.check_gpu), ("models downloaded", self.sc.check_models)):
            with self.subTest(name):
                self.check(name, fn)

    def test_2_apple_vision_reads_an_image(self):
        self.check("Apple Vision", lambda: self.sc.check_apple_vision(self.tmp))

    def test_3_document_reader_reads_a_scan(self):
        self.check("document reader", lambda: self.sc.check_vlm(self.tmp))

    def test_4_docling_converts_a_digital_pdf(self):
        self.check("docling", lambda: self.sc.check_docling(self.tmp))

    def test_5_playground_indexes_a_scan_and_a_digital_pdf(self):
        self.check("Playground index", lambda: self.sc.check_e2e(self.tmp))

    def test_6_embedding_model_and_reranker_verify(self):
        self.check("models verify", self.sc.check_models_verify)


if __name__ == "__main__":
    unittest.main()
