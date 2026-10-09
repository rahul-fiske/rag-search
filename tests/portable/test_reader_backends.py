"""The reader's backends: which exist, and what each needs before a process is started."""

from __future__ import annotations

import unittest
from unittest import mock

from rag_search.core.conversion import vlm, vlm_worker


class RegistryTests(unittest.TestCase):
    def test_the_real_backends_are_registered(self):
        self.assertEqual(set(vlm_worker.BACKENDS), {"mlx", "transformers"})

    def test_an_unknown_backend_names_the_choices(self):
        with self.assertRaises(RuntimeError) as cm:
            vlm_worker.load_backend("nonsense", "m")
        self.assertIn("transformers", str(cm.exception))

    def test_a_module_attr_backend_still_loads(self):
        b = vlm_worker.load_backend("tests.helpers:FakeVlmBackend", "m")
        self.assertTrue(hasattr(b, "read"))


class PreflightTests(unittest.TestCase):
    def reader(self):
        return vlm.VlmReader("some/model", backend="transformers")

    def test_transformers_needs_the_package(self):
        with mock.patch("importlib.util.find_spec", return_value=None):
            with self.assertRaises(vlm.ReaderUnavailable) as cm:
                self.reader().preflight()
        self.assertIn("transformers", str(cm.exception))

    def test_transformers_needs_the_model_on_disk(self):
        with mock.patch("importlib.util.find_spec", return_value=object()), \
                mock.patch("rag_search.models.cache_state", return_value={"cached": False}):
            with self.assertRaises(vlm.ReaderUnavailable) as cm:
                self.reader().preflight()
        self.assertIn("not downloaded", str(cm.exception))

    def test_transformers_has_no_platform_restriction(self):
        with mock.patch("importlib.util.find_spec", return_value=object()), \
                mock.patch("rag_search.models.cache_state", return_value={"cached": True}), \
                mock.patch("rag_search.machine.mlx_possible", return_value=False):
            self.reader().preflight()


if __name__ == "__main__":
    unittest.main()
