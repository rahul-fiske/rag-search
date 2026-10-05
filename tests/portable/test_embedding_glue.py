"""The glue around the model libraries in core/embedding.py: which device and precision are chosen, how texts
are batched, what is refused (NaN, wrong output), how a cross-encoder's raw scores become 0..1.  Tier A: `torch`,
`sentence_transformers` and the cached-model check are fakes put in `sys.modules`; no weights are loaded.
(That the real libraries do what the fakes assume is tier B's job, `tests/real/`.)"""

from __future__ import annotations

import sys
import types
import unittest
from unittest import mock

import numpy as np

from tests.helpers import TempHome

from rag_search import models
from rag_search.core import embedding


class FakeTorch(types.ModuleType):
    """Just what embedding.py touches of torch."""

    float16, bfloat16, float32 = "fp16", "bf16", "fp32"
    __version__ = "2.5.1+cpu"

    def __init__(self, mps=False, cuda=False):
        super().__init__("torch")
        self.backends = types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: mps))
        self.cuda = types.SimpleNamespace(is_available=lambda: cuda, empty_cache=mock.Mock())
        self.mps = types.SimpleNamespace(empty_cache=mock.Mock())
        self.nn = types.SimpleNamespace(Sigmoid=lambda: "sigmoid")


class FakeST(types.ModuleType):
    """sentence_transformers with a bi-encoder and a cross-encoder that record how they were built."""

    def __init__(self, encode=None, predict=None, reject_activation=False):
        super().__init__("sentence_transformers")
        outer = self
        self.built: list[tuple[str, dict]] = []

        class SentenceTransformer:
            def __init__(self, name, **kw):
                outer.built.append((name, kw))
                self.max_seq_length = 0
                self.batch_sizes: list[int] = []

            def encode(self, part, batch_size, **kw):
                self.batch_sizes.append(len(part))
                return encode(part) if encode else np.ones((len(part), 4), dtype=np.float32)

            def parameters(self):
                return [types.SimpleNamespace(numel=lambda: 10, element_size=lambda: 4)]

        class CrossEncoder:
            def __init__(self, name, **kw):
                if reject_activation and "activation_fn" in kw:
                    raise TypeError("unexpected keyword argument 'activation_fn'")
                outer.built.append((name, kw))

            def predict(self, pairs, **kw):
                return predict(pairs) if predict else np.full(len(pairs), 0.5)

        self.SentenceTransformer, self.CrossEncoder = SentenceTransformer, CrossEncoder


class DeviceAndPrecisionTests(TempHome):
    def test_the_device_is_forced_or_the_best_one_there_is(self):
        import os

        os.environ["RAG_SEARCH_DEVICE"] = "cpu"
        self.addCleanup(os.environ.pop, "RAG_SEARCH_DEVICE", None)
        self.assertEqual(embedding.pick_device(), "cpu")
        del os.environ["RAG_SEARCH_DEVICE"]
        for torch, system, machine, want in ((FakeTorch(mps=True), "Darwin", "arm64", "mps"),
                                             (FakeTorch(mps=True), "Darwin", "x86_64", "cpu"),   # an Intel Mac's MPS is not used
                                             (FakeTorch(cuda=True), "Linux", "x86_64", "cuda"),
                                             (FakeTorch(), "Linux", "x86_64", "cpu")):
            with self.subTest(want, system=system, machine=machine), \
                    mock.patch.dict(sys.modules, {"torch": torch}), \
                    mock.patch("platform.system", return_value=system), mock.patch("platform.machine", return_value=machine):
                self.assertEqual(embedding.pick_device(), want)

    def test_half_precision_only_on_a_gpu_unless_forced(self):
        import os

        torch = FakeTorch()
        self.assertEqual(embedding.torch_dtype(torch, "mps"), "fp16")
        self.assertIsNone(embedding.torch_dtype(torch, "cpu"))
        self.assertIsNone(embedding.torch_dtype(torch, "mps", default_half=False))
        for forced, want in (("bf16", "bf16"), ("FLOAT32", "fp32"), ("half", "fp16")):
            os.environ["RAG_SEARCH_DTYPE"] = forced
            self.assertEqual(embedding.torch_dtype(torch, "cpu"), want)
        os.environ["RAG_SEARCH_DTYPE"] = "int4"
        with self.assertRaises(ValueError):
            embedding.torch_dtype(torch, "cpu")
        os.environ.pop("RAG_SEARCH_DTYPE")

    def test_dropping_a_model_gives_the_gpu_cache_back(self):
        torch = FakeTorch(mps=True, cuda=True)
        with mock.patch.dict(sys.modules, {"torch": torch}):
            embedding.release_memory()
        torch.mps.empty_cache.assert_called_once()
        torch.cuda.empty_cache.assert_called_once()
        with mock.patch.dict(sys.modules, {"torch": None}):      # torch never imported: nothing to do
            embedding.release_memory()

    def test_the_torch_version_and_the_weight_size_are_read_defensively(self):
        with mock.patch.dict(sys.modules, {"torch": FakeTorch()}):
            self.assertEqual(embedding._torch_version(), (2, 5))
        self.assertIsNone(embedding._param_bytes(None))
        self.assertIsNone(embedding._param_bytes(object()))                 # no parameters(): never raises
        m = FakeST().SentenceTransformer("x")
        self.assertEqual(embedding._param_bytes(m), 40)
        self.assertEqual(embedding._param_bytes(types.SimpleNamespace(model=m)), 40)


class EmbedderTests(TempHome):
    def build(self, **kw):
        st = FakeST(**kw)
        p = mock.patch.dict(sys.modules, {"torch": FakeTorch(), "sentence_transformers": st})
        p.start()
        self.addCleanup(p.stop)
        for q in (mock.patch.object(models, "cache_state", return_value={"cached": False}),
                  mock.patch.object(embedding, "model_revision", return_value="abc123")):
            q.start()
            self.addCleanup(q.stop)
        return st

    def test_loading_builds_the_model_once_with_the_chosen_device_and_sequence_length(self):
        st = self.build()
        emb = embedding.Embedder("x/y", batch_size=2, max_seq_length=77)
        model = emb.load()
        self.assertIs(emb.load(), model)                                     # once
        self.assertEqual((st.built[0][0], st.built[0][1]["device"], model.max_seq_length), ("x/y", "cpu", 77))
        self.assertEqual(emb.revision, "abc123")
        self.assertEqual(emb.memory_bytes(), 40)

    def test_texts_go_through_in_batches_with_progress_and_nothing_gives_an_empty_array(self):
        st = self.build()
        emb = embedding.Embedder("x/y", batch_size=2)
        seen = []
        vecs = emb.encode([f"t{i}" for i in range(19)], progress=lambda done, total: seen.append((done, total)))
        self.assertEqual(vecs.shape, (19, 4))
        self.assertEqual(emb._model.batch_sizes, [8, 8, 3])                  # step = 4 batches
        self.assertEqual(seen, [(8, 19), (16, 19), (19, 19)])
        self.assertEqual(emb.encode([]).shape, (0, 0))
        del st

    def test_a_query_gets_the_instruction_prefix_and_passages_do_not(self):
        seen = []
        self.build(encode=lambda part: (seen.extend(part), np.ones((len(part), 4), dtype=np.float32))[1])
        emb = embedding.Embedder("x/y", query_prefix="Instruct: find it\nQuery: ")
        emb.encode_query("hello")
        emb.encode(["passage"])
        self.assertEqual(seen, ["Instruct: find it\nQuery: hello", "passage"])

    def test_a_model_that_returns_nan_is_refused_with_advice(self):
        self.build(encode=lambda part: np.full((len(part), 4), np.nan, dtype=np.float32))
        with self.assertRaises(RuntimeError) as cm:
            embedding.Embedder("x/y").encode(["a"])
        self.assertIn("RAG_SEARCH_DTYPE=float32", str(cm.exception))


class RerankerTests(TempHome):
    def build(self, **kw):
        st = FakeST(**kw)
        for p in (mock.patch.dict(sys.modules, {"torch": FakeTorch(), "sentence_transformers": st}),
                  mock.patch.object(models, "cache_state", return_value={"cached": False})):
            p.start()
            self.addCleanup(p.stop)
        return st

    def test_a_cross_encoder_is_built_with_a_sigmoid_activation(self):
        st = self.build()
        rer = embedding.Reranker("x/r", max_length=128, batch_size=4)
        rer.load()
        kw = st.built[0][1]
        self.assertEqual((kw["max_length"], kw["activation_fn"], rer._needs_sigmoid), (128, "sigmoid", False))
        self.assertEqual(rer.score("q", ["a", "b"]), [0.5, 0.5])
        self.assertEqual(rer.score("q", []), [])
        self.assertEqual(rer.memory_bytes(), None)

    def test_a_library_without_activation_fn_gets_its_raw_scores_squashed_into_zero_to_one(self):
        st = self.build(reject_activation=True, predict=lambda pairs: np.array([-4.0, 0.0, 4.0][:len(pairs)]))
        rer = embedding.Reranker("x/r")
        scores = rer.score("q", ["a", "b", "c"])
        self.assertTrue(rer._needs_sigmoid)
        self.assertEqual([round(s, 3) for s in scores], [0.018, 0.5, 0.982])
        self.assertNotIn("activation_fn", st.built[0][1])

    def test_scores_that_are_already_probabilities_are_left_alone_and_nan_is_refused(self):
        self.build(reject_activation=True, predict=lambda pairs: np.array([0.2, 0.9][:len(pairs)]))
        self.assertEqual(embedding.Reranker("x/r").score("q", ["a", "b"]), [0.2, 0.9])
        self.build(predict=lambda pairs: np.array([np.nan] * len(pairs)))
        with self.assertRaises(RuntimeError) as cm:
            embedding.Reranker("x/r2").score("q", ["a"])
        self.assertIn("NaN/inf", str(cm.exception))


class LoadingFallbackTests(TempHome):
    def test_a_cached_model_that_fails_to_load_offline_is_loaded_the_normal_way(self):
        calls = []

        def factory(**kw):
            calls.append(kw)
            if kw.get("local_files_only"):
                raise OSError("incomplete cache")
            return "model"

        with mock.patch.object(models, "cache_state", return_value={"cached": True}):
            self.assertEqual(embedding._load_model("x/y", factory), "model")
        self.assertEqual([bool(c.get("local_files_only")) for c in calls], [True, False])

    def test_a_cached_model_that_fails_the_normal_way_too_raises_its_own_error(self):
        def factory(**kw):
            raise ValueError("corrupt weights")

        with mock.patch.object(models, "cache_state", return_value={"cached": True}), self.assertRaises(ValueError):
            embedding._load_model("x/y", factory)

    def test_a_model_that_cannot_be_downloaded_gets_an_explanation(self):
        def factory(**kw):
            raise OSError("Connection refused")

        with mock.patch.object(models, "cache_state", return_value={"cached": False}), \
                self.assertRaises(RuntimeError) as cm:
            embedding._load_model("x/y", factory)
        self.assertIn("x/y", str(cm.exception))

    def test_a_backend_spec_must_name_a_module_and_a_class(self):
        with self.assertRaises(ValueError):
            embedding._from_spec("justaname")
        self.assertEqual(embedding._from_spec("tests.helpers:FakeEmbedder").__class__.__name__, "FakeEmbedder")

    def test_an_unsupported_option_is_not_reported_as_a_failed_download(self):
        def factory(**kw):
            raise TypeError("unexpected keyword argument 'activation_fn'")

        with mock.patch.object(models, "cache_state", return_value={"cached": False}), self.assertRaises(TypeError):
            embedding._load_model("x/y", factory)

    def test_the_revision_of_a_model_is_empty_when_the_cache_cannot_say(self):
        with mock.patch.object(models, "cached_revision", side_effect=OSError("no cache")):
            self.assertEqual(embedding.model_revision("x/y"), "")


if __name__ == "__main__":
    unittest.main()
