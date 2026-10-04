"""The document VLM reader, child side: ``python -m rag_search.core.conversion.vlm_worker``.

Loads one model through a *backend*, then answers read requests on stdin until told to quit (see
``vlm.py`` for the protocol).  Standard output carries the protocol only: whatever a library prints
is moved to standard error.

A backend is a class with ``__init__(model_id)``, ``read(image_path, prompt, max_tokens) ->
{"md": str, "tokens": int}`` and optionally ``close()``.  ``mlx`` is the real one; any
``module:attr`` works (the tests use a fake).  Nothing here is imported by the rest of rag-search
except through a subprocess, so mlx-vlm is needed only on the machine that runs a reader.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import resource
import sys
import time
from typing import Any


def rss_mb() -> float:
    """Peak resident memory of this process, MB (ru_maxrss is bytes on macOS, KB on Linux)."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return round(peak / (1024 * 1024) if sys.platform == "darwin" else peak / 1024, 1)


class MlxBackend:
    """mlx-vlm on Apple Silicon (``pip install 'rag-search[mac-vlm]'``).  Greedy decoding."""

    def __init__(self, model_id: str) -> None:
        try:
            from mlx_vlm import generate, load
            from mlx_vlm.prompt_utils import apply_chat_template
            from mlx_vlm.utils import load_config
        except ImportError as exc:
            raise RuntimeError("mlx-vlm is not installed (pip install 'rag-search[mac-vlm]'; "
                               f"Apple Silicon only): {exc}") from exc
        self._generate, self._template = generate, apply_chat_template
        self.model, self.processor = load(model_id)
        self.config = load_config(model_id)

    def read(self, image_path: str, prompt: str, max_tokens: int) -> dict[str, Any]:
        formatted = self._template(self.processor, self.config, prompt, num_images=1)
        out = self._generate(self.model, self.processor, formatted, [image_path],
                             max_tokens=max_tokens, temperature=0.0, verbose=False)
        text = getattr(out, "text", out)                      # newer mlx-vlm: a result object
        tokens = getattr(out, "generation_tokens", None)
        text = str(text or "")
        return {"md": text, "tokens": int(tokens) if tokens else max(1, len(text) // 4)}

    def close(self) -> None:
        try:
            import mlx.core as mx

            self.model = self.processor = None
            mx.clear_cache()
        except Exception:  # noqa: BLE001
            pass


def load_backend(spec: str, model: str) -> Any:
    if spec == "mlx":
        return MlxBackend(model)
    mod, _, attr = spec.partition(":")
    if not attr:
        raise RuntimeError(f"unknown reader backend {spec!r} (use mlx or module:attr)")
    return getattr(importlib.import_module(mod), attr)(model)


def serve(backend_spec: str, model: str) -> int:
    proto = os.fdopen(os.dup(1), "w", buffering=1, encoding="utf-8")   # the protocol channel
    os.dup2(2, 1)                          # native code that writes to fd 1 lands on stderr, not in the protocol
    sys.stdout = sys.stderr                                            # library noise goes to stderr

    def send(obj: dict[str, Any]) -> None:
        proto.write(json.dumps(obj, ensure_ascii=False) + "\n")
        proto.flush()

    t0 = time.perf_counter()
    try:
        backend = load_backend(backend_spec, model)
    except BaseException as exc:  # noqa: BLE001 - the parent shows this message
        send({"event": "error", "error": f"{type(exc).__name__}: {exc}"})
        return 2
    send({"event": "ready", "model": model, "backend": backend_spec,
          "load_s": round(time.perf_counter() - t0, 2), "rss_mb": rss_mb()})
    for line in sys.stdin:
        try:
            req = json.loads(line)
        except ValueError:
            continue
        if not isinstance(req, dict):
            continue
        if req.get("op") == "quit":
            break
        if req.get("op") != "read":
            continue
        t1 = time.perf_counter()
        try:
            res = backend.read(req["image"], req.get("prompt", ""), int(req.get("max_tokens") or 4096))
            send({"id": req.get("id"), "ok": True, "md": res.get("md", ""), "tokens": res.get("tokens", 0),
                  "seconds": round(time.perf_counter() - t1, 3), "rss_mb": rss_mb()})
        except Exception as exc:  # noqa: BLE001 - one page failed; the process stays up
            send({"id": req.get("id"), "ok": False, "error": f"{type(exc).__name__}: {exc}"})
    close = getattr(backend, "close", None)
    if callable(close):
        close()
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="document VLM reader process")
    ap.add_argument("--backend", default="mlx")
    ap.add_argument("--model", required=True)
    a = ap.parse_args(argv)
    return serve(a.backend, a.model)


if __name__ == "__main__":
    sys.exit(main())
