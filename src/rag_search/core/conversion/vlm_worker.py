"""The document VLM reader, child side: ``python -m rag_search.core.conversion.vlm_worker``.

Loads one model through a *backend*, then answers read requests on stdin until told to quit (see
``vlm.py`` for the protocol).  Standard output carries the protocol only: whatever a library prints
is moved to standard error.

A backend is a class with ``__init__(model_id)``, ``read(image_path, prompt, max_tokens[, repetition_penalty]) ->
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


LOOP_CHECK_EVERY = 64                # tokens between two looks at the text for a loop


def rss_mb() -> float:
    """Peak resident memory of this process, MB (ru_maxrss is bytes on macOS, KB on Linux)."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return round(peak / (1024 * 1024) if sys.platform == "darwin" else peak / 1024, 1)


class MlxBackend:
    """mlx-vlm on Apple Silicon (``pip install 'rag-search[mac-vlm]'``).  Greedy decoding."""

    def __init__(self, model_id: str) -> None:
        try:
            from mlx_vlm import load, stream_generate
            from mlx_vlm.prompt_utils import apply_chat_template
            from mlx_vlm.utils import load_config
        except ImportError as exc:
            raise RuntimeError("mlx-vlm is not installed (pip install 'rag-search[mac-vlm]'; "
                               f"Apple Silicon only): {exc}") from exc
        self._stream, self._template = stream_generate, apply_chat_template
        self.model, self.processor = load(model_id)
        self.config = load_config(model_id)

    def read(self, image_path: str, prompt: str, max_tokens: int,
             repetition_penalty: float | None = None) -> dict[str, Any]:
        """Greedy decoding; the text is checked while it is generated and generation stops as soon as the
        model starts repeating itself (``stopped: "loop"``) instead of running to the token limit."""
        from rag_search.core.conversion import degenerate

        formatted = self._template(self.processor, self.config, prompt, num_images=1)
        kw: dict[str, Any] = {"max_tokens": max_tokens, "temperature": 0.0}
        if repetition_penalty:
            kw["repetition_penalty"] = float(repetition_penalty)
        parts: list[str] = []
        n, stopped, checked = 0, "", 0
        for chunk in self._stream(self.model, self.processor, formatted, [image_path], **kw):
            parts.append(str(getattr(chunk, "text", "") or ""))
            n = int(getattr(chunk, "generation_tokens", 0) or (n + 1))
            if n - checked >= LOOP_CHECK_EVERY:            # not "n % 64 == 0": a chunk may carry several tokens
                checked = n
                if degenerate.looping("".join(parts)):
                    stopped = "loop"
                    break
        return {"md": "".join(parts), "tokens": n, "stopped": stopped}

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
            extra = {"repetition_penalty": float(req["repetition_penalty"])} if req.get("repetition_penalty") else {}
            res = backend.read(req["image"], req.get("prompt", ""), int(req.get("max_tokens") or 4096), **extra)
            send({"id": req.get("id"), "ok": True, "md": res.get("md", ""), "tokens": res.get("tokens", 0),
                  "stopped": res.get("stopped", ""),
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
