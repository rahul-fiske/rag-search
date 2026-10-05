"""Keeps the portable tier (A) independent of what only the real tier (B) installs.

``block()`` makes ``import docling`` (and torch, sentence_transformers, transformers, huggingface_hub) fail and
puts Hugging Face in offline mode, so a portable test that quietly needs a real library, or tries to download a
model, fails at once -- on a machine that has them as well as on one that has not.  ``importlib.util.find_spec``
then reports them as not installed.  ``unblock()`` undoes it for the real tier, which shares the process when
everything is run with ``discover -s tests``.  See docs/design/test-strategy.md.
"""

from __future__ import annotations

import os
import sys

BLOCKED = ("docling", "docling_core", "docling_parse", "torch", "sentence_transformers", "transformers",
           "huggingface_hub")
OFFLINE = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")

_blocked: set[str] = set()
_env_set: set[str] = set()


def block() -> None:
    for name in BLOCKED:
        if name not in sys.modules:                 # something already imported stays: it cannot be un-imported
            sys.modules[name] = None                # type: ignore[assignment]  # import raises ImportError
            _blocked.add(name)
    for key in OFFLINE:
        if key not in os.environ:
            os.environ[key] = "1"
            _env_set.add(key)


def unblock() -> None:
    for name in list(_blocked):
        if sys.modules.get(name, 0) is None:
            del sys.modules[name]
        _blocked.discard(name)
    for key in list(_env_set):
        os.environ.pop(key, None)
        _env_set.discard(key)


def active() -> bool:
    return bool(_blocked)
