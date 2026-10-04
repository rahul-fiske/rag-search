"""Tokenizer + a small vectorised BM25 (Okapi).  Needs only numpy."""

from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np

from ..spec import BM25_B, BM25_K1, TOKENIZER_VERSION  # noqa: F401  (re-exported)

_TOKEN_RE = re.compile(r"\w+(?:[.\-_/:]\w+)*", re.UNICODE)
_SPLIT_RE = re.compile(r"[.\-_/:]")


def tokenize(text: str) -> list[str]:
    """Lower-cased word tokens.  Compound tokens ('svm-name', '9.16.1', 'a/b')
    are kept whole *and* split into their parts so both spellings match."""
    out: list[str] = []
    for m in _TOKEN_RE.finditer(text.lower()):
        tok = m.group(0)
        out.append(tok)
        if _SPLIT_RE.search(tok):
            out.extend(p for p in _SPLIT_RE.split(tok) if p)
    return out


class BM25:
    def __init__(self, docs_tokens: list[list[str]], k1: float = BM25_K1, b: float = BM25_B):
        self.k1, self.b = k1, b
        self.n = len(docs_tokens)
        lengths = np.array([len(t) for t in docs_tokens], dtype=np.float32)
        self.avgdl = float(lengths.mean()) if self.n else 0.0
        self._norm = k1 * (1.0 - b + b * lengths / (self.avgdl or 1.0))
        postings: dict[str, tuple[list[int], list[int]]] = {}
        for i, toks in enumerate(docs_tokens):
            for term, tf in Counter(toks).items():
                ids, tfs = postings.setdefault(term, ([], []))
                ids.append(i)
                tfs.append(tf)
        self._post: dict[str, tuple[np.ndarray, np.ndarray, float]] = {}
        for term, (ids, tfs) in postings.items():
            df = len(ids)
            idf = math.log(1.0 + (self.n - df + 0.5) / (df + 0.5))
            self._post[term] = (
                np.asarray(ids, dtype=np.int64),
                np.asarray(tfs, dtype=np.float32),
                idf,
            )

    def scores(self, query_tokens: list[str]) -> np.ndarray:
        s = np.zeros(self.n, dtype=np.float32)
        for term in set(query_tokens):
            entry = self._post.get(term)
            if entry is None:
                continue
            ids, tfs, idf = entry
            s[ids] += idf * (tfs * (self.k1 + 1.0)) / (tfs + self._norm[ids])
        return s
