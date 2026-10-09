#!/usr/bin/env python3
"""Does a faster configuration find what the full one finds?  (run in rag-search's own environment)

    "$(uv tool dir)/rag-search-local/bin/python" scripts/retrieval_parity.py [--queries 40] [--k 5] [--seed 1]
                                                                             [--variant NAME=ENV=VAL,ENV=VAL ...]

Needs no labels.  Queries are made from chunks of the published index: a stretch of 8-14 words from the middle of
a prose chunk, so the chunk it came from is the right answer ("self-retrieval").  Each variant is a set of environment
variables (device, quantization ...); it runs in its own process on the same queries and the same index, read-only, and
the report says, against the first variant (the reference):

* hit@k and MRR of the source chunk: how often and how high the right chunk comes (the accuracy that can be measured
  without a person),
* top-1 agreement and overlap@k with the reference: how much the ranking moves,
* the time to load the models and the mean time of a query.

By default the dense and rerank stages are measured (BM25 would find an exact stretch of words by itself and hide the
difference between model variants); ``--stages`` changes that.  Nothing from the documents is printed or stored: the
queries live in a temporary file that is removed, the report holds numbers only.  Speed figures are only comparable
between variants on the same machine; accuracy figures do not depend on the machine.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

DEFAULT_VARIANTS = ["reference", "cpu_fp32=RAG_SEARCH_DEVICE=cpu,RAG_SEARCH_QUANTIZE=off",
                    "cpu_int8=RAG_SEARCH_DEVICE=cpu,RAG_SEARCH_QUANTIZE=int8"]


def _key(collection: str, file: str, snippet: str) -> str:
    return hashlib.md5(f"{collection}|{file}|{snippet}".encode()).hexdigest()[:16]


def make_queries(n: int, seed: int) -> list[dict]:
    """*n* queries from chunks of the published index (collection, source key, the words)."""
    from rag_search.core.search import SNIPPET_CHARS
    from rag_search.paths import ALL_DIR, NODES_FILE, get_paths, read_json

    root = get_paths().current_gen()
    if root is None:
        raise SystemExit("nothing is published: index some documents first")
    rng = random.Random(seed)
    pool = []
    for coll_dir in sorted(p for p in (root / "index").iterdir() if (p / ALL_DIR / NODES_FILE).exists()):
        nodes = read_json(coll_dir / ALL_DIR / NODES_FILE).get("nodes", [])
        pool += [(coll_dir.name, nd) for nd in nodes]
    rng.shuffle(pool)
    out = []
    for coll, nd in pool:
        text = re.sub(r"<[^>]+>|[#*|_`>-]+", " ", nd["text"])
        words = text.split()
        letters = sum(c.isalpha() for c in text) / max(1, len(text))
        if len(words) < 60 or letters < 0.75:
            continue
        start = rng.randint(len(words) // 4, max(len(words) // 4, len(words) * 3 // 4 - 14))
        query = " ".join(words[start:start + rng.randint(8, 14)])
        meta = nd["metadata"]
        snippet = nd["text"] if len(nd["text"]) <= SNIPPET_CHARS else nd["text"][:SNIPPET_CHARS] + " …"
        out.append({"q": query, "key": _key(coll, meta.get("file_name", ""), snippet)})
        if len(out) >= n:
            break
    return out


def run_variant(queries_file: str, out_file: str, k: int, stages: str, rerank_pool: int) -> None:
    """The child: load the models, run every query, write the keys and the times."""
    from rag_search.core.search import SearchEngine
    from rag_search.paths import get_paths

    queries = json.loads(Path(queries_file).read_text())
    t0 = time.perf_counter()
    eng = SearchEngine(get_paths())
    eng.load_models()
    gen = eng.prepare_generation()
    eng.install(gen)
    load_s = time.perf_counter() - t0
    from rag_search.core.search import SNIPPET_CHARS  # noqa: F401  (same constant as make_queries)

    results, times = [], []
    for q in queries:
        t = time.perf_counter()
        res = eng.search(q["q"], top_k=k, stages=stages, rerank_pool_n=rerank_pool or None)
        times.append(time.perf_counter() - t)
        results.append([_key(r["collection"], r["file"], r["text"]) for r in res["results"]])
    Path(out_file).write_text(json.dumps({"load_s": load_s, "times": times, "results": results,
                                          "device": getattr(eng.embedder, "device", "?")}))


def report(names: list[str], runs: dict[str, dict], queries: list[dict], k: int) -> list[str]:
    ref = runs[names[0]]["results"]
    lines = [f"{'variant':<22}{'device':<7}{'hit@%d' % k:>7}{'MRR':>7}{'top1=ref':>10}{'overlap@%d' % k:>11}{'load s':>8}{'s/query':>9}"]
    for name in names:
        r = runs[name]
        hits = mrr = top1 = overlap = 0.0
        for q, got, want in zip(queries, r["results"], ref):
            if q["key"] in got:
                hits += 1
                mrr += 1 / (got.index(q["key"]) + 1)
            top1 += bool(got and want and got[0] == want[0])
            overlap += len(set(got) & set(want)) / max(1, len(set(want)))
        n = max(1, len(queries))
        lines.append(f"{name:<22}{r['device']:<7}{hits / n:>7.2f}{mrr / n:>7.2f}{top1 / n:>10.2f}{overlap / n:>11.2f}"
                     f"{r['load_s']:>8.1f}{sum(r['times']) / n:>9.2f}")
    return lines


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--queries", type=int, default=40)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--stages", default="dense,rerank")
    ap.add_argument("--rerank-pool", type=int, default=0, help="rerank this many candidates (0 = the default)")
    ap.add_argument("--variant", action="append", help="NAME or NAME=ENV=VAL,ENV=VAL (the first is the reference)")
    ap.add_argument("--run", nargs=2, metavar=("QUERIES", "OUT"), help=argparse.SUPPRESS)
    a = ap.parse_args(argv)
    if a.run:
        run_variant(a.run[0], a.run[1], a.k, a.stages, a.rerank_pool)
        return 0
    variants = a.variant or DEFAULT_VARIANTS
    queries = make_queries(a.queries, a.seed)
    print(f"{len(queries)} queries from the published index; stages {a.stages}; k={a.k}", flush=True)
    names, runs = [], {}
    with tempfile.TemporaryDirectory(prefix="parity-") as tmp:
        qf = Path(tmp) / "queries.json"
        qf.write_text(json.dumps(queries))
        for spec in variants:
            name, _, envspec = spec.partition("=")
            env = dict(os.environ)
            for pair in filter(None, envspec.split(",")):
                key, _, val = pair.partition("=")
                env[key] = val
            out = Path(tmp) / f"{name}.json"
            t0 = time.perf_counter()
            cp = subprocess.run([sys.executable, __file__, "--run", str(qf), str(out), "--k", str(a.k), "--stages", a.stages,
                                 "--rerank-pool", str(a.rerank_pool)], env=env, capture_output=True, text=True, check=False)
            if cp.returncode:
                print(f"{name}: failed\n{cp.stderr[-600:]}", file=sys.stderr)
                return 1
            names.append(name)
            runs[name] = json.loads(out.read_text())
            print(f"  {name}: done in {time.perf_counter() - t0:.0f}s", flush=True)
    print("\n".join(report(names, runs, queries, a.k)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
