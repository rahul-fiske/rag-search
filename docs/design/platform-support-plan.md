# Platform support: Apple Silicon, Intel Mac, Linux

Work log and decisions for the 2.0.0 platform work (all decisions about the machine live in `rag_search/machine.py`).

## CPU quantization (measured)

`models.quantize = int8` runs the embedding and rerank models with 8-bit dynamic quantization on the CPU.
`scripts/retrieval_parity.py` compares it with full precision on the same index, using chunks of the published index as
queries (24 queries, dense + rerank, k=5), x86_64 under Rosetta on an Apple Silicon Mac:

| variant | hit@5 | MRR | top1 = reference | overlap@5 | load s | s/query |
|---|---|---|---|---|---|---|
| fp32 (reference) | 0.38 | 0.35 | 1.00 | 1.00 | 104.9 | 18.2 |
| int8 | 0.38 | 0.35 | 0.75 | 0.83 | 24.3 | 26.5 |

Findings: hit and MRR are unchanged, rankings move (top-1 differs in a quarter of the queries), loading is 4x faster,
queries are slower. Rosetta has no AVX, so the query timing is not representative of a real Intel CPU.

Decision: the default stays `off` on every machine (`machine.quantize_default()`); `int8` is an opt-in setting until it
can be timed on real Intel hardware (the `macos-15-intel` CI runner). Accuracy is best effort, not a blocker.
