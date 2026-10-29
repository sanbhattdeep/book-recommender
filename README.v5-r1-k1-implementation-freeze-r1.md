# V5 R1/K1 Implementation Source Freeze r1

This package implements the already-frozen V5 R1/K1 architecture without authoring or executing V5D.

## Frozen components

- R1: `BAAI/bge-m3` revision `5617a9f61b028005a4858fdac845db406aefb181`, dense embeddings only, normalized dot-product ranking, depth 50.
- K1: `BAAI/bge-reranker-v2-m3` revision `953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e`, raw-logit score-only ranking with exact-score tie-breaks.
- R1 batch size is frozen at 16; K1 batch size remains the architecture-frozen 4.

## Dependency boundary

Before running the freeze, add exactly:

```powershell
uv add sentence-transformers==6.2.0
```

The freeze requires:

- sentence-transformers 6.2.0
- transformers 5.15.0
- torch 2.11.0+cu128

It also verifies `books_with_emotions.csv` and `tagged_description.txt` are unchanged from HEAD, then captures their SHA-256 hashes into the implementation lock.

## What this step executes

Only deterministic unit tests over synthetic vectors/scores. It does **not** load/download either model, run retrieval/reranking, access V5D, execute the judge, or calculate selection/release metrics.

## Run

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File .\freeze_semantic_relevance_v5_r1_k1_implementation.ps1 2>&1 | `
  Tee-Object -FilePath .\semantic_relevance_v5_r1_k1_implementation_freeze_output.txt
```

After this passes, a separate qualification step will load the exact pinned model revisions and require two repeated inference passes on a fixed non-V5D probe set to produce identical rank-order signatures.
