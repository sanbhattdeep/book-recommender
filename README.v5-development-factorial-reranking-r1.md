# V5D Factorial Reranking r1

This package freezes and executes the ranking-only V5 development stage over the already-frozen candidate run `20261026T124615Z_candidate_collection`.

Systems are materialized exactly as preregistered: B=R0K0, R=R1K0, K=R0K1, RK=R1K1. No retrieval is rerun and candidate membership is not changed. K0 and K1 each run twice on both candidate unions; complete per-query rank order must be identical across the two passes. Judge execution and development selection remain disabled.

The package also records first-pass reranker latency and combines it with the already-frozen R0/R1 + shared-BM25 first-pass latency only to preserve the preregistered exact-tie fallback evidence.

Candidate-collection lock: `d6c863d5dcd69ac6230ac00e60ee05aa9867701b2d15e42b91de0d05c8cd2e03`  
Reranking contract SHA-256: `622f2fbb599d455d1641535884a0adf08c7e4b6b52f7db227a2b3845022c1683`

Run the preflight first. Do not run the execution script until the preflight output has been reviewed.
