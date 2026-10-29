# Semantic Relevance V5 — R1/K1 Architecture Freeze v0.1.0

## Goal
Freeze one retrieval-only hypothesis (`R1`) and one ranking-only hypothesis (`K1`) for the already-frozen V5 B/R/K/RK factorial program. This step performs no retrieval, model, reranker, judge, or V5D execution.

## R1 — dense retriever replacement only

R1 replaces the `nomic-embed-text` dense branch with `BAAI/bge-m3` pinned at revision `5617a9f61b028005a4858fdac845db406aefb181`.

Everything else stays fixed: dense depth 50, the existing deterministic BM25 title+description branch at depth 50, BM25 k1=1.5/b=0.75, RRF k=60/equal weights, ISBN deduplication, no pre-reranker truncation, and a hard maximum of 100 unique candidates/query.

BGE-M3 is used in **dense-only** mode. Sparse and ColBERT modes are explicitly prohibited for R1. The raw query is embedded without an instruction, and the exact current tagged-description payload is used as the document representation. Embeddings are L2-normalized and ranked by dot product/cosine similarity.

This makes R1 a single causal question: **does a stronger dense retriever improve candidate discovery while the lexical branch and downstream ranking remain unchanged?**

## K1 — cross-encoder replacement only

K1 replaces `BAAI/bge-reranker-base` with `BAAI/bge-reranker-v2-m3` pinned at revision `953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e`.

The V4B ranking policy remains unchanged: raw reranker logit descending, no calibration/normalization/fusion, exact-score ties only broken by retrieval rank then ISBN. The candidate text remains exactly `Title: {title}
Description: {description}` and max sequence length remains 512.

K1 reranks whichever candidate set its R factor supplies and performs no retrieval itself.

## Factorial systems

| System | Retrieval | Ranking |
|---|---|---|
| B | R0 | K0 |
| R | R1 | K0 |
| K | R0 | K1 |
| RK | R1 | K1 |

`RK` is only the mechanical composition of the frozen R1 and K1 implementations.

## Why this is intentionally narrow

The V4H diagnostic established that some failures have ranking/display headroom while others have no clear/strong candidate anywhere in the frozen V4B union. It justifies testing both factors, but V4H is consumed evidence and cannot be used to tune V5 parameters. Therefore this architecture changes one model per factor and freezes all surrounding mechanics.

## Next boundary

After this architecture freeze passes, implement R1 and K1, add deterministic unit/repeatability tests, and freeze their exact source/model/dependency hashes. **Only after both implementation locks exist may V5D01–V5D24 be authored.**
