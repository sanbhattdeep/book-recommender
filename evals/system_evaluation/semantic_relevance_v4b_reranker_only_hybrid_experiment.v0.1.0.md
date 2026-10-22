# Semantic Relevance V4B — Reranker-only ranking over the frozen hybrid union

## Decision

V4B keeps the useful part of V4A — candidate expansion from the exact frozen
dense50 + BM25-50 union — and removes the part now most suspect after the V4A
development result: feeding hybrid retrieval rank back into final ranking via
an equal-weight reciprocal-rank fusion with reranker rank.

V4B final ordering is therefore:

1. `reranker_score` descending;
2. `hybrid_retrieval_rank` ascending **only for exact reranker-score ties**;
3. `isbn13` ascending.

There is no epsilon tie bucket, score rounding, calibrated blend, or retrieval
weight in ordinary non-tie ordering.

## Why this is the next isolated hypothesis

The frozen 17-query retrieval diagnostic shows that BM25 candidate discovery is
real but concentrated:

- 33–34 BM25-only clear/strong candidates;
- 7/17 queries gain at least one BM25-only clear/strong candidate;
- the limitation-class mix changes from 7 retrieval-limited / 4 reranker-limited
  / 6 mixed to 3 retrieval-limited / 6 reranker-limited / 8 mixed;
- among the original seven retrieval-limited queries, three move to mixed, one
  moves to reranker-limited, and three remain retrieval-limited;
- one original mixed query moves to reranker-limited;
- the conclusion is invariant to every allowed score 0–4 for the single
  quarantined H14 case.

This supports keeping BM25 in candidate generation. Because V4A itself was not
selected on the frozen development comparison, the clean next experiment is to
remove retrieval-rank influence after cross-encoder scoring and ask whether the
same expanded pool can be exploited better by the frozen reranker.

## Execution economy

V4B development does not require rerunning dense retrieval, BM25 retrieval, or
the cross-encoder. It must reuse the exact frozen V4A hybrid union and the exact
frozen V4A full-union reranker-score artifact. The V4B ranking step is therefore
deterministic.

## Authority boundary

H01-H24 are development-only. V4B is compared against frozen V3B under the same
lexicographic selection rule used for V4A. Retrieval diagnostics may explain the
outcome but cannot override selection.

If V4B is selected, a new unseen V4H01-V4H24 holdout is required before any
final system claim or release decision.
