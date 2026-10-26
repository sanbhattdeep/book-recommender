# V4H Headroom Stage-B Judging Preparation r1

This package performs **no judge, model, retrieval, or reranker execution**.

It consumes the frozen 252-pair Stage-B candidate inventory for the six
`NO_CLEAR_THROUGH_50` V4H queries and partitions those pairs into:

1. exact frozen-score reuse, allowed only for the same `(query_id, isbn13)`;
2. fresh judging input for every remaining pair.

Two already-frozen score sources are checked:

- the original V4H displayed-union judge scores;
- the completed Stage-A fresh judge scores.

The Stage-A source is included for complete provenance even though the frozen
rank windows are disjoint; any unexpected overlap or conflicting provenance is
rejected rather than silently reused.

Cross-query reuse and approximate matching are forbidden. The unchanged
`semantic_relevance_v0.29.0-r5` measurement instrument remains pinned for any
fresh Stage-B judging that is required later.

V4H is consumed diagnostic evidence. This package cannot recalculate release
metrics or modify the frozen V4B `FAIL` decision.
