# V4H Headroom Stage-B Classification r1

This package performs the deterministic classification that follows the completed
Stage-B judge run.

It combines:

- 1 exact same-query/same-ISBN frozen score, and
- 251 frozen fresh Stage-B judge scores,

into the complete 252-pair ranks 51-end score map for the six V4H queries that
remained `NO_CLEAR_THROUGH_50` after Stage A.

The Stage-B outcome rule was frozen before candidate selection:

- `HEADROOM_FOUND_51_END`: at least one frozen V4B rank 51-end candidate has
  `judge_score >= 3`.
- `NO_CLEAR_THROUGH_FULL_UNION`: no frozen V4B rank 51-end candidate has
  `judge_score >= 3`. Combined with the frozen Stage-A result, this establishes
  that the frozen V4B full-union candidate set contains no clear/strong result
  for that consumed-holdout query.

The package also creates a 10-query overall disposition while preserving the four
already-frozen `HEADROOM_FOUND_11_50` Stage-A outcomes.

No judge, model, retrieval, or reranker calls occur. No human relevance labels,
release metrics, release gates, or release-decision changes are allowed. V4H remains
consumed diagnostic data and the frozen V4B release decision remains `FAIL`.

## Order

1. Run `prepare_semantic_relevance_v4h_headroom_stage_b_classification.ps1`.
2. Inspect the zero-call preflight output.
3. Run `classify_semantic_relevance_v4h_headroom_stage_b.ps1` only after the preflight passes.

Classification contract SHA-256: `90c0d5295989f5517fa079ec16215cbb1b44f661fe5919cc3065babd89f9dda5`
