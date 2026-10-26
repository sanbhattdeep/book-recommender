# V4H Headroom Stage-A Classification r1

This package performs the deterministic classification that follows the completed
Stage-A judge run.

It combines:

- 60 exact same-query/same-ISBN frozen V4H scores, and
- 340 frozen fresh Stage-A judge scores,

into the complete 400-pair ranks 11-50 score map for the 10 V4H queries that had
`NO_CLEAR_TOP10` under V4B.

The classification rule was already frozen by the Stage-A definition contract:

- `HEADROOM_FOUND_11_50`: at least one ranks 11-50 candidate has `judge_score >= 3`.
  This proves displayed-ranking headroom for that query.
- `NO_CLEAR_THROUGH_50`: no ranks 11-50 candidate has `judge_score >= 3`.
  This remains unresolved. It does **not** prove full-union retrieval limitation;
  optional Stage B must inspect ranks 51-end before that conclusion.

No judge, model, retrieval, or reranker calls occur. No human relevance labels,
release metrics, release gates, or release-decision changes are allowed. V4H remains
consumed diagnostic data and the frozen V4B release decision remains `FAIL`.

## Order

1. Run `prepare_semantic_relevance_v4h_headroom_stage_a_classification.ps1`.
2. Inspect the zero-call preflight output.
3. Run `classify_semantic_relevance_v4h_headroom_stage_a.ps1` only after the preflight passes.

Classification contract SHA-256: `16228e03a1148d298aa0e2529265d3ddeee021062882b6c043e6cf9bdd33a5a9`
