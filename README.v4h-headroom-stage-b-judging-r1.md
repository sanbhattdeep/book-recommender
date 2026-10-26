# V4H Headroom Stage-B Frozen Judge Execution r1

This package executes the unchanged `semantic_relevance_v0.29.0-r5` judge on
only the **251 fresh** Stage-B ranks-51-to-end pairs frozen by the Stage-B
judging-preparation boundary. The remaining 1/252 pair reuses an exact
same-query/same-ISBN score from the frozen V4H displayed-union score set and is
not re-judged.

The six eligible queries are exactly the Stage-A `NO_CLEAR_THROUGH_50` set:
`V4H10`, `V4H11`, `V4H13`, `V4H16`, `V4H21`, and `V4H23`.

The execution is resume-safe and persists after every completed fresh case. A
3-case smoke test and the later continuation use the same score artifact.

The already accepted non-semantic prominence span-echo and duplicate candidate
ID normalizations are carried forward unchanged. No semantic rule, judge prompt,
model, temperature, facet specification, deterministic scoring rule, candidate
set, release metric, release gate, or release decision is changed here.

No retrieval or reranking occurs. Human relevance labels remain unused. V4H is
consumed diagnostic evidence and the frozen V4B release `FAIL` is immutable.
