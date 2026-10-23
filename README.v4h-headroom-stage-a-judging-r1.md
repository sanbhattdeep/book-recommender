# V4H Headroom Stage-A Frozen Judge Execution r1

This package executes the unchanged `semantic_relevance_v0.29.0-r5` judge on
the 340 novel Stage-A ranks 11–50 pairs. The other 60/400 Stage-A pairs reuse
exact same-query/same-ISBN scores already frozen in the V4H displayed-union
evaluation and are not re-judged here.

The execution is resume-safe and persists after every completed fresh case.
A limited smoke test and the later continuation use the same score artifact.

Two previously accepted non-semantic execution normalizations are installed:
the prominence span-echo normalization and the exact duplicate real-S# ID
normalization. The latter is carried forward via a new provenance amendment
because this same frozen judge has already exhibited and reproduced that
output-format failure on V4H.

No retrieval, reranking, release metrics, human labels, or release decision
changes occur in this package. V4H remains consumed diagnostic data and the
V4B release FAIL is immutable.
