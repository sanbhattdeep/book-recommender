# V4H Headroom Stage-A Judging Preparation r1

This package does not execute the judge.

It takes the frozen 400-pair ranks 11–50 Stage-A candidate inventory and first
checks whether any exact `(query_id, isbn13)` pairs were already scored in the
frozen 382-pair V4H displayed-union evaluation.

Exact same-query/same-book scores are reused. All remaining pairs are placed in
a fresh judging input for the unchanged `semantic_relevance_v0.29.0-r5`
measurement instrument.

No cross-query reuse and no approximate matching are allowed.

V4H remains consumed diagnostic data and the frozen V4B release FAIL cannot be
changed by this stage or any later headroom result.
