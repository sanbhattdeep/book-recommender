# V5H final release analysis r1

Authoritative system: **K=R0K1**. B=R0K0 is secondary and cannot override K.

Frozen judge scores SHA-256:
`336e4374bbafdd50ebf7149d6edce00895bb072b8f4c258ec070c2ad6b530a83`

Frozen judge-scoring lock SHA-256:
`35ea4964e039aae36efb92771180531125f5236ae4328e66f632ed10e03a6ae6`

Original eight gates are verified from `semantic_relevance_system_eval.v1.0.0.json`.
Bootstrap: 5000 query-cluster replicates, seed 20261002.
Analysis makes zero judge/model/retrieval/reranker calls.

Contract SHA-256: `e9e5a3b327e796982cc05b9fee16cbcfb0e2725dbdbc90d7d9726397abe46ac7`
Verifier SHA-256: `027bedb6c9712eb0f15feec1d81392c936a9880dc43338ec65715ee77ecf437e`
Analyzer SHA-256: `02d00d9d13447e97fc10fb8a34e9536573e9579a30c14227205e341b699243cc`


## r2 preflight-schema repair

The r1 preflight verified all frozen artifact hashes and then failed because it
assumed a historical JSON field name inside the immutable V5D development
selection lock.

r2 removes only that brittle schema dependency:

- the V5D selection lock is still required and verified by its exact frozen
  SHA-256;
- the selected holdout identity `K=R0K1` is verified from the later frozen
  V5H authoring lock, which is independently SHA-bound and is the identity
  carried through V5H retrieval, reranking, judging, and final analysis.

No frozen scores, candidate selection, release gates, bootstrap settings,
metric definitions, or decision policy are changed.


## r3 analyzer numeric repair

The r2 analyzer passed all frozen-boundary checks and then failed during the
first nDCG calculation because it called:

`np.arange(2, len(s)+2, float)`

NumPy interprets the third positional argument as `step`, not `dtype`.
r3 changes only that expression to:

`np.arange(2, len(s)+2, dtype=float)`

No score evidence, metric definition, release gate, bootstrap setting,
selected candidate, decision policy, or verifier logic is changed.
