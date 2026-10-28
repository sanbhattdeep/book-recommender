# V5D factorial development analysis r2

This is a narrow pandas column-access repair of r1.

## Observed r1 failure

`g.query` resolved to the pandas `DataFrame.query()` method instead of the
`"query"` column, causing:

`AttributeError: 'function' object has no attribute 'iloc'`

The failure occurred during in-memory query-metric construction, before any
development metric/selection artifacts were written.

## r2 repair

Use explicit bracket-based column access (`frame["column"]`) in analysis code,
including the `query` field and related dataframe/series accesses. This is a
mechanical Python/pandas repair only.

## Unchanged boundaries

- analysis contract
- frozen B/R/K/RK recommendations
- frozen 452 judge scores and score lock
- metric definitions
- bootstrap: 5000 query-level replicates, seed 20261002
- factorial contrasts
- lexicographic selection order
- exact tie policy
- candidate-depth diagnostic boundary
- no model/judge/retrieval/reranker calls
- no release decision

Analysis contract SHA-256 (unchanged):
`1d920bc842eb0dd540259620110e29706dcdca5427fa21ea4ae92daf03b753b5`

Verifier SHA-256 (unchanged):
`34295163ee2beea6d17e1b5d6ede7dd42e5ce4fc84ae556579522d7c13dee947`

Corrected analyzer SHA-256:
`411db007171fd1f9a064161df92f2a74cce7c0cb0e4bebec9dbf1eea82d40a7a`
