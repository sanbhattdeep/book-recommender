# V5D judging-input / exact score-reuse freeze r1

This package prepares and freezes the relevance-judge input for the already-frozen
V5D B/R/K/RK displayed-pair universe.

## Boundary

- frozen displayed pairs: 452
- allowed historical reuse: exact same query text + same canonical ISBN only
- expected reusable historical scores: 0
- expected fresh V5D judge cases: 452
- judge candidate to be used later: `semantic_relevance_v0.29.0-r5`
- retrieval/reranking: disabled
- judge execution: disabled
- human relevance labels: not used
- development metrics/selection: not calculated

The zero-reuse result is established from the frozen prior-query manifest:
V5D01-V5D24 are exactly disjoint from the prior Q01-Q12, H01-H24 and
V4H01-V4H24 query texts used by compatible historical score families.
No cross-query, approximate-query or ISBN-only score reuse is permitted.

## Sequence

1. Run `prepare_semantic_relevance_v5_development_judging_input.ps1`.
2. Review the preflight output.
3. Run `freeze_semantic_relevance_v5_development_judging_input.ps1` only after preflight passes.
4. Do not execute the judge until `v5d_judging_input.lock.json` has been reviewed.

Contract SHA-256: `d2e013323ddfc8d87154277c79165d242038b29978b6fc20243ebcd6bf5a69dd`
Verifier canonical-LF SHA-256: `f6e252442e0768a7d7647faddb8e769df0ed07bc195d53821cc6764db4ebffbd`
