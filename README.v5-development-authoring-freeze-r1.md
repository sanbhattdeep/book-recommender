# Semantic relevance V5 — fresh V5D24 authoring freeze r1

This package authors and freezes the fresh one-shot V5 development selection set **before any V5D retrieval**.

## Boundary

- Depends on successful pinned-model qualification lock `67b1dd15403b5825086cce4c9e5c5af14cc883945f5a9626abaa4e5894b77211`.
- 24 fresh queries: `V5D01`–`V5D24`.
- Six frozen strata, four queries each.
- Query/facet authoring is candidate blind: no catalog inspection, V5D retrieval output, reranker output, judge output, or human relevance label is used.
- Prior Q/H/V4H and mechanical-probe query text is included only as an anti-overlap reference; it is not a model/parameter-selection source.
- Exact and token-near-duplicate reuse is mechanically rejected at Jaccard >= 0.35; a manual semantic-overlap declaration is also frozen.
- No retrieval, model/reranker inference, judging, development metric, or release metric is executed by this package.

## Order

1. Run `prepare_semantic_relevance_v5_development_authoring.ps1` and inspect the zero-call preflight.
2. Only after the preflight passes, run `freeze_semantic_relevance_v5_development_authoring.ps1`.
3. Only after the authoring lock exists may V5D R0/R1 candidate collection be prepared.

Frozen package source hashes before lock creation:

- query JSON: `1d93979d21d081afaecac63ee46c56330e80a2e7dbeeeea5d3bfa55a4b56387a`
- facet JSON: `355ac59604fc73039fe394ba36b9ef37c04cd0be447ff470bce669d8adeb4746`
- prior-query manifest: `32015f2426c1d72b2eb65db0e11c4be7ffdf390e14e00e00f29d0307e9f89056`
- preflight verifier: `95abd23e68771fd6d1a1276e90cfcac0a5de2cef103b6a1a7666df6eeec96776`
