# V5D v0.29.0-r5 judge execution r2

This is a narrow Windows line-ending repair of r1.

## What changed

The r1 preflight verified the historical duplicate-candidate-ID JSON patch using
a raw-byte SHA-256 check. On Windows the tracked file is CRLF, while the frozen
expected hash is its canonical-LF hash. r2 verifies that text artifact using the
existing raw-or-canonical-LF policy.

The same newline-safe verification is applied to the V5D carry-forward patch and
the V5D judge-execution contract to prevent the same failure after a Git checkout.

## What did not change

- frozen 452-case V5D judging input
- score inventory and judging-input lock
- `semantic_relevance_v0.29.0-r5`
- V5D facet specification
- judge prompt, model, temperature, or deterministic scoring
- span-echo normalization
- duplicate-ID normalization semantics
- execution-contract content
- smoke query IDs
- resume/failure policy
- development selection boundary

Execution contract SHA-256 (unchanged):
`aac315cbea6a1994f0ecd897267e34db38558a87c5d8a3184f1b9cd2d3abc033`

V5D duplicate-ID carry-forward SHA-256 (unchanged):
`d079bb55f2dfd98b5ab78ee2138faf1f1f2d5d4d90681616f962686817461e57`

Corrected verifier canonical-LF SHA-256:
`0bf32c2f6899475314baf20542a035a3b34cfed3149d1cdd6554cbe9be9699d6`
