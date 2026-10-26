# V5 R1/K1 pinned-model qualification r1

This package qualifies the exact frozen R1 and K1 model revisions **before any V5D query is authored or executed**.

It uses only three fixed synthetic probe queries and eight synthetic probe documents embedded in the frozen contract. These probes are not taken from V4H, the book catalog, or V5D and are not used for semantic-quality selection.

Pass/fail is limited to pinned revision resolution, valid finite outputs, and repeatability across two identical inference passes. No expected semantic winner is asserted.

## Order

1. Run `prepare_semantic_relevance_v5_r1_k1_model_qualification.ps1` and inspect the zero-call preflight.
2. Only after approval, run `qualify_semantic_relevance_v5_r1_k1_models.ps1`.

The execution may download/cache the two pinned Hugging Face model revisions if they are not already local. It does not run catalog retrieval, BM25, V5D, the judge, development selection, or release analysis.

Contract SHA-256: `36bef237ae7b196c39022ce098fc158ee7cd02ffc97393b3984a22113db1194b`
Verifier SHA-256: `08d66428303f7a41062636642cb2c20a99ed3267476dc18e1f36bee7c19c9f43`
Execution source SHA-256: `6213db9b9056ff12927e0395b5901c34b07216c45bfce3bae9b352bd516d97cd`
