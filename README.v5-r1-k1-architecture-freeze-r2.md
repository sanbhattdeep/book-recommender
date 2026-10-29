# V5 R1/K1 Architecture Freeze r2

This package freezes the exact architecture hypotheses for the R1 retrieval factor and K1 ranking factor before component implementation or V5D authoring.

- **R1:** replace only the dense `nomic-embed-text` branch with pinned `BAAI/bge-m3`; retain BM25-50, dense-50 depth, RRF k=60, no pre-reranker truncation, and <=100 unique candidates.
- **K1:** replace only `BAAI/bge-reranker-base` with pinned `BAAI/bge-reranker-v2-m3`; retain the V4B raw-logit score-only final ordering policy and exact input template.

No V5D queries are authored or executed. No retrieval/model/reranker/judge calls occur.

Run:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File .\freeze_semantic_relevance_v5_r1_k1_architecture.ps1 2>&1 |
  Tee-Object -FilePath .\semantic_relevance_v5_r1_k1_architecture_freeze_output.txt
```

JSON spec SHA-256: `7b2c9d7916b62f4b65e01c97b7b2a751b07bbc051af6cf8c5304eafb2f31b1d2`
Markdown spec SHA-256: `3b4a9fd427c7658775fbc1adb9212cd2a3e67e79728b985848bddbb7bd18e4ec`


## r2 verifier repair

r2 changes only the freeze verifier. Text-source hashes are verified with the same raw-or-canonical-LF rule used by the original V4B implementation freeze, so Windows CRLF checkout does not look like semantic source drift. JSON lock artifacts remain raw-byte verified. The R1/K1 architecture specs and their frozen SHA-256 values are unchanged from r1.
