$ErrorActionPreference = "Stop"
Write-Host "============================================================"
Write-Host "SEMANTIC RELEVANCE V4H - FINAL HOLDOUT RETRIEVAL PREFLIGHT"
Write-Host "============================================================"
Write-Host "Dense/BM25 retrieval:       DISABLED"
Write-Host "Reranker inference:         DISABLED"
Write-Host "Judge execution:            DISABLED"
Write-Host "Release analysis:           DISABLED"
Write-Host "============================================================"

uv run python .\evals\system_evaluation\verify_semantic_relevance_v4h_retrieval_preflight.py
if ($LASTEXITCODE -ne 0) {
    throw "V4H retrieval preflight failed. Do not run V4H retrieval."
}
