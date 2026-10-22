$ErrorActionPreference = "Stop"
Write-Host "============================================================"
Write-Host "SEMANTIC RELEVANCE V4H - FINAL HOLDOUT CANDIDATE RETRIEVAL"
Write-Host "============================================================"
Write-Host "Dense retrieval:            ENABLED"
Write-Host "BM25 retrieval:             ENABLED"
Write-Host "Hybrid union:               ENABLED"
Write-Host "Reranker inference:         DISABLED"
Write-Host "Judge execution:            DISABLED"
Write-Host "Release analysis:           DISABLED"
Write-Host "============================================================"

uv run python .\evals\system_evaluation\collect_semantic_relevance_v4h_retrieval.py
if ($LASTEXITCODE -ne 0) {
    throw "V4H retrieval failed. Preserve any partial run directory and review before retrying."
}
