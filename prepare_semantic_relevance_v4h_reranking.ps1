$ErrorActionPreference = "Stop"
Write-Host "============================================================"
Write-Host "SEMANTIC RELEVANCE V4H - V3B/V4B RERANKING PREFLIGHT"
Write-Host "============================================================"
Write-Host "Cross-encoder inference:    DISABLED"
Write-Host "Retrieval calls:            DISABLED"
Write-Host "Judge execution:            DISABLED"
Write-Host "Release analysis:           DISABLED"
Write-Host "============================================================"

uv run python .\evals\system_evaluation\verify_semantic_relevance_v4h_reranking.py
if ($LASTEXITCODE -ne 0) {
    throw "V4H reranking preflight failed. Do not run reranker inference."
}
