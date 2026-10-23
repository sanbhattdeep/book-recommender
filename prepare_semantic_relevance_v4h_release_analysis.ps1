$ErrorActionPreference = "Stop"
Write-Host "============================================================"
Write-Host "SEMANTIC RELEVANCE V4H - FINAL RELEASE ANALYSIS PREFLIGHT"
Write-Host "============================================================"
Write-Host "Judge execution:            DISABLED"
Write-Host "Model/retrieval/reranker:   DISABLED"
Write-Host "Human relevance labels:     NOT USED"
Write-Host "Release metrics:            DISABLED"
Write-Host "Release decision:           DISABLED"
Write-Host "============================================================"

uv run python .\evals\system_evaluation\verify_semantic_relevance_v4h_release_analysis.py
if ($LASTEXITCODE -ne 0) {
    throw "V4H final release-analysis preflight failed. Do not run analysis."
}
