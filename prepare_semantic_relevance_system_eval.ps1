$ErrorActionPreference = "Stop"

Write-Host "============================================================"
Write-Host "SEMANTIC RELEVANCE SYSTEM EVALUATION v1.0 - CONTRACT FREEZE"
Write-Host "============================================================"
Write-Host "System under test: book recommender"
Write-Host "Measurement instrument: semantic_relevance_v0.29.0-r5"
Write-Host "Human relevance labels: NOT USED"
Write-Host "Queries: 12"
Write-Host "Recommendations/query: 10"
Write-Host "Expected judged pairs: 120"
Write-Host "Bootstrap: 5000 query-level replicates, 95% CI"
Write-Host "Recommender calls: DISABLED"
Write-Host "Judge calls: DISABLED"
Write-Host "============================================================"

uv run python .\evals\system_evaluation\freeze_semantic_relevance_system_eval.py
if ($LASTEXITCODE -ne 0) {
    throw "System evaluation contract freeze failed."
}

Write-Host ""
Write-Host "============================================================"
Write-Host "SYSTEM EVALUATION CONTRACT FREEZE: PASS"
Write-Host "No recommender or judge calls were made."
Write-Host "Next: collect the frozen top-10 recommendations for all 12 queries."
Write-Host "============================================================"
