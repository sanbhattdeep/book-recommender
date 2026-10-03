$ErrorActionPreference = "Stop"

Write-Host "============================================================"
Write-Host "SEMANTIC RELEVANCE SYSTEM EVAL - RECOMMENDATION COLLECTION"
Write-Host "============================================================"
Write-Host "System under test: frozen book recommender"
Write-Host "Queries: 12"
Write-Host "Top recommendations/query: 10"
Write-Host "Expected query-book pairs: 120"
Write-Host "Judge execution: DISABLED"
Write-Host "Human relevance labels: NOT USED"
Write-Host "============================================================"

uv run python `
  .\evals\system_evaluation\collect_semantic_relevance_system_recommendations.py

if ($LASTEXITCODE -ne 0) {
    throw "System recommendation collection failed."
}

Write-Host ""
Write-Host "============================================================"
Write-Host "RECOMMENDATION COLLECTION COMPLETE"
Write-Host "Judge has NOT been called."
Write-Host "Upload this console output for review before judge scoring."
Write-Host "============================================================"
