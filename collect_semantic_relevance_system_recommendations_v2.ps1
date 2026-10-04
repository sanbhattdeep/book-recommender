$ErrorActionPreference = "Stop"

Write-Host "============================================================"
Write-Host "SEMANTIC RELEVANCE SYSTEM EVAL V2 - RECOMMENDATION COLLECTION"
Write-Host "============================================================"
Write-Host "System under test: semantic-recommender-v2-orderfix"
Write-Host "Comparison benchmark: frozen v1 contract + 12 queries"
Write-Host "Queries: 12"
Write-Host "Top recommendations/query: 10"
Write-Host "Expected query-book pairs: 120"
Write-Host "Judge execution: DISABLED"
Write-Host "Human relevance labels: NOT USED"
Write-Host "v1 run namespace: PRESERVED / NOT WRITTEN"
Write-Host "============================================================"

uv run python `
  .\evals\system_evaluation\collect_semantic_relevance_system_recommendations_v2.py

if ($LASTEXITCODE -ne 0) {
    throw "V2 system recommendation collection failed."
}

Write-Host ""
Write-Host "============================================================"
Write-Host "V2 RECOMMENDATION COLLECTION COMPLETE"
Write-Host "Judge has NOT been called."
Write-Host "Upload this console output for review before judge scoring."
Write-Host "============================================================"
