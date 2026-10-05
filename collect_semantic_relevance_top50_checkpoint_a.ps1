$ErrorActionPreference = "Stop"

Write-Host "============================================================"
Write-Host "SEMANTIC RELEVANCE TOP-50 DIAGNOSTIC - CHECKPOINT A"
Write-Host "============================================================"
Write-Host "Purpose: reconstruct + freeze top-50 from frozen v2 candidate generator"
Write-Host "Queries: 12"
Write-Host "Candidates/query: 50"
Write-Host "Expected query-book pairs: 600"
Write-Host "Frozen v2 top-10 reproduction: REQUIRED"
Write-Host "Historical v1 top-10 containment: DIAGNOSTIC / NON-BLOCKING"
Write-Host "Judge execution: DISABLED"
Write-Host "Human relevance labels: NOT USED"
Write-Host "Recommender modification: NONE"
Write-Host "============================================================"

uv run python `
  .\evals\system_evaluation\collect_semantic_relevance_top50_candidates.py

if ($LASTEXITCODE -ne 0) {
    throw "Top-50 diagnostic Checkpoint A failed. Do NOT proceed to score reuse or judging."
}

Write-Host ""
Write-Host "============================================================"
Write-Host "TOP-50 DIAGNOSTIC CHECKPOINT A COMPLETE"
Write-Host "No judge calls were performed."
Write-Host "Upload the captured console output for review before Checkpoint B."
Write-Host "============================================================"
