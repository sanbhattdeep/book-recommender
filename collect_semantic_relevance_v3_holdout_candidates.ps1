$ErrorActionPreference = "Stop"

Write-Host "============================================================"
Write-Host "SEMANTIC RELEVANCE V3 - UNSEEN HOLDOUT RETRIEVAL"
Write-Host "============================================================"
Write-Host "Candidate generator: frozen v2 vector retrieval"
Write-Host "Candidate depth: 50"
Write-Host "Holdout queries: 24"
Write-Host "Reranker inference: DISABLED"
Write-Host "Judge execution: DISABLED"
Write-Host "Human labels: NOT USED"
Write-Host "============================================================"

uv run python `
  .\evals\system_evaluation\collect_semantic_relevance_v3_holdout_candidates.py

if ($LASTEXITCODE -ne 0) {
    throw "V3 unseen holdout retrieval failed."
}
