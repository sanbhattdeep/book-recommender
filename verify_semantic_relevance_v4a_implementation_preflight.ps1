$ErrorActionPreference = "Stop"

Write-Host "============================================================"
Write-Host "SEMANTIC RELEVANCE V4A - IMPLEMENTATION CANDIDATE PREFLIGHT"
Write-Host "============================================================"
Write-Host "Experiment retrieval: DISABLED"
Write-Host "Catalog indexing:     DISABLED"
Write-Host "Reranker inference:   DISABLED"
Write-Host "Judge execution:      DISABLED"
Write-Host "============================================================"

uv run python `
  .\evals\system_evaluation\verify_semantic_relevance_v4a_implementation_preflight.py

if ($LASTEXITCODE -ne 0) {
    throw "V4a implementation candidate preflight failed."
}
