$ErrorActionPreference = "Stop"

Write-Host "============================================================"
Write-Host "SEMANTIC RELEVANCE V4A - DEV SCORE REUSE PACKAGE PREFLIGHT"
Write-Host "============================================================"
Write-Host "Score inventory:   DISABLED"
Write-Host "Judge execution:   DISABLED"
Write-Host "Human labels:      DISABLED"
Write-Host "Release analysis:  DISABLED"
Write-Host "============================================================"

uv run python `
  .\evals\system_evaluation\verify_semantic_relevance_v4a_dev_score_reuse_package.py

if ($LASTEXITCODE -ne 0) {
    throw "V4a development score-reuse package preflight failed."
}
