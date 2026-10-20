$ErrorActionPreference = "Stop"

Write-Host "============================================================"
Write-Host "SEMANTIC RELEVANCE V4A - BUILD/FREEZE DEV SCORE REUSE"
Write-Host "============================================================"
Write-Host "New judge execution: DISABLED"
Write-Host "Human labels:        DISABLED"
Write-Host "Release analysis:    DISABLED"
Write-Host "============================================================"

uv run python `
  .\evals\system_evaluation\prepare_semantic_relevance_v4a_dev_score_reuse.py

if ($LASTEXITCODE -ne 0) {
    throw "V4a development score-reuse preparation failed."
}
