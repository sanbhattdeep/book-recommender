$ErrorActionPreference = "Stop"

Write-Host "============================================================"
Write-Host "SEMANTIC RELEVANCE V3 - POST-HOC STAGE C CLASSIFICATION"
Write-Host "============================================================"
Write-Host "Evidence: frozen 17x50 Stage-B score map"
Write-Host "Judge execution: DISABLED"
Write-Host "Model execution: DISABLED"
Write-Host "Release metrics/gates: NOT RECALCULATED"
Write-Host "Frozen v3 release decision modification: DISABLED"
Write-Host "============================================================"

uv run python `
  .\evals\system_evaluation\classify_semantic_relevance_v3_posthoc_stage_c.py

if ($LASTEXITCODE -ne 0) {
    throw "V3 post-hoc Stage C classification failed."
}
