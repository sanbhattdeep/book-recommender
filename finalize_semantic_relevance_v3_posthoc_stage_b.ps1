$ErrorActionPreference = "Stop"

Write-Host "============================================================"
Write-Host "SEMANTIC RELEVANCE V3 - POST-HOC STAGE B FINALIZATION"
Write-Host "============================================================"
Write-Host "Persisted novel judgments: 624/624"
Write-Host "Judge execution: DISABLED"
Write-Host "Model execution: DISABLED"
Write-Host "Classification: NOT CALCULATED"
Write-Host "Finalization patch: pandas3 dtype normalization1"
Write-Host "============================================================"

uv run python `
  .\evals\system_evaluation\finalize_semantic_relevance_v3_posthoc_stage_b.py

if ($LASTEXITCODE -ne 0) {
    throw "V3 post-hoc Stage B deterministic finalization failed."
}
