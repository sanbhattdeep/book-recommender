$ErrorActionPreference = "Stop"

Write-Host "============================================================"
Write-Host "SEMANTIC RELEVANCE V3 - FINAL VISUAL LAYER"
Write-Host "============================================================"
Write-Host "Source: frozen final-analysis artifacts"
Write-Host "Judge execution: DISABLED"
Write-Host "Metric recalculation: DISABLED"
Write-Host "============================================================"

uv run python `
  .\evals\system_evaluation\render_semantic_relevance_v3_final_dashboard.py

if ($LASTEXITCODE -ne 0) {
    throw "V3 final visual dashboard generation failed."
}
