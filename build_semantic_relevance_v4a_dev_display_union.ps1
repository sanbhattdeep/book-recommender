$ErrorActionPreference = "Stop"

Write-Host "============================================================"
Write-Host "SEMANTIC RELEVANCE V4A - BUILD/FREEZE DEV DISPLAY UNION"
Write-Host "============================================================"
Write-Host "Judge execution:  DISABLED"
Write-Host "Human labels:     DISABLED"
Write-Host "Release analysis: DISABLED"
Write-Host "============================================================"

uv run python `
  .\evals\system_evaluation\build_semantic_relevance_v4a_dev_display_union.py

if ($LASTEXITCODE -ne 0) {
    throw "V4a development display-union build failed."
}
