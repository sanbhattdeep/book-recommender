$ErrorActionPreference = "Stop"

Write-Host "============================================================"
Write-Host "SEMANTIC RELEVANCE V4A - DEV DISPLAY UNION PREFLIGHT"
Write-Host "============================================================"
Write-Host "Union creation:   DISABLED"
Write-Host "Judge execution:  DISABLED"
Write-Host "Human labels:     DISABLED"
Write-Host "Release analysis: DISABLED"
Write-Host "============================================================"

uv run python `
  .\evals\system_evaluation\verify_semantic_relevance_v4a_dev_display_union_package.py

if ($LASTEXITCODE -ne 0) {
    throw "V4a development display-union package preflight failed."
}
