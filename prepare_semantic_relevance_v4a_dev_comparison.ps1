$ErrorActionPreference = "Stop"
Write-Host "============================================================"
Write-Host "SEMANTIC RELEVANCE V4A - DEV COMPARISON PACKAGE PREFLIGHT"
Write-Host "============================================================"
Write-Host "Development metrics:    DISABLED"
Write-Host "Development selection:  DISABLED"
Write-Host "Judge execution:        DISABLED"
Write-Host "Model execution:        DISABLED"
Write-Host "Release gates/decision: DISABLED"
Write-Host "============================================================"

uv run python .\evals\system_evaluation\verify_semantic_relevance_v4a_dev_comparison_package.py
if ($LASTEXITCODE -ne 0) {
    throw "V4a development comparison package preflight failed."
}
