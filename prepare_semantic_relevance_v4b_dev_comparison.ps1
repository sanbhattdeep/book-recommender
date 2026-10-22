$ErrorActionPreference = "Stop"
Write-Host "============================================================"
Write-Host "SEMANTIC RELEVANCE V4B - DEVELOPMENT COMPARISON PREFLIGHT"
Write-Host "============================================================"
Write-Host "Development metrics:        DISABLED"
Write-Host "Development selection:      DISABLED"
Write-Host "Judge/model execution:      DISABLED"
Write-Host "Release gates/decision:     DISABLED"
Write-Host "============================================================"

uv run python .\evals\system_evaluation\verify_semantic_relevance_v4b_dev_comparison_package.py
if ($LASTEXITCODE -ne 0) {
    throw "V4B development-comparison preflight failed. Preserve frozen score-map artifacts and review before analysis."
}
