$ErrorActionPreference = "Stop"
Write-Host "============================================================"
Write-Host "SEMANTIC RELEVANCE V4A - V3B VS V4A DEVELOPMENT COMPARISON"
Write-Host "============================================================"
Write-Host "Judge execution:        DISABLED"
Write-Host "Model execution:        DISABLED"
Write-Host "Human labels:           DISABLED"
Write-Host "Release gates/decision: DISABLED"
Write-Host "Development metrics:    ENABLED"
Write-Host "Development selection:  ENABLED (frozen lexicographic rule)"
Write-Host "============================================================"

uv run python .\evals\system_evaluation\analyze_semantic_relevance_v4a_dev_comparison.py
if ($LASTEXITCODE -ne 0) {
    throw "V4a development comparison failed."
}
