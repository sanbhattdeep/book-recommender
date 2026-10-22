$ErrorActionPreference = "Stop"
Write-Host "============================================================"
Write-Host "SEMANTIC RELEVANCE V4B - V3B VS V4B DEVELOPMENT COMPARISON"
Write-Host "============================================================"
Write-Host "Judge/model execution:      DISABLED"
Write-Host "Selection rule:             FROZEN LEXICOGRAPHIC"
Write-Host "Release gates/decision:     DISABLED"
Write-Host "============================================================"

uv run python .\evals\system_evaluation\analyze_semantic_relevance_v4b_dev_comparison.py
if ($LASTEXITCODE -ne 0) {
    throw "V4B development comparison failed. Preserve frozen inputs and partial outputs for review."
}
