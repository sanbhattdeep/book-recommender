$ErrorActionPreference = "Stop"
Write-Host "============================================================"
Write-Host "SEMANTIC RELEVANCE V4A - RETRIEVAL DIAGNOSTIC SENSITIVITY ANALYSIS PREFLIGHT"
Write-Host "============================================================"
Write-Host "Judge execution:          DISABLED"
Write-Host "Model execution:          DISABLED"
Write-Host "Diagnostic calculations:  DISABLED"
Write-Host "Actual quarantine score:  UNASSIGNED"
Write-Host "Development selection:    FROZEN / NOT REOPENED"
Write-Host "Release analysis:         DISABLED"
Write-Host "============================================================"

uv run python .\evals\system_evaluation\verify_semantic_relevance_v4a_retrieval_diagnostic_sensitivity_analysis.py
if ($LASTEXITCODE -ne 0) {
    throw "V4a retrieval-diagnostic sensitivity analysis preflight failed."
}
