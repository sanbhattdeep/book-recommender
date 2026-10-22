$ErrorActionPreference = "Stop"
Write-Host "============================================================"
Write-Host "SEMANTIC RELEVANCE V4A - RUN RETRIEVAL DIAGNOSTIC SENSITIVITY ANALYSIS"
Write-Host "============================================================"
Write-Host "Scenarios:                S0=0 S1=1 S2=2 S3=3 S4=4"
Write-Host "Judge execution:          DISABLED"
Write-Host "Model execution:          DISABLED"
Write-Host "Actual quarantine score:  UNASSIGNED"
Write-Host "Development selection:    FROZEN / NOT REOPENED"
Write-Host "Release analysis:         DISABLED"
Write-Host "============================================================"

uv run python .\evals\system_evaluation\analyze_semantic_relevance_v4a_retrieval_diagnostic_sensitivity.py
if ($LASTEXITCODE -ne 0) {
    throw "V4a retrieval-diagnostic sensitivity analysis failed. Preserve frozen inputs and review before retrying."
}
