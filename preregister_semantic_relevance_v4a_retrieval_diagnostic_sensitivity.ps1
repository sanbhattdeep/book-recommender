$ErrorActionPreference = "Stop"
Write-Host "============================================================"
Write-Host "SEMANTIC RELEVANCE V4A - PRE-REGISTER RETRIEVAL-DIAGNOSTIC SENSITIVITY"
Write-Host "============================================================"
Write-Host "Hypothetical scores:      {0,1,2,3,4}"
Write-Host "Judge execution:          DISABLED"
Write-Host "Model execution:          DISABLED"
Write-Host "Diagnostic calculations:  DISABLED"
Write-Host "Quarantine score assign:  DISABLED"
Write-Host "Development selection:    FROZEN / NOT REOPENED"
Write-Host "Release analysis:         DISABLED"
Write-Host "============================================================"

uv run python .\evals\system_evaluation\preregister_semantic_relevance_v4a_retrieval_diagnostic_sensitivity.py
if ($LASTEXITCODE -ne 0) {
    throw "V4a retrieval-diagnostic sensitivity pre-registration failed. Preserve frozen artifacts and review before analysis."
}
