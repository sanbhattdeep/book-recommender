$ErrorActionPreference = "Stop"
Write-Host "============================================================"
Write-Host "SEMANTIC RELEVANCE V4A - FREEZE PARTIAL RETRIEVAL-DIAGNOSTIC JUDGING STATE"
Write-Host "============================================================"
Write-Host "Expected scored cases:  674"
Write-Host "Expected quarantine:    1 / UNSCORED"
Write-Host "Judge execution:        DISABLED"
Write-Host "Model execution:        DISABLED"
Write-Host "Diagnostic analysis:    DISABLED"
Write-Host "Development selection:  FROZEN / NOT REOPENED"
Write-Host "Release analysis:       DISABLED"
Write-Host "============================================================"

uv run python .\evals\system_evaluation\freeze_semantic_relevance_v4a_retrieval_diagnostic_partial_judging.py
if ($LASTEXITCODE -ne 0) {
    throw "V4a retrieval-diagnostic partial judging state freeze failed. Preserve existing artifacts and review before continuing."
}
