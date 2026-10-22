$ErrorActionPreference = "Stop"
Write-Host "============================================================"
Write-Host "SEMANTIC RELEVANCE V4B - SCORE FINALIZATION PREFLIGHT"
Write-Host "============================================================"
Write-Host "Complete-map generation:    DISABLED"
Write-Host "Judge/model execution:      DISABLED"
Write-Host "Development comparison:     NOT CALCULATED"
Write-Host "Release analysis:           DISABLED"
Write-Host "============================================================"

uv run python .\evals\system_evaluation\verify_semantic_relevance_v4b_dev_score_finalization.py
if ($LASTEXITCODE -ne 0) {
    throw "V4B score-finalization preflight failed. Preserve frozen scores and review before finalization."
}
