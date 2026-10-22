$ErrorActionPreference = "Stop"
Write-Host "============================================================"
Write-Host "SEMANTIC RELEVANCE V4B - DETERMINISTIC SCORE FINALIZATION"
Write-Host "============================================================"
Write-Host "Judge/model execution:      DISABLED"
Write-Host "Development comparison:     NOT CALCULATED"
Write-Host "Release analysis:           DISABLED"
Write-Host "============================================================"

uv run python .\evals\system_evaluation\finalize_semantic_relevance_v4b_dev_scores.py
if ($LASTEXITCODE -ne 0) {
    throw "V4B deterministic score finalization failed. Preserve frozen inputs and review before retrying."
}
