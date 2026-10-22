$ErrorActionPreference = "Stop"
Write-Host "============================================================"
Write-Host "SEMANTIC RELEVANCE V4B - PREPARE SCORE REUSE / NOVEL INPUT"
Write-Host "============================================================"
Write-Host "Judge execution:           DISABLED"
Write-Host "Human relevance labels:    NOT USED"
Write-Host "Development comparison:    NOT CALCULATED"
Write-Host "Release analysis:          DISABLED"
Write-Host "============================================================"

uv run python .\evals\system_evaluation\prepare_semantic_relevance_v4b_dev_score_reuse.py
if ($LASTEXITCODE -ne 0) {
    throw "V4B score-reuse preparation failed. Preserve frozen inputs and review before retrying."
}
