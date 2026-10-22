$ErrorActionPreference = "Stop"
Write-Host "============================================================"
Write-Host "SEMANTIC RELEVANCE V4B - SCORE REUSE PREFLIGHT"
Write-Host "============================================================"
Write-Host "Inventory generation:      DISABLED"
Write-Host "Judge execution:           DISABLED"
Write-Host "Human relevance labels:    NOT USED"
Write-Host "Development comparison:    NOT CALCULATED"
Write-Host "Release analysis:          DISABLED"
Write-Host "============================================================"

uv run python .\evals\system_evaluation\verify_semantic_relevance_v4b_dev_score_reuse.py
if ($LASTEXITCODE -ne 0) {
    throw "V4B score-reuse preflight failed. Preserve frozen displayed union and score artifacts before retrying."
}
