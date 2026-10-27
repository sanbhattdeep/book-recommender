$ErrorActionPreference = "Stop"

Write-Host "============================================================"
Write-Host "SEMANTIC RELEVANCE V5 - JUDGING-INPUT FREEZE"
Write-Host "============================================================"
Write-Host "Retrieval/reranking:       DISABLED"
Write-Host "Judge execution:           DISABLED"
Write-Host "Development selection:     NOT CALCULATED"
Write-Host "============================================================"

uv run python .\evals\system_evaluation\freeze_semantic_relevance_v5_development_judging_input.py
if ($LASTEXITCODE -ne 0) {
    throw "V5D judging-input freeze failed."
}
