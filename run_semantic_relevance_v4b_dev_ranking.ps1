$ErrorActionPreference = "Stop"
Write-Host "============================================================"
Write-Host "SEMANTIC RELEVANCE V4B - DETERMINISTIC DEVELOPMENT RANKING"
Write-Host "============================================================"
Write-Host "Input:                     Frozen V4A reranker scores"
Write-Host "Retrieval rerun:           NO"
Write-Host "Model inference:           NO"
Write-Host "Judge execution:           NO"
Write-Host "Relevance labels:          NOT LOADED"
Write-Host "Development selection:     NOT CALCULATED"
Write-Host "Release analysis:          DISABLED"
Write-Host "============================================================"

uv run python .\evals\system_evaluation\run_semantic_relevance_v4b_dev_ranking.py
if ($LASTEXITCODE -ne 0) {
    throw "V4B deterministic development ranking failed. Preserve frozen artifacts and review before retrying."
}
