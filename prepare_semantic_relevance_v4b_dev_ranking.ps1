$ErrorActionPreference = "Stop"
Write-Host "============================================================"
Write-Host "SEMANTIC RELEVANCE V4B - DEVELOPMENT RANKING PREFLIGHT"
Write-Host "============================================================"
Write-Host "Recommendation generation: DISABLED"
Write-Host "Retrieval/model/judge:      DISABLED"
Write-Host "Relevance labels:           NOT LOADED"
Write-Host "Development selection:      NOT CALCULATED"
Write-Host "Release analysis:           DISABLED"
Write-Host "============================================================"

uv run python .\evals\system_evaluation\verify_semantic_relevance_v4b_dev_ranking.py
if ($LASTEXITCODE -ne 0) {
    throw "V4B development ranking preflight failed. Preserve frozen artifacts and review before ranking."
}
