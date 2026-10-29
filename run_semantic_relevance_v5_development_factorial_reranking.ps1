$ErrorActionPreference = "Stop"
Write-Host "============================================================"
Write-Host "SEMANTIC RELEVANCE V5 - B/R/K/RK FACTORIAL RERANKING"
Write-Host "============================================================"
Write-Host "New retrieval:            DISABLED"
Write-Host "K0/K1 reranker inference: ENABLED"
Write-Host "Judge execution:          DISABLED"
Write-Host "Development selection:    DISABLED"
Write-Host "============================================================"
uv run python .\evals\system_evaluation\run_semantic_relevance_v5_development_factorial_reranking.py --device auto
if ($LASTEXITCODE -ne 0) { throw "V5D factorial reranking failed." }
