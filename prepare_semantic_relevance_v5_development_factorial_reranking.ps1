$ErrorActionPreference = "Stop"
Write-Host "============================================================"
Write-Host "SEMANTIC RELEVANCE V5 - FACTORIAL RERANKING PREFLIGHT"
Write-Host "============================================================"
Write-Host "New retrieval:            DISABLED"
Write-Host "K0/K1 reranker inference: DISABLED"
Write-Host "Judge execution:          DISABLED"
Write-Host "Development selection:    NOT CALCULATED"
Write-Host "============================================================"
uv run python .\evals\system_evaluation\verify_semantic_relevance_v5_development_factorial_reranking.py
if ($LASTEXITCODE -ne 0) { throw "V5D factorial reranking preflight failed." }
