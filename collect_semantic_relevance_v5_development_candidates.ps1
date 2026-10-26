$ErrorActionPreference = "Stop"
Write-Host "============================================================"
Write-Host "SEMANTIC RELEVANCE V5 - V5D R0/R1 CANDIDATE COLLECTION"
Write-Host "============================================================"
Write-Host "R0/R1 retrieval:          ENABLED"
Write-Host "K0/K1 reranker inference: DISABLED"
Write-Host "Judge execution:          DISABLED"
Write-Host "Development selection:    DISABLED"
Write-Host "============================================================"
uv run python .\evals\system_evaluation\collect_semantic_relevance_v5_development_candidates.py
if ($LASTEXITCODE -ne 0) { throw "V5D R0/R1 candidate collection failed." }
