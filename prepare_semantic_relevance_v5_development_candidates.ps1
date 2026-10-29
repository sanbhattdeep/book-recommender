$ErrorActionPreference = "Stop"
Write-Host "============================================================"
Write-Host "SEMANTIC RELEVANCE V5 - V5D R0/R1 CANDIDATE PREFLIGHT"
Write-Host "============================================================"
Write-Host "R0/R1 retrieval:          DISABLED"
Write-Host "K0/K1 reranker inference: DISABLED"
Write-Host "Judge execution:          DISABLED"
Write-Host "Development selection:    NOT CALCULATED"
Write-Host "============================================================"
uv run python .\evals\system_evaluation\verify_semantic_relevance_v5_development_candidate_collection.py
if ($LASTEXITCODE -ne 0) { throw "V5D R0/R1 candidate-collection preflight failed." }
