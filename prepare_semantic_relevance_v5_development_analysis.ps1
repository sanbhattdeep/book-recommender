$ErrorActionPreference = "Stop"
Write-Host "============================================================"
Write-Host "SEMANTIC RELEVANCE V5 - DEVELOPMENT ANALYSIS PREFLIGHT"
Write-Host "============================================================"
Write-Host "Judge/model/retrieval/reranker: DISABLED"
Write-Host "Development selection:          NOT CALCULATED"
Write-Host "Release decision:               NOT CALCULATED"
Write-Host "============================================================"
uv run python .\evals\system_evaluation\verify_semantic_relevance_v5_development_analysis.py
if ($LASTEXITCODE -ne 0) { throw "V5D development-analysis preflight failed." }
