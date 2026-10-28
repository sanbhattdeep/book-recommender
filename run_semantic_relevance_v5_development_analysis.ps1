$ErrorActionPreference = "Stop"
Write-Host "============================================================"
Write-Host "SEMANTIC RELEVANCE V5 - DEVELOPMENT ANALYSIS + SELECTION"
Write-Host "============================================================"
Write-Host "Judge/model/retrieval/reranker: DISABLED"
Write-Host "Selection rule:                 FROZEN LEXICOGRAPHIC"
Write-Host "Release decision:               NOT CALCULATED"
Write-Host "============================================================"
uv run python .\evals\system_evaluation\analyze_semantic_relevance_v5_development.py
if ($LASTEXITCODE -ne 0) { throw "V5D development analysis/selection failed." }
