$ErrorActionPreference = "Stop"
Write-Host "============================================================"
Write-Host "SEMANTIC RELEVANCE V5 - JUDGE EXECUTION PREFLIGHT"
Write-Host "============================================================"
Write-Host "Judge execution:          DISABLED"
Write-Host "Retrieval/reranking:      DISABLED"
Write-Host "Development selection:    NOT CALCULATED"
Write-Host "============================================================"
uv run python .\evals\system_evaluation\verify_semantic_relevance_v5_development_judge_execution.py
if ($LASTEXITCODE -ne 0) { throw "V5D judge execution preflight failed." }
