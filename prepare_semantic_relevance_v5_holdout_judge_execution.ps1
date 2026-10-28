$ErrorActionPreference = "Stop"
Write-Host "============================================================"
Write-Host "SEMANTIC RELEVANCE V5H - JUDGE EXECUTION PREFLIGHT"
Write-Host "============================================================"
Write-Host "Judge execution:              DISABLED DURING PREFLIGHT"
Write-Host "Retrieval/reranker:           DISABLED"
Write-Host "Release analysis:             NOT CALCULATED"
Write-Host "============================================================"
uv run python .\evals\system_evaluation\verify_semantic_relevance_v5_holdout_judge_execution.py
if ($LASTEXITCODE -ne 0) { throw "V5H judge-execution preflight failed." }
