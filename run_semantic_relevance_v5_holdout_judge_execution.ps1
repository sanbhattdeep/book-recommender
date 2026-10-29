$ErrorActionPreference = "Stop"
Write-Host "============================================================"
Write-Host "SEMANTIC RELEVANCE V5H - JUDGE FULL/RESUME"
Write-Host "============================================================"
Write-Host "Judge execution:              ENABLED - RESUME SAFE"
Write-Host "Retrieval/reranker:           DISABLED"
Write-Host "Release analysis:             NOT CALCULATED"
Write-Host "============================================================"
uv run python .\evals\system_evaluation\run_semantic_relevance_v5_holdout_judge.py --mode full
if ($LASTEXITCODE -ne 0) { throw "V5H judge full/resume failed." }
