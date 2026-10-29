$ErrorActionPreference = "Stop"
Write-Host "============================================================"
Write-Host "SEMANTIC RELEVANCE V5H - FINAL RELEASE ANALYSIS PREFLIGHT"
Write-Host "============================================================"
uv run python .\evals\system_evaluation\verify_semantic_relevance_v5_holdout_release_analysis.py
if ($LASTEXITCODE -ne 0) { throw "V5H release-analysis preflight failed." }
