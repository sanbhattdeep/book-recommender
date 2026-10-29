$ErrorActionPreference = "Stop"
Write-Host "============================================================"
Write-Host "SEMANTIC RELEVANCE V5H - FINAL RELEASE ANALYSIS"
Write-Host "============================================================"
uv run python .\evals\system_evaluation\analyze_semantic_relevance_v5_holdout_final_release.py
if ($LASTEXITCODE -ne 0) { throw "V5H final release analysis failed." }
