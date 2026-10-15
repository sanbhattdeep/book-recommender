$ErrorActionPreference = "Stop"
Write-Host "============================================================"
Write-Host "SEMANTIC RELEVANCE V3 - FINAL UNSEEN HOLDOUT ANALYSIS"
Write-Host "============================================================"
Write-Host "Judge execution: DISABLED"
Write-Host "Human labels: NOT USED"
Write-Host "Primary decision: standalone v3 unseen holdout"
Write-Host "Paired v2-v3 comparison: SECONDARY"
Write-Host "============================================================"
uv run python .\evals\system_evaluation\analyze_semantic_relevance_v3_final_holdout.py
if ($LASTEXITCODE -ne 0) { throw "V3 final unseen holdout analysis failed." }
