$ErrorActionPreference = "Stop"

Write-Host "============================================================"
Write-Host "SEMANTIC RELEVANCE SYSTEM EVAL - METRICS + 95% CI + GATES"
Write-Host "============================================================"
Write-Host "Input: frozen 120-pair judge_scores.csv"
Write-Host "Judge execution: DISABLED"
Write-Host "Human relevance labels: NOT USED"
Write-Host "Bootstrap: 5000 query-level cluster replicates"
Write-Host "============================================================"

uv run python .\evals\system_evaluation\analyze_semantic_relevance_system_eval.py

if ($LASTEXITCODE -ne 0) {
    throw "System evaluation metric analysis failed."
}

uv run python .\evals\system_evaluation\render_semantic_relevance_system_eval_dashboard.py

if ($LASTEXITCODE -ne 0) {
    throw "System evaluation dashboard rendering failed."
}

Write-Host ""
Write-Host "============================================================"
Write-Host "SYSTEM EVALUATION ANALYSIS COMPLETE"
Write-Host "Next: review release decision and publish frozen results to Confident AI."
Write-Host "============================================================"
