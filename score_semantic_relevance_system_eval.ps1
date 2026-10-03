$ErrorActionPreference = "Stop"

$env:DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS_OVERRIDE = "300"
$runDir = ".\evals\runs\semantic_relevance_system_eval_v1\20261002T152856Z_recommendations"

Write-Host "============================================================"
Write-Host "SEMANTIC RELEVANCE SYSTEM EVAL - CALIBRATED JUDGE SCORING"
Write-Host "============================================================"
Write-Host "Frozen collection run: $runDir"
Write-Host "Judge: semantic_relevance_v0.29.0-r5"
Write-Host "Expected pairs: 120"
Write-Host "Human relevance labels: NOT USED"
Write-Host "Aggregate metrics: NOT CALCULATED IN THIS STAGE"
Write-Host "Bootstrap CIs: NOT CALCULATED IN THIS STAGE"
Write-Host "============================================================"

uv run python `
  .\evals\system_evaluation\score_frozen_semantic_relevance_recommendations.py `
  --collection-run $runDir

if ($LASTEXITCODE -ne 0) {
    throw "Calibrated judge scoring failed or is partial. Rerun SAME command to resume."
}

Write-Host ""
Write-Host "============================================================"
Write-Host "CALIBRATED JUDGE SCORING COMPLETE"
Write-Host "Next stage: aggregate metrics + bootstrap CIs + release gates."
Write-Host "============================================================"
