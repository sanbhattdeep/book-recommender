$ErrorActionPreference = "Stop"

Write-Host "============================================================"
Write-Host "SEMANTIC RELEVANCE TOP-50 DIAGNOSTIC - CHECKPOINT D"
Write-Host "============================================================"
Write-Host "Purpose: frozen top-50 depth/headroom/oracle analysis"
Write-Host "Judge calls: 0"
Write-Host "Human relevance labels: NOT USED"
Write-Host "Release decision: NOT PRODUCED"
Write-Host "Benchmark role: development/diagnostic"
Write-Host "============================================================"

uv run python `
  .\evals\system_evaluation\analyze_semantic_relevance_top50_checkpoint_d.py

if ($LASTEXITCODE -ne 0) {
    throw "Top-50 Checkpoint D failed. Review the error before retrying."
}

Write-Host ""
Write-Host "============================================================"
Write-Host "CHECKPOINT D INVOCATION COMPLETE"
Write-Host "Upload the captured console output for review."
Write-Host "============================================================"
