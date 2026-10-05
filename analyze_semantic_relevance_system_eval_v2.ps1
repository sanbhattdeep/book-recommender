$ErrorActionPreference = "Stop"

Write-Host "============================================================"
Write-Host "SEMANTIC RELEVANCE SYSTEM EVAL V2 - CORE ANALYSIS R2"
Write-Host "============================================================"
Write-Host "Layer 1: original frozen benchmark metrics + CIs + gates"
Write-Host "Layer 2: paired v1-v2 deltas + paired bootstrap CIs + W/T/L"
Write-Host "Layer 3: frozen release-gate transitions"
Write-Host "Layer 4: visual dashboard"
Write-Host "Judge execution: DISABLED"
Write-Host "Human relevance labels: NOT USED"
Write-Host "============================================================"

uv run python `
  .\evals\system_evaluation\analyze_semantic_relevance_system_eval_v2.py

if ($LASTEXITCODE -ne 0) {
    throw "V2 benchmark analysis failed."
}

Write-Host ""
Write-Host "============================================================"
Write-Host "PAIRED V1 VS V2 COMPARISON"
Write-Host "============================================================"

uv run python `
  .\evals\system_evaluation\compare_semantic_relevance_system_eval_v1_v2.py

if ($LASTEXITCODE -ne 0) {
    throw "Paired v1-v2 comparison failed."
}

Write-Host ""
Write-Host "============================================================"
Write-Host "VISUAL DASHBOARD"
Write-Host "============================================================"

uv run python `
  .\evals\system_evaluation\render_semantic_relevance_system_eval_dashboard_v2.py

if ($LASTEXITCODE -ne 0) {
    throw "V2 visual dashboard rendering failed."
}

Write-Host ""
Write-Host "============================================================"
Write-Host "SYSTEM EVAL V2 CORE ANALYSIS R2 COMPLETE"
Write-Host "No judge calls were performed."
Write-Host "============================================================"
