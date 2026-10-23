$ErrorActionPreference = "Stop"
Write-Host "============================================================"
Write-Host "SEMANTIC RELEVANCE V4H - FINAL V4B RELEASE ANALYSIS"
Write-Host "============================================================"
Write-Host "Authoritative system:       V4B"
Write-Host "V3B paired comparison:      SECONDARY"
Write-Host "Judge/model calls:          DISABLED"
Write-Host "Human relevance labels:     NOT USED"
Write-Host "Release metrics/decision:   ENABLED"
Write-Host "============================================================"

uv run python .\evals\system_evaluation\analyze_semantic_relevance_v4h_final_release.py
if ($LASTEXITCODE -ne 0) {
    throw "V4H final release analysis failed."
}
