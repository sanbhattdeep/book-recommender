$ErrorActionPreference = "Stop"
Write-Host "============================================================"
Write-Host "SEMANTIC RELEVANCE V4A - RETRIEVAL DIAGNOSTIC JUDGE R3 PREFLIGHT"
Write-Host "============================================================"
Write-Host "Judge execution:       DISABLED"
Write-Host "Timeout patch:         900s / RESUME ONLY"
Write-Host "Case quarantine:       H14/9781582972947 / UNSCORED"
Write-Host "Human labels:          DISABLED"
Write-Host "Development selection: FROZEN / NOT REOPENED"
Write-Host "Release analysis:      DISABLED"
Write-Host "============================================================"

uv run python .\evals\system_evaluation\verify_semantic_relevance_v4a_retrieval_diagnostic_judge_package.py
if ($LASTEXITCODE -ne 0) {
    throw "V4a retrieval-diagnostic judge r3 package preflight failed."
}
