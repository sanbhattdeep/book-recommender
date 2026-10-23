$ErrorActionPreference = "Stop"
Write-Host "============================================================"
Write-Host "SEMANTIC RELEVANCE V4H - FREEZE JUDGING INPUT"
Write-Host "============================================================"
Write-Host "Judge execution:            DISABLED"
Write-Host "Historical score reuse:     DISABLED"
Write-Host "Human relevance labels:     NOT USED"
Write-Host "Release analysis:           DISABLED"
Write-Host "============================================================"

uv run python .\evals\system_evaluation\prepare_semantic_relevance_v4h_judging.py
if ($LASTEXITCODE -ne 0) {
    throw "V4H judge-input preparation failed. Do not run judging."
}
