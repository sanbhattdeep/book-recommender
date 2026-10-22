$ErrorActionPreference = "Stop"
Write-Host "============================================================"
Write-Host "SEMANTIC RELEVANCE V4H - FREEZE DISPLAYED-PAIR UNION"
Write-Host "============================================================"
Write-Host "Union generation:           ENABLED"
Write-Host "Score reuse:                DISABLED"
Write-Host "Judge execution:            DISABLED"
Write-Host "Release analysis:           DISABLED"
Write-Host "============================================================"

uv run python .\evals\system_evaluation\build_semantic_relevance_v4h_display_union.py
if ($LASTEXITCODE -ne 0) {
    throw "V4H display-union build failed. Do not run judging."
}
