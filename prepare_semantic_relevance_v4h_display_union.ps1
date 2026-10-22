$ErrorActionPreference = "Stop"
Write-Host "============================================================"
Write-Host "SEMANTIC RELEVANCE V4H - DISPLAY UNION PREFLIGHT"
Write-Host "============================================================"
Write-Host "Union generation:           DISABLED"
Write-Host "Score reuse:                DISABLED"
Write-Host "Judge execution:            DISABLED"
Write-Host "Release analysis:           DISABLED"
Write-Host "============================================================"

uv run python .\evals\system_evaluation\verify_semantic_relevance_v4h_display_union.py
if ($LASTEXITCODE -ne 0) {
    throw "V4H display-union preflight failed. Do not build or judge the union."
}
