$ErrorActionPreference = "Stop"
Write-Host "============================================================"
Write-Host "SEMANTIC RELEVANCE V4B - DISPLAY UNION PREFLIGHT"
Write-Host "============================================================"
Write-Host "Union generation:          DISABLED"
Write-Host "Score reuse/judge:         DISABLED"
Write-Host "Relevance labels:          NOT LOADED"
Write-Host "Development selection:     NOT CALCULATED"
Write-Host "Release analysis:          DISABLED"
Write-Host "============================================================"

uv run python .\evals\system_evaluation\verify_semantic_relevance_v4b_dev_display_union.py
if ($LASTEXITCODE -ne 0) {
    throw "V4B display-union preflight failed. Preserve frozen ranking artifacts and review before building union."
}
