$ErrorActionPreference = "Stop"
Write-Host "============================================================"
Write-Host "SEMANTIC RELEVANCE V4B - FREEZE DISPLAYED-PAIR UNION"
Write-Host "============================================================"
Write-Host "Score reuse:               DISABLED"
Write-Host "Judge execution:           DISABLED"
Write-Host "Relevance labels:          NOT LOADED"
Write-Host "Development selection:     NOT CALCULATED"
Write-Host "Release analysis:          DISABLED"
Write-Host "============================================================"

uv run python .\evals\system_evaluation\build_semantic_relevance_v4b_dev_display_union.py
if ($LASTEXITCODE -ne 0) {
    throw "V4B displayed-pair union freeze failed. Preserve frozen inputs and review before retrying."
}
