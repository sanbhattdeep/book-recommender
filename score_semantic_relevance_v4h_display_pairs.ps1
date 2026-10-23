param(
    [int]$Limit = 0
)

$ErrorActionPreference = "Stop"
Write-Host "============================================================"
Write-Host "SEMANTIC RELEVANCE V4H - FROZEN JUDGE EXECUTION"
Write-Host "============================================================"
Write-Host "Judge:                      semantic_relevance_v0.29.0-r5"
Write-Host "Historical score reuse:     DISABLED"
Write-Host "Resume-safe persistence:    ENABLED"
Write-Host "Human relevance labels:     NOT USED"
Write-Host "Release analysis:           DISABLED"
Write-Host "============================================================"

$argsList = @()
if ($Limit -gt 0) {
    $argsList += "--limit"
    $argsList += "$Limit"
}

uv run python .\evals\system_evaluation\score_semantic_relevance_v4h_display_pairs.py @argsList
if ($LASTEXITCODE -ne 0) {
    throw "V4H judge execution stopped with an error. Persisted completed cases are safe; review output and rerun to resume."
}
