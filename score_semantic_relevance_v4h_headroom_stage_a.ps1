param(
    [int]$Limit = 0
)

$ErrorActionPreference = "Stop"
Write-Host "============================================================"
Write-Host "V4H HEADROOM STAGE A - FROZEN JUDGE EXECUTION"
Write-Host "============================================================"
Write-Host "Judge:                   semantic_relevance_v0.29.0-r5"
Write-Host "Fresh cases:             340"
Write-Host "Exact reused cases:      60 (external; not re-judged)"
Write-Host "Resume persistence:      ENABLED"
Write-Host "Human relevance labels:  NOT USED"
Write-Host "Release decision:        IMMUTABLE FAIL"
Write-Host "============================================================"

$argsList = @()
if ($Limit -gt 0) {
    $argsList += "--limit"
    $argsList += "$Limit"
}

uv run python .\evals\system_evaluation\score_semantic_relevance_v4h_headroom_stage_a.py @argsList
if ($LASTEXITCODE -ne 0) {
    throw "V4H headroom Stage-A judge execution stopped with an error. Persisted completed fresh cases are safe; review output and rerun to resume."
}
