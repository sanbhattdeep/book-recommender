$ErrorActionPreference = "Stop"
Write-Host "============================================================"
Write-Host "V4H HEADROOM STAGE A - JUDGE EXECUTION PREFLIGHT"
Write-Host "============================================================"
Write-Host "Fresh cases:             340"
Write-Host "Judge execution:         DISABLED"
Write-Host "Resume persistence:      ENABLED"
Write-Host "Release decision:        IMMUTABLE FAIL"
Write-Host "============================================================"
uv run python .\evals\system_evaluation\verify_semantic_relevance_v4h_headroom_stage_a_judging.py
if ($LASTEXITCODE -ne 0) { throw "V4H headroom Stage-A judge preflight failed." }
