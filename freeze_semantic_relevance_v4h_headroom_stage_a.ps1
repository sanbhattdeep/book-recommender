$ErrorActionPreference = "Stop"
Write-Host "============================================================"
Write-Host "V4H NO-CLEAR HEADROOM STAGE A - CANDIDATE FREEZE"
Write-Host "============================================================"
Write-Host "Judge/model calls:      DISABLED"
Write-Host "Release decision:       IMMUTABLE FAIL"
Write-Host "============================================================"
uv run python .\evals\system_evaluation\freeze_semantic_relevance_v4h_headroom_stage_a.py
if ($LASTEXITCODE -ne 0) { throw "V4H headroom Stage-A candidate freeze failed." }
