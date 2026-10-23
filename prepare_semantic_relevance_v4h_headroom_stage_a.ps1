$ErrorActionPreference = "Stop"
Write-Host "============================================================"
Write-Host "V4H NO-CLEAR HEADROOM STAGE A - PREFLIGHT"
Write-Host "============================================================"
Write-Host "V4H:                    CONSUMED / DIAGNOSTIC ONLY"
Write-Host "Target queries:         10 NO_CLEAR_TOP10"
Write-Host "Candidate window:       V4B ranks 11-50"
Write-Host "Judge/model calls:      DISABLED"
Write-Host "Release decision:       IMMUTABLE FAIL"
Write-Host "============================================================"
uv run python .\evals\system_evaluation\verify_semantic_relevance_v4h_headroom_stage_a.py
if ($LASTEXITCODE -ne 0) { throw "V4H headroom Stage-A preflight failed." }
