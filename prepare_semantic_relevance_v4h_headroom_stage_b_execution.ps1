$ErrorActionPreference = "Stop"
Write-Host "============================================================"
Write-Host "V4H HEADROOM STAGE B - JUDGE EXECUTION PREFLIGHT"
Write-Host "============================================================"
Write-Host "Fresh cases:             251"
Write-Host "Exact reused cases:      1 (external; not re-judged)"
Write-Host "Judge execution:         DISABLED"
Write-Host "Resume persistence:      ENABLED"
Write-Host "Release decision:        IMMUTABLE FAIL"
Write-Host "============================================================"
uv run python .\evals\system_evaluation\verify_semantic_relevance_v4h_headroom_stage_b_judging.py
if ($LASTEXITCODE -ne 0) { throw "V4H headroom Stage-B judge preflight failed." }
