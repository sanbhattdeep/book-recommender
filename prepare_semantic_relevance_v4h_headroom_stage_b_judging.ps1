$ErrorActionPreference = "Stop"
Write-Host "============================================================"
Write-Host "V4H HEADROOM STAGE B - JUDGING PREPARATION"
Write-Host "============================================================"
Write-Host "V4H:                    CONSUMED / DIAGNOSTIC ONLY"
Write-Host "Candidate inventory:    252 frozen ranks-51-to-end pairs"
Write-Host "Exact score reuse:      SAME QUERY + SAME ISBN ONLY"
Write-Host "Judge/model calls:      DISABLED"
Write-Host "Release decision:       IMMUTABLE FAIL"
Write-Host "============================================================"

uv run python .\evals\system_evaluation\prepare_semantic_relevance_v4h_headroom_stage_b_judging.py
if ($LASTEXITCODE -ne 0) {
    throw "V4H headroom Stage-B judging preparation failed. Do not execute the judge."
}
