param(
    [int]$MaxNew = 0
)

$ErrorActionPreference = "Stop"

$RunDir = ".\evals\runs\semantic_relevance_top50_diagnostic_v1\20261005T113402Z_candidates"

Write-Host "============================================================"
Write-Host "SEMANTIC RELEVANCE TOP-50 DIAGNOSTIC - CHECKPOINT C"
Write-Host "============================================================"
Write-Host "Run directory: $RunDir"
Write-Host "Judge: semantic_relevance_v0.29.0-r5"
Write-Host "Reusable frozen scores: 213"
Write-Host "Novel pairs to judge: 387"
Write-Host "Human relevance labels: NOT USED"
Write-Host "Aggregate metrics: NOT CALCULATED"
Write-Host "Oracle metrics: NOT CALCULATED"
Write-Host "Restart-safe persistence: ENABLED"
Write-Host "Windows sleep protection: ENABLED"
if ($MaxNew -gt 0) {
    Write-Host "This invocation new-case cap: $MaxNew"
}
else {
    Write-Host "This invocation new-case cap: NONE"
}
Write-Host "============================================================"

Add-Type @"
using System;
using System.Runtime.InteropServices;

public static class NativePowerTop50C
{
    [DllImport("kernel32.dll", SetLastError = true)]
    public static extern uint SetThreadExecutionState(uint esFlags);
}
"@

$ES_CONTINUOUS       = [Convert]::ToUInt32("80000000", 16)
$ES_SYSTEM_REQUIRED  = [uint32]0x00000001
$ES_DISPLAY_REQUIRED = [uint32]0x00000002

$keepAwakeFlags = (
    $ES_CONTINUOUS -bor
    $ES_SYSTEM_REQUIRED -bor
    $ES_DISPLAY_REQUIRED
)

try {
    $result = [NativePowerTop50C]::SetThreadExecutionState($keepAwakeFlags)

    if ($result -eq 0) {
        throw "SetThreadExecutionState failed."
    }

    Write-Host "PASS  Windows execution-state sleep protection enabled."

    $pythonArgs = @(
        "run",
        "python",
        ".\evals\system_evaluation\score_semantic_relevance_top50_novel_candidates.py"
    )

    if ($MaxNew -gt 0) {
        $pythonArgs += @("--max-new", "$MaxNew")
    }

    & uv @pythonArgs

    if ($LASTEXITCODE -ne 0) {
        throw (
            "Top-50 Checkpoint C failed or stopped on an error. " +
            "Rerun the SAME command to resume persisted cases."
        )
    }
}
finally {
    [void][NativePowerTop50C]::SetThreadExecutionState($ES_CONTINUOUS)
    Write-Host "PASS  Windows sleep protection cleared; normal power behavior restored."
}

Write-Host ""
Write-Host "============================================================"
Write-Host "CHECKPOINT C INVOCATION COMPLETE"
Write-Host "If fewer than 387 novel cases are persisted, rerun the same command."
Write-Host "When 387/387 completes, upload the captured console output for review."
Write-Host "============================================================"
