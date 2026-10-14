param(
    [switch]$Resume
)

$ErrorActionPreference = "Stop"

Write-Host "============================================================"
Write-Host "SEMANTIC RELEVANCE V3 - POST-HOC STAGE B DEEP JUDGING"
Write-Host "============================================================"
Write-Host "Target queries: 17"
Write-Host "Reusable frozen scores: 226"
Write-Host "Novel deep candidates: 624"
Write-Host "Judge: semantic_relevance_v0.29.0-r5"
Write-Host "Human relevance labels: NOT USED"
Write-Host "Classification: NOT CALCULATED"
Write-Host "Frozen v3 release decision modification: DISABLED"
if ($Resume) {
    Write-Host "Mode: RESUME persisted cases after execution failure"
}
else {
    Write-Host "Mode: INITIAL post-hoc diagnostic execution"
}
Write-Host "Windows sleep protection: ENABLED"
Write-Host "============================================================"

Add-Type @"
using System;
using System.Runtime.InteropServices;

public static class NativePowerV3PosthocB
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
    $result = [NativePowerV3PosthocB]::SetThreadExecutionState($keepAwakeFlags)
    if ($result -eq 0) {
        throw "SetThreadExecutionState failed."
    }

    Write-Host "PASS  Windows execution-state sleep protection enabled."

    $pythonArgs = @(
        "run",
        "python",
        ".\evals\system_evaluation\score_semantic_relevance_v3_posthoc_stage_b.py"
    )

    if ($Resume) {
        $pythonArgs += "--resume"
    }

    & uv @pythonArgs

    if ($LASTEXITCODE -ne 0) {
        throw (
            "V3 post-hoc Stage B judging failed or stopped. " +
            "Do not restart from scratch or inspect/tune interim semantic results. " +
            "After confirming a pure execution failure, resume persisted cases only."
        )
    }
}
finally {
    [void][NativePowerV3PosthocB]::SetThreadExecutionState($ES_CONTINUOUS)
    Write-Host "PASS  Windows sleep protection cleared; normal power behavior restored."
}
