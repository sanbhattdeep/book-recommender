param(
    [switch]$Resume
)

$ErrorActionPreference = "Stop"

$RunDir = ".\evals\runs\semantic_relevance_v3_holdout\20261007T124150Z_retrieval"

Write-Host "============================================================"
Write-Host "SEMANTIC RELEVANCE V3 - ONE-TIME UNSEEN HOLDOUT JUDGING"
Write-Host "============================================================"
Write-Host "Run directory: $RunDir"
Write-Host "Frozen unique pairs: 322"
Write-Host "Judge: semantic_relevance_v0.29.0-r5"
Write-Host "Human relevance labels: NOT USED"
Write-Host "Aggregate metrics: NOT CALCULATED"
Write-Host "Paired analysis: NOT CALCULATED"
Write-Host "Release decision: NOT PRODUCED"
if ($Resume) {
    Write-Host "Mode: RESUME persisted cases after execution failure"
}
else {
    Write-Host "Mode: INITIAL one-time execution"
}
Write-Host "Windows sleep protection: ENABLED"
Write-Host "============================================================"

Add-Type @"
using System;
using System.Runtime.InteropServices;

public static class NativePowerV3Holdout
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
    $result = [NativePowerV3Holdout]::SetThreadExecutionState($keepAwakeFlags)
    if ($result -eq 0) {
        throw "SetThreadExecutionState failed."
    }

    Write-Host "PASS  Windows execution-state sleep protection enabled."

    $pythonArgs = @(
        "run",
        "python",
        ".\evals\system_evaluation\score_semantic_relevance_v3_holdout_union.py"
    )

    if ($Resume) {
        $pythonArgs += "--resume"
    }

    & uv @pythonArgs

    if ($LASTEXITCODE -ne 0) {
        throw (
            "V3 unseen holdout judge scoring failed or stopped. " +
            "Do not restart from scratch or inspect/tune semantic results. " +
            "After reviewing the execution error, resume persisted cases only."
        )
    }
}
finally {
    [void][NativePowerV3Holdout]::SetThreadExecutionState($ES_CONTINUOUS)
    Write-Host "PASS  Windows sleep protection cleared; normal power behavior restored."
}
