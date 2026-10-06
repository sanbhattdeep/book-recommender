$ErrorActionPreference = "Stop"

$RunDir = ".\evals\runs\semantic_relevance_system_eval_v2\20261004T140218Z_recommendations"

Write-Host "============================================================"
Write-Host "SEMANTIC RELEVANCE SYSTEM EVAL V2 - CALIBRATED JUDGE SCORING"
Write-Host "============================================================"
Write-Host "Frozen collection run: $RunDir"
Write-Host "System variant: v2-orderfix"
Write-Host "Judge: semantic_relevance_v0.29.0-r5"
Write-Host "Expected pairs: 120"
Write-Host "Human relevance labels: NOT USED"
Write-Host "Aggregate metrics: NOT CALCULATED IN THIS STAGE"
Write-Host "Bootstrap CIs: NOT CALCULATED IN THIS STAGE"
Write-Host "Windows sleep protection: ENABLED for this process"
Write-Host "============================================================"

Add-Type @"
using System;
using System.Runtime.InteropServices;

public static class NativePowerV2
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
    $result = [NativePowerV2]::SetThreadExecutionState($keepAwakeFlags)

    if ($result -eq 0) {
        throw "SetThreadExecutionState failed."
    }

    Write-Host "PASS  Windows execution-state sleep protection enabled."

    uv run python `
      .\evals\system_evaluation\score_frozen_semantic_relevance_recommendations_v2.py `
      --collection-run $RunDir

    if ($LASTEXITCODE -ne 0) {
        throw (
            "V2 calibrated judge scoring failed or is partial. " +
            "Rerun SAME command to resume."
        )
    }
}
finally {
    [void][NativePowerV2]::SetThreadExecutionState($ES_CONTINUOUS)
    Write-Host "PASS  Windows sleep protection cleared; normal power behavior restored."
}
