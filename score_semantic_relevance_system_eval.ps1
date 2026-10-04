$ErrorActionPreference = "Stop"

$env:DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS_OVERRIDE = "300"
$runDir = ".\evals\runs\semantic_relevance_system_eval_v1\20261002T152856Z_recommendations"

# ---------------------------------------------------------------------------
# Windows sleep / Modern Standby protection
# ---------------------------------------------------------------------------
# Keep both the system and display awake for the duration of this long-running
# evaluation. This changes only the execution state of this PowerShell process;
# it does not permanently alter the Windows power plan.
#
# ES_CONTINUOUS       0x80000000
# ES_SYSTEM_REQUIRED  0x00000001
# ES_DISPLAY_REQUIRED 0x00000002
#
# The state is always cleared in finally, even if judge scoring fails.
# ---------------------------------------------------------------------------

if (-not ("SleepProtection.NativeMethods" -as [type])) {
    Add-Type @"
using System;
using System.Runtime.InteropServices;

namespace SleepProtection {
    public static class NativeMethods {
        [DllImport("kernel32.dll", SetLastError = true)]
        public static extern uint SetThreadExecutionState(uint esFlags);
    }
}
"@
}

$ES_CONTINUOUS       = [Convert]::ToUInt32("80000000", 16)
$ES_SYSTEM_REQUIRED  = [uint32]0x00000001
$ES_DISPLAY_REQUIRED = [uint32]0x00000002

$keepAwakeFlags = (
    $ES_CONTINUOUS `
    -bor $ES_SYSTEM_REQUIRED `
    -bor $ES_DISPLAY_REQUIRED
)

# Windows PowerShell 5.1 treats hexadecimal 0x80000000 as a signed Int32.
# The Convert.ToUInt32() construction above avoids that overflow behavior.
if (
    $ES_CONTINUOUS -ne [uint32]2147483648 `
    -or $keepAwakeFlags -ne [uint32]2147483651
) {
    throw "Windows sleep-protection flag calculation failed."
}

Write-Host "============================================================"
Write-Host "SEMANTIC RELEVANCE SYSTEM EVAL - CALIBRATED JUDGE SCORING"
Write-Host "============================================================"
Write-Host "Frozen collection run: $runDir"
Write-Host "Judge: semantic_relevance_v0.29.0-r5"
Write-Host "Expected pairs: 120"
Write-Host "Human relevance labels: NOT USED"
Write-Host "Aggregate metrics: NOT CALCULATED IN THIS STAGE"
Write-Host "Bootstrap CIs: NOT CALCULATED IN THIS STAGE"
Write-Host "Windows sleep protection: ENABLED for this process"
Write-Host "System sleep / Modern Standby: BLOCKED while scoring"
Write-Host "Display timeout: BLOCKED while scoring"
Write-Host "============================================================"

$executionStateSet = $false

try {
    $result = [SleepProtection.NativeMethods]::SetThreadExecutionState(
        $keepAwakeFlags
    )

    if ($result -eq 0) {
        $win32Error = [Runtime.InteropServices.Marshal]::GetLastWin32Error()
        throw "Failed to enable Windows sleep protection. SetThreadExecutionState returned 0. Win32 error=$win32Error"
    }

    $executionStateSet = $true
    Write-Host "PASS  Windows execution-state sleep protection enabled."

    uv run python `
      .\evals\system_evaluation\score_frozen_semantic_relevance_recommendations.py `
      --collection-run $runDir

    if ($LASTEXITCODE -ne 0) {
        throw "Calibrated judge scoring failed or is partial. Rerun SAME command to resume."
    }

    Write-Host ""
    Write-Host "============================================================"
    Write-Host "CALIBRATED JUDGE SCORING COMPLETE"
    Write-Host "Next stage: aggregate metrics + bootstrap CIs + release gates."
    Write-Host "============================================================"
}
finally {
    if ($executionStateSet) {
        # ES_CONTINUOUS alone clears SYSTEM_REQUIRED / DISPLAY_REQUIRED and
        # restores normal Windows power-management behavior.
        $clearResult = [SleepProtection.NativeMethods]::SetThreadExecutionState(
            $ES_CONTINUOUS
        )

        if ($clearResult -eq 0) {
            $clearError = [Runtime.InteropServices.Marshal]::GetLastWin32Error()
            Write-Warning "Could not explicitly clear Windows execution-state protection. It will also clear when this PowerShell process exits. Win32 error=$clearError"
        }
        else {
            Write-Host "PASS  Windows sleep protection cleared; normal power behavior restored."
        }
    }
}
