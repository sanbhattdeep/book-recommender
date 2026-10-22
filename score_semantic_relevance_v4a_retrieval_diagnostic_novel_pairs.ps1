param(
    [switch]$Resume
)
$ErrorActionPreference = "Stop"
Write-Host "============================================================"
Write-Host "SEMANTIC RELEVANCE V4A - RETRIEVAL DIAGNOSTIC NOVEL JUDGING"
Write-Host "============================================================"
Write-Host "Frozen novel pairs:    675"
Write-Host "Diagnostic role:       EXPLANATORY ONLY"
Write-Host "Execution patch:       DeepEval per-attempt timeout = 900s"
Write-Host "Resume-only package:   YES"
Write-Host "Case quarantine:       H14/9781582972947 / UNSCORED; continue others"
Write-Host "Human labels:          DISABLED"
Write-Host "Development selection: FROZEN / NOT REOPENED"
Write-Host "Release analysis:      DISABLED"
Write-Host "============================================================"

if (-not $Resume) {
    throw "This r3 timeout/quarantine-patched package is resume-only. Re-run with -Resume to preserve the frozen partial score file."
}

$timeoutVar = "DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS_OVERRIDE"
$previousTimeout = [Environment]::GetEnvironmentVariable($timeoutVar, "Process")
[Environment]::SetEnvironmentVariable($timeoutVar, "900", "Process")

try {
    $argsList = @("run", "python", ".\evals\system_evaluation\score_semantic_relevance_v4a_retrieval_diagnostic_novel_pairs.py", "--resume")
    & uv @argsList
    $exitCode = $LASTEXITCODE
}
finally {
    [Environment]::SetEnvironmentVariable($timeoutVar, $previousTimeout, "Process")
}

if ($exitCode -ne 0) {
    throw "V4a retrieval-diagnostic novel judging encountered an unhandled failure under the frozen timeout/quarantine execution patch. Preserve partial and quarantine artifacts; do not change semantic settings before review."
}
