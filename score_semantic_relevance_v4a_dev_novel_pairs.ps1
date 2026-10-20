param(
    [switch]$Resume
)
$ErrorActionPreference = "Stop"
Write-Host "============================================================"
Write-Host "SEMANTIC RELEVANCE V4A - DEVELOPMENT NOVEL-PAIR JUDGING"
Write-Host "============================================================"
Write-Host "Frozen novel pairs:    79"
Write-Host "Human labels:          DISABLED"
Write-Host "Release analysis:      DISABLED"
Write-Host "Development selection: DISABLED"
Write-Host "============================================================"

$argsList = @("run", "python", ".\evals\system_evaluation\score_semantic_relevance_v4a_dev_novel_pairs.py")
if ($Resume) { $argsList += "--resume" }
& uv @argsList
if ($LASTEXITCODE -ne 0) {
    throw "V4a development novel-pair judging failed. Preserve partial artifacts and use -Resume only after a pure execution failure."
}
