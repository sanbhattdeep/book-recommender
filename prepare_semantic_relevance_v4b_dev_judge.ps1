$ErrorActionPreference = "Stop"
Write-Host "============================================================"
Write-Host "SEMANTIC RELEVANCE V4B - NOVEL JUDGE PREFLIGHT"
Write-Host "============================================================"
Write-Host "Judge/model execution:      DISABLED"
Write-Host "Human relevance labels:     NOT USED"
Write-Host "Development comparison:     NOT CALCULATED"
Write-Host "Release analysis:           DISABLED"
Write-Host "============================================================"

uv run python .\evals\system_evaluation\verify_semantic_relevance_v4b_dev_judge_package.py
if ($LASTEXITCODE -ne 0) {
    throw "V4B novel-judge preflight failed. Preserve frozen judging inputs and review before execution."
}
