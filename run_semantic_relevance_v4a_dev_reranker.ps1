param(
    [int]$BatchSize = 8,
    [ValidateSet("auto", "cuda", "cpu")]
    [string]$Device = "auto"
)

$ErrorActionPreference = "Stop"

Write-Host "============================================================"
Write-Host "SEMANTIC RELEVANCE V4A - DEVELOPMENT RERANKER"
Write-Host "============================================================"
Write-Host "Retrieval rerun:    DISABLED"
Write-Host "Reranker inference: ENABLED (two passes)"
Write-Host "Judge execution:    DISABLED"
Write-Host "Release analysis:   DISABLED"
Write-Host "============================================================"

uv run python `
  .\evals\system_evaluation\run_semantic_relevance_v4a_dev_reranker.py `
  --batch-size $BatchSize `
  --device $Device

if ($LASTEXITCODE -ne 0) {
    throw "V4a development reranker execution failed."
}
