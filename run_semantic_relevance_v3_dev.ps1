param(
    [int]$BatchSize = 8,
    [ValidateSet("auto", "cuda", "cpu")]
    [string]$Device = "auto"
)

$ErrorActionPreference = "Stop"

Write-Host "============================================================"
Write-Host "SEMANTIC RELEVANCE V3 DEVELOPMENT RERANKER"
Write-Host "============================================================"
Write-Host "Batch size: $BatchSize"
Write-Host "Device: $Device"
Write-Host "Judge labels loaded by reranker: NO"
Write-Host "Judge calls: 0"
Write-Host "============================================================"

uv run python `
  .\evals\system_evaluation\run_semantic_relevance_v3_dev.py `
  --batch-size $BatchSize `
  --device $Device

if ($LASTEXITCODE -ne 0) {
    throw "V3 development reranker execution failed."
}
