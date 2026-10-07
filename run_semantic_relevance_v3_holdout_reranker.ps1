param(
    [int]$BatchSize = 8,
    [ValidateSet("auto", "cuda", "cpu")]
    [string]$Device = "auto"
)

$ErrorActionPreference = "Stop"

$RetrievalRun = ".\evals\runs\semantic_relevance_v3_holdout\20261007T124150Z_retrieval"

Write-Host "============================================================"
Write-Host "SEMANTIC RELEVANCE V3 - UNSEEN HOLDOUT RERANKING"
Write-Host "============================================================"
Write-Host "Selected variant: v3b_rrf_hybrid"
Write-Host "Retrieval run: $RetrievalRun"
Write-Host "Batch size: $BatchSize"
Write-Host "Device: $Device"
Write-Host "Judge labels: NOT LOADED"
Write-Host "Human labels: NOT LOADED"
Write-Host "Judge calls: 0"
Write-Host "============================================================"

uv run python `
  .\evals\system_evaluation\run_semantic_relevance_v3_holdout_reranker.py `
  --retrieval-run $RetrievalRun `
  --batch-size $BatchSize `
  --device $Device

if ($LASTEXITCODE -ne 0) {
    throw "V3 unseen holdout reranking failed."
}
