param(
    [ValidateSet("auto","cuda","cpu")]
    [string]$Device = "auto",

    [int]$BatchSize = 8
)

$ErrorActionPreference = "Stop"
Write-Host "============================================================"
Write-Host "SEMANTIC RELEVANCE V4H - FROZEN V3B/V4B RERANKING"
Write-Host "============================================================"
Write-Host "Cross-encoder inference:    ENABLED (2 passes)"
Write-Host "Retrieval calls:            DISABLED"
Write-Host "Judge execution:            DISABLED"
Write-Host "Release analysis:           DISABLED"
Write-Host "============================================================"

uv run python .\evals\system_evaluation\run_semantic_relevance_v4h_reranking.py `
  --device $Device `
  --batch-size $BatchSize

if ($LASTEXITCODE -ne 0) {
    throw "V4H reranking failed. Preserve any partial artifacts and review before retrying."
}
