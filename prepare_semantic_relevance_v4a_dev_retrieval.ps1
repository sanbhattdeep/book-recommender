$ErrorActionPreference = "Stop"

Write-Host "============================================================"
Write-Host "SEMANTIC RELEVANCE V4A - DEV RETRIEVAL PACKAGE PREFLIGHT"
Write-Host "============================================================"
Write-Host "Runtime catalog read:  DISABLED"
Write-Host "BM25 catalog indexing: DISABLED"
Write-Host "BM25 retrieval:        DISABLED"
Write-Host "Reranker inference:    DISABLED"
Write-Host "Judge execution:       DISABLED"
Write-Host "============================================================"

uv run python `
  .\evals\system_evaluation\verify_semantic_relevance_v4a_dev_retrieval_package.py

if ($LASTEXITCODE -ne 0) {
    throw "V4a development retrieval package preflight failed."
}
