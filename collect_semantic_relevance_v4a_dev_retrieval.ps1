$ErrorActionPreference = "Stop"

Write-Host "============================================================"
Write-Host "SEMANTIC RELEVANCE V4A - DEVELOPMENT HYBRID RETRIEVAL"
Write-Host "============================================================"
Write-Host "Dense retrieval rerun: NO"
Write-Host "BM25 catalog indexing: ENABLED"
Write-Host "BM25 retrieval:        ENABLED"
Write-Host "Hybrid RRF:            ENABLED"
Write-Host "Reranker inference:    DISABLED"
Write-Host "Judge execution:       DISABLED"
Write-Host "============================================================"

uv run python `
  .\evals\system_evaluation\collect_semantic_relevance_v4a_dev_retrieval.py

if ($LASTEXITCODE -ne 0) {
    throw "V4a development retrieval execution failed."
}
