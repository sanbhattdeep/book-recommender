$ErrorActionPreference = "Stop"

Write-Host "============================================================"
Write-Host "SEMANTIC RELEVANCE V4A - DEV RERANKER PACKAGE PREFLIGHT"
Write-Host "============================================================"
Write-Host "Retrieval rerun:    DISABLED"
Write-Host "Reranker inference: DISABLED"
Write-Host "Judge execution:    DISABLED"
Write-Host "Release analysis:   DISABLED"
Write-Host "============================================================"

uv run python `
  .\evals\system_evaluation\verify_semantic_relevance_v4a_dev_reranker_package.py

if ($LASTEXITCODE -ne 0) {
    throw "V4a development reranker package preflight failed."
}
