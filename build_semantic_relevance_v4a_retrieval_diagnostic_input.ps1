$ErrorActionPreference = "Stop"

Write-Host "============================================================"
Write-Host "SEMANTIC RELEVANCE V4A - BUILD 17-QUERY RETRIEVAL DIAGNOSTIC INPUT"
Write-Host "============================================================"
Write-Host "Judge execution:           DISABLED"
Write-Host "Model execution:           DISABLED"
Write-Host "Development selection:     FROZEN / NOT REOPENED"
Write-Host "Release analysis:          DISABLED"
Write-Host "============================================================"

uv run python .\evals\system_evaluation\prepare_semantic_relevance_v4a_retrieval_diagnostic.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
