$ErrorActionPreference = "Stop"

Write-Host "============================================================"
Write-Host "SEMANTIC RELEVANCE V4A - 17-QUERY RETRIEVAL DIAGNOSTIC PREFLIGHT"
Write-Host "============================================================"
Write-Host "Diagnostic input creation: DISABLED"
Write-Host "Judge execution:           DISABLED"
Write-Host "Model execution:           DISABLED"
Write-Host "Development selection:     FROZEN / NOT REOPENED"
Write-Host "Release analysis:          DISABLED"
Write-Host "============================================================"

uv run python .\evals\system_evaluation\verify_semantic_relevance_v4a_retrieval_diagnostic_package.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
