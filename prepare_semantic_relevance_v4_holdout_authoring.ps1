$ErrorActionPreference = "Stop"
Write-Host "============================================================"
Write-Host "SEMANTIC RELEVANCE V4H - UNSEEN HOLDOUT AUTHORING PREFLIGHT"
Write-Host "============================================================"
Write-Host "Holdout retrieval:          DISABLED"
Write-Host "Judge/model execution:      DISABLED"
Write-Host "Release analysis:           DISABLED"
Write-Host "============================================================"

uv run python .\evals\system_evaluation\verify_semantic_relevance_v4_holdout_authoring.py
if ($LASTEXITCODE -ne 0) {
    throw "V4H authoring preflight failed. Do not run any V4H retrieval."
}
