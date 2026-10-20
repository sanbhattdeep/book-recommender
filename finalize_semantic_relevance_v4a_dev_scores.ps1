$ErrorActionPreference = "Stop"
Write-Host "============================================================"
Write-Host "SEMANTIC RELEVANCE V4A - FINALIZE DEV SCORE MAP"
Write-Host "============================================================"
Write-Host "Judge execution:       DISABLED"
Write-Host "Model execution:       DISABLED"
Write-Host "Development comparison: DISABLED"
Write-Host "Development selection:  DISABLED"
Write-Host "Release analysis:        DISABLED"
Write-Host "============================================================"
uv run python .\evals\system_evaluation\finalize_semantic_relevance_v4a_dev_scores.py
if ($LASTEXITCODE -ne 0) { throw "V4a development score finalization failed." }
