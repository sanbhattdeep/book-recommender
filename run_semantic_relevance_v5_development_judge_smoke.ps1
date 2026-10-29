$ErrorActionPreference = "Stop"
Write-Host "============================================================"
Write-Host "SEMANTIC RELEVANCE V5 - 3-CASE JUDGE SMOKE"
Write-Host "============================================================"
Write-Host "Smoke queries: V5D01, V5D17, V5D21"
Write-Host "Development metrics/selection: DISABLED"
Write-Host "============================================================"
uv run python .\evals\system_evaluation\score_semantic_relevance_v5_development_display_pairs.py --smoke
if ($LASTEXITCODE -ne 0) { throw "V5D 3-case judge smoke failed." }
