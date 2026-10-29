$ErrorActionPreference = 'Stop'
Write-Host '============================================================'
Write-Host 'SEMANTIC RELEVANCE V5 - V5D24 AUTHORING FREEZE'
Write-Host '============================================================'
Write-Host 'V5D retrieval/reranking: DISABLED'
Write-Host 'Judge execution:         DISABLED'
Write-Host 'Development selection:   NOT CALCULATED'
Write-Host '============================================================'
uv run python .\evals\system_evaluation\freeze_semantic_relevance_v5_development_authoring.py
if ($LASTEXITCODE -ne 0) { throw 'V5D24 authoring freeze failed.' }
