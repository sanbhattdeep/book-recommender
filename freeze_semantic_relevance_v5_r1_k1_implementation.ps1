$ErrorActionPreference = "Stop"
Write-Host "============================================================"
Write-Host "SEMANTIC RELEVANCE V5 - R1/K1 IMPLEMENTATION SOURCE FREEZE"
Write-Host "============================================================"
Write-Host "V5D authoring/execution:    DISABLED"
Write-Host "Model/retrieval/reranker:   DISABLED"
Write-Host "Judge execution:            DISABLED"
Write-Host "Development selection:      NOT CALCULATED"
Write-Host "============================================================"
uv run python .\evals\system_evaluation\freeze_semantic_relevance_v5_r1_k1_implementation.py
if ($LASTEXITCODE -ne 0) { throw "V5 R1/K1 implementation source freeze failed." }
