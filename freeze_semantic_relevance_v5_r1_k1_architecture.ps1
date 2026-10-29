$ErrorActionPreference = "Stop"
Write-Host "============================================================"
Write-Host "SEMANTIC RELEVANCE V5 - R1/K1 ARCHITECTURE FREEZE"
Write-Host "============================================================"
Write-Host "V5D authoring/execution:    DISABLED"
Write-Host "Retrieval/model/reranker:   DISABLED"
Write-Host "Judge execution:            DISABLED"
Write-Host "Development selection:      NOT CALCULATED"
Write-Host "Release analysis:           DISABLED"
Write-Host "============================================================"
uv run python .\evals\system_evaluation\freeze_semantic_relevance_v5_r1_k1_architecture.py
if ($LASTEXITCODE -ne 0) { throw "V5 R1/K1 architecture freeze failed." }
