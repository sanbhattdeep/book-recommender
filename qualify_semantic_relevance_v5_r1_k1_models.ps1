$ErrorActionPreference = "Stop"
Write-Host "============================================================"
Write-Host "V5 R1/K1 - PINNED MODEL QUALIFICATION"
Write-Host "============================================================"
Write-Host "Scope:                     FIXED SYNTHETIC NON-V5D PROBES"
Write-Host "R1/K1 model inference:     ENABLED ONLY FOR FROZEN PROBES"
Write-Host "Catalog/BM25 retrieval:    DISABLED"
Write-Host "Judge execution:           DISABLED"
Write-Host "Development selection:     DISABLED"
Write-Host "============================================================"
uv run python .\evals\system_evaluation\qualify_semantic_relevance_v5_r1_k1_models.py
if ($LASTEXITCODE -ne 0) { throw "V5 R1/K1 pinned-model qualification failed." }
