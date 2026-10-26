$ErrorActionPreference = "Stop"
Write-Host "============================================================"
Write-Host "V5 R1/K1 - PINNED MODEL QUALIFICATION PREFLIGHT"
Write-Host "============================================================"
Write-Host "Model inference:           DISABLED"
Write-Host "V5D authoring/execution:   DISABLED"
Write-Host "Judge execution:           DISABLED"
Write-Host "Development selection:     NOT CALCULATED"
Write-Host "============================================================"
uv run python .\evals\system_evaluation\verify_semantic_relevance_v5_r1_k1_model_qualification.py
if ($LASTEXITCODE -ne 0) { throw "V5 R1/K1 model qualification preflight failed." }
