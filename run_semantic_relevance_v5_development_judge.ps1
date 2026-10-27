$ErrorActionPreference = "Stop"
Write-Host "============================================================"
Write-Host "SEMANTIC RELEVANCE V5 - FULL/RESUME JUDGE EXECUTION"
Write-Host "============================================================"
Write-Host "Frozen cases:              452"
Write-Host "Resume-safe persistence:   ENABLED"
Write-Host "Development selection:     DISABLED UNTIL COMPLETE"
Write-Host "============================================================"
uv run python .\evals\system_evaluation\score_semantic_relevance_v5_development_display_pairs.py
if ($LASTEXITCODE -ne 0) { throw "V5D full/resume judge execution failed." }
