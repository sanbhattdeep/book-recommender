$ErrorActionPreference = "Stop"

Write-Host "============================================================"
Write-Host "SEMANTIC RELEVANCE TOP-50 DIAGNOSTIC - CHECKPOINT B R2"
Write-Host "============================================================"
Write-Host "Purpose: reuse frozen v1/v2 judge evidence + prepare novel-only input"
Write-Host "Frozen Checkpoint A pairs: 600"
Write-Host "Expected reusable pairs: 213"
Write-Host "Expected novel pairs: 387"
Write-Host "Expected historical score conflicts: 0"
Write-Host "Expected historical payload mismatches: 0"
Write-Host "Optional author null normalization: ENABLED"
Write-Host "Judge execution: DISABLED"
Write-Host "Human relevance labels: NOT USED"
Write-Host "============================================================"

uv run python `
  .\evals\system_evaluation\prepare_semantic_relevance_top50_score_reuse.py

if ($LASTEXITCODE -ne 0) {
    throw "Top-50 diagnostic Checkpoint B failed. Do NOT proceed to novel judging."
}

Write-Host ""
Write-Host "============================================================"
Write-Host "TOP-50 DIAGNOSTIC CHECKPOINT B COMPLETE"
Write-Host "No judge calls were performed."
Write-Host "Upload the captured console output for review before Checkpoint C."
Write-Host "============================================================"
