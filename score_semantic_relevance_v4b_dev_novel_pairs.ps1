$ErrorActionPreference = "Stop"
Write-Host "============================================================"
Write-Host "SEMANTIC RELEVANCE V4B - JUDGE 22 NOVEL DISPLAYED PAIRS"
Write-Host "============================================================"
Write-Host "Frozen judge:               semantic_relevance_v0.29.0-r5"
Write-Host "Human relevance labels:     NOT USED"
Write-Host "Development comparison:     NOT CALCULATED"
Write-Host "Release analysis:           DISABLED"
Write-Host "============================================================"

uv run python .\evals\system_evaluation\score_semantic_relevance_v4b_dev_novel_pairs.py
if ($LASTEXITCODE -ne 0) {
    throw "V4B novel judging stopped. Preserve persisted partial scores; review before using the resume command."
}
