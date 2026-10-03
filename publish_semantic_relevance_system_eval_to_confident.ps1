$ErrorActionPreference = "Stop"

# Prevent DeepEval/Confident AI from opening a browser automatically.
$env:CONFIDENT_OPEN_BROWSER = "0"

Write-Host "============================================================"
Write-Host "PUBLISH SEMANTIC RELEVANCE SYSTEM EVAL TO CONFIDENT AI"
Write-Host "============================================================"
Write-Host "Source: frozen local judge_scores.csv"
Write-Host "Expected test cases: 120"
Write-Host "Judge re-execution: NO"
Write-Host "Human relevance labels: NOT USED"
Write-Host "CONFIDENT_API_KEY: read from environment / DeepEval dotenv"
Write-Host "============================================================"

uv run python `
  .\evals\system_evaluation\publish_semantic_relevance_system_eval_to_confident.py

if ($LASTEXITCODE -ne 0) {
    throw (
        "Confident AI publication failed. Local frozen evaluation results "
        + "remain valid and unchanged."
    )
}

Write-Host ""
Write-Host "============================================================"
Write-Host "CONFIDENT AI PUBLICATION COMPLETE"
Write-Host "Run: uv run deepeval view"
Write-Host "============================================================"
