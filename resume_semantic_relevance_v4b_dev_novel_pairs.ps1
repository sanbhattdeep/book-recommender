$ErrorActionPreference = "Stop"
uv run python .\evals\system_evaluation\score_semantic_relevance_v4b_dev_novel_pairs.py --resume
if ($LASTEXITCODE -ne 0) {
    throw "V4B novel judging resume stopped. Preserve persisted output and review."
}
