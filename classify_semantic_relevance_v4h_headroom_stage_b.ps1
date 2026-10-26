$ErrorActionPreference='Stop'
uv run python .\evals\system_evaluation\classify_semantic_relevance_v4h_headroom_stage_b.py
if ($LASTEXITCODE -ne 0) { throw 'V4H Stage-B classification failed.' }
