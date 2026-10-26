$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Push-Location $root
try {
    uv run python evals/system_evaluation/close_semantic_relevance_v4h_headroom_diagnostic.py
}
finally {
    Pop-Location
}
