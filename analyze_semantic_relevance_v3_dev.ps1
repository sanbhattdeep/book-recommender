param(
    [Parameter(Mandatory=$true)]
    [string]$Run
)

$ErrorActionPreference = "Stop"

uv run python `
  .\evals\system_evaluation\analyze_semantic_relevance_v3_dev.py `
  --run $Run

if ($LASTEXITCODE -ne 0) {
    throw "V3 development analysis failed."
}
