"""Static/unit contract for the v0.27.0 r7 stability spot-check. No judge calls."""
from pathlib import Path
import json

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
manifest_path = HERE / "datasets" / "semantic_relevance_v0.27_r7_stability_manifest.v1.0.0.json"
analyzer_path = HERE / "analyze_v0_27_r7_stability.py"
runner_path = ROOT / "run_v027_r7_stability.ps1"

manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
assert manifest["required_fresh_runs"] == 3
expected = manifest["expectations"]
assert len(expected) == 7
assert expected["U3_Q01_T03"]["judge_min"] == expected["U3_Q01_T03"]["judge_max"] == 0
assert expected["U3_Q02_T05"]["judge_min"] == expected["U3_Q02_T05"]["judge_max"] == 0
assert expected["U3_Q07_T01"]["judge_min"] == expected["U3_Q07_T01"]["judge_max"] == 1
assert expected["U3_Q05_T01"]["judge_min"] == 2 and expected["U3_Q05_T01"]["judge_max"] == 3
assert "U_Q01_T02" in expected and "U_Q02_T02" in expected and "U_Q07_T02" in expected

analyzer = analyzer_path.read_text(encoding="utf-8")
for token in ["exactly {required_runs} fresh --run directories", "duplicate run directory", "Observed score sequences", "Stability spot-check:"]:
    assert token in analyzer, token

runner = runner_path.read_text(encoding="utf-8")
for token in ["verify_v0_27_package.py", "build_v0_27_r7_development_dataset.py", "analyze_v0_27_r7_stability.py", "Final holdout remains LOCKED_DO_NOT_RUN."]:
    assert token in runner, token
assert "final_holdout" not in runner.lower().replace("final holdout", "")
print("v0.27 r7 stability contract passed")
