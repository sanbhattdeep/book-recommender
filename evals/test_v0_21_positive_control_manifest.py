import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
path = ROOT / "evals/datasets/semantic_relevance_v0.21_regression_manifest.v1.0.0.json"
manifest = json.loads(path.read_text(encoding="utf-8"))

assert manifest["judge_version"] == "0.21.0"
assert manifest["q11_q12_positive_controls"] == ["U_Q11_T02", "U_Q12_T02"]
for case_id in manifest["q11_q12_positive_controls"]:
    assert manifest["targeted_expectations"][case_id]["judge_min"] >= 2

print("v0.21 Q11/Q12 positive-control manifest tests passed.")
