from __future__ import annotations
import json
from pathlib import Path

R = Path(__file__).resolve().parents[1]
SPEC = R / "evals/facets/semantic_relevance/semantic_relevance_query_facets.v0.9.12.json"
CFG = R / "evals/judge_configs/semantic_relevance_judge.v0.28.0-r7.json"
REG = R / "evals/datasets/semantic_relevance_v0.28_regression_manifest.v1.6.0.json"

def main():
    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    cfg = json.loads(CFG.read_text(encoding="utf-8"))
    reg = json.loads(REG.read_text(encoding="utf-8"))
    assert spec["version"] == "0.9.12"
    assert cfg["version"] == "0.28.0-r7"
    assert cfg["facet_spec_version"] == "0.9.12"
    assert len(reg["targeted_expectations"]) == 52
    assert reg["targeted_expectations"]["U2_Q11_T02"] == {"judge_min": 3, "judge_max": 4}
    assert reg["r7_new_target"]["U2_Q11_T02"]["r6_judge_score"] == 2
    assert set(reg["diagnostic_only_cases"]) == {"U3_Q10_T01", "U3_Q11_T04"}
    print("v0.28 r7 semantic/provenance contract passed")

if __name__ == "__main__":
    main()
