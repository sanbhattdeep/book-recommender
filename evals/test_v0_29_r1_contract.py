from __future__ import annotations
import json
from pathlib import Path

R = Path(__file__).resolve().parents[1]
SPEC = R / "evals/facets/semantic_relevance/semantic_relevance_query_facets.v0.10.0.json"
CFG = R / "evals/judge_configs/semantic_relevance_judge.v0.29.0-r1.json"
REG = R / "evals/datasets/semantic_relevance_v0.29_regression_manifest.v2.0.0.json"
JUDGE = R / "evals/semantic_relevance_facet_judge.py"
SCORING = R / "evals/semantic_relevance_facet_scoring.py"
RUNNER = R / "evals/run_judge_v0_29_r1_development.py"

def main():
    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    cfg = json.loads(CFG.read_text(encoding="utf-8"))
    reg = json.loads(REG.read_text(encoding="utf-8"))
    judge = JUDGE.read_text(encoding="utf-8")
    scoring = SCORING.read_text(encoding="utf-8")
    runner = RUNNER.read_text(encoding="utf-8")

    assert spec["version"] == "0.10.0"
    assert spec["development_dataset_version"] == "6.1.0"
    assert cfg["version"] == "0.29.0-r1"
    assert cfg["facet_spec_version"] == "0.10.0"
    assert len(reg["targeted_expectations"]) == 58

    assert reg["targeted_expectations"]["U4_Q03_T01"] == {"judge_min": 2, "judge_max": 2}
    assert reg["targeted_expectations"]["U4_Q03_T02"] == {"judge_min": 2, "judge_max": 2}
    assert reg["targeted_expectations"]["U4_Q04_T02"] == {"judge_max": 0}
    assert reg["targeted_expectations"]["U4_Q06_T01"] == {"judge_max": 0}
    assert reg["targeted_expectations"]["U4_Q09_T04"] == {"judge_min": 2, "judge_max": 2}
    assert reg["targeted_expectations"]["U2_Q09_T02"] == {"judge_min": 3, "judge_max": 4}

    assert "_q03_r1_purpose_component_check" in judge
    assert "_q04_r1_movement_text_anchor_guard" in judge
    assert "_q04_r1_recovery_movement_text_anchor_guard" in judge
    assert "q09_missing_resistance_cap_to_partial" in scoring

    assert 'EVALUATION_DATASET_VERSION = "6.1.0"' in runner
    assert 'FACET_SPEC_VERSION = "0.10.0"' in runner
    assert 'JUDGE_CONFIG_VERSION = "0.29.0-r1"' in runner
    assert "exactly 240 consumed cases" in runner
    print("v0.29 r1 semantic/provenance contract passed")

if __name__ == "__main__":
    main()
