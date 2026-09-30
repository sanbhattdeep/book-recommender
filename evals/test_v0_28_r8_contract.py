from __future__ import annotations
import json
from pathlib import Path

R = Path(__file__).resolve().parents[1]
SPEC = R / "evals/facets/semantic_relevance/semantic_relevance_query_facets.v0.9.13.json"
CFG = R / "evals/judge_configs/semantic_relevance_judge.v0.28.0-r8.json"
REG = R / "evals/datasets/semantic_relevance_v0.28_regression_manifest.v1.7.0.json"
JUDGE = R / "evals/semantic_relevance_facet_judge.py"

def main():
    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    cfg = json.loads(CFG.read_text(encoding="utf-8"))
    reg = json.loads(REG.read_text(encoding="utf-8"))
    judge = JUDGE.read_text(encoding="utf-8")
    assert spec["version"] == "0.9.13"
    assert cfg["version"] == "0.28.0-r8"
    assert cfg["facet_spec_version"] == "0.9.13"
    assert len(reg["targeted_expectations"]) == 52
    assert reg["targeted_expectations"]["U2_Q11_T02"] == {"judge_min": 3, "judge_max": 4}
    assert reg["r8_repair"]["U2_Q11_T02"]["r7_judge_score"] == 2
    assert "_q11_r8_cross_span_classical_myth_component_check" in judge
    assert "r8_cross_span_deterministic" in judge
    print("v0.28 r8 semantic/provenance contract passed")

if __name__ == "__main__":
    main()
