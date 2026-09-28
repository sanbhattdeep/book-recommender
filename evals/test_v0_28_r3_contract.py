from pathlib import Path
import json

E = Path(__file__).resolve().parent
spec = json.loads(
    (E / "facets/semantic_relevance/semantic_relevance_query_facets.v0.9.9.json")
    .read_text(encoding="utf-8")
)
cfg = json.loads(
    (E / "judge_configs/semantic_relevance_judge.v0.28.0.json")
    .read_text(encoding="utf-8")
)
reg = json.loads(
    (E / "datasets/semantic_relevance_v0.28_regression_manifest.v1.2.0.json")
    .read_text(encoding="utf-8")
)
judge = (E / "semantic_relevance_facet_judge.py").read_text(encoding="utf-8")
runner = (E / "run_judge_development.py").read_text(encoding="utf-8")

assert spec["version"] == "0.9.9"
assert cfg["version"] == "0.28.0"
assert cfg["facet_spec_version"] == "0.9.9"
assert 'FACET_SPEC_VERSION = "0.9.9"' in runner

assert len(reg["targeted_expectations"]) == 40
assert reg["targeted_expectations"]["U_Q11_T70"] == {"judge_min": 1}
assert reg["targeted_expectations"]["U2_Q11_T70"] == {"judge_min": 1}
assert reg["targeted_expectations"]["U3_Q11_T03"] == {"judge_min": 1}
assert reg["targeted_expectations"]["U3_Q03_T04"] == {"judge_max": 1}
assert reg["targeted_expectations"]["U3_Q03_T01"] == {"judge_max": 0}
assert reg["targeted_expectations"]["U3_Q11_T02"] == {"judge_max": 1}
assert set(reg["diagnostic_only_cases"]) == {"U3_Q10_T01", "U3_Q11_T04"}

q03 = next(q for q in spec["queries"] if q["query_id"] == "Q03")
q03f1 = next(f for f in q03["facets"] if f["facet_id"] == "F1")
q03_text = q03f1["semantic_definition"] + " " + " ".join(
    q03f1["required_components"][0]["negative_boundaries"]
)
assert "Generic intellectual learning" in q03_text
assert "wising up" in q03_text
assert "Potential for happiness" in q03_text

q11 = next(q for q in spec["queries"] if q["query_id"] == "Q11")
q11f1 = next(f for f in q11["facets"] if f["facet_id"] == "F1")
q11_text = q11f1["semantic_definition"] + " " + " ".join(
    q11f1["required_components"][0]["negative_boundaries"]
)
assert "explicit narrative quest, journey, or expedition" in q11_text.lower()
assert "taking up arms" in q11_text
assert "dangerous target" in q11_text
assert "bare mission" in q11_text.lower()

for token in [
    "_q03_r3_learning_only_guard",
    "_q11_r3_adventure_positive_guard",
    "_apply_v028_r3_isolated_component_guards",
    "_apply_v028_r3_recovery_guard",
]:
    assert token in judge

# r1 global wording filter remains rolled back.
assert "speculative_entailment_guard" not in judge

print("v0.28 r3 localized regression contract passed")
