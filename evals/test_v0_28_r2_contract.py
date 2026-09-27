from pathlib import Path
import json

E = Path(__file__).resolve().parent
spec = json.loads(
    (E / "facets/semantic_relevance/semantic_relevance_query_facets.v0.9.8.json")
    .read_text(encoding="utf-8")
)
cfg = json.loads(
    (E / "judge_configs/semantic_relevance_judge.v0.28.0.json")
    .read_text(encoding="utf-8")
)
reg = json.loads(
    (E / "datasets/semantic_relevance_v0.28_regression_manifest.v1.1.0.json")
    .read_text(encoding="utf-8")
)
judge = (E / "semantic_relevance_facet_judge.py").read_text(encoding="utf-8")

assert spec["version"] == "0.9.8"
assert cfg["version"] == "0.28.0"
assert cfg["facet_spec_version"] == "0.9.8"
assert len(reg["targeted_expectations"]) == 36
assert reg["targeted_expectations"]["U3_Q03_T01"] == {"judge_max": 0}
assert reg["targeted_expectations"]["U3_Q11_T02"] == {"judge_max": 1}
assert set(reg["diagnostic_only_cases"]) == {"U3_Q10_T01", "U3_Q11_T04"}

# Local Q03 repair remains.
q03 = next(q for q in spec["queries"] if q["query_id"] == "Q03")
q03f1 = next(f for f in q03["facets"] if f["facet_id"] == "F1")
assert "Potential for happiness" in " ".join(
    q03f1["required_components"][0]["negative_boundaries"]
)

# Local Q11 adventure / legendary-hero repairs remain.
q11 = next(q for q in spec["queries"] if q["query_id"] == "Q11")
adventure = next(f for f in q11["facets"] if f["facet_id"] == "F1")
hero = next(f for f in q11["facets"] if f["facet_id"] == "F4")
assert "mission, duty, objective" in " ".join(
    adventure["required_components"][0]["negative_boundaries"]
)
assert "Fantasy-sounding names or settings" in " ".join(
    hero["required_components"][0]["negative_boundaries"]
)

# Frozen r7 positive boundaries must remain explicit in the facet spec.
q07 = next(q for q in spec["queries"] if q["query_id"] == "Q07")
q07f1 = next(f for f in q07["facets"] if f["facet_id"] == "F1")
assert "Explicit parental opposition" in q07f1["semantic_definition"]

q04 = next(q for q in spec["queries"] if q["query_id"] == "Q04")
q04f3 = next(f for f in q04["facets"] if f["facet_id"] == "F3")
assert "shipwrecked" in " ".join(
    q04f3["required_components"][1]["negative_boundaries"]
)

# r2 structural change: lexical direct cue list mentions receive a deterministic
# role cap only when no independent role span exists.
assert "_apply_direct_cue_list_mention_guard" in judge
assert "independent_support" in judge

# r1 global speculative guard must be gone.
assert "speculative_entailment_guard" not in judge

print("v0.28 r2 localized repair contract passed")
