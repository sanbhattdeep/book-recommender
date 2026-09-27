from pathlib import Path
import json

E = Path(__file__).resolve().parent
judge = (E / "semantic_relevance_facet_judge.py").read_text(encoding="utf-8")
spec = json.loads(
    (E / "facets/semantic_relevance/semantic_relevance_query_facets.v0.9.8.json")
    .read_text(encoding="utf-8")
)

# r2 explicitly rolls back the r1 global wording-based speculative-entailment
# filter. Entailment discipline is expressed only through localized facet
# definitions / negative boundaries and narrow deterministic post-processing.
assert "_SPECULATIVE_ENTAILMENT_MARKERS" not in judge
assert "_apply_speculative_isolated_entailment_guard" not in judge
assert "_apply_speculative_recovery_entailment_guard" not in judge
assert "If your rationale would need words such as could" not in judge
assert "speculative_entailment_guard" not in judge

# Preserve the successful localized Q03 repair from r1.
q03 = next(q for q in spec["queries"] if q["query_id"] == "Q03")
f1 = next(f for f in q03["facets"] if f["facet_id"] == "F1")
boundaries = " ".join(f1["required_components"][0]["negative_boundaries"])
assert "Potential for happiness" in boundaries
assert "can be interpreted as contributing to growth" in boundaries

print("v0.28 r2 localized-entailment rollback contract passed")
