"""v0.26.0 r6: parental opposition can establish a separate parent-child relation."""
from __future__ import annotations

import json
from pathlib import Path

from semantic_relevance_facet_judge import IsolatedComponentVerification, verify_candidate_evidence
from semantic_relevance_facet_scoring import QueryFacet, QueryFacetSpec, VerificationRelation

E = Path(__file__).resolve().parent
CONFIG = json.loads((E / "judge_configs/semantic_relevance_judge.v0.26.0.json").read_text(encoding="utf-8"))
FACETS = json.loads((E / "facets/semantic_relevance/semantic_relevance_query_facets.v0.9.3.json").read_text(encoding="utf-8"))


def facet(query_id: str, facet_id: str) -> QueryFacet:
    raw = next(q for q in FACETS["queries"] if q["query_id"] == query_id)
    spec = QueryFacetSpec(query_id=raw["query_id"], query=raw["query"], facets=[QueryFacet(**f) for f in raw["facets"]])
    return next(f for f in spec.facets if f.facet_id == facet_id)


class QueueModel:
    def __init__(self, outputs):
        self.outputs = list(outputs)
    def generate(self, prompt, schema):
        return self.outputs.pop(0)


item = facet("Q07", "F1")
assert [c.component_id for c in item.required_components] == [
    "parent_child_relationship",
    "difficult_or_complex_dynamic",
]
text = " ".join(item.required_components[0].negative_boundaries).lower()
assert "same person may separately be a spouse/partner and a child" in text
assert "parental" in item.required_components[0].definition.lower()

# Positive fixture: parental opposition targets the young couple; this establishes
# a parent-child relation involving one/both members without conflating it with marriage.
model = QueueModel([
    IsolatedComponentVerification(
        component_id="parent_child_relationship",
        grounding_relation="entailed",
        negative_boundary_applied=False,
        reason="Explicit bitter parental opposition toward the young couple necessarily involves parents opposing their child or children, separate from the couple's marriage.",
    ),
    IsolatedComponentVerification(
        component_id="difficult_or_complex_dynamic",
        grounding_relation="entailed",
        negative_boundary_applied=False,
        reason="Bitter parental opposition establishes meaningful conflict in that parent-child dynamic.",
    ),
])
result, retries = verify_candidate_evidence(
    model,
    item,
    "S1",
    "A young couple marry and face bitter parental opposition.",
    CONFIG,
)
assert retries == 0
assert result.verification_relation == VerificationRelation.ENTAILED
assert all(c.established for c in result.component_checks)

# Negative control: a turbulent marriage alone still cannot substitute for parent-child.
model = QueueModel([
    IsolatedComponentVerification(
        component_id="parent_child_relationship",
        grounding_relation="missing",
        negative_boundary_applied=True,
        reason="Only a marriage is described; no parental action or parent-child relation is present.",
    ),
    IsolatedComponentVerification(
        component_id="difficult_or_complex_dynamic",
        grounding_relation="explicit",
        negative_boundary_applied=False,
        reason="The marriage is explicitly turbulent.",
    ),
])
result, _ = verify_candidate_evidence(
    model,
    item,
    "S1",
    "The forty-year marriage is turbulent and emotionally complex.",
    CONFIG,
)
assert result.verification_relation == VerificationRelation.UNSUPPORTED
print("v0.26.0 r6 parent-child referent-binding contract passed")
