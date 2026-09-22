"""v0.26.0 r4: explicit participant growth does not require autobiographical author growth."""
from __future__ import annotations

import json
from pathlib import Path

from semantic_relevance_facet_judge import IsolatedComponentVerification, verify_candidate_evidence
from semantic_relevance_facet_scoring import QueryFacet, QueryFacetSpec, VerificationRelation

E = Path(__file__).resolve().parent
CONFIG = json.loads((E / "judge_configs/semantic_relevance_judge.v0.26.0.json").read_text())
FACETS = json.loads((E / "facets/semantic_relevance/semantic_relevance_query_facets.v0.9.2.json").read_text())


def facet(query_id: str, facet_id: str) -> QueryFacet:
    raw = next(q for q in FACETS["queries"] if q["query_id"] == query_id)
    spec = QueryFacetSpec(query_id=raw["query_id"], query=raw["query"], facets=[QueryFacet(**f) for f in raw["facets"]])
    return next(f for f in spec.facets if f.facet_id == facet_id)


class QueueModel:
    def __init__(self, outputs):
        self.outputs = list(outputs)
    def generate(self, prompt, schema):
        return self.outputs.pop(0)


item = facet("Q03", "F1")
component = item.required_components[0]
assert component.component_id == "actual_self_development_or_self_understanding"
assert "need not be the author" in item.semantic_definition.lower()
assert "intended participants/readers" in component.definition.lower()

# Positive fixture: the book explicitly says the path promotes maturity and growth
# in young people, even though the authors are not the people developing.
model = QueueModel([
    IsolatedComponentVerification(
        component_id=component.component_id,
        grounding_relation="explicit",
        negative_boundary_applied=False,
        reason="The evidence explicitly says the dating path promotes maturity and growth in young people.",
    )
])
result, retries = verify_candidate_evidence(
    model,
    item,
    "S1",
    "The authors show young people how dating can be a spiritual path that promotes maturity and growth.",
    CONFIG,
)
assert retries == 0
assert result.verification_relation == VerificationRelation.DIRECT

# Negative control: spirituality alone still does not establish personal growth.
model = QueueModel([
    IsolatedComponentVerification(
        component_id=component.component_id,
        grounding_relation="missing",
        negative_boundary_applied=True,
        reason="The evidence describes spirituality but states no development, maturity, transformation, or self-understanding.",
    )
])
result, _ = verify_candidate_evidence(
    model,
    item,
    "S1",
    "The book presents dating as a spiritual path and explores faith.",
    CONFIG,
)
assert result.verification_relation == VerificationRelation.UNSUPPORTED
print("v0.26.0 r4 personal-growth participant-scope contract passed")
