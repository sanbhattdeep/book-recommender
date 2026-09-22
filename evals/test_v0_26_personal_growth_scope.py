"""v0.26.0 r5: personal-growth promotion mode is independently sufficient."""
from __future__ import annotations

import json
from pathlib import Path

from semantic_relevance_facet_judge import (
    IsolatedComponentVerification,
    build_isolated_component_prompt,
    verify_candidate_evidence,
)
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
        self.prompts = []
    def generate(self, prompt, schema):
        self.prompts.append(prompt)
        return self.outputs.pop(0)


item = facet("Q03", "F1")
component = item.required_components[0]
assert component.component_id == "actual_self_development_or_self_understanding"
assert "two independently sufficient" in item.semantic_definition.lower()
assert "mode a" in component.definition.lower()
assert "mode b" in component.definition.lower()
assert "promotes maturity and growth" in component.definition.lower()
assert "do not require proof that a named individual has already completed" in component.definition.lower()

# The real model prompt must surface the disjunctive contract and explicitly forbid
# silently replacing Mode B with an already-completed-outcome requirement.
prompt = build_isolated_component_prompt(
    facet=item,
    component=component,
    evidence_span_id="S1",
    evidence_text="The authors show young people how dating can be a spiritual path that promotes maturity and growth.",
)
pl = prompt.lower()
assert "two independently sufficient positive modes" in pl
assert "either mode establishes this component" in pl
assert "promotes maturity and growth" in pl
assert "do not silently replace it with a stricter criterion" in pl

# Positive Mode B: explicit promotion of maturity/growth in intended participants.
for evidence in [
    "The authors show young people how dating can be a spiritual path that promotes maturity and growth.",
    "The program helps adolescents develop confidence and emotional maturity.",
]:
    model = QueueModel([
        IsolatedComponentVerification(
            component_id=component.component_id,
            grounding_relation="explicit",
            negative_boundary_applied=False,
            reason="The evidence explicitly states personal development/maturity as the intended effect for participants.",
        )
    ])
    result, retries = verify_candidate_evidence(model, item, "S1", evidence, CONFIG)
    assert retries == 0
    assert result.verification_relation == VerificationRelation.DIRECT

# Negative controls: spirituality/peace without explicit development remain insufficient.
for evidence in [
    "The book explores prayer, faith, wisdom, and spiritual devotion.",
    "Meditation may bring peace and a deeper connection to God.",
]:
    model = QueueModel([
        IsolatedComponentVerification(
            component_id=component.component_id,
            grounding_relation="missing",
            negative_boundary_applied=True,
            reason="No personal growth, maturity, development, transformation, or self-understanding is established.",
        )
    ])
    result, _ = verify_candidate_evidence(model, item, "S1", evidence, CONFIG)
    assert result.verification_relation == VerificationRelation.UNSUPPORTED

print("v0.26.0 r5 personal-growth promotion-mode contract passed")
