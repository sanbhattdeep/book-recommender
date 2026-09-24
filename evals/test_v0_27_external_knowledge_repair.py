"""v0.27 r5 repairs/fails closed on external-knowledge audit contradictions."""
from __future__ import annotations

import json
from pathlib import Path

from semantic_relevance_facet_judge import (
    IsolatedComponentVerification,
    verify_isolated_component,
)
from semantic_relevance_facet_scoring import QueryFacet, QueryFacetSpec

E = Path(__file__).resolve().parent
FACETS = json.loads(
    (E / "facets/semantic_relevance/semantic_relevance_query_facets.v0.9.5.json")
    .read_text(encoding="utf-8")
)


def facet(query_id: str, facet_id: str) -> QueryFacet:
    raw = next(q for q in FACETS["queries"] if q["query_id"] == query_id)
    spec = QueryFacetSpec(
        query_id=raw["query_id"],
        query=raw["query"],
        facets=[QueryFacet(**f) for f in raw["facets"]],
    )
    return next(f for f in spec.facets if f.facet_id == facet_id)


f = facet("Q06", "F1")
component = f.required_components[0]


class RepairingModel:
    def __init__(self) -> None:
        self.calls = 0
        self.prompts: list[str] = []

    def generate(self, prompt, schema):
        self.calls += 1
        self.prompts.append(prompt)
        assert schema is IsolatedComponentVerification
        if self.calls == 1:
            return IsolatedComponentVerification(
                component_id=component.component_id,
                grounding_relation="entailed",
                negative_boundary_applied=False,
                external_knowledge_required=True,
                reason="Invalid contradiction: the positive claim relies on outside knowledge.",
            )
        return IsolatedComponentVerification(
            component_id=component.component_id,
            grounding_relation="missing",
            negative_boundary_applied=False,
            external_knowledge_required=True,
            reason="Repaired: supplied text alone does not establish the component.",
        )


model = RepairingModel()
result, retries = verify_isolated_component(
    judge_model=model,
    facet=f,
    component=component,
    evidence_span_id="S1",
    evidence_text="A deliberately underspecified span.",
)
assert model.calls == 2
assert retries == 1
assert result.grounding_relation == "missing"
assert result.external_knowledge_required is True
assert "PREVIOUS VALIDATION FAILURE" in model.prompts[1]
assert "external_knowledge_required=true" in model.prompts[1]
assert "grounding_relation='missing'" in model.prompts[1]


class StubbornModel:
    def __init__(self) -> None:
        self.calls = 0

    def generate(self, prompt, schema):
        self.calls += 1
        assert schema is IsolatedComponentVerification
        return IsolatedComponentVerification(
            component_id=component.component_id,
            grounding_relation="entailed",
            negative_boundary_applied=False,
            external_knowledge_required=True,
            reason="Still contradictory after repair instruction.",
        )


model = StubbornModel()
result, retries = verify_isolated_component(
    judge_model=model,
    facet=f,
    component=component,
    evidence_span_id="S1",
    evidence_text="Another deliberately underspecified span.",
)
assert model.calls == 2
assert retries == 1
assert result.grounding_relation == "missing"
assert result.external_knowledge_required is True
assert result.reason.startswith(
    "isolated_component_external_knowledge_conflict_rejected_after_bounded_repairs:"
)

print("v0.27 r5 external-knowledge contradiction repair contract passed")
