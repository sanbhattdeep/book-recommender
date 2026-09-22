"""v0.26 full-context recovery asks only about components still missing."""
from __future__ import annotations

import json
from pathlib import Path

from semantic_relevance_facet_judge import (
    ComponentEvidenceCheck,
    EvidenceVerification,
    FullContextComponentRecovery,
    verify_composite_evidence,
)
from semantic_relevance_facet_scoring import (
    EvidenceInferenceKind,
    QueryFacet,
    QueryFacetSpec,
    VerificationRelation,
)

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
        self.prompts: list[str] = []

    def generate(self, prompt, schema):
        self.prompts.append(prompt)
        assert schema is FullContextComponentRecovery
        return self.outputs.pop(0)


# A Time to Embrace style case: keep loss context from S8; recover only rebuilding from new S11.
f = facet("Q06", "F3")
spans = {
    "S8": "Suddenly devastation tears at the heart of their family, and the depth of their existence.",
    "S11": "Together, they're determined to move on with their lives.",
}
prior = {
    "S8": EvidenceVerification(
        verification_relation=VerificationRelation.ADJACENT,
        inference_kind=EvidenceInferenceKind.INCOMPLETE_CONNECTION,
        component_checks=[
            ComponentEvidenceCheck(
                component_id="prior_grief_or_major_loss",
                established=True,
                grounding_relation="entailed",
                supporting_span_ids=["S8"],
                reason="Major-loss context established.",
            ),
            ComponentEvidenceCheck(
                component_id="rebuilding_or_resuming_own_life_after_loss",
                established=False,
                grounding_relation="missing",
                supporting_span_ids=[],
                reason="No rebuilding in S8.",
            ),
        ],
        all_required_components_established=False,
        missing_semantic_component="rebuilding_or_resuming_own_life_after_loss",
        reason="Partial.",
    )
}
model = QueueModel([
    FullContextComponentRecovery(
        component_id="rebuilding_or_resuming_own_life_after_loss",
        grounding_relation="explicit",
        supporting_span_ids=["S11"],
        reason="Move on with their lives explicitly grounds continuation/rebuilding.",
    )
])
result, retries = verify_composite_evidence(model, f, spans, CONFIG, prior)
assert retries == 0
assert result.verification_relation == VerificationRelation.ENTAILED
assert result.supporting_span_ids == ["S8", "S11"]
assert len(model.prompts) == 1
assert "rebuilding_or_resuming_own_life_after_loss" in model.prompts[0]
assert "prior_grief_or_major_loss" not in model.prompts[0]

# Same-span resurrection is rejected and, after bounded repair attempts, remains missing rather than crashing.
f = facet("Q07", "F1")
spans = {"S1": "The story of a forty-year marriage follows turbulence and emotional fallout."}
prior = {
    "S1": EvidenceVerification(
        verification_relation=VerificationRelation.UNSUPPORTED,
        inference_kind=EvidenceInferenceKind.NONE,
        component_checks=[
            ComponentEvidenceCheck(
                component_id="parent_child_relationship",
                established=False,
                grounding_relation="missing",
                supporting_span_ids=[],
                negative_boundary_applied=True,
                reason="Marriage is not parent-child.",
            ),
            ComponentEvidenceCheck(
                component_id="difficult_or_complex_dynamic",
                established=True,
                grounding_relation="explicit",
                supporting_span_ids=["S1"],
                reason="Turbulence is complex.",
            ),
        ],
        all_required_components_established=False,
        semantic_definition_exclusion_applied=True,
        reason="Blocked relationship type.",
    )
}
invalid = FullContextComponentRecovery(
    component_id="parent_child_relationship",
    grounding_relation="explicit",
    supporting_span_ids=["S1"],
    reason="Invalid attempt to flip same span.",
)
model = QueueModel([invalid, invalid])
result, retries = verify_composite_evidence(model, f, spans, CONFIG, prior)
assert retries == 1
assert result.verification_relation == VerificationRelation.UNSUPPORTED
assert next(c for c in result.component_checks if c.component_id == "parent_child_relationship").established is False

print("v0.26.0 missing-component-only recovery contract passed")
