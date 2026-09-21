"""v0.25 canonical component IDs are frozen by the facet spec, not model-authored."""
from __future__ import annotations

import json
from pathlib import Path

from semantic_relevance_facet_judge import (
    ComponentEvidenceCheck,
    EvidenceVerification,
    JudgeOutputValidationError,
    _validate_verification_result,
)
from semantic_relevance_facet_scoring import (
    EvidenceInferenceKind,
    QueryFacet,
    QueryFacetSpec,
    VerificationRelation,
)

E = Path(__file__).resolve().parent
CONFIG = json.loads((E / "judge_configs/semantic_relevance_judge.v0.26.0.json").read_text())
FACETS = json.loads((E / "facets/semantic_relevance/semantic_relevance_query_facets.v0.9.0.json").read_text())

q07_raw = next(q for q in FACETS["queries"] if q["query_id"] == "Q07")
q07 = QueryFacetSpec(
    query_id=q07_raw["query_id"],
    query=q07_raw["query"],
    facets=[QueryFacet(**f) for f in q07_raw["facets"]],
)
facet = q07.facets[0]
assert [c.component_id for c in facet.required_components] == [
    "parent_child_relationship",
    "difficult_or_complex_dynamic",
]

# Regression for The Law of Enclosures failure mode: a model may not omit the
# relationship-type component and substitute easy evidence-derived labels.
omitted_relationship_type = EvidenceVerification(
    verification_relation=VerificationRelation.DIRECT,
    inference_kind=EvidenceInferenceKind.EXPLICIT_COMPONENTS,
    component_checks=[
        ComponentEvidenceCheck(
            component_id="difficult_or_complex_dynamic",
            established=True,
            supporting_span_ids=["S1"],
            reason="Turbulence and emotional complexity are present.",
        )
    ],
    all_required_components_established=True,
    reason="Incorrectly treats a complicated marriage as sufficient.",
)
try:
    _validate_verification_result(
        omitted_relationship_type,
        facet,
        CONFIG,
        evidence_span_id="S1",
    )
except JudgeOutputValidationError as error:
    assert "frozen canonical component IDs" in str(error)
else:
    raise AssertionError("Verifier accepted a ledger that omitted parent_child_relationship")

# Correct incomplete relation: complex dynamics are grounded, relationship type is not.
valid_adjacent = EvidenceVerification(
    verification_relation=VerificationRelation.ADJACENT,
    inference_kind=EvidenceInferenceKind.INCOMPLETE_CONNECTION,
    component_checks=[
        ComponentEvidenceCheck(
            component_id="parent_child_relationship",
            established=False,
            supporting_span_ids=[],
            negative_boundary_applied=True,
            reason="The evidence explicitly describes a marriage, not a parent-child relationship.",
        ),
        ComponentEvidenceCheck(
            component_id="difficult_or_complex_dynamic",
            established=True,
            supporting_span_ids=["S1"],
            reason="The marriage is turbulent and emotionally complex.",
        ),
    ],
    all_required_components_established=False,
    missing_semantic_component="parent_child_relationship",
    reason="The difficult dynamic is real, but the required relationship type is absent.",
)
_validate_verification_result(valid_adjacent, facet, CONFIG, evidence_span_id="S1")

# A component cannot be established while admitting its own negative boundary applies.
boundary_contradiction = valid_adjacent.model_copy(deep=True)
boundary_contradiction.component_checks[0].established = True
boundary_contradiction.component_checks[0].supporting_span_ids = ["S1"]
boundary_contradiction.all_required_components_established = True
boundary_contradiction.verification_relation = VerificationRelation.DIRECT
boundary_contradiction.inference_kind = EvidenceInferenceKind.EXPLICIT_COMPONENTS
boundary_contradiction.missing_semantic_component = None
try:
    _validate_verification_result(boundary_contradiction, facet, CONFIG, evidence_span_id="S1")
except JudgeOutputValidationError as error:
    assert "negative_boundary_applied=true" in str(error)
else:
    raise AssertionError("Established component ignored its own negative boundary")

print("v0.25.0 canonical component contract passed")
