"""Composite verification may connect grounded components but not resurrect missing ones."""
from __future__ import annotations

import json
from pathlib import Path

from semantic_relevance_facet_judge import (
    ComponentEvidenceCheck,
    CompositeEvidenceVerification,
    EvidenceVerification,
    JudgeOutputValidationError,
    _validate_composite_result,
)
from semantic_relevance_facet_scoring import EvidenceInferenceKind, QueryFacet, VerificationRelation

E = Path(__file__).resolve().parent
CONFIG = json.loads((E / "judge_configs/semantic_relevance_judge.v0.26.0.json").read_text())
FACET = QueryFacet(facet_id="F1", text="redemption", facet_type="core", semantic_definition="restoration after damage; generic recovery alone is insufficient")


def single(span: str) -> EvidenceVerification:
    return EvidenceVerification(
        verification_relation=VerificationRelation.UNSUPPORTED,
        inference_kind=EvidenceInferenceKind.NONE,
        component_checks=[
            ComponentEvidenceCheck(component_id="restoration after damage", established=False, supporting_span_ids=[], reason=f"{span} says recovery only")
        ],
        all_required_components_established=False,
        missing_semantic_component="restoration after damage",
        semantic_definition_exclusion_applied=True,
        reason="generic recovery is excluded",
    )

prior = {"S1": single("S1"), "S2": single("S2")}
resurrected = CompositeEvidenceVerification(
    supporting_span_ids=["S1", "S2"],
    verification_relation=VerificationRelation.ENTAILED,
    inference_kind=EvidenceInferenceKind.NECESSARY_SEMANTIC_INFERENCE,
    component_checks=[
        ComponentEvidenceCheck(component_id="restoration after damage", established=True, supporting_span_ids=["S1", "S2"], reason="incorrectly combines recovery")
    ],
    all_required_components_established=True,
    combined_evidence_summary="Two recovery references.",
    reason="Incorrect resurrection.",
)
try:
    _validate_composite_result(
        resurrected,
        FACET,
        {"S1":"recovery", "S2":"sobriety"},
        CONFIG,
        verification_results_by_span=prior,
    )
except JudgeOutputValidationError as error:
    assert "resurrection" in str(error)
else:
    raise AssertionError("composite resurrected a universally missing component")

# A genuinely new span can supply the missing component and is not blocked by
# the monotonicity guard (semantic validity remains the model's job).
with_new_span = CompositeEvidenceVerification(
    supporting_span_ids=["S1", "S3"],
    verification_relation=VerificationRelation.ENTAILED,
    inference_kind=EvidenceInferenceKind.NECESSARY_SEMANTIC_INFERENCE,
    component_checks=[
        ComponentEvidenceCheck(component_id="restoration after damage", established=True, supporting_span_ids=["S3"], reason="new span supplies restoration")
    ],
    all_required_components_established=True,
    combined_evidence_summary="Recovery plus explicit restoration.",
    reason="New evidence supplies the missing component.",
)
_validate_composite_result(
    with_new_span,
    FACET,
    {"S1":"recovery", "S3":"explicit restoration after damage"},
    CONFIG,
    verification_results_by_span={"S1": prior["S1"]},
)

print("v0.24.0 composite monotonicity contracts passed")
