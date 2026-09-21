"""Mechanical/prompt contracts inherited from v0.23 and strengthened in v0.24."""
from __future__ import annotations

import json
from pathlib import Path

from semantic_relevance_facet_judge import (
    ComponentEvidenceCheck,
    EvidenceVerification,
    CompositeEvidenceVerification,
    JudgeOutputValidationError,
    _validate_verification_result,
    _validate_composite_result,
)
from semantic_relevance_facet_scoring import (
    EvidenceInferenceKind,
    QueryFacet,
    VerificationRelation,
)

E = Path(__file__).resolve().parent
CONFIG = json.loads((E / "judge_configs/semantic_relevance_judge.v0.25.0.json").read_text(encoding="utf-8"))


def facet(text: str, definition: str) -> QueryFacet:
    return QueryFacet(facet_id="F1", text=text, facet_type="core", semantic_definition=definition)


def check(component: str, established: bool, span_id: str = "S1") -> ComponentEvidenceCheck:
    return ComponentEvidenceCheck(
        component_id=component,
        established=established,
        supporting_span_ids=[span_id] if established else [],
        reason="synthetic component audit",
    )


def must_reject(result: EvidenceVerification, f: QueryFacet) -> None:
    try:
        _validate_verification_result(result=result, facet=f, config=CONFIG, evidence_span_id="S1")
    except JudgeOutputValidationError:
        return
    raise AssertionError(f"Expected verifier contract rejection: {result}")


parent_child = facet(
    "complicated parent-child relationship",
    "A parent-child relationship with meaningful conflict or emotional complexity.",
)
must_reject(
    EvidenceVerification(
        verification_relation=VerificationRelation.ENTAILED,
        inference_kind=EvidenceInferenceKind.NECESSARY_SEMANTIC_INFERENCE,
        component_checks=[
            check("complicated relationship", True),
            check("parent-child relationship", False),
        ],
        all_required_components_established=False,
        missing_semantic_component="parent-child relationship",
        semantic_definition_exclusion_applied=False,
        reason="The evidence only describes a turbulent marriage.",
    ),
    parent_child,
)

dangerous_journey = facet(
    "dangerous journeys",
    "Dangerous travel; general danger without travel is insufficient.",
)
must_reject(
    EvidenceVerification(
        verification_relation=VerificationRelation.ENTAILED,
        inference_kind=EvidenceInferenceKind.NECESSARY_SEMANTIC_INFERENCE,
        component_checks=[check("danger", True), check("travel or movement", False)],
        all_required_components_established=False,
        missing_semantic_component="travel or movement",
        semantic_definition_exclusion_applied=False,
        reason="Danger is present but movement is absent.",
    ),
    dangerous_journey,
)

redemption = facet(
    "redemption",
    "Restoration after wrongdoing/failure/damage. Generic recovery alone does not establish redemption.",
)
must_reject(
    EvidenceVerification(
        verification_relation=VerificationRelation.DIRECT,
        inference_kind=EvidenceInferenceKind.EXPLICIT_COMPONENTS,
        component_checks=[check("restoration after damage", True)],
        all_required_components_established=True,
        missing_semantic_component=None,
        semantic_definition_exclusion_applied=True,
        reason="The evidence states recovery only.",
    ),
    redemption,
)

complete = EvidenceVerification(
    verification_relation=VerificationRelation.DIRECT,
    inference_kind=EvidenceInferenceKind.EXPLICIT_COMPONENTS,
    component_checks=[check("synthetic component", True)],
    all_required_components_established=True,
    missing_semantic_component=None,
    semantic_definition_exclusion_applied=False,
    reason="The evidence explicitly states the complete synthetic facet.",
)
_validate_verification_result(
    result=complete,
    facet=facet("synthetic", "synthetic"),
    config=CONFIG,
    evidence_span_id="S1",
)

adjacent = EvidenceVerification(
    verification_relation=VerificationRelation.ADJACENT,
    inference_kind=EvidenceInferenceKind.INCOMPLETE_CONNECTION,
    component_checks=[check("required outcome", False)],
    all_required_components_established=False,
    missing_semantic_component="required outcome",
    semantic_definition_exclusion_applied=False,
    reason="There is a genuine incomplete connection.",
)
_validate_verification_result(
    result=adjacent,
    facet=facet("synthetic", "synthetic"),
    config=CONFIG,
    evidence_span_id="S1",
)

comp = CompositeEvidenceVerification(
    supporting_span_ids=["S1", "S2"],
    verification_relation=VerificationRelation.ENTAILED,
    inference_kind=EvidenceInferenceKind.NECESSARY_SEMANTIC_INFERENCE,
    component_checks=[
        ComponentEvidenceCheck(component_id="danger", established=True, supporting_span_ids=["S1"], reason="danger"),
        ComponentEvidenceCheck(component_id="travel or movement", established=False, supporting_span_ids=[], reason="missing"),
    ],
    all_required_components_established=False,
    semantic_definition_exclusion_applied=False,
    combined_evidence_summary="Danger plus a location, but no movement.",
    missing_semantic_component="travel or movement",
    reason="Incomplete dangerous-journey evidence.",
)
try:
    _validate_composite_result(comp, dangerous_journey, {"S1": "danger", "S2": "a location"}, CONFIG)
except JudgeOutputValidationError:
    pass
else:
    raise AssertionError("Composite positive with a missing required component must fail.")

instructions = "\n".join(CONFIG["verification_stage"]["instructions"]).lower()
for phrase in [
    "complicated marriage",
    "danger while stationary",
    "recovery is not redemption",
    "learning new information",
    "location is not movement",
    "lost time",
    "strongly affective presentation",
]:
    assert phrase in instructions, phrase

print("v0.24.0 component-completeness contracts passed")
