"""Mechanical and prompt-contract tests for v0.23.0 component-complete verification."""
from __future__ import annotations

import json
from pathlib import Path

from semantic_relevance_facet_judge import (
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
CONFIG = json.loads((E / "judge_configs/semantic_relevance_judge.v0.23.0.json").read_text(encoding="utf-8"))


def facet(text: str, definition: str) -> QueryFacet:
    return QueryFacet(facet_id="F1", text=text, facet_type="core", semantic_definition=definition)


def must_reject(result: EvidenceVerification, f: QueryFacet) -> None:
    try:
        _validate_verification_result(result=result, facet=f, config=CONFIG)
    except JudgeOutputValidationError:
        return
    raise AssertionError(f"Expected verifier contract rejection: {result}")

# Missing relationship type: a complex marriage may not be upgraded to parent-child.
parent_child = facet(
    "complicated parent-child relationship",
    "A parent-child relationship with meaningful conflict or emotional complexity.",
)
must_reject(
    EvidenceVerification(
        verification_relation=VerificationRelation.ENTAILED,
        inference_kind=EvidenceInferenceKind.NECESSARY_SEMANTIC_INFERENCE,
        all_required_components_established=False,
        missing_semantic_component="parent-child relationship",
        semantic_definition_exclusion_applied=False,
        reason="The evidence only describes a turbulent marriage.",
    ),
    parent_child,
)

# Missing movement component: danger alone may not become a dangerous journey.
dangerous_journey = facet(
    "dangerous journeys",
    "Dangerous travel, a quest, voyage, expedition, or movement through places; general danger without travel is insufficient.",
)
must_reject(
    EvidenceVerification(
        verification_relation=VerificationRelation.ENTAILED,
        inference_kind=EvidenceInferenceKind.NECESSARY_SEMANTIC_INFERENCE,
        all_required_components_established=False,
        missing_semantic_component="travel or movement",
        semantic_definition_exclusion_applied=False,
        reason="The character is in danger but no journey is established.",
    ),
    dangerous_journey,
)

# Explicit semantic-definition exclusion: generic recovery cannot be promoted to redemption.
redemption = facet(
    "redemption",
    "Restoration after wrongdoing/failure/damage. Generic recovery alone does not establish redemption.",
)
must_reject(
    EvidenceVerification(
        verification_relation=VerificationRelation.DIRECT,
        inference_kind=EvidenceInferenceKind.EXPLICIT_COMPONENTS,
        all_required_components_established=True,
        missing_semantic_component=None,
        semantic_definition_exclusion_applied=True,
        reason="The evidence states recovery only.",
    ),
    redemption,
)

# Legitimate complete positive remains valid.
complete = EvidenceVerification(
    verification_relation=VerificationRelation.DIRECT,
    inference_kind=EvidenceInferenceKind.EXPLICIT_COMPONENTS,
    all_required_components_established=True,
    missing_semantic_component=None,
    semantic_definition_exclusion_applied=False,
    reason="The evidence explicitly states the complete synthetic facet.",
)
_validate_verification_result(result=complete, facet=facet("synthetic", "synthetic"), config=CONFIG)

# ADJACENT remains available only when the missing component is explicit.
adjacent = EvidenceVerification(
    verification_relation=VerificationRelation.ADJACENT,
    inference_kind=EvidenceInferenceKind.INCOMPLETE_CONNECTION,
    all_required_components_established=False,
    missing_semantic_component="required outcome",
    semantic_definition_exclusion_applied=False,
    reason="There is a genuine incomplete connection.",
)
_validate_verification_result(result=adjacent, facet=facet("synthetic", "synthetic"), config=CONFIG)

# Composite ENTAILED is also mechanically barred when any component is missing.
comp = CompositeEvidenceVerification(
    supporting_span_ids=["S1", "S2"],
    verification_relation=VerificationRelation.ENTAILED,
    inference_kind=EvidenceInferenceKind.NECESSARY_SEMANTIC_INFERENCE,
    all_required_components_established=False,
    semantic_definition_exclusion_applied=False,
    combined_evidence_summary="Danger plus a location, but no movement.",
    missing_semantic_component="travel or movement",
    reason="Incomplete dangerous-journey evidence.",
)
try:
    _validate_composite_result(comp, dangerous_journey, {"S1":"danger", "S2":"a location"}, CONFIG)
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
    "hardship",
]:
    assert phrase in instructions, phrase

print("v0.23.0 component-completeness contracts passed")
