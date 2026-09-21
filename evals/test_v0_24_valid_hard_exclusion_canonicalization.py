"""Regression: valid triggered hard exclusions canonically force negative audit fields."""
from __future__ import annotations

import json
from pathlib import Path

from semantic_relevance_facet_judge import (
    ComponentEvidenceCheck,
    CompositeEvidenceVerification,
    EvidenceVerification,
    JudgeOutputValidationError,
    verify_candidate_evidence,
    verify_composite_evidence,
)
from semantic_relevance_facet_scoring import (
    EvidenceInferenceKind,
    QueryFacet,
    VerificationRelation,
)

E = Path(__file__).resolve().parent
CONFIG = json.loads(
    (E / "judge_configs/semantic_relevance_judge.v0.26.0.json").read_text(
        encoding="utf-8"
    )
)

facet = QueryFacet(
    facet_id="F1",
    text="test facet",
    facet_type="core",
    semantic_definition="A test semantic definition.",
    hard_exclusions=[
        {
            "exclusion_id": "HX_TEST",
            "rule": "Evidence matching this boundary must be unsupported.",
        }
    ],
)


class SingleContradictoryModel:
    def __init__(self) -> None:
        self.calls = 0

    def generate(self, prompt, schema, **kwargs):
        self.calls += 1
        assert "verification_relation MUST be unsupported" in prompt
        assert "inference_kind MUST be none" in prompt
        return EvidenceVerification(
            verification_relation=VerificationRelation.ENTAILED,
            inference_kind=EvidenceInferenceKind.NECESSARY_SEMANTIC_INFERENCE,
            component_checks=[ComponentEvidenceCheck(component_id="test component", established=True, supporting_span_ids=["S1"], reason="grounded")],
            all_required_components_established=True,
            missing_semantic_component=None,
            semantic_definition_exclusion_applied=False,
            hard_exclusion_triggered=True,
            hard_exclusion_id="HX_TEST",
            reason="The configured hard exclusion applies, despite inconsistent auxiliary fields.",
        )


single_model = SingleContradictoryModel()
single, retries = verify_candidate_evidence(
    judge_model=single_model,
    facet=facet,
    evidence_span_id="S1",
    evidence_text="Boundary-matching evidence.",
    config=CONFIG,
)
assert single_model.calls == 1
assert retries == 0
assert single.hard_exclusion_triggered is True
assert single.hard_exclusion_id == "HX_TEST"
assert single.verification_relation == VerificationRelation.UNSUPPORTED
assert single.inference_kind == EvidenceInferenceKind.NONE


class CompositeContradictoryModel:
    def __init__(self) -> None:
        self.calls = 0

    def generate(self, prompt, schema, **kwargs):
        self.calls += 1
        return CompositeEvidenceVerification(
            supporting_span_ids=["S1", "S2"],
            verification_relation=VerificationRelation.ENTAILED,
            inference_kind=EvidenceInferenceKind.NECESSARY_SEMANTIC_INFERENCE,
            component_checks=[ComponentEvidenceCheck(component_id="test component", established=True, supporting_span_ids=["S1", "S2"], reason="grounded")],
            all_required_components_established=True,
            semantic_definition_exclusion_applied=False,
            hard_exclusion_triggered=True,
            hard_exclusion_id="HX_TEST",
            combined_evidence_summary="The two spans match the configured hard exclusion.",
            missing_semantic_component=None,
            reason="The hard exclusion applies, despite inconsistent positive relation fields.",
        )


composite_model = CompositeContradictoryModel()
composite, retries = verify_composite_evidence(
    judge_model=composite_model,
    facet=facet,
    spans={"S1": "First boundary span.", "S2": "Second boundary span."},
    config=CONFIG,
)
assert composite_model.calls == 1
assert retries == 0
assert composite.hard_exclusion_triggered is True
assert composite.hard_exclusion_id == "HX_TEST"
assert composite.verification_relation == VerificationRelation.UNSUPPORTED
assert composite.inference_kind == EvidenceInferenceKind.NONE
assert composite.supporting_span_ids == []


class InvalidExclusionModel:
    def __init__(self) -> None:
        self.calls = 0

    def generate(self, prompt, schema, **kwargs):
        self.calls += 1
        return EvidenceVerification(
            verification_relation=VerificationRelation.UNSUPPORTED,
            inference_kind=EvidenceInferenceKind.NONE,
            component_checks=[ComponentEvidenceCheck(component_id="test component", established=False, supporting_span_ids=[], reason="missing")],
            all_required_components_established=False,
            missing_semantic_component=None,
            semantic_definition_exclusion_applied=False,
            hard_exclusion_triggered=True,
            hard_exclusion_id="HX_INVENTED",
            reason="Invented exclusion IDs must still fail validation.",
        )


invalid_model = InvalidExclusionModel()
try:
    verify_candidate_evidence(
        judge_model=invalid_model,
        facet=facet,
        evidence_span_id="S1",
        evidence_text="Boundary-matching evidence.",
        config=CONFIG,
    )
except JudgeOutputValidationError as error:
    assert "must name one of" in str(error)
else:
    raise AssertionError("Invented hard exclusion ID should remain invalid.")
assert invalid_model.calls == 2

print("v0.24.0 valid hard-exclusion canonicalization passed")
