"""Deterministic pipeline tests for v0.18.0 self-selecting full-context composition."""

from __future__ import annotations

import json
from pathlib import Path

from semantic_relevance_facet_judge import (
    CompositeEvidenceVerification,
    EvidenceSelection,
    EvidenceVerification,
    ProminenceAssessment,
    build_composite_verification_prompt,
    evaluate_one_facet,
)
from semantic_relevance_facet_scoring import (
    FacetProminence,
    FacetSupport,
    QueryFacet,
    VerificationRelation,
    derive_facet_support,
)

EVALS_DIR = Path(__file__).resolve().parent
CONFIG_FILE = EVALS_DIR / "judge_configs" / "semantic_relevance_judge.v0.18.0.json"
config = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))

facet = QueryFacet(
    facet_id="F3",
    text="learning to live again",
    facet_type="core",
    semantic_definition=(
        "Rebuilding, resuming, or finding a way to continue life after grief "
        "or a major loss."
    ),
)

spans = {
    "S1": "A devastating tragedy shatters the family.",
    "S2": "The immediate aftermath leaves everyone stunned.",
    "S3": "They ask friends for advice.",
    "S4": "Months later they are determined to move forward with their lives.",
}


# ----------------------------------------------------------------------
# Full-context verifier can recover a useful span omitted by Stage A.
# ----------------------------------------------------------------------
class FullContextRecoveryModel:
    def __init__(self) -> None:
        self.composite_calls = 0
        self.prominence_calls = 0

    def generate(self, prompt, schema, **kwargs):
        if schema is EvidenceSelection:
            # Stage A deliberately misses S4.
            return EvidenceSelection(
                evidence_span_ids=["S1", "S2", "S3"],
                reason="Best standalone candidates.",
            )

        if schema is EvidenceVerification:
            return EvidenceVerification(
                verification_relation=VerificationRelation.UNSUPPORTED,
                reason="No single Stage-A span establishes recovery.",
            )

        if schema is CompositeEvidenceVerification:
            self.composite_calls += 1
            # The full-context verifier sees S4 directly and cites it itself.
            assert "S4: Months later they are determined to move forward with their lives." in prompt
            return CompositeEvidenceVerification(
                supporting_span_ids=["S1", "S4"],
                verification_relation=VerificationRelation.ENTAILED,
                combined_evidence_summary=(
                    "S1 establishes the major disruption and S4 establishes "
                    "later continuation of life after it."
                ),
                missing_semantic_component=None,
                reason=(
                    "Together the two spans supply the disruption and the later "
                    "continuation/rebuilding component required by the facet."
                ),
            )

        if schema is ProminenceAssessment:
            self.prominence_calls += 1
            return ProminenceAssessment(
                prominence=FacetProminence.SUBSTANTIVE,
                reason="The recovery arc is a meaningful part of the description.",
            )

        raise AssertionError(schema)


model = FullContextRecoveryModel()
assessment, evidence, candidate_texts, _ = evaluate_one_facet(
    judge_model=model,
    query_id="Q06",
    facet=facet,
    spans=spans,
    config=config,
)

assert assessment.candidate_evidence_span_ids == ["S1", "S2", "S3"]
assert "S4" not in assessment.candidate_evidence_span_ids
assert assessment.composite_verification_attempted is True
assert assessment.composite_evidence_span_ids == ["S1", "S4"]
assert assessment.composite_verification_relation == VerificationRelation.ENTAILED
assert assessment.composite_missing_semantic_component is None
assert assessment.verification_relation == VerificationRelation.ENTAILED
assert assessment.prominence == FacetProminence.SUBSTANTIVE
assert "S1:" in evidence and "S4:" in evidence
assert "S4" in candidate_texts
assert model.composite_calls == 1
assert model.prominence_calls == 1
assert derive_facet_support(facet, assessment) == FacetSupport.MEANINGFUL


# ----------------------------------------------------------------------
# Full-context composition can recover even when Stage A returns NONE.
# ----------------------------------------------------------------------
class StageANoneRecoveryModel:
    def generate(self, prompt, schema, **kwargs):
        if schema is EvidenceSelection:
            return EvidenceSelection(
                evidence_span_ids=["NONE"],
                reason="No standalone evidence.",
            )

        if schema is CompositeEvidenceVerification:
            return CompositeEvidenceVerification(
                supporting_span_ids=["S1", "S4"],
                verification_relation=VerificationRelation.ENTAILED,
                combined_evidence_summary="Disruption followed by explicit continuation.",
                missing_semantic_component=None,
                reason="The combined spans entail the recovery facet.",
            )

        if schema is ProminenceAssessment:
            return ProminenceAssessment(
                prominence=FacetProminence.SUBSTANTIVE,
                reason="Recovery is substantive.",
            )

        raise AssertionError(schema)


assessment, evidence, candidate_texts, _ = evaluate_one_facet(
    judge_model=StageANoneRecoveryModel(),
    query_id="Q06",
    facet=facet,
    spans=spans,
    config=config,
)

assert assessment.candidate_evidence_span_ids == []
assert assessment.candidate_evidence_span_id == "NONE"
assert assessment.verification_relation == VerificationRelation.ENTAILED
assert assessment.composite_evidence_span_ids == ["S1", "S4"]
assert "S4" in candidate_texts
assert derive_facet_support(facet, assessment) == FacetSupport.MEANINGFUL


# ----------------------------------------------------------------------
# DIRECT standalone support must skip composite recovery.
# ----------------------------------------------------------------------
class DirectSingleSpanModel:
    def __init__(self) -> None:
        self.composite_calls = 0

    def generate(self, prompt, schema, **kwargs):
        if schema is EvidenceSelection:
            return EvidenceSelection(
                evidence_span_ids=["S4", "S1"],
                reason="synthetic",
            )
        if schema is EvidenceVerification:
            return EvidenceVerification(
                verification_relation=VerificationRelation.DIRECT,
                reason="synthetic direct",
            )
        if schema is CompositeEvidenceVerification:
            self.composite_calls += 1
            raise AssertionError("composite verifier should not run after DIRECT")
        if schema is ProminenceAssessment:
            return ProminenceAssessment(
                prominence=FacetProminence.CENTRAL,
                reason="synthetic central",
            )
        raise AssertionError(schema)


model = DirectSingleSpanModel()
assessment, *_ = evaluate_one_facet(
    judge_model=model,
    query_id="Q06",
    facet=facet,
    spans=spans,
    config=config,
)
assert assessment.verification_relation == VerificationRelation.DIRECT
assert assessment.composite_verification_attempted is False
assert assessment.composite_evidence_span_ids == []
assert model.composite_calls == 0


# ----------------------------------------------------------------------
# Prompt contract: full description + explicit entailment/adjacency rules.
# ----------------------------------------------------------------------
prompt = build_composite_verification_prompt(
    facet=facet,
    spans=spans,
    config=config,
)
assert "FULL NUMBERED BOOK-DESCRIPTION SPANS" in prompt
assert "S4: Months later they are determined to move forward with their lives." in prompt
assert "Do not require exact facet wording for ENTAILED." in prompt
assert "missing_semantic_component" in prompt

print("All v0.18.0 self-selecting full-context composition tests passed.")
