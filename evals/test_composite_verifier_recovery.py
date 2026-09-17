"""Regression test for v0.19 full-context composite verifier recovery."""

from __future__ import annotations

import json
from pathlib import Path

from semantic_relevance_facet_judge import (
    CompositeEvidenceVerification,
    EvidenceSelection,
    EvidenceVerification,
    evaluate_one_facet,
)
from semantic_relevance_facet_scoring import (
    EvidenceInferenceKind,
    FacetProminence,
    QueryFacet,
    VerificationRelation,
)

EVALS_DIR = Path(__file__).resolve().parent
CONFIG_FILE = EVALS_DIR / "judge_configs" / "semantic_relevance_judge.v0.21.1.json"
config = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))

facet = QueryFacet(
    facet_id="F1",
    text="synthetic recovery facet",
    facet_type="core",
    semantic_definition="A synthetic facet used only for structured-output testing.",
)

spans = {
    "S1": "One weak statement.",
    "S2": "Another related statement.",
    "S3": "A later response.",
}


class RepeatedMalformedCompositeModel:
    def __init__(self) -> None:
        self.composite_calls = 0

    def generate(self, prompt, schema, **kwargs):
        if schema is EvidenceSelection:
            return EvidenceSelection(
                evidence_span_ids=["S1"],
                reason="synthetic standalone candidate",
            )

        if schema is EvidenceVerification:
            return EvidenceVerification(
                verification_relation=VerificationRelation.UNSUPPORTED,
                inference_kind=EvidenceInferenceKind.NONE,
                reason="synthetic unsupported",
            )

        if schema is CompositeEvidenceVerification:
            self.composite_calls += 1
            # Invalid: positive relation with only one cited span and an
            # ENTAILED result that declares a missing component.
            return CompositeEvidenceVerification(
                supporting_span_ids=["S2"],
                verification_relation=VerificationRelation.ENTAILED,
                inference_kind=EvidenceInferenceKind.NECESSARY_SEMANTIC_INFERENCE,
                combined_evidence_summary="synthetic malformed positive",
                missing_semantic_component="still missing something",
                reason="malformed",
            )

        raise AssertionError(schema)


model = RepeatedMalformedCompositeModel()
assessment, evidence, candidate_texts, retries = evaluate_one_facet(
    judge_model=model,
    query_id="SYN",
    facet=facet,
    spans=spans,
    config=config,
)

assert model.composite_calls == 2
assert assessment.verification_relation == VerificationRelation.UNSUPPORTED
assert assessment.prominence == FacetProminence.NOT_APPLICABLE
assert assessment.composite_verification_attempted is True
assert assessment.composite_evidence_span_ids == []
assert assessment.composite_verification_relation == VerificationRelation.UNSUPPORTED
assert assessment.composite_missing_semantic_component is None
assert assessment.composite_verification_reason.startswith(
    "structured_output_recovery_fallback:"
)
assert evidence == spans["S1"]
assert candidate_texts == {"S1": spans["S1"]}

print("v0.21.1 malformed full-context composite fallback test passed.")
