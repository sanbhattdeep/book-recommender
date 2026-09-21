"""Regression test for v0.19 full-context composite verifier recovery."""

from __future__ import annotations

import json
from pathlib import Path

from semantic_relevance_facet_judge import (
    ComponentEvidenceCheck,
    CompositeEvidenceVerification,
    EvidenceSelection,
    EvidenceVerification,
    JudgeOutputValidationError,
    evaluate_one_facet,
)
from semantic_relevance_facet_scoring import (
    EvidenceInferenceKind,
    FacetProminence,
    QueryFacet,
    VerificationRelation,
)

EVALS_DIR = Path(__file__).resolve().parent
CONFIG_FILE = EVALS_DIR / "judge_configs" / "semantic_relevance_judge.v0.26.0.json"
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
                component_checks=[
                    ComponentEvidenceCheck(component_id="full synthetic facet", established=False, supporting_span_ids=[], reason="missing"),
                ],
                all_required_components_established=False,
                missing_semantic_component="full synthetic facet",
                semantic_definition_exclusion_applied=False,
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
                component_checks=[
                    ComponentEvidenceCheck(component_id="full synthetic facet", established=False, supporting_span_ids=[], reason="missing"),
                ],
                all_required_components_established=False,
                semantic_definition_exclusion_applied=False,
                combined_evidence_summary="synthetic malformed positive",
                missing_semantic_component="still missing something",
                reason="malformed",
            )

        raise AssertionError(schema)


model = RepeatedMalformedCompositeModel()
try:
    evaluate_one_facet(
        judge_model=model,
        query_id="SYN",
        facet=facet,
        spans=spans,
        config=config,
    )
except JudgeOutputValidationError as error:
    assert "Unable to obtain a structurally valid full-context verification" in str(error)
else:
    raise AssertionError("Malformed full-context output was silently converted to semantic UNSUPPORTED")

assert model.composite_calls == 2
print("v0.25.0 malformed full-context output propagates as evaluation failure")
