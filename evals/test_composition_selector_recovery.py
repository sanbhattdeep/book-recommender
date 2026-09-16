"""Regression test for the v0.17.1 composition-selector reliability patch."""

from __future__ import annotations

import json
from pathlib import Path

from semantic_relevance_facet_judge import (
    CompositeEvidenceSelection,
    EvidenceSelection,
    EvidenceVerification,
    evaluate_one_facet,
)
from semantic_relevance_facet_scoring import (
    FacetProminence,
    QueryFacet,
    VerificationRelation,
)

EVALS_DIR = Path(__file__).resolve().parent
CONFIG_FILE = (
    EVALS_DIR
    / "judge_configs"
    / "semantic_relevance_judge.v0.17.1.json"
)

config = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))

facet = QueryFacet(
    facet_id="F1",
    text="synthetic recovery facet",
    facet_type="core",
    semantic_definition="A synthetic facet used only for structured-output testing.",
)

spans = {
    "S1": "One weak statement.",
    "S2": "Another unrelated statement.",
    "S3": "A third statement.",
}


class RepeatedSingletonSelectorModel:
    """
    Stage A returns weak unsupported evidence.

    The composition selector then violates its contract twice by returning
    exactly one real span. v0.17.1 must NOT abort the case; it must
    conservatively fall back to NONE/no-composition.
    """

    def __init__(self) -> None:
        self.selector_calls = 0

    def generate(self, prompt, schema, **kwargs):
        if schema is EvidenceSelection:
            return EvidenceSelection(
                evidence_span_ids=["S1"],
                reason="synthetic standalone candidate",
            )

        if schema is EvidenceVerification:
            return EvidenceVerification(
                verification_relation=VerificationRelation.UNSUPPORTED,
                reason="synthetic unsupported",
            )

        if schema is CompositeEvidenceSelection:
            self.selector_calls += 1
            return CompositeEvidenceSelection(
                evidence_span_ids=["S2"],
                reason="malformed: only one real span",
            )

        raise AssertionError(
            f"Unexpected schema after conservative fallback: {schema}"
        )


model = RepeatedSingletonSelectorModel()

assessment, evidence, candidate_texts, retries = evaluate_one_facet(
    judge_model=model,
    query_id="SYN",
    facet=facet,
    spans=spans,
    config=config,
)

assert model.selector_calls == 2
assert assessment.verification_relation == VerificationRelation.UNSUPPORTED
assert assessment.prominence == FacetProminence.NOT_APPLICABLE
assert assessment.composite_selection_attempted is True
assert assessment.composite_candidate_span_ids == []
assert assessment.composite_evidence_span_ids == []
assert assessment.composite_verification_relation is None
assert assessment.composite_selection_reason is not None
assert assessment.composite_selection_reason.startswith(
    "structured_output_recovery_fallback:"
)
assert evidence == spans["S1"]
assert candidate_texts == {"S1": spans["S1"]}

print(
    "v0.17.1 malformed composition-selector fallback test passed."
)
