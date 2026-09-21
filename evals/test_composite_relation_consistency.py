"""Contract tests for v0.19 composite relation consistency."""

from __future__ import annotations

import json
from pathlib import Path

from semantic_relevance_facet_judge import (
    ComponentEvidenceCheck,
    CompositeEvidenceVerification,
    JudgeOutputValidationError,
    _validate_composite_result,
)
from semantic_relevance_facet_scoring import EvidenceInferenceKind, VerificationRelation, QueryFacet

EVALS_DIR = Path(__file__).resolve().parent
config = json.loads(
    (
        EVALS_DIR
        / "judge_configs"
        / "semantic_relevance_judge.v0.26.0.json"
    ).read_text(encoding="utf-8")
)

facet = QueryFacet(
    facet_id="F1",
    text="synthetic",
    facet_type="core",
    semantic_definition="synthetic",
)

spans = {
    "S1": "A major disruption occurs.",
    "S2": "Later the characters resume their lives.",
}

entailed = CompositeEvidenceVerification(
    supporting_span_ids=["S1", "S2"],
    verification_relation=VerificationRelation.ENTAILED,
    inference_kind=EvidenceInferenceKind.NECESSARY_SEMANTIC_INFERENCE,
    component_checks=[
        ComponentEvidenceCheck(component_id="synthetic component", established=True, supporting_span_ids=["S1", "S2"], reason="grounded"),
    ],
    all_required_components_established=True,
    semantic_definition_exclusion_applied=False,
    combined_evidence_summary="The disruption is followed by explicit resumption.",
    missing_semantic_component=None,
    reason="All required components are supplied.",
)
_validate_composite_result(entailed, facet, spans, config)

adjacent = CompositeEvidenceVerification(
    supporting_span_ids=["S1", "S2"],
    verification_relation=VerificationRelation.ADJACENT,
    inference_kind=EvidenceInferenceKind.INCOMPLETE_CONNECTION,
    component_checks=[
        ComponentEvidenceCheck(component_id="synthetic component", established=False, supporting_span_ids=[], reason="missing"),
    ],
    all_required_components_established=False,
    semantic_definition_exclusion_applied=False,
    combined_evidence_summary="The spans show disruption and later activity.",
    missing_semantic_component="The later activity is not established as recovery from the disruption.",
    reason="One required causal/recovery component remains missing.",
)
_validate_composite_result(adjacent, facet, spans, config)

bad_adjacent = CompositeEvidenceVerification(
    supporting_span_ids=["S1", "S2"],
    verification_relation=VerificationRelation.ADJACENT,
    inference_kind=EvidenceInferenceKind.INCOMPLETE_CONNECTION,
    component_checks=[
        ComponentEvidenceCheck(component_id="synthetic component", established=False, supporting_span_ids=[], reason="missing"),
    ],
    all_required_components_established=False,
    semantic_definition_exclusion_applied=False,
    combined_evidence_summary="The full facet is established.",
    missing_semantic_component=None,
    reason="Malformed adjacent output.",
)
try:
    _validate_composite_result(bad_adjacent, facet, spans, config)
except JudgeOutputValidationError:
    pass
else:
    raise AssertionError("ADJACENT without a missing semantic component must fail.")

bad_entailed = CompositeEvidenceVerification(
    supporting_span_ids=["S1", "S2"],
    verification_relation=VerificationRelation.ENTAILED,
    inference_kind=EvidenceInferenceKind.NECESSARY_SEMANTIC_INFERENCE,
    component_checks=[
        ComponentEvidenceCheck(component_id="synthetic component", established=True, supporting_span_ids=["S1", "S2"], reason="grounded"),
    ],
    all_required_components_established=True,
    semantic_definition_exclusion_applied=False,
    combined_evidence_summary="Nearly complete.",
    missing_semantic_component="A required component is still absent.",
    reason="Malformed entailed output.",
)
try:
    _validate_composite_result(bad_entailed, facet, spans, config)
except JudgeOutputValidationError:
    pass
else:
    raise AssertionError("ENTAILED with a missing semantic component must fail.")

print("All v0.22.0 composite relation-consistency tests passed.")
