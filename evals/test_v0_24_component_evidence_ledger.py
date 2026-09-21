"""v0.24 component ledger must ground every established component to real spans."""
from __future__ import annotations

import json
from pathlib import Path

from semantic_relevance_facet_judge import (
    ComponentEvidenceCheck,
    CompositeEvidenceVerification,
    EvidenceVerification,
    JudgeOutputValidationError,
    _validate_composite_result,
    _validate_verification_result,
)
from semantic_relevance_facet_scoring import EvidenceInferenceKind, QueryFacet, VerificationRelation

E = Path(__file__).resolve().parent
CONFIG = json.loads((E / "judge_configs/semantic_relevance_judge.v0.25.0.json").read_text())
FACET = QueryFacet(facet_id="F1", text="dangerous journeys", facet_type="core", semantic_definition="danger + movement")

# Single-span positive cannot cite a different span for a component.
bad_single = EvidenceVerification(
    verification_relation=VerificationRelation.DIRECT,
    inference_kind=EvidenceInferenceKind.EXPLICIT_COMPONENTS,
    component_checks=[ComponentEvidenceCheck(component_id="movement", established=True, supporting_span_ids=["S2"], reason="wrong span")],
    all_required_components_established=True,
    reason="bad grounding",
)
try:
    _validate_verification_result(bad_single, FACET, CONFIG, evidence_span_id="S1")
except JudgeOutputValidationError:
    pass
else:
    raise AssertionError("single-span ledger accepted evidence from another span")

# Composite positive must ground every established component inside top-level support.
bad_composite = CompositeEvidenceVerification(
    supporting_span_ids=["S1", "S2"],
    verification_relation=VerificationRelation.ENTAILED,
    inference_kind=EvidenceInferenceKind.NECESSARY_SEMANTIC_INFERENCE,
    component_checks=[ComponentEvidenceCheck(component_id="movement", established=True, supporting_span_ids=["S3"], reason="outside winner set")],
    all_required_components_established=True,
    combined_evidence_summary="synthetic",
    reason="synthetic",
)
try:
    _validate_composite_result(bad_composite, FACET, {"S1":"a", "S2":"b", "S3":"c"}, CONFIG)
except JudgeOutputValidationError:
    pass
else:
    raise AssertionError("composite ledger accepted component evidence outside top-level support")

# Aggregate completeness is mechanically derived from the ledger.
bad_flag = EvidenceVerification(
    verification_relation=VerificationRelation.UNSUPPORTED,
    inference_kind=EvidenceInferenceKind.NONE,
    component_checks=[ComponentEvidenceCheck(component_id="movement", established=False, supporting_span_ids=[], reason="missing")],
    all_required_components_established=True,
    reason="inconsistent flag",
)
try:
    _validate_verification_result(bad_flag, FACET, CONFIG, evidence_span_id="S1")
except JudgeOutputValidationError:
    pass
else:
    raise AssertionError("aggregate completeness flag diverged from component ledger")

print("v0.24.0 component-evidence-ledger contracts passed")
