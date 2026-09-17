from __future__ import annotations
import json
from pathlib import Path

from semantic_relevance_facet_judge import (
    CompositeEvidenceVerification,
    EvidenceVerification,
    JudgeOutputValidationError,
    ProminenceAssessment,
    _validate_composite_result,
    _validate_prominence_result,
    _validate_verification_result,
)
from semantic_relevance_facet_scoring import (
    EvidenceInferenceKind,
    FacetContextRole,
    FacetProminence,
    VerificationRelation,
)

ROOT = Path(__file__).resolve().parent
CFG = json.loads((ROOT / "judge_configs" / "semantic_relevance_judge.v0.19.0.json").read_text(encoding="utf-8"))

# Necessary entailment passes.
ok = EvidenceVerification(
    verification_relation=VerificationRelation.ENTAILED,
    inference_kind=EvidenceInferenceKind.NECESSARY_SEMANTIC_INFERENCE,
    reason="The exact evidence necessarily supplies the complete facet in one short semantic step.",
)
_validate_verification_result(ok, CFG)

# Symbolic possibility can never masquerade as entailment.
bad_kind = EvidenceVerification(
    verification_relation=VerificationRelation.ENTAILED,
    inference_kind=EvidenceInferenceKind.SYMBOLIC_POSSIBILITY,
    reason="The image necessarily establishes the facet.",
)
try:
    _validate_verification_result(bad_kind, CFG)
except JudgeOutputValidationError:
    pass
else:
    raise AssertionError("ENTAILED + symbolic_possibility must fail")

# Even a mislabeled necessary inference is rejected if its reason is possibility language.
bad_reason = EvidenceVerification(
    verification_relation=VerificationRelation.ENTAILED,
    inference_kind=EvidenceInferenceKind.NECESSARY_SEMANTIC_INFERENCE,
    reason="The rain could symbolize the presence of gods.",
)
try:
    _validate_verification_result(bad_reason, CFG)
except JudgeOutputValidationError:
    pass
else:
    raise AssertionError("Speculative ENTAILED reason must fail")

# Composite contract uses the same backstop.
comp = CompositeEvidenceVerification(
    supporting_span_ids=["S1", "S2"],
    verification_relation=VerificationRelation.ENTAILED,
    inference_kind=EvidenceInferenceKind.NECESSARY_SEMANTIC_INFERENCE,
    combined_evidence_summary="The two spans necessarily establish the complete facet.",
    missing_semantic_component=None,
    reason="Both required semantic components are explicitly supplied across the two spans.",
)
_validate_composite_result(comp, {"S1":"a", "S2":"b"}, CFG)

# Context role -> prominence is deterministic.
for role, prominence in [
    (FacetContextRole.CENTRAL_SUBJECT, FacetProminence.CENTRAL),
    (FacetContextRole.SUBSTANTIVE_SUBJECT, FacetProminence.SUBSTANTIVE),
    (FacetContextRole.BACKGROUND_CAUSE, FacetProminence.INCIDENTAL),
    (FacetContextRole.EXAMPLE_OR_ILLUSTRATION, FacetProminence.INCIDENTAL),
    (FacetContextRole.META_DISCUSSION, FacetProminence.INCIDENTAL),
    (FacetContextRole.INCIDENTAL_MENTION, FacetProminence.INCIDENTAL),
]:
    _validate_prominence_result(
        ProminenceAssessment(context_role=role, prominence=prominence, reason="test"),
        CFG,
    )

bad_prominence = ProminenceAssessment(
    context_role=FacetContextRole.BACKGROUND_CAUSE,
    prominence=FacetProminence.SUBSTANTIVE,
    reason="Background cause was incorrectly inflated.",
)
try:
    _validate_prominence_result(bad_prominence, CFG)
except JudgeOutputValidationError:
    pass
else:
    raise AssertionError("background_cause -> substantive must fail")

print("v0.19 inference-kind and context-role contract tests passed.")
