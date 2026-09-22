"""v0.25 full-context recovery may find missing components in Stage-A-unseen spans."""
from __future__ import annotations

import json
from pathlib import Path

from semantic_relevance_facet_judge import (
    ComponentEvidenceCheck,
    CompositeEvidenceVerification,
    EvidenceVerification,
    JudgeOutputValidationError,
    _validate_composite_result,
)
from semantic_relevance_facet_scoring import (
    EvidenceInferenceKind,
    QueryFacet,
    QueryFacetSpec,
    VerificationRelation,
)

E = Path(__file__).resolve().parent
CONFIG = json.loads((E / "judge_configs/semantic_relevance_judge.v0.26.0.json").read_text(encoding="utf-8"))
FACETS = json.loads((E / "facets/semantic_relevance/semantic_relevance_query_facets.v0.9.0.json").read_text(encoding="utf-8"))

q06_raw = next(q for q in FACETS["queries"] if q["query_id"] == "Q06")
q06 = QueryFacetSpec(
    query_id=q06_raw["query_id"],
    query=q06_raw["query"],
    facets=[QueryFacet(**f) for f in q06_raw["facets"]],
)
facet = next(f for f in q06.facets if f.facet_id == "F3")

spans = {
    "S8": "Suddenly devastation tears at the heart of their family, and the depth of their existence.",
    "S11": "Together, they're determined to move on with their lives.",
}

# Stage A audited only S8. It can ground the prerequisite loss context while
# still rejecting the full facet because rebuilding/resuming is absent.
prior = {
    "S8": EvidenceVerification(
        verification_relation=VerificationRelation.ADJACENT,
        inference_kind=EvidenceInferenceKind.INCOMPLETE_CONNECTION,
        component_checks=[
            ComponentEvidenceCheck(
                component_id="prior_grief_or_major_loss",
                established=True,
                supporting_span_ids=["S8"],
                reason="The devastation affecting the family grounds a major-loss context.",
            ),
            ComponentEvidenceCheck(
                component_id="rebuilding_or_resuming_own_life_after_loss",
                established=False,
                supporting_span_ids=[],
                reason="No rebuilding is stated in this span.",
            ),
        ],
        all_required_components_established=False,
        missing_semantic_component="rebuilding_or_resuming_own_life_after_loss",
        reason="Loss context only.",
    )
}

# Full-context recovery may cite new S11 and combine the two canonical components.
recovered = CompositeEvidenceVerification(
    supporting_span_ids=["S8", "S11"],
    verification_relation=VerificationRelation.ENTAILED,
    inference_kind=EvidenceInferenceKind.NECESSARY_SEMANTIC_INFERENCE,
    component_checks=[
        ComponentEvidenceCheck(
            component_id="prior_grief_or_major_loss",
            established=True,
            supporting_span_ids=["S8"],
            reason="S8 grounds the loss context.",
        ),
        ComponentEvidenceCheck(
            component_id="rebuilding_or_resuming_own_life_after_loss",
            established=True,
            supporting_span_ids=["S11"],
            reason="S11 explicitly states determination to move on with their lives.",
        ),
    ],
    all_required_components_established=True,
    combined_evidence_summary="Loss context plus explicit continuation of life.",
    reason="The canonical components are distributed across S8 and newly selected S11.",
)
_validate_composite_result(recovered, facet, spans, CONFIG, verification_results_by_span=prior)

# The same already-audited span may not flip a missing component to established.
resurrected = recovered.model_copy(deep=True)
resurrected.supporting_span_ids = ["S8"]
resurrected.component_checks[1].supporting_span_ids = ["S8"]
try:
    _validate_composite_result(resurrected, facet, spans, CONFIG, verification_results_by_span=prior)
except JudgeOutputValidationError as error:
    assert "resurrection" in str(error)
else:
    raise AssertionError("Full-context recovery resurrected a missing component from the same audited span")

# v0.25 permits a one-span full-context positive when one newly found span
# genuinely supplies every canonical component; this avoids schema-shape loss.
synthetic = QueryFacet(
    facet_id="F1",
    text="synthetic",
    facet_type="core",
    semantic_definition="synthetic",
    required_components=[
        {
            "component_id": "complete_fact",
            "definition": "The complete synthetic fact is stated.",
            "negative_boundaries": [],
        }
    ],
)
one_span = CompositeEvidenceVerification(
    supporting_span_ids=["S2"],
    verification_relation=VerificationRelation.ENTAILED,
    inference_kind=EvidenceInferenceKind.NECESSARY_SEMANTIC_INFERENCE,
    component_checks=[
        ComponentEvidenceCheck(
            component_id="complete_fact",
            established=True,
            supporting_span_ids=["S2"],
            reason="S2 supplies the complete fact.",
        )
    ],
    all_required_components_established=True,
    combined_evidence_summary="Complete fact.",
    reason="Full-context scan found the missed span.",
)
_validate_composite_result(one_span, synthetic, {"S2": "complete fact"}, CONFIG)

print("v0.25.0 full-context component recovery contract passed")
