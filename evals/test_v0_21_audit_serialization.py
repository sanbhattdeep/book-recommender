import json

from semantic_relevance_facet_judge import serialize_facet_assessments
from semantic_relevance_facet_scoring import (
    CandidateVerificationRecord,
    EvidenceInferenceKind,
    FacetContextRole,
    FacetJudgeVerdict,
    FacetPipelineAssessment,
    FacetProminence,
    QueryFacet,
    QueryFacetSpec,
    VerificationRelation,
)


spec = QueryFacetSpec(
    query_id="QX",
    query="x",
    facets=[
        QueryFacet(
            facet_id="F1",
            text="x",
            facet_type="core",
            semantic_definition="x",
        )
    ],
)
assessment = FacetPipelineAssessment(
    facet_id="F1",
    candidate_evidence_span_ids=["S1"],
    candidate_verifications=[
        CandidateVerificationRecord(
            candidate_rank=1,
            evidence_span_id="S1",
            verification_relation=VerificationRelation.ENTAILED,
            inference_kind=EvidenceInferenceKind.NECESSARY_SEMANTIC_INFERENCE,
            verification_reason="x",
        )
    ],
    candidate_evidence_span_id="S1",
    verification_relation=VerificationRelation.ENTAILED,
    inference_kind=EvidenceInferenceKind.NECESSARY_SEMANTIC_INFERENCE,
    prominence=FacetProminence.INCIDENTAL,
    context_role=FacetContextRole.BACKGROUND_CAUSE,
    primary_subject_summary="primary subject",
    is_background_cause_or_factor=True,
    role_supporting_span_ids=["S1"],
    evidence_selection_reason="x",
    verification_reason="x",
    prominence_reason="x",
    hard_exclusion_precheck_attempted=True,
    hard_exclusion_precheck_reason="no exclusion triggered",
    composite_inference_kind=EvidenceInferenceKind.NONE,
)
payload = json.loads(
    serialize_facet_assessments(
        spec,
        FacetJudgeVerdict(assessments=[assessment], overall_reason="x"),
    )
)[0]

assert payload["inference_kind"] == "necessary_semantic_inference"
assert payload["context_role"] == "background_cause"
assert payload["is_background_cause_or_factor"] is True
assert payload["role_supporting_span_ids"] == ["S1"]
assert payload["hard_exclusion_precheck_attempted"] is True
assert "composite_inference_kind" in payload

print("v0.21 audit serialization test passed.")
