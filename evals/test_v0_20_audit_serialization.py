import json
from semantic_relevance_facet_judge import serialize_facet_assessments
from semantic_relevance_facet_scoring import *
spec=QueryFacetSpec(query_id='QX',query='x',facets=[QueryFacet(facet_id='F1',text='x',facet_type='core',semantic_definition='x')])
a=FacetPipelineAssessment(facet_id='F1',candidate_evidence_span_ids=['S1'],candidate_verifications=[CandidateVerificationRecord(candidate_rank=1,evidence_span_id='S1',verification_relation=VerificationRelation.ENTAILED,inference_kind=EvidenceInferenceKind.NECESSARY_SEMANTIC_INFERENCE,verification_reason='x')],candidate_evidence_span_id='S1',verification_relation=VerificationRelation.ENTAILED,inference_kind=EvidenceInferenceKind.NECESSARY_SEMANTIC_INFERENCE,prominence=FacetProminence.INCIDENTAL,context_role=FacetContextRole.BACKGROUND_CAUSE,evidence_selection_reason='x',verification_reason='x',prominence_reason='x',composite_inference_kind=EvidenceInferenceKind.NONE)
v=FacetJudgeVerdict(assessments=[a],overall_reason='x')
p=json.loads(serialize_facet_assessments(spec,v))[0]
assert p['inference_kind']=='necessary_semantic_inference'
assert p['context_role']=='background_cause'
assert 'composite_inference_kind' in p
assert 'hard_exclusion_triggered' in p
print('v0.20 audit serialization test passed.')
