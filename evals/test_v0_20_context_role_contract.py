import json
from pathlib import Path
from semantic_relevance_facet_judge import ProminenceAssessment, _validate_prominence_result, _prominence_from_context_role
from semantic_relevance_facet_scoring import FacetContextRole, FacetProminence
ROOT=Path(__file__).resolve().parents[1]
c=json.loads((ROOT/'evals/judge_configs/semantic_relevance_judge.v0.20.0.json').read_text())
for role, expected in [
 (FacetContextRole.CENTRAL_SUBJECT,FacetProminence.CENTRAL),
 (FacetContextRole.SUBSTANTIVE_SUBJECT,FacetProminence.SUBSTANTIVE),
 (FacetContextRole.BACKGROUND_CAUSE,FacetProminence.INCIDENTAL),
 (FacetContextRole.EXAMPLE_OR_ILLUSTRATION,FacetProminence.INCIDENTAL),
 (FacetContextRole.META_DISCUSSION,FacetProminence.INCIDENTAL),
 (FacetContextRole.INCIDENTAL_MENTION,FacetProminence.INCIDENTAL),
]:
    r=ProminenceAssessment(primary_subject_summary='primary subject',context_role=role,reason='test')
    _validate_prominence_result(r,c)
    assert _prominence_from_context_role(role,c)==expected
print('v0.20 primary-subject/context-role contract tests passed.')
