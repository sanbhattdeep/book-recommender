"""Static/unit r7 contracts from the consumed r6 targeted diagnostics."""
from pathlib import Path
import json, sys
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from semantic_relevance_facet_judge import (
    ProminenceAssessment,
    _apply_constituent_item_scope_guard,
)
from semantic_relevance_facet_scoring import QueryFacet, SubjectRelation

p96=ROOT/'facets/semantic_relevance/semantic_relevance_query_facets.v0.9.6.json'
p97=ROOT/'facets/semantic_relevance/semantic_relevance_query_facets.v0.9.7.json'
a=json.loads(p96.read_text(encoding='utf-8')); b=json.loads(p97.read_text(encoding='utf-8'))
assert b['version']=='0.9.7' and b['development_dataset_version']=='3.1.0'
qa={q['query_id']:q for q in a['queries']}; qb={q['query_id']:q for q in b['queries']}
assert set(qa)==set(qb)
# r7 changes only Q01/Q02 semantics; the successful r6 Q05 repair is byte-structurally frozen.
for qid in qa:
    if qid not in {'Q01','Q02'}:
        assert qa[qid]==qb[qid], f'unexpected r7 facet-spec semantic change in {qid}'
for qid in qa:
    for fa,fb in zip(qa[qid]['facets'],qb[qid]['facets'],strict=True):
        assert fa['facet_id']==fb['facet_id']
        assert [x['component_id'] for x in fa['required_components']]==[x['component_id'] for x in fb['required_components']]
q01=json.dumps(qb['Q01'],ensure_ascii=False).lower()
for token in ['directional transition','finding a way out','restoration/atonement']:
    assert token in q01
q02=json.dumps(qb['Q02'],ensure_ascii=False).lower()
for token in ['future-tense plot progression','trigger, cause, attract, ignite, or set in motion','do not infer suspense']:
    assert token in q02
cfg=json.loads((ROOT/'judge_configs/semantic_relevance_judge.v0.27.0.json').read_text(encoding='utf-8'))
assert cfg['facet_spec_version']=='0.9.7'
prom='\n'.join(cfg['prominence_stage']['instructions']).lower()
assert 'title story is still one constituent item' in prom
judge=(ROOT/'semantic_relevance_facet_judge.py').read_text(encoding='utf-8')
for token in [
    'LOCAL Q01 REDEMPTION-RESTORATION CONTRACT',
    'negative-state -> better-state trajectory',
    'LOCAL Q02 SUSPENSE CONTRACT',
    'Future-tense plot progression',
    '_apply_constituent_item_scope_guard',
    'LOCAL Q05 WAR CONTRACT',
]:
    assert token in judge

facet=QueryFacet(
    facet_id='F1', text='complicated parent-child relationship', facet_type='core',
    semantic_definition='test', hard_exclusions=[], required_components=[]
)
model_result=ProminenceAssessment(
    subject_relation=SubjectRelation.DEFINING_CONTENT_OR_NARRATIVE_DRIVER,
    is_substantively_examined=True, supporting_span_ids=['S1'], reason='important inside the title story'
)
item_text=('A parent whose choices damage the children, in the title story in a collection of tales about immigrant families.')
guarded=_apply_constituent_item_scope_guard(facet=facet,evidence_text=item_text,spans={'S1':item_text},result=model_result)
assert guarded.subject_relation==SubjectRelation.OTHER
assert guarded.is_substantively_examined is False
# Explicit collection-level facet wording disables the guard.
whole=('A collection about complicated parent-child relationship dynamics; in the title story, one family faces conflict.')
not_guarded=_apply_constituent_item_scope_guard(facet=facet,evidence_text=whole,spans={'S1':whole},result=model_result)
assert not_guarded.subject_relation==SubjectRelation.DEFINING_CONTENT_OR_NARRATIVE_DRIVER
print('v0.27 r7 localized validation-boundary contracts passed')
