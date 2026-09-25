"""Static r6 contracts derived from the consumed 30-case validation review."""
from pathlib import Path
import json
ROOT=Path(__file__).resolve().parent
p95=ROOT/'facets/semantic_relevance/semantic_relevance_query_facets.v0.9.5.json'
p96=ROOT/'facets/semantic_relevance/semantic_relevance_query_facets.v0.9.6.json'
a=json.loads(p95.read_text(encoding='utf-8')); b=json.loads(p96.read_text(encoding='utf-8'))
assert b['version']=='0.9.6' and b['development_dataset_version']=='3.1.0'
qa={q['query_id']:q for q in a['queries']}; qb={q['query_id']:q for q in b['queries']}
assert set(qa)==set(qb)
# Only Q01/Q02/Q05 facet semantics change in v0.9.6.
for qid in qa:
    if qid not in {'Q01','Q02','Q05'}:
        assert qa[qid]==qb[qid], f'unexpected facet-spec semantic change in {qid}'
# Canonical component IDs are immutable everywhere.
for qid in qa:
    for fa,fb in zip(qa[qid]['facets'],qb[qid]['facets'],strict=True):
        assert fa['facet_id']==fb['facet_id']
        assert [x['component_id'] for x in fa['required_components']]==[x['component_id'] for x in fb['required_components']]
q01=json.dumps(qb['Q01'],ensure_ascii=False).lower(); assert 'fall-and-rise' in q01 and 'return to greatness' in q01
q02=json.dumps(qb['Q02'],ensure_ascii=False).lower(); assert 'action-packed' in q02 and 'battle' in q02 and 'does not' in q02
q05=json.dumps(qb['Q05'],ensure_ascii=False).lower(); assert 'mode a' in q05 and 'mode b' in q05 and 'meaningful historical/analytical subject' in q05
cfg=json.loads((ROOT/'judge_configs/semantic_relevance_judge.v0.27.0.json').read_text(encoding='utf-8'))
assert cfg['facet_spec_version'] in {'0.9.6','0.9.7'}
prom='\n'.join(cfg['prominence_stage']['instructions']).lower()
assert 'constituent-item scope' in prom and 'broader collection/anthology' in prom and 'use other' in prom
judge=(ROOT/'semantic_relevance_facet_judge.py').read_text(encoding='utf-8')
for token in ['LOCAL Q01 REDEMPTION-DAMAGE CONTRACT','LOCAL Q01 REDEMPTION-RESTORATION CONTRACT','LOCAL Q02 SUSPENSE CONTRACT','LOCAL Q05 WAR CONTRACT']:
    assert token in judge
print('v0.27 r6 localized validation-boundary contracts passed')
