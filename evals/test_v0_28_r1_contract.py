from pathlib import Path
import json
E=Path(__file__).resolve().parent
spec=json.loads((E/'facets/semantic_relevance/semantic_relevance_query_facets.v0.9.8.json').read_text(encoding='utf-8'))
cfg=json.loads((E/'judge_configs/semantic_relevance_judge.v0.28.0.json').read_text(encoding='utf-8'))
reg=json.loads((E/'datasets/semantic_relevance_v0.28_regression_manifest.v1.0.0.json').read_text(encoding='utf-8'))
assert spec['version']=='0.9.8' and cfg['version']=='0.28.0' and cfg['facet_spec_version']=='0.9.8'
q03=next(q for q in spec['queries'] if q['query_id']=='Q03');f1=next(f for f in q03['facets'] if f['facet_id']=='F1')
assert 'Potential for happiness' in ' '.join(f1['required_components'][0]['negative_boundaries'])
q11=next(q for q in spec['queries'] if q['query_id']=='Q11');a=next(f for f in q11['facets'] if f['facet_id']=='F1');h=next(f for f in q11['facets'] if f['facet_id']=='F4')
assert 'mission, duty, objective' in ' '.join(a['required_components'][0]['negative_boundaries'])
assert 'Fantasy-sounding names or settings' in ' '.join(h['required_components'][0]['negative_boundaries'])
assert any('DIRECT-CUE / LIST-MENTION BOUNDARY' in x for x in cfg['prominence_stage']['instructions'])
assert len(reg['targeted_expectations'])==36
assert reg['targeted_expectations']['U3_Q03_T01']=={'judge_max':0}
assert reg['targeted_expectations']['U3_Q11_T02']=={'judge_max':1}
assert set(reg['diagnostic_only_cases'])=={'U3_Q10_T01','U3_Q11_T04'}
print('v0.28 r1 localized contract passed')
