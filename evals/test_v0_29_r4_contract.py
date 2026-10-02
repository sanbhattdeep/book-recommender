import json
from pathlib import Path
R=Path(__file__).resolve().parents[1]
def j(p): return json.loads(p.read_text(encoding='utf-8-sig'))
def t(p): return p.read_text(encoding='utf-8-sig')
spec=j(R/'evals/facets/semantic_relevance/semantic_relevance_query_facets.v0.10.1.json')
cfg=j(R/'evals/judge_configs/semantic_relevance_judge.v0.29.0-r4.json')
reg=j(R/'evals/datasets/semantic_relevance_v0.29_regression_manifest.v2.3.0.json')
judge=t(R/'evals/semantic_relevance_facet_judge.py')
runner=t(R/'evals/run_judge_v0_29_r4_development.py')
assert spec['version']=='0.10.1'
assert cfg['version']=='0.29.0-r4' and cfg['facet_spec_version']=='0.10.1'
assert reg['version']=='2.3.0' and len(reg['targeted_expectations'])==60
assert reg['targeted_expectations']['U_Q04_T10']=={'judge_min':3,'judge_max':4}
assert reg['targeted_expectations']['U4_Q07_T03']=={'judge_max':1}
assert reg['targeted_expectations']['U2_Q04_T70']['judge_min']==3
assert reg['targeted_expectations']['U2_Q09_T02']['judge_min']==3
assert reg['targeted_expectations']['U4_Q03_T02']=={'judge_min':2,'judge_max':2}
assert reg['targeted_expectations']['U4_Q04_T02']=={'judge_max':0}
assert reg['targeted_expectations']['U4_Q09_T04']=={'judge_min':2,'judge_max':2}
assert '_Q04_R4_DIRECTIONAL_DEPARTURE_RE' in judge
assert '_q04_r4_has_directional_departure_anchor' in judge
assert '_q07_r4' not in judge.lower()
assert 'JUDGE_CONFIG_VERSION = "0.29.0-r4"' in runner
assert 'FACET_SPEC_VERSION = "0.10.1"' in runner
print('v0.29 r4 semantic/provenance contract passed')
