import json
from pathlib import Path
from semantic_relevance_facet_judge import build_hard_exclusion_precheck_prompt
from semantic_relevance_facet_scoring import QueryFacetSpec

ROOT=Path(__file__).resolve().parents[1]
config=json.loads((ROOT/'evals/judge_configs/semantic_relevance_judge.v0.21.1.json').read_text())
specs=json.loads((ROOT/'evals/facets/semantic_relevance/semantic_relevance_query_facets.v0.8.0.json').read_text())
q11=next(q for q in specs['queries'] if q['query_id']=='Q11')
spec=QueryFacetSpec(**q11)
f4=next(f for f in spec.facets if f.facet_id=='F4')
prompt=build_hard_exclusion_precheck_prompt(f4, {'S1':'Greek and Norse myths and legends, the stories of gods and heroes.'}, config)
assert 'negative semantic guard, not a lexical veto' in prompt
assert 'Heroes explicitly situated within myths or legends are not generic heroes' in prompt
assert 'Q11_F4_HX_GENERIC_HERO' in prompt
print('v0.21.1 hard-exclusion precheck contract passed')
