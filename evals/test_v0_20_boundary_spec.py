import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
p=ROOT/'evals/facets/semantic_relevance/semantic_relevance_query_facets.v0.8.0.json'
d=json.loads(p.read_text())
assert d['version']=='0.8.0'
qs={q['query_id']:q for q in d['queries']}
for qid,fid in [('Q02','F1'),('Q05','F1'),('Q05','F2'),('Q11','F1'),('Q11','F2'),('Q11','F3'),('Q11','F4')]:
    f=next(x for x in qs[qid]['facets'] if x['facet_id']==fid)
    assert f.get('hard_exclusions'), (qid,fid)
config=json.loads((ROOT/'evals/judge_configs/semantic_relevance_judge.v0.20.0.json').read_text())
assert not any(r.get('cue_id')=='Q11_F1_adventure_lexical' for r in config['deterministic_direct_cue_stage']['rules'])
print('v0.20 boundary-spec tests passed.')
