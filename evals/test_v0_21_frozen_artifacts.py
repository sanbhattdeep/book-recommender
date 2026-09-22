import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
cfg=json.loads((ROOT/'evals/judge_configs/semantic_relevance_judge.v0.21.2.json').read_text(encoding="utf-8"))
facets=json.loads((ROOT/'evals/facets/semantic_relevance/semantic_relevance_query_facets.v0.8.0.json').read_text(encoding="utf-8"))
rubric=json.loads((ROOT/'evals/rubrics/semantic_relevance/semantic_relevance_rubric.v0.1.0.json').read_text(encoding="utf-8"))
assert cfg['version']=='0.21.2'
assert cfg['facet_spec_version']=='0.8.0'
assert cfg['rubric_version']=='0.1.0'
assert facets['version']=='0.8.0' and facets['status']=='frozen'
assert rubric['version']=='0.1.0'
print('v0.21.2 frozen artifact contract passed')
