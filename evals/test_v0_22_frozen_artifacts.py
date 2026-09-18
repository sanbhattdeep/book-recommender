from pathlib import Path
import json
ROOT=Path(__file__).resolve().parents[1]
cfg=json.loads((ROOT/'evals/judge_configs/semantic_relevance_judge.v0.23.0.json').read_text())
facet=json.loads((ROOT/'evals/facets/semantic_relevance/semantic_relevance_query_facets.v0.8.0.json').read_text())
rubric=json.loads((ROOT/'evals/rubrics/semantic_relevance/semantic_relevance_rubric.v0.1.0.json').read_text())
assert cfg['version']=='0.23.0'
assert cfg['facet_spec_version']=='0.8.0'
assert cfg['rubric_version']=='0.1.0'
assert facet['version']=='0.8.0' and facet['status']=='frozen'
assert rubric['version']=='0.1.0'
print('v0.23.0 frozen artifact contract passed')
