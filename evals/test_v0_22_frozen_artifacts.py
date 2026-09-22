from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]
cfg = json.loads((ROOT / 'evals/judge_configs/semantic_relevance_judge.v0.26.0.json').read_text())
facet = json.loads((ROOT / 'evals/facets/semantic_relevance/semantic_relevance_query_facets.v0.9.2.json').read_text())
rubric = json.loads((ROOT / 'evals/rubrics/semantic_relevance/semantic_relevance_rubric.v0.1.0.json').read_text())

assert cfg['version'] == '0.26.0'
assert cfg['facet_spec_version'] == '0.9.2'
assert cfg['rubric_version'] == '0.1.0'
assert facet['version'] == '0.9.2' and facet['status'] == 'frozen'
assert rubric['version'] == '0.1.0'

for query in facet['queries']:
    for item in query['facets']:
        components = item.get('required_components', [])
        assert components, (query['query_id'], item['facet_id'])
        ids = [component['component_id'] for component in components]
        assert len(ids) == len(set(ids)), (query['query_id'], item['facet_id'], ids)
        for component in components:
            assert component['definition'].strip()
            assert isinstance(component.get('negative_boundaries', []), list)

print('v0.26.0 frozen facet/rubric contract passed')
