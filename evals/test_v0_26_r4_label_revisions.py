"""v0.26.0 r4 blind-review label revisions."""
from pathlib import Path
import json

E = Path(__file__).resolve().parent
p = E / 'datasets/semantic_relevance_label_revisions.v1.1.0.json'
d = json.loads(p.read_text(encoding='utf-8'))
assert d['version'] == '1.1.0'
items = {x['case_id']: x for x in d['revisions']}
assert set(items) == {'U2_Q03_T30', 'U2_Q01_T02', 'U2_Q05_T10'}
assert items['U2_Q03_T30']['previous_human_score'] == 2 and items['U2_Q03_T30']['new_human_score'] == 0
assert items['U2_Q01_T02']['previous_human_score'] == 2 and items['U2_Q01_T02']['new_human_score'] == 1
assert items['U2_Q05_T10']['previous_human_score'] == 4 and items['U2_Q05_T10']['new_human_score'] == 2
assert 'weak match with redemption' in items['U2_Q01_T02']['reason'].lower()
assert 'political-conflict aspect is missing' in items['U2_Q05_T10']['reason'].lower()
assert all(x['review_method'] == 'blind_review' for x in items.values())
print('v0.26.0 r4 blind-review label revision contract passed')
