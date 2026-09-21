"""Contract for the blind-review development-label revision used by v0.24.0."""
from pathlib import Path
import json
ROOT=Path(__file__).resolve().parents[1]
p=ROOT/'evals/datasets/semantic_relevance_label_revisions.v1.0.0.json'
d=json.loads(p.read_text(encoding='utf-8'))
assert d['version']=='1.0.0'
items=d['revisions']
assert len(items)==1
r=items[0]
assert r['case_id']=='U2_Q03_T30'
assert r['previous_human_score']==2
assert r['new_human_score']==0
assert r['review_method']=='blind_review'
assert 'not automatic indicators of personal growth' in r['reason'].lower()
print('v0.24.0 blind-review label revision manifest passed')
