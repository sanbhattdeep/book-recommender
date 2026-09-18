"""Static contracts for v0.22.2 subject-relation semantics."""
from pathlib import Path
import json
ROOT=Path(__file__).resolve().parents[1]
cfg=json.loads((ROOT/'evals/judge_configs/semantic_relevance_judge.v0.22.2.json').read_text())
stage=cfg['prominence_stage']
rels=stage['subject_relation_scale']
assert set(rels)=={
    'same_as_primary_subject',
    'defining_content_or_narrative_driver',
    'causal_or_contextual_background',
    'example_or_meta',
    'other',
}
text=' '.join(stage['instructions']).lower()
assert 'constituent content' in text
assert 'do not override the frozen subject' in text
assert 'story-defining content' in text
assert 'causal/contextual background only' in text
print('v0.22.2 subject-relation semantic invariants passed')
