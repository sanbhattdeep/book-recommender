from pathlib import Path
import json
ROOT=Path(__file__).resolve().parents[1]
config=json.loads((ROOT/'evals/judge_configs/semantic_relevance_judge.v0.22.1.json').read_text())
stage=config['hard_exclusion_precheck_stage']
text='\n'.join(stage['instructions'])
assert 'negative semantic guard' in text
assert 'not a lexical veto' in text
assert 'Generic-hero exclusions' in text
print('v0.22.0 hard-exclusion precheck contract passed')
