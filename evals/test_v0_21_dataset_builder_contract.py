from pathlib import Path
p=Path(__file__).resolve().parent/'build_v0_21_development_dataset.py'
s=p.read_text(encoding='utf-8')
assert 'semantic_relevance_fresh_holdout' not in s
assert 'FRESH_VALIDATION' in s
assert 'holdout_read' in s
assert 'semantic_relevance_v0.21_development.v2.0.0.csv' in s
print('v0.21 dataset-builder contract test passed.')
