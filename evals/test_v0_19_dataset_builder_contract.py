from pathlib import Path
P=Path(__file__).resolve().parent/"build_v0_19_development_dataset.py"
s=P.read_text(encoding="utf-8")
assert 'semantic_relevance_v0.18_development.v1.0.0.csv' in s
assert 'semantic_relevance_fresh_validation.v2.0.0.csv' in s
assert 'semantic_relevance_v0.19_development.v2.0.0.csv' in s
# The locked holdout filename/path must never be opened or referenced as an input.
assert 'semantic_relevance_fresh_holdout' not in s
assert 'DO_NOT_RUN_YET' not in s
assert 'if len(old) != 60' in s
assert 'if len(fresh) != 30' in s
assert 'len(combined) != 90' in s
print("v0.19 dataset-builder contract test passed.")
