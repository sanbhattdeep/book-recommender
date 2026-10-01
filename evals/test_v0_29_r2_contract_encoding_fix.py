from pathlib import Path

R = Path(__file__).resolve().parents[1]
P = R / "evals/test_v0_29_r2_contract.py"
T = P.read_text(encoding="utf-8")

assert 'encoding="utf-8-sig"' in T
assert "def read_json" in T
assert "def read_utf8" in T
assert ".read_text()" not in T
assert 'semantic_relevance_query_facets.v0.10.1.json' in T
assert 'semantic_relevance_judge.v0.29.0-r2.json' in T
assert 'semantic_relevance_v0.29_regression_manifest.v2.1.0.json' in T

compile(T, str(P), "exec")
print("v0.29 r2 contract-encoding fix contract passed")
