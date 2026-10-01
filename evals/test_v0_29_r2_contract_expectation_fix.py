from pathlib import Path

R = Path(__file__).resolve().parents[1]
P = R / "evals/test_v0_29_r2_contract.py"
T = P.read_text(encoding="utf-8")

assert 'encoding="utf-8-sig"' in T
assert 'q04 = reg["targeted_expectations"]["U2_Q04_T70"]' in T
assert 'assert int(q04["judge_min"]) == 3' in T
assert '"judge_max" not in q04 or int(q04["judge_max"]) >= 3' in T
assert '== {"judge_min": 3, "judge_max": 4}' not in T
compile(T, str(P), "exec")
print("v0.29 r2 contract-expectation fix contract passed")
