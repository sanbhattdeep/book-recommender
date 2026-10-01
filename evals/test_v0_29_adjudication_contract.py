from pathlib import Path
R = Path(__file__).resolve().parents[1]
A = (R/"evals/apply_v0_29_adjudications.py").read_text(encoding="utf-8")
B = (R/"evals/recalculate_v0_29_adjudicated_r8_baseline.py").read_text(encoding="utf-8")
P = (R/"apply_v029_adjudications.ps1").read_text(encoding="utf-8-sig")

for token in [
    "semantic_relevance_v0.29_development.v6.1.0.csv",
    '"U4_Q03_T01"',
    '"old_score": 0',
    '"new_score": 2',
    '"U4_Q06_T01"',
    '"old_score": 2',
    '"new_score": 0',
    '"U4_Q09_T04"',
    '"old_score": 1',
    "label_spec_tension_plus_judge_overpromotion",
    "only the three approved adjudication rows changed",
    "all non-label fields remain unchanged across all 240 rows",
    "human score/reason unchanged for all 237 non-adjudicated rows",
    "all output rows use dataset_version 6.1.0",
    "NO JUDGE CALLS WERE MADE",
    "NO SEMANTIC CHANGES WERE MADE",
]:
    assert token in A, token

for token in [
    "U4_Q03_T02",
    "U4_Q04_T02",
    "U4_Q09_T04",
    "post-adjudication U4 severe cases are exactly Q03_T02 and Q04_T02",
    "one-point over-promotion repair target",
]:
    assert token in B, token

for token in [
    "test_v0_29_adjudication_contract.py",
    "apply_v0_29_adjudications.py",
    "recalculate_v0_29_adjudicated_r8_baseline.py",
    "JUDGE EXECUTION: DISABLED",
    "SEMANTIC CHANGES: NONE",
]:
    assert token in P, token

compile(A, str(R/"evals/apply_v0_29_adjudications.py"), "exec")
compile(B, str(R/"evals/recalculate_v0_29_adjudicated_r8_baseline.py"), "exec")
print("v0.29 adjudication contract passed")
