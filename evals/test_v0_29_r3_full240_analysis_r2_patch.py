from pathlib import Path

R = Path(__file__).resolve().parents[1]
A = (R / "evals/analyze_v0_29_r3_development.py").read_text(encoding="utf-8")
W = (R / "analyze_v029_r3_full240.ps1").read_text(encoding="utf-8")

assert 'closeout.get("closeout_status")' in A
assert 'closeout.get("final_holdout_decision")' in A
assert '"consumed_v0_26_final_holdout_30"' in A
assert '"consumed_v0_27_final_holdout_30"' not in A
assert 'r8_final_holdout_may_be_reused_as_independent_evidence' in A

assert W.lstrip().startswith("param(")
assert W.index("param(") < W.index('$ErrorActionPreference = "Stop"')

compile(A, str(R / "evals/analyze_v0_29_r3_development.py"), "exec")
print("v0.29 r3 full-240 analysis r2 patch contract passed")
