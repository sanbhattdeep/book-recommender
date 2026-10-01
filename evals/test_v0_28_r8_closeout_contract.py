from pathlib import Path
R = Path(__file__).resolve().parents[1]
C = (R / "evals/closeout_v0_28_r8_final_evaluation.py").read_text(encoding="utf-8")
P = (R / "closeout_v028_r8.ps1").read_text(encoding="utf-8-sig")
for token in [
    '"EVALUATED_NOT_FINAL_QUALIFIED"',
    '"FINAL_HOLDOUT_REVIEW"',
    '"semantic_relevance_v0.29.0"',
    '"consumed_final_evidence_may_be_diagnostic_or_development_only"',
    '"r8_final_holdout_may_be_reused_as_independent_evidence": False',
    '"r8_final_holdout_may_be_rerun_for_r8": False',
    '"r8_may_be_retuned_after_final_result": False',
    "new unseen human-labelled independent evidence set",
    "FAILED =",
    "SEVERE =",
    "NO JUDGE CALLS WERE MADE BY THIS CLOSEOUT TOOL",
]:
    assert token in C, token
for token in [
    "verify_v0_28_r8_final_holdout_execution_package.py",
    "closeout_v0_28_r8_final_evaluation.py",
    "JUDGE EXECUTION: DISABLED",
    "DO NOT RUN FINAL HOLDOUT",
    "v0.29.0",
]:
    assert token in P, token
compile(C, str(R / "evals/closeout_v0_28_r8_final_evaluation.py"), "exec")
print("v0.28 r8 final-evaluation closeout contract passed")
