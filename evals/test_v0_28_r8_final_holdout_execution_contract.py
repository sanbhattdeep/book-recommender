"""Static safety contract for one-time r8 final-holdout execution."""
from __future__ import annotations
from pathlib import Path

R = Path(__file__).resolve().parents[1]
RUNNER = (R / "evals/run_judge_v0_28_r8_final_holdout.py").read_text(encoding="utf-8")
ANALYZER = (R / "evals/analyze_v0_28_r8_final_holdout.py").read_text(encoding="utf-8")
PS = (R / "run_v028_r8_final_holdout.ps1").read_text(encoding="utf-8-sig")

for token in [
    'CONFIRM_TOKEN = "FINAL_V0_28_R8"',
    'FINALITY_MARKER = FINAL_RUNS_DIR / "FINAL_HOLDOUT_STARTED.json"',
    '"FINAL_HOLDOUT_STARTED"',
    '"FINAL_HOLDOUT_COMPLETED"',
    "judge_run_count",
    "Fresh targeted final-holdout runs are forbidden",
    "Final holdout was already started. Resume only",
    "write_json_atomic(FINALITY_MARKER, marker)",
    "transition_final_lock_started(dataset, release_manifest, protocol, run_dir)",
]:
    assert token in RUNNER, token

# Marker + lock transition must be textually before the base runner can execute.
assert RUNNER.index("write_json_atomic(FINALITY_MARKER, marker)") < RUNNER.index("base.main()")
assert RUNNER.index("transition_final_lock_started(dataset, release_manifest, protocol, run_dir)") < RUNNER.index("base.main()")

for token in [
    'protocol["decision_policy"]["all_checks_pass"]',
    'protocol["decision_policy"]["any_check_fails"]',
    '"FINAL_HOLDOUT_COMPLETED"',
    '"judge_run_count": 1',
    "This result is final evidence whether PASS or REVIEW",
    "must not be rerun",
]:
    assert token in ANALYZER, token

for token in [
    "ConfirmFinalHoldout",
    "FINAL_V0_28_R8",
    "preflight_v028_r8_final_holdout.ps1",
    "verify_v0_28_r8_final_holdout_execution_package.py",
    "run_judge_v0_28_r8_final_holdout.py",
    "analyze_v0_28_r8_final_holdout.py",
    "FINAL_HOLDOUT_STARTED.json",
    "resume recorded run only",
    "Do not retune, relabel, or rerun",
]:
    assert token in PS, token

compile(RUNNER, str(R / "evals/run_judge_v0_28_r8_final_holdout.py"), "exec")
compile(ANALYZER, str(R / "evals/analyze_v0_28_r8_final_holdout.py"), "exec")
print("v0.28 r8 one-time final-holdout execution contract passed")
