"""Static contract for v0.28.0 r8 freeze overlay."""
from pathlib import Path

R = Path(__file__).resolve().parents[1]
F = (R / "evals/freeze_v0_28_r8_release_candidate.py").read_text(encoding="utf-8")
P = (R / "run_v028_r8_freeze.ps1").read_text(encoding="utf-8-sig")

for token in [
    'release_status": "FROZEN_BEFORE_FINAL_HOLDOUT"',
    '"decision": "READY_FOR_FINAL_HOLDOUT_EXECUTION_REVIEW"',
    "semantic_relevance_v0.28_r8_targeted_state.v1.0.0.json",
    "semantic_relevance_v0.28_r8_full210_state.v1.0.0.json",
    "semantic_relevance_v0.28_r8_stability_state.v1.0.0.json",
    "exactly three fresh stability run directories recorded",
    "all 20 stability cases are identical across all three fresh runs",
    "independent final holdout remains LOCKED_DO_NOT_RUN",
    "judge_run_count = 0",
    "NO JUDGE CALLS WERE MADE BY THIS FREEZE TOOL",
]:
    assert token in F, token

for token in [
    "verify_v0_28_r8_package.py",
    "build_v0_28_r8_development_dataset.py",
    "analyze_v0_28_r8_targeted.py",
    "analyze_v0_28_r8_development.py",
    "analyze_v0_28_r8_stability.py",
    "freeze_v0_28_r8_release_candidate.py",
]:
    assert token in P, token

compile(F, str(R / "evals/freeze_v0_28_r8_release_candidate.py"), "exec")
print("v0.28 r8 freeze overlay contract passed")
