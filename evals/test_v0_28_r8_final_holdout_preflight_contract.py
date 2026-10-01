"""Static contract for the v0.28.0 r8 final-holdout preflight overlay."""
from __future__ import annotations
import json
from pathlib import Path

R = Path(__file__).resolve().parents[1]
P = R / "evals/releases/semantic_relevance_v0.28.0_r8_final_holdout_protocol.v1.0.0.json"
V = R / "evals/preflight_v0_28_r8_final_holdout.py"
PS = R / "preflight_v028_r8_final_holdout.ps1"

def main():
    protocol = json.loads(P.read_text(encoding="utf-8"))
    assert protocol["candidate_release"] == "semantic_relevance_v0.28.0-r8"
    assert protocol["evidence_role"] == "independent_final_holdout"
    assert protocol["execution_policy"]["fresh_runs_permitted"] == 1
    assert protocol["execution_policy"]["case_count"] == 30
    assert protocol["execution_policy"]["recovery_policy"] == "resume_recorded_run_only"
    assert protocol["execution_policy"]["confirmation_token"] == "FINAL_V0_28_R8"
    assert protocol["execution_policy"]["rerun_after_completed_result_permitted"] is False
    assert protocol["execution_policy"]["retune_after_result_permitted"] is False
    assert protocol["execution_policy"]["relabel_after_result_permitted"] is False
    assert len(protocol["preregistered_checks"]) == 8

    verifier = V.read_text(encoding="utf-8")
    for token in [
        "c25759ccaba3a3258552a5c138170821b3a9afd48acb968859a5943b4fc91cab",
        "0c7b55065629b6c1fbdb3f0d530cf8bdc8e15011b0d213ab4be6259b95a43e85",
        "FROZEN_BEFORE_FINAL_HOLDOUT",
        "READY_FOR_FINAL_HOLDOUT_EXECUTION_REVIEW",
        "INDEPENDENT_VALIDATION_COMPLETED",
        "REVIEW_STOP_FINAL_HOLDOUT",
        "LOCKED_DO_NOT_RUN",
        "judge_run_count = 0",
        "repository final-holdout finality marker does not exist",
        "FINAL_V0_28_R8",
    ]:
        assert token in verifier, token

    ps = PS.read_text(encoding="utf-8-sig")
    for token in [
        "verify_v0_28_r8_package.py",
        "preflight_v0_28_r8_final_holdout.py",
        "JUDGE EXECUTION: DISABLED",
        "DO NOT RUN FINAL HOLDOUT",
    ]:
        assert token in ps, token

    compile(verifier, str(V), "exec")
    print("v0.28 r8 final-holdout preflight contract passed")

if __name__ == "__main__":
    main()
