from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
EVALS = REPO_ROOT / "evals"

SOURCE = REPO_ROOT / "gradio-dashboard.py"
FIX_MANIFEST = EVALS / "system_evaluation/semantic_retrieval_order_fix_v2_manifest.json"
REGRESSION_TEST = EVALS / "system_evaluation/test_semantic_retrieval_order_regression.py"
LOCK = EVALS / "system_evaluation/semantic_recommender_v2_lock.json"

EXPECTED_SOURCE_SHA = "cdd0458ff23bd669185a8c882ceb132438d037af1a29f4dd60bd7fef9c405b60"
BASELINE_SOURCE_SHA = "b8f7c1223a525345a7e278fd29e0cd32a5e0c644a9a0b88e5cc9ac319f749974"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_output(*args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return None


def main() -> None:
    if not SOURCE.exists():
        raise FileNotFoundError(SOURCE)
    if not FIX_MANIFEST.exists():
        raise FileNotFoundError(FIX_MANIFEST)
    if not REGRESSION_TEST.exists():
        raise FileNotFoundError(REGRESSION_TEST)

    actual_source_sha = sha256(SOURCE)
    if actual_source_sha != EXPECTED_SOURCE_SHA:
        raise ValueError(
            "Corrected recommender source hash mismatch: "
            f"expected={EXPECTED_SOURCE_SHA} actual={actual_source_sha}"
        )

    fix = json.loads(FIX_MANIFEST.read_text(encoding="utf-8-sig"))

    if fix.get("status") != "APPLIED":
        raise ValueError("Semantic-order fix manifest is not APPLIED.")
    if fix.get("fix_id") != "semantic-retrieval-order-v2":
        raise ValueError("Unexpected semantic-order fix id.")
    if fix.get("before_sha256") != BASELINE_SOURCE_SHA:
        raise ValueError("Fix manifest baseline SHA mismatch.")
    if fix.get("after_sha256") != EXPECTED_SOURCE_SHA:
        raise ValueError("Fix manifest corrected SHA mismatch.")
    if fix.get("category_behavior_changed") is not False:
        raise ValueError("Category behavior unexpectedly changed.")
    if fix.get("tone_behavior_changed") is not False:
        raise ValueError("Tone behavior unexpectedly changed.")
    if fix.get("judge_changed") is not False:
        raise ValueError("Judge unexpectedly changed.")
    if fix.get("evaluation_gates_changed") is not False:
        raise ValueError("Evaluation gates unexpectedly changed.")
    if fix.get("evaluation_queries_changed") is not False:
        raise ValueError("Evaluation queries unexpectedly changed.")

    commit = git_output("rev-parse", "HEAD")
    status = git_output("status", "--porcelain")

    payload = {
        "status": "FROZEN_FOR_SYSTEM_EVALUATION",
        "candidate": "semantic-recommender-v2-orderfix",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_file": "gradio-dashboard.py",
        "source_sha256": actual_source_sha,
        "baseline_v1_source_sha256": BASELINE_SOURCE_SHA,
        "fix_id": fix["fix_id"],
        "fix_manifest_sha256": sha256(FIX_MANIFEST),
        "regression_test_file": str(REGRESSION_TEST.relative_to(REPO_ROOT)).replace("\\", "/"),
        "regression_test_sha256": sha256(REGRESSION_TEST),
        "baseline_system_eval_run": (
            "evals/runs/semantic_relevance_system_eval_v1/"
            "20261002T152856Z_recommendations"
        ),
        "controlled_comparison_policy": {
            "same_queries": True,
            "same_judge": True,
            "same_rubric": True,
            "same_scoring_logic": True,
            "same_initial_top_k": 50,
            "same_final_top_k": 10,
            "same_bootstrap_method": True,
            "same_release_gates": True,
            "only_intended_system_change": (
                "preserve vector-store semantic rank through metadata join"
            ),
        },
        "git_commit_at_freeze": commit,
        "git_worktree_clean_at_freeze": (status == "") if status is not None else None,
        "git_status_porcelain_at_freeze": status,
    }

    LOCK.write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )

    print("SEMANTIC RECOMMENDER V2 FREEZE: PASS")
    print(f"Source SHA-256: {actual_source_sha}")
    print(f"Fix manifest SHA-256: {sha256(FIX_MANIFEST)}")
    print(f"Regression test SHA-256: {sha256(REGRESSION_TEST)}")
    print(f"Git commit: {commit or '<unavailable>'}")
    print(
        "Git worktree clean: "
        + (
            str(status == "")
            if status is not None
            else "<unavailable>"
        )
    )
    print(f"Lock: {LOCK}")
    print("Recommender calls performed: 0")
    print("Judge calls performed: 0")


if __name__ == "__main__":
    main()
