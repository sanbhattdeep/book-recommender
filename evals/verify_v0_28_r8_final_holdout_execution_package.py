"""Verify the frozen r8 final-holdout execution package. No judge calls."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

R = Path(__file__).resolve().parents[1]
E = R / "evals"
D = E / "datasets"
REL = E / "releases"
RUN_ROOT = E / "runs" / "semantic_relevance_v0_28_final_holdout"

RELEASE = REL / "semantic_relevance_v0.28.0_r8_release_candidate.json"
PROTOCOL = REL / "semantic_relevance_v0.28.0_r8_final_holdout_protocol.v1.0.0.json"
HOLDOUT = D / "semantic_relevance_final_holdout.v4.0.0.DO_NOT_RUN_YET.csv"
LOCK = D / "semantic_relevance_final_holdout_lock.v4.0.0.json"
VALIDATION_LOCK = D / "semantic_relevance_validation_lock.v4.0.0.json"
MARKER = RUN_ROOT / "FINAL_HOLDOUT_STARTED.json"

EXPECTED = {
    RELEASE: "c25759ccaba3a3258552a5c138170821b3a9afd48acb968859a5943b4fc91cab",
    PROTOCOL: "7aa066815dd7c084be20773daec70c09981feb6c646061b658ed4fea1fca87f8",
    HOLDOUT: "0c7b55065629b6c1fbdb3f0d530cf8bdc8e15011b0d213ab4be6259b95a43e85",
}

SEMANTIC = {
    "evals/semantic_relevance_facet_judge.py": "e4efec54d03601e3016b376741c7f14fc1259ab3d63d1fdff75925cc2bfb43c1",
    "evals/semantic_relevance_facet_scoring.py": "8447a75aee29d1ebf34147e625a7fe8c1f8ccf4b409029e917ec4c47285e8808",
    "evals/facets/semantic_relevance/semantic_relevance_query_facets.v0.9.13.json": "0952d80ebed8fbdebfb3b625f5648be9ccc08148d87d6753adc83d010953e289",
    "evals/judge_configs/semantic_relevance_judge.v0.28.0-r8.json": "bdc4157b457e3453ff94b860cb45022a7c7c83f311432ef9841a1c1fe5a9bb8d",
    "evals/datasets/semantic_relevance_v0.28_regression_manifest.v1.7.0.json": "da8d7ffe0f9131dd5ab976a0af72d7c478ee592b4cf1b495e105e483d908c585",
    "evals/rubrics/semantic_relevance/semantic_relevance_rubric.v0.1.0.json": "658fe8ef49f1b73e4d6d6bdb143ab4d43a040ed7ea347fc2d4a41a3051ee7d2e",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def require(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)
    print("PASS ", msg)


def main() -> None:
    print("v0.28.0 r8 final-holdout execution package verification")
    print("--------------------------------------------------------")
    print("JUDGE EXECUTION: DISABLED")

    for path, expected in EXPECTED.items():
        require(path.exists(), f"required frozen file exists: {path.relative_to(R)}")
        require(sha(path) == expected, f"hash pinned: {path.relative_to(R)}")

    for rel, expected in SEMANTIC.items():
        path = R / rel
        require(path.exists(), f"semantic artifact exists: {rel}")
        require(sha(path) == expected, f"semantic hash pinned: {rel}")

    release = load(RELEASE)
    require(release.get("release_candidate") == "semantic_relevance_v0.28.0-r8",
            "release candidate identity correct")
    require(release.get("release_status") == "FROZEN_BEFORE_FINAL_HOLDOUT",
            "release status frozen before final holdout")

    protocol = load(PROTOCOL)
    policy = protocol["execution_policy"]
    require(policy["fresh_runs_permitted"] == 1, "protocol permits exactly one fresh run")
    require(policy["recovery_policy"] == "resume_recorded_run_only",
            "protocol requires resume-only recovery")
    require(policy["rerun_after_completed_result_permitted"] is False,
            "protocol forbids completed rerun")

    validation = load(VALIDATION_LOCK)
    require(validation.get("status") == "INDEPENDENT_VALIDATION_COMPLETED",
            "historical validation remains completed")
    require(int(validation.get("judge_run_count", -1)) == 1,
            "historical validation judge_run_count remains 1")

    lock = load(LOCK)
    require(lock.get("evidence_role") == "independent_final_holdout",
            "final-holdout lock role correct")

    status = lock.get("status")
    count = int(lock.get("judge_run_count", -1))
    marker_exists = MARKER.exists()

    if status == "LOCKED_DO_NOT_RUN":
        require(count == 0, "pre-start final-holdout judge_run_count = 0")
        require(not marker_exists, "pre-start finality marker absent")
        require(lock.get("finality_marker_present") is False,
                "pre-start lock records no finality marker")
    elif status in {"FINAL_HOLDOUT_STARTED", "FINAL_HOLDOUT_COMPLETED"}:
        require(count == 1, "started/completed final-holdout judge_run_count = 1")
        require(marker_exists, "started/completed finality marker exists")
        require(lock.get("finality_marker_present") is True,
                "started/completed lock records finality marker")
        marker = load(MARKER)
        require(marker.get("release_candidate_id") == "semantic_relevance_v0.28.0-r8",
                "marker release candidate identity correct")
        require(marker.get("dataset_sha256") == "0c7b55065629b6c1fbdb3f0d530cf8bdc8e15011b0d213ab4be6259b95a43e85",
                "marker final-holdout SHA pinned")
        require(marker.get("release_manifest_sha256") == "c25759ccaba3a3258552a5c138170821b3a9afd48acb968859a5943b4fc91cab",
                "marker release-manifest SHA pinned")
        require(marker.get("final_holdout_protocol_sha256") == "7aa066815dd7c084be20773daec70c09981feb6c646061b658ed4fea1fca87f8",
                "marker protocol SHA pinned")
        require(str(lock.get("run_directory")) == str(marker.get("run_directory")),
                "lock and marker point to same canonical run")
        if status == "FINAL_HOLDOUT_COMPLETED":
            require(marker.get("completed") is True,
                    "completed marker records completed=true")
            require(lock.get("final_holdout_decision") in {"FINAL_HOLDOUT_PASS", "FINAL_HOLDOUT_REVIEW"},
                    "completed lock records a final decision")
    else:
        raise AssertionError(f"Unexpected final-holdout lifecycle status: {status!r}")

    print("v0.28.0 r8 final-holdout execution package verification passed.")
    print("NO JUDGE CALLS WERE MADE.")


if __name__ == "__main__":
    main()
