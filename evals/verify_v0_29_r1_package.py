"""Static verifier for semantic relevance v0.29.0 r1. No judge calls.

Verifier-fix revision:
- Do not hard-code the generated adjudication-manifest or adjudicated-review
  file SHA.
- Instead, verify the adjudication manifest's exact contents and verify that
  its internally recorded artifact hashes match the actual files.
"""
from __future__ import annotations

import hashlib
import json
import py_compile
import subprocess
import sys
from pathlib import Path

R = Path(__file__).resolve().parents[1]

EXPECTED = {
    "evals/semantic_relevance_facet_judge.py":
        "69a0e01d3ef81b83f5a4702df1a6f2048aff7170bf51766c0e757942f843655d",
    "evals/semantic_relevance_facet_scoring.py":
        "8d40b3f773e3c763d9b6436ff036e63966ddc2b3877ecf4bf5f8989664d2587b",
    "evals/facets/semantic_relevance/semantic_relevance_query_facets.v0.10.0.json":
        "f99bb372778eb6941254077d9e1749ebe0198da67c90ef25264058fa469b2d81",
    "evals/judge_configs/semantic_relevance_judge.v0.29.0-r1.json":
        "5b88447f58cb623a4c633a777f58bbdc16f77ed96690c4e426092e4c98048ebe",
    "evals/datasets/semantic_relevance_v0.29_regression_manifest.v2.0.0.json":
        "202f323992333de2c7cf5f819039280384bc39ca17766180a107496b5923da68",
}

DATASET_SHA = "cfcc465497079944fec6da17f14bbe5f3350d9385ea9ddc646610b7751d72da7"
SOURCE_DATASET_SHA = "c8087f1ed80fce92e2b666f61732f727e0bfe6342ad5960227cd8217b92122c6"
SOURCE_REVIEW_SHA = "2155f3defb582b7830a7b9c04d79619725909515ab4d8e6faf1eb33f2a25eced"
R8_CLOSEOUT_SHA = "24f90865503f4572d54e168d7f2cca33647a7c9a3c294bfb1ab727fd544cb2de"

EXPECTED_REVISIONS = {
    "U4_Q03_T01": (0, 2),
    "U4_Q06_T01": (2, 0),
    "U4_Q09_T04": (1, 2),
}
EXPECTED_UNCHANGED = {
    "U4_Q03_T02": 2,
    "U4_Q04_T02": 0,
}
EXPECTED_REPAIR_TARGETS = {
    "U4_Q03_T02",
    "U4_Q04_T02",
    "U4_Q09_T04",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(cond: bool, message: str) -> None:
    if not cond:
        raise AssertionError(message)
    print("PASS ", message)


def main() -> None:
    print("v0.29.0 r1 package verification")
    print("--------------------------------")
    print("JUDGE EXECUTION: DISABLED")

    for rel, expected in EXPECTED.items():
        path = R / rel
        require(path.exists(), f"artifact exists: {rel}")
        require(sha(path) == expected, f"hash pinned: {rel}")

    dataset = R / "evals/datasets/semantic_relevance_v0.29_development.v6.1.0.csv"
    adj = R / "evals/datasets/semantic_relevance_v0.29_adjudications.v1.0.0.json"
    review = R / "evals/datasets/semantic_relevance_v0.29_severe_review.adjudicated.v1.0.0.csv"
    source_review = R / "evals/datasets/semantic_relevance_v0.29_severe_review.v1.0.0.csv"
    closeout = R / "evals/releases/semantic_relevance_v0.28.0_r8_final_evaluation_closeout.json"

    require(dataset.exists(), "adjudicated 240-case v0.29 development dataset exists")
    require(sha(dataset) == DATASET_SHA,
            "adjudicated 240-case v0.29 development dataset hash pinned")
    require(adj.exists(), "v0.29 adjudication manifest exists")
    require(review.exists(), "adjudicated five-case review exists")
    require(source_review.exists(), "source five-case review exists")
    require(sha(source_review) == SOURCE_REVIEW_SHA,
            "source five-case review hash pinned")
    require(closeout.exists() and sha(closeout) == R8_CLOSEOUT_SHA,
            "v0.28 r8 closeout hash pinned")

    payload = json.loads(adj.read_text(encoding="utf-8-sig"))

    require(payload.get("schema_version") == "1.0.0",
            "adjudication manifest schema version pinned")
    require(payload.get("lineage") == "semantic_relevance_v0.29.0",
            "adjudication manifest lineage pinned")
    require(payload.get("source_dataset_sha256") == SOURCE_DATASET_SHA,
            "adjudication manifest source dataset SHA pinned")
    require(payload.get("output_dataset_sha256") == DATASET_SHA,
            "adjudication manifest output dataset SHA pinned")
    require(payload.get("source_review_sha256") == SOURCE_REVIEW_SHA,
            "adjudication manifest source review SHA pinned")
    require(payload.get("judge_calls_made") is False,
            "adjudication records no judge calls")
    require(payload.get("semantic_changes_made") is False,
            "adjudication records no semantic changes")

    require(
        payload.get("output_dataset") ==
        "evals\\datasets\\semantic_relevance_v0.29_development.v6.1.0.csv",
        "adjudication manifest output dataset path pinned"
    )
    require(
        payload.get("adjudicated_review") ==
        "evals\\datasets\\semantic_relevance_v0.29_severe_review.adjudicated.v1.0.0.csv",
        "adjudication manifest adjudicated-review path pinned"
    )
    require(
        payload.get("adjudicated_review_sha256") == sha(review),
        "adjudicated-review actual SHA matches adjudication manifest"
    )

    revisions = {}
    for rec in payload.get("approved_label_revisions", []):
        revisions[str(rec["case_id"])] = (int(rec["old_score"]), int(rec["new_score"]))
    require(revisions == EXPECTED_REVISIONS,
            "exact three approved adjudication score transitions pinned")

    unchanged = {}
    for rec in payload.get("reviewed_without_label_change", []):
        unchanged[str(rec["case_id"])] = int(rec["score"])
    require(unchanged == EXPECTED_UNCHANGED,
            "exact two reviewed-but-unchanged labels pinned")

    require(
        set(payload.get("semantic_repair_targets", [])) == EXPECTED_REPAIR_TARGETS,
        "semantic repair targets pinned"
    )

    lock_path = R / "evals/datasets/semantic_relevance_final_holdout_lock.v4.0.0.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8-sig"))
    require(lock.get("status") == "FINAL_HOLDOUT_COMPLETED",
            "v0.28 final holdout remains FINAL_HOLDOUT_COMPLETED")
    require(int(lock.get("judge_run_count", -1)) == 1,
            "v0.28 final holdout remains consumed exactly once")
    require(lock.get("final_holdout_decision") == "FINAL_HOLDOUT_REVIEW",
            "v0.28 final decision preserved")

    boundary = R / "evals/datasets/semantic_relevance_v0.29_evidence_boundary.v1.0.0.json"
    require(boundary.exists(), "v0.29 evidence boundary exists")
    boundary_payload = json.loads(boundary.read_text(encoding="utf-8-sig"))
    require(boundary_payload.get("new_independent_evidence_required") is True,
            "v0.29 still requires new independent evidence after candidate freeze")

    for rel in [
        "evals/semantic_relevance_facet_judge.py",
        "evals/semantic_relevance_facet_scoring.py",
        "evals/run_judge_v0_29_r1_development.py",
        "evals/analyze_v0_29_r1_targeted.py",
        "evals/test_v0_29_r1_contract.py",
        "evals/test_v0_29_r1_localized_guards.py",
    ]:
        py_compile.compile(str(R / rel), doraise=True)

    for script in [
        "evals/test_v0_28_r5_localized_guards.py",
        "evals/test_v0_28_r8_localized_guards.py",
        "evals/test_v0_29_r1_contract.py",
        "evals/test_v0_29_r1_localized_guards.py",
    ]:
        subprocess.run([sys.executable, str(R / script)], check=True, cwd=R)

    print("v0.29.0 r1 package verification passed.")
    print("No judge calls performed.")


if __name__ == "__main__":
    main()
