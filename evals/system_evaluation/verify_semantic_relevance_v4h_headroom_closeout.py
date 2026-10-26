from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "evals/runs/semantic_relevance_v4_holdout/20261022T103659Z_retrieval"
SYS = ROOT / "evals/system_evaluation"

FAILURE_LOCK = RUN / "v4h_failure_analysis.lock.json"
FAILURE_QUERY_DIAGNOSTICS = RUN / "v4h_failure_query_diagnostics.csv"
FINAL_RELEASE_LOCK = RUN / "v4h_final_release_analysis.lock.json"
STAGE_A_LOCK = RUN / "v4h_headroom_stage_a_classification.lock.json"
STAGE_B_LOCK = RUN / "v4h_headroom_stage_b_classification.lock.json"
STAGE_B_OVERALL = RUN / "v4h_headroom_overall_query_disposition.csv"
CONTRACT = SYS / "semantic_relevance_v4h_headroom_closeout_contract.v0.1.0.json"

QUERY_COHORTS = RUN / "v4h_headroom_diagnostic_closeout_query_cohorts.csv"
SUMMARY = RUN / "v4h_headroom_diagnostic_closeout_summary.json"
REPORT = RUN / "v4h_headroom_diagnostic_closeout_report.md"
METADATA = RUN / "v4h_headroom_diagnostic_closeout_metadata.json"
CLOSEOUT_LOCK = RUN / "v4h_headroom_diagnostic_closeout.lock.json"

EXPECTED = {
    "failure_lock": "4e0f10f3d70fae6154c20d32622c21f437749d30835e892736cafa7d6ab3fc90",
    "final_release_lock": "7e7d73920452ec39ea72947d2b3762a02aaa1b2bb0bcd7470c7e21dbb3a25ab0",
    "stage_a_lock": "906b5975345c9d9eed1273ecc4ccd0611880b14d6326bf9f0e74b46f7d63d948",
    "stage_b_lock": "d942a01c9af384e4c0a9e065c7e441f4d2d123229110d2606b2ec722a657f55f",
    "contract": "318cbdad92e36635e12b998ad95a553310877107dc2355ee980e3de036a87d15",
}

HEADROOM_11_50 = ["V4H07", "V4H09", "V4H19", "V4H22"]
NO_CLEAR_FULL_UNION = ["V4H10", "V4H11", "V4H13", "V4H16", "V4H21", "V4H23"]
TARGETS = HEADROOM_11_50 + NO_CLEAR_FULL_UNION


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def req(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def verify_raw(path: Path, expected: str, label: str) -> None:
    req(path.exists(), f"Missing {label}: {path}")
    actual = sha256(path)
    req(actual == expected, f"{label} hash mismatch: expected={expected} actual={actual}")
    print(f"PASS  {label} raw SHA-256 verified")


def verify_bound_output(lock: dict[str, Any], path: Path, label: str) -> None:
    expected = lock.get("outputs", {}).get(path.name)
    req(isinstance(expected, str) and len(expected) == 64,
        f"{label} is not bound by its lock: {path.name}")
    verify_raw(path, expected, label)


def verify_boundary() -> dict[str, Any]:
    verify_raw(FAILURE_LOCK, EXPECTED["failure_lock"], "V4H failure-analysis lock")
    verify_raw(FINAL_RELEASE_LOCK, EXPECTED["final_release_lock"], "V4H final release-analysis lock")
    verify_raw(STAGE_A_LOCK, EXPECTED["stage_a_lock"], "Stage-A classification lock")
    verify_raw(STAGE_B_LOCK, EXPECTED["stage_b_lock"], "Stage-B classification lock")
    verify_raw(CONTRACT, EXPECTED["contract"], "headroom closeout contract")

    failure = load_json(FAILURE_LOCK)
    req(failure.get("status") == "V4H_POST_RELEASE_FAILURE_ANALYSIS_FROZEN",
        "Unexpected V4H failure-analysis lock status")
    req(failure.get("frozen_release_decision") == "FAIL", "Frozen release decision changed")
    req(failure.get("release_decision_modified") is False, "Failure analysis changed release decision")
    req(failure.get("v4h_consumed") is True, "V4H is not marked consumed")
    verify_bound_output(failure, FAILURE_QUERY_DIAGNOSTICS, "V4H failure query diagnostics")

    final = load_json(FINAL_RELEASE_LOCK)
    req(final.get("status") == "V4H_FINAL_RELEASE_ANALYSIS_FROZEN",
        "Unexpected V4H final release-analysis lock status")
    req(final.get("v4b_final_release_decision") == "FAIL", "V4B release decision changed")

    stage_a = load_json(STAGE_A_LOCK)
    req(stage_a.get("status") == "V4H_HEADROOM_STAGE_A_CLASSIFICATION_FROZEN",
        "Unexpected Stage-A classification lock status")
    req(stage_a.get("classification_counts") == {
        "HEADROOM_FOUND_11_50": 4,
        "NO_CLEAR_THROUGH_50": 6,
    }, f"Unexpected Stage-A classification counts: {stage_a.get('classification_counts')}")
    req(stage_a.get("release_decision_modified") is False, "Stage-A changed release decision")

    stage_b = load_json(STAGE_B_LOCK)
    req(stage_b.get("status") == "V4H_HEADROOM_STAGE_B_CLASSIFICATION_FROZEN",
        "Unexpected Stage-B classification lock status")
    req(stage_b.get("stage_b_classification_counts") == {
        "HEADROOM_FOUND_51_END": 0,
        "NO_CLEAR_THROUGH_FULL_UNION": 6,
    }, f"Unexpected Stage-B classification counts: {stage_b.get('stage_b_classification_counts')}")
    req(stage_b.get("overall_10_query_disposition_counts") == {
        "HEADROOM_FOUND_11_50": 4,
        "HEADROOM_FOUND_51_END": 0,
        "NO_CLEAR_THROUGH_FULL_UNION": 6,
    }, f"Unexpected overall headroom counts: {stage_b.get('overall_10_query_disposition_counts')}")
    req(stage_b.get("release_decision_modified") is False, "Stage-B changed release decision")
    req(stage_b.get("new_judge_calls") == 0 and stage_b.get("new_model_calls") == 0,
        "Stage-B classification lock reports judge/model calls")
    verify_bound_output(stage_b, STAGE_B_OVERALL, "10-query overall headroom disposition")

    overall = pd.read_csv(STAGE_B_OVERALL, dtype=str, keep_default_na=False, encoding="utf-8")
    req(len(overall) == 10, "Expected exactly 10 overall headroom rows")
    req(not overall["query_id"].duplicated().any(), "Duplicate overall headroom query_id")
    found_11 = overall.loc[overall["overall_headroom_disposition"] == "HEADROOM_FOUND_11_50", "query_id"].tolist()
    found_51 = overall.loc[overall["overall_headroom_disposition"] == "HEADROOM_FOUND_51_END", "query_id"].tolist()
    no_clear = overall.loc[overall["overall_headroom_disposition"] == "NO_CLEAR_THROUGH_FULL_UNION", "query_id"].tolist()
    req(found_11 == HEADROOM_11_50, f"Unexpected ranks-11-50 cohort: {found_11}")
    req(found_51 == [], f"Unexpected ranks-51-end headroom cohort: {found_51}")
    req(no_clear == NO_CLEAR_FULL_UNION, f"Unexpected full-union no-clear cohort: {no_clear}")

    failure_q = pd.read_csv(FAILURE_QUERY_DIAGNOSTICS, dtype=str, keep_default_na=False, encoding="utf-8")
    req(len(failure_q) == 24, "Expected 24 V4H failure-diagnostic query rows")
    req(not failure_q["query_id"].duplicated().any(), "Duplicate failure-diagnostic query_id")
    no_clear_top10 = failure_q.loc[failure_q["v4_failure_signature"] == "NO_CLEAR_TOP10", "query_id"].tolist()
    req(no_clear_top10 == ["V4H07","V4H09","V4H10","V4H11","V4H13","V4H16","V4H19","V4H21","V4H22","V4H23"],
        f"Original NO_CLEAR_TOP10 target set changed: {no_clear_top10}")

    print("PASS  original 10-query NO_CLEAR_TOP10 cohort reconciled with final headroom disposition")
    print("PASS  four ranks-11-50 headroom queries verified")
    print("PASS  zero ranks-51-end-only headroom queries verified")
    print("PASS  six no-clear-through-full-union queries verified")

    return {"failure": failure, "final": final, "stage_a": stage_a, "stage_b": stage_b,
            "overall": overall, "failure_q": failure_q}


def main() -> None:
    print("Semantic relevance V4H - headroom diagnostic closeout preflight")
    print("----------------------------------------------------------------")
    print("Role: consumed-holdout diagnostic closeout only")
    print("Judge/model/retrieval/reranker calls: DISABLED")
    print("Release metrics/gates: NOT CALCULATED")
    print("Frozen V4B release decision modification: DISABLED")
    print()
    verify_boundary()
    req(not CLOSEOUT_LOCK.exists(), "V4H headroom closeout lock already exists")
    for path in [QUERY_COHORTS, SUMMARY, REPORT, METADATA]:
        req(not path.exists(), f"Refusing to overwrite existing closeout output: {path}")
    print()
    print("V4H HEADROOM DIAGNOSTIC CLOSEOUT PREFLIGHT: PASS")
    print("Target NO_CLEAR_TOP10 queries: 10")
    print("HEADROOM_FOUND_11_50: 4")
    print("HEADROOM_FOUND_51_END: 0")
    print("NO_CLEAR_THROUGH_FULL_UNION: 6")
    print("Judge/model/retrieval/reranker calls performed: 0")
    print("Release metrics calculated: NO")
    print("Frozen V4B release decision modified: NO")
    print("Next: write deterministic V4H headroom diagnostic closeout artifacts.")


if __name__ == "__main__":
    main()
