"""Freeze semantic relevance v0.28.0 r8 as the final-holdout release candidate.

This command is non-judging. It records and pins evidence that already exists:
- targeted r8 regression PASS;
- canonical full-210 consumed-development PASS;
- three-run 20-case stability PASS;
- exact semantic/config/data hashes;
- independent final holdout still locked with judge_run_count=0.

The final-holdout dataset is verified only by its preregistered SHA-256 and lock
state. It is not scored or incorporated into development evidence by this tool.
"""
from __future__ import annotations

from datetime import datetime, timezone
import csv
import hashlib
import json
from pathlib import Path

R = Path(__file__).resolve().parents[1]
E = R / "evals"
RUN_ROOT = E / "runs" / "semantic_relevance_v0_28_r8_development"
RELEASES = E / "releases"

TARGETED_STATE = RUN_ROOT / "semantic_relevance_v0.28_r8_targeted_state.v1.0.0.json"
FULL_STATE = RUN_ROOT / "semantic_relevance_v0.28_r8_full210_state.v1.0.0.json"
STABILITY_STATE = RUN_ROOT / "semantic_relevance_v0.28_r8_stability_state.v1.0.0.json"

STABILITY_MANIFEST = E / "datasets" / "semantic_relevance_v0.28_r8_stability_manifest.v1.0.0.json"
DEVELOPMENT = E / "datasets" / "semantic_relevance_v0.28_development.v5.0.0.csv"
FINAL_HOLDOUT = E / "datasets" / "semantic_relevance_final_holdout.v4.0.0.DO_NOT_RUN_YET.csv"
FINAL_HOLDOUT_LOCK = E / "datasets" / "semantic_relevance_final_holdout_lock.v4.0.0.json"
VALIDATION_LOCK = E / "datasets" / "semantic_relevance_validation_lock.v4.0.0.json"
SPLIT_MANIFEST = E / "datasets" / "semantic_relevance_unseen_split_manifest.v4.0.0.json"

RELEASE = RELEASES / "semantic_relevance_v0.28.0_r8_release_candidate.json"

STATIC_HASHES = {
    "evals/semantic_relevance_facet_judge.py": "e4efec54d03601e3016b376741c7f14fc1259ab3d63d1fdff75925cc2bfb43c1",
    "evals/semantic_relevance_facet_scoring.py": "8447a75aee29d1ebf34147e625a7fe8c1f8ccf4b409029e917ec4c47285e8808",
    "evals/facets/semantic_relevance/semantic_relevance_query_facets.v0.9.13.json": "0952d80ebed8fbdebfb3b625f5648be9ccc08148d87d6753adc83d010953e289",
    "evals/judge_configs/semantic_relevance_judge.v0.28.0-r8.json": "bdc4157b457e3453ff94b860cb45022a7c7c83f311432ef9841a1c1fe5a9bb8d",
    "evals/datasets/semantic_relevance_v0.28_regression_manifest.v1.7.0.json": "da8d7ffe0f9131dd5ab976a0af72d7c478ee592b4cf1b495e105e483d908c585",
    "evals/datasets/semantic_relevance_v0.28_post_validation_label_adjudications.v1.0.0.json": "68f5d85fd5905034d813234b9f9df8202252ef1598e9676b5247902e55a7b893",
    "evals/rubrics/semantic_relevance/semantic_relevance_rubric.v0.1.0.json": "658fe8ef49f1b73e4d6d6bdb143ab4d43a040ed7ea347fc2d4a41a3051ee7d2e",
    "evals/datasets/semantic_relevance_v0.28_development.v5.0.0.csv": "21e53ee575fab0f99e69ffabee705f8afcc843878a33d640d6ada795be1ea520",
    "evals/datasets/semantic_relevance_final_holdout.v4.0.0.DO_NOT_RUN_YET.csv": "0c7b55065629b6c1fbdb3f0d530cf8bdc8e15011b0d213ab4be6259b95a43e85",
    "evals/datasets/semantic_relevance_unseen_split_manifest.v4.0.0.json": "889673f565cc7c377f6f0c11b690c709c10e93b574e9ad404c852cb2f2d742df",
    "evals/datasets/semantic_relevance_v0.28_r8_stability_manifest.v1.0.0.json": "c1f5ad5a6e4a92cc0bd4aa3121566384c81952633b293fddd7841e39b87d94a0",
}

EXPECTED_RUN_METADATA = {
    "judge_config_version": "0.28.0-r8",
    "evaluation_dataset_version": "5.0.0",
    "facet_spec_version": "0.9.13",
    "rubric_version": "0.1.0",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def require(cond: bool, message: str) -> None:
    if not cond:
        raise AssertionError(message)
    print("PASS ", message)


def evidence_for_run(run_dir: Path, expected_rows: int) -> dict:
    results = run_dir / "judge_results.csv"
    metadata_path = run_dir / "run_metadata.json"
    require(run_dir.exists(), f"run directory exists: {run_dir}")
    require(results.exists(), f"judge_results.csv exists: {run_dir.name}")
    require(metadata_path.exists(), f"run_metadata.json exists: {run_dir.name}")

    with results.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    require(len(rows) == expected_rows, f"{run_dir.name} contains exactly {expected_rows} judge rows")
    require(len({r['case_id'] for r in rows}) == expected_rows, f"{run_dir.name} case IDs are unique")

    metadata = load_json(metadata_path)
    for key, expected in EXPECTED_RUN_METADATA.items():
        require(str(metadata.get(key)) == expected, f"{run_dir.name} metadata {key}={expected}")

    return {
        "run_directory": str(run_dir.relative_to(R)),
        "case_count": expected_rows,
        "judge_results_sha256": sha256(results),
        "run_metadata_sha256": sha256(metadata_path),
    }


def verify_targeted() -> dict:
    state = load_json(TARGETED_STATE)
    run_dir = Path(state["run_directory"])
    if not run_dir.is_absolute():
        run_dir = R / run_dir
    evidence = evidence_for_run(run_dir, 54)

    manifest = load_json(E / "datasets" / "semantic_relevance_v0.28_regression_manifest.v1.7.0.json")
    expectations = manifest["targeted_expectations"]
    diagnostics = manifest["diagnostic_only_cases"]
    require(len(expectations) == 52, "targeted manifest contains 52 gated expectations")
    require(len(diagnostics) == 2, "targeted manifest contains 2 diagnostic-only cases")

    with (run_dir / "judge_results.csv").open("r", encoding="utf-8-sig", newline="") as f:
        by_id = {r["case_id"]: int(r["judge_score"]) for r in csv.DictReader(f)}

    failed = []
    for case_id, spec in expectations.items():
        if case_id not in by_id:
            failed.append(f"{case_id}:missing")
            continue
        score = by_id[case_id]
        if "judge_min" in spec and score < int(spec["judge_min"]):
            failed.append(f"{case_id}:{score}<min")
        if "judge_max" in spec and score > int(spec["judge_max"]):
            failed.append(f"{case_id}:{score}>max")
    require(not failed, f"all 52 targeted expectations pass: {failed}")
    evidence["gated_expectations"] = 52
    evidence["diagnostic_only_cases"] = 2
    return evidence


def verify_full() -> dict:
    state = load_json(FULL_STATE)
    run_dir = Path(state["run_directory"])
    if not run_dir.is_absolute():
        run_dir = R / run_dir
    evidence = evidence_for_run(run_dir, 210)

    summary_path = run_dir / "development_summary.v0_28_r8.json"
    require(summary_path.exists(), "r8 full-run analyzer summary exists")
    summary = load_json(summary_path)
    require(summary.get("development_regression_gate_pass") is True, "r8 full-210 development regression gate = PASS")
    checks = summary.get("prior150_reference_checks", {})
    require(len(checks) == 8 and all(checks.values()), "all eight frozen 150-case reference checks pass")
    require(summary.get("consumed_v027_final_no_new_severe_regressions") is True, "consumed v0.27 final-30 no-new-severe guard passes")
    require(summary.get("consumed_r4_validation_no_new_severe_regressions") is True, "consumed r4 validation-30 no-new-severe guard passes")
    require(len(summary.get("targeted_checks", {})) == 52 and all(summary["targeted_checks"].values()), "all 52 targeted checks pass inside full-210 run")

    evidence["development_summary_sha256"] = sha256(summary_path)
    evidence["overall_210"] = summary.get("overall_210")
    evidence["original_development_150"] = summary.get("original_development_150")
    return evidence


def verify_stability() -> dict:
    state = load_json(STABILITY_STATE)
    require(state.get("status") == "STABILITY_PASS", "r8 stability state = STABILITY_PASS")
    run_dirs = [Path(x) for x in state.get("run_dirs", [])]
    require(len(run_dirs) == 3, "exactly three fresh stability run directories recorded")

    manifest = load_json(STABILITY_MANIFEST)
    expectations = manifest["expectations"]
    require(len(expectations) == 20, "stability manifest contains 20 critical cases")
    max_span = int(manifest["max_observed_score_span"])

    histories = {case_id: [] for case_id in expectations}
    run_evidence = []
    for raw in run_dirs:
        run_dir = raw if raw.is_absolute() else R / raw
        ev = evidence_for_run(run_dir, 20)
        run_evidence.append(ev)
        with (run_dir / "judge_results.csv").open("r", encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
        by_id = {r["case_id"]: int(r["judge_score"]) for r in rows}
        require(set(by_id) == set(expectations), f"{run_dir.name} contains exactly the 20 stability case IDs")
        for case_id, spec in expectations.items():
            score = by_id[case_id]
            minimum = int(spec["judge_min"])
            maximum = int(spec["judge_max"])
            require(minimum <= score <= maximum, f"{run_dir.name} {case_id} score {score} inside {minimum}-{maximum}")
            histories[case_id].append(score)

    spans = {case_id: max(v)-min(v) for case_id, v in histories.items()}
    require(all(span <= max_span for span in spans.values()), f"all stability score spans <= {max_span}")
    identical = sum(1 for values in histories.values() if len(set(values)) == 1)
    require(identical == 20, "all 20 stability cases are identical across all three fresh runs")

    return {
        "state_file": str(STABILITY_STATE.relative_to(R)),
        "state_sha256": sha256(STABILITY_STATE),
        "manifest_sha256": sha256(STABILITY_MANIFEST),
        "required_runs": 3,
        "cases_per_run": 20,
        "total_observations": 60,
        "max_allowed_score_span": max_span,
        "identical_cases": identical,
        "score_sequences": histories,
        "runs": run_evidence,
    }


def main() -> None:
    print("v0.28.0 r8 release-candidate freeze")
    print("------------------------------------")
    print("JUDGE EXECUTION: DISABLED")

    for rel, expected in STATIC_HASHES.items():
        path = R / rel
        require(path.exists(), f"frozen artifact exists: {rel}")
        require(sha256(path) == expected, f"frozen hash pinned: {rel}")

    validation_lock = load_json(VALIDATION_LOCK)
    require(validation_lock.get("status") == "INDEPENDENT_VALIDATION_COMPLETED", "r4 validation remains consumed")
    require(int(validation_lock.get("judge_run_count", -1)) == 1, "r4 validation judge_run_count = 1")
    require(validation_lock.get("validation_decision") == "REVIEW_STOP_FINAL_HOLDOUT", "r4 validation failure decision remains preserved")

    final_lock = load_json(FINAL_HOLDOUT_LOCK)
    require(final_lock.get("status") == "LOCKED_DO_NOT_RUN", "independent final holdout remains LOCKED_DO_NOT_RUN")
    require(int(final_lock.get("judge_run_count", -1)) == 0, "independent final holdout judge_run_count = 0")

    targeted = verify_targeted()
    full = verify_full()
    stability = verify_stability()

    RELEASES.mkdir(parents=True, exist_ok=True)

    manifest = {
        "schema_version": "1.0.0",
        "release_candidate": "semantic_relevance_v0.28.0-r8",
        "release_status": "FROZEN_BEFORE_FINAL_HOLDOUT",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "semantic_identity": {
            "judge_config_version": "0.28.0-r8",
            "facet_spec_version": "0.9.13",
            "rubric_version": "0.1.0",
            "regression_manifest_version": "1.7.0",
            "development_dataset_version": "5.0.0",
        },
        "frozen_hashes": STATIC_HASHES,
        "evidence": {
            "targeted_regression": targeted,
            "full_210_consumed_development": full,
            "stability": stability,
        },
        "independent_evidence_state": {
            "r4_validation": {
                "status": validation_lock.get("status"),
                "judge_run_count": int(validation_lock.get("judge_run_count", -1)),
                "validation_decision": validation_lock.get("validation_decision"),
                "role_from_r5_onward": "consumed_development_only",
            },
            "final_holdout": {
                "dataset": str(FINAL_HOLDOUT.relative_to(R)),
                "dataset_sha256": sha256(FINAL_HOLDOUT),
                "status": final_lock.get("status"),
                "judge_run_count": int(final_lock.get("judge_run_count", -1)),
                "included_in_development": False,
                "judge_output_used_for_r8_tuning": False,
            },
        },
        "decision": "READY_FOR_FINAL_HOLDOUT_EXECUTION_REVIEW",
        "methodology_note": (
            "r8 is frozen only after targeted regression PASS, canonical full-210 "
            "consumed-development PASS, and three fresh 20-case stability runs with "
            "identical scores. The 30-case independent final holdout remains locked "
            "and has not been executed or used for r8 tuning. Any semantic/config/"
            "rubric/regression change after this freeze creates a new candidate and "
            "invalidates this release manifest for final-holdout execution."
        ),
    }

    # Freeze is intentionally one-way/idempotent: never silently overwrite a release manifest.
    if RELEASE.exists():
        existing = load_json(RELEASE)
        comparable_existing = dict(existing)
        comparable_new = dict(manifest)
        comparable_existing.pop("frozen_at_utc", None)
        comparable_new.pop("frozen_at_utc", None)
        require(comparable_existing == comparable_new, "existing r8 release manifest matches current frozen evidence")
        print("Release candidate was already frozen; manifest left unchanged.")
    else:
        RELEASE.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print("v0.28.0 r8 release candidate frozen.")
    print("Manifest:", RELEASE)
    print("Manifest SHA-256:", sha256(RELEASE))
    print("PASS  targeted regression: 52/52 gated expectations")
    print("PASS  full 210 consumed-development gate")
    print("PASS  all eight frozen 150-case reference checks")
    print("PASS  both consumed-30 no-new-severe-regression guards")
    print("PASS  three fresh 20-case stability runs; 20/20 identical score sequences")
    print("PASS  final holdout remains LOCKED_DO_NOT_RUN / judge_run_count=0")
    print("NO JUDGE CALLS WERE MADE BY THIS FREEZE TOOL.")


if __name__ == "__main__":
    main()
