"""Freeze v0.27.0 r7 as the release candidate for the one-time final holdout.

This command performs NO judge calls. It validates the completed 150-case
consumed-development regression, the three fresh r7 stability spot-check runs,
the exact frozen semantic artifacts, and the still-untouched 30-case final
holdout. Only then does it write an immutable release-candidate evidence
manifest under evals/releases/.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
E = REPO_ROOT / "evals"
D = E / "datasets"
RELEASES = E / "releases"
OUT = RELEASES / "semantic_relevance_v0.27.0_r7_release_candidate.json"
FINALITY_MARKER = E / "runs" / "semantic_relevance_v0_27_final_holdout" / "FINAL_HOLDOUT_STARTED.json"

FINAL_HOLDOUT = D / "semantic_relevance_final_holdout.v3.0.0.DO_NOT_RUN_YET.csv"
FINAL_LOCK = D / "semantic_relevance_final_holdout_lock.v3.0.0.json"
DEV_DATASET = D / "semantic_relevance_v0.27_development.v3.1.0.csv"
STABILITY_MANIFEST = D / "semantic_relevance_v0.27_r7_stability_manifest.v1.0.0.json"

EXPECTED_FINAL_HOLDOUT_SHA = "cce944bd48b102452c024134560b7b87866b85f5fa1bc1194775138b8976eb2e"
EXPECTED_DEV_DATASET_SHA = "72810fa61cc71e28f05beaec9664f2ccdee55d5834272c44e2f1cc97cafeee7d"

EXPECTED_HASHES = {
    "evals/semantic_relevance_facet_judge.py": "d7504a62721f7120dda50cdc6cea759fced8ea85c98d1766baf62bfd2fb716d1",
    "evals/semantic_relevance_facet_scoring.py": "8447a75aee29d1ebf34147e625a7fe8c1f8ccf4b409029e917ec4c47285e8808",
    "evals/facets/semantic_relevance/semantic_relevance_query_facets.v0.9.7.json": "2f54203336c8125e9bec6751afba6fc8aa45e046527e992550468f6a8aaa25de",
    "evals/judge_configs/semantic_relevance_judge.v0.27.0.json": "d96347852d982d1f9242520519eac55ff7e0dd8c598b0eb02f82eeeb6c17bff2",
    "evals/rubrics/semantic_relevance/semantic_relevance_rubric.v0.1.0.json": "658fe8ef49f1b73e4d6d6bdb143ab4d43a040ed7ea347fc2d4a41a3051ee7d2e",
    "evals/datasets/semantic_relevance_v0.27_regression_manifest.v1.3.0.json": "8e3280827881c85c3f1975fe57f17b50de06ec55226dbfd9c10b39e486645c2c",
    "evals/datasets/semantic_relevance_v0.27_r7_stability_manifest.v1.0.0.json": "fa4d3b82b3573e9301d3768d821704ef7d057feb4e095d83ec292ef285b47122",
}

RELEASE_CHECKS = {
    "within_one_ge_0_95": {"metric": "within_one_agreement", "op": ">=", "value": 0.95},
    "exact_ge_0_50": {"metric": "exact_agreement", "op": ">=", "value": 0.50},
    "quadratic_kappa_ge_0_80": {"metric": "quadratic_weighted_cohens_kappa", "op": ">=", "value": 0.80},
    "linear_kappa_ge_0_60": {"metric": "linear_weighted_cohens_kappa", "op": ">=", "value": 0.60},
    "absolute_difference_ge_2_le_1_case": {"metric": "error_counts.absolute_difference_ge_2", "op": "<=", "value": 1},
    "binary_precision_ge_0_90": {"metric": "binary_relevance_0_vs_positive.precision", "op": ">=", "value": 0.90},
    "binary_recall_ge_0_80": {"metric": "binary_relevance_0_vs_positive.recall", "op": ">=", "value": 0.80},
    "deterministic_cue_severe_false_positives_eq_0": {"metric": "deterministic_cue_severe_false_positives", "op": "==", "value": 0},
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resolve(raw: str) -> Path:
    path = Path(raw)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def git_sha() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return None


def verify_holdout_untouched() -> dict:
    if not FINAL_HOLDOUT.exists():
        raise FileNotFoundError(FINAL_HOLDOUT)
    if not FINAL_LOCK.exists():
        raise FileNotFoundError(FINAL_LOCK)
    actual = sha256(FINAL_HOLDOUT)
    if actual != EXPECTED_FINAL_HOLDOUT_SHA:
        raise ValueError(f"Final holdout hash changed: {actual} != {EXPECTED_FINAL_HOLDOUT_SHA}")
    lock = load_json(FINAL_LOCK)
    if lock.get("status") != "LOCKED_DO_NOT_RUN":
        raise ValueError(f"Final holdout status is {lock.get('status')!r}, expected LOCKED_DO_NOT_RUN")
    if int(lock.get("judge_run_count", -1)) != 0:
        raise ValueError(f"Final holdout judge_run_count={lock.get('judge_run_count')!r}, expected 0")
    if FINALITY_MARKER.exists():
        raise RuntimeError("A v0.27 final-holdout finality marker already exists; refusing to freeze a new release candidate.")
    return {
        "path": str(FINAL_HOLDOUT.relative_to(REPO_ROOT)),
        "sha256": actual,
        "status": "LOCKED_DO_NOT_RUN",
        "judge_run_count": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--development-run", required=True)
    parser.add_argument("--stability-run", action="append", required=True)
    args = parser.parse_args()

    if len(args.stability_run) != 3:
        raise ValueError("Exactly three --stability-run directories are required.")
    if OUT.exists():
        raise RuntimeError(f"Release candidate manifest already exists: {OUT}")

    artifacts: dict[str, dict[str, object]] = {}
    for rel, expected in EXPECTED_HASHES.items():
        path = REPO_ROOT / rel
        if not path.exists():
            raise FileNotFoundError(path)
        actual = sha256(path)
        if actual != expected:
            raise ValueError(f"Frozen r7 artifact hash mismatch for {rel}: {actual} != {expected}")
        artifacts[rel] = {"sha256": actual, "size_bytes": path.stat().st_size}

    if not DEV_DATASET.exists():
        raise FileNotFoundError(DEV_DATASET)
    dev_dataset_sha = sha256(DEV_DATASET)
    if dev_dataset_sha != EXPECTED_DEV_DATASET_SHA:
        raise ValueError(f"150-case development dataset hash mismatch: {dev_dataset_sha} != {EXPECTED_DEV_DATASET_SHA}")

    holdout = verify_holdout_untouched()

    dev = resolve(args.development_run)
    summary_file = dev / "development_summary.json"
    results_file = dev / "judge_results.csv"
    metadata_file = dev / "run_metadata.json"
    for path in (summary_file, results_file, metadata_file):
        if not path.exists():
            raise FileNotFoundError(path)
    summary = load_json(summary_file)
    metadata = load_json(metadata_file)

    if int(summary.get("cases_compared", 0)) != 150:
        raise ValueError("Release freeze requires a complete 150-case development analysis.")
    if not bool(summary.get("all_pre_registered_checks_pass")):
        raise ValueError("Development reference checks did not all pass.")
    acceptance = summary.get("acceptance_checks", {})
    if set(acceptance) != set(RELEASE_CHECKS) or not all(bool(v) for v in acceptance.values()):
        raise ValueError("Release freeze requires all eight pre-registered development checks to pass.")
    targeted = summary.get("targeted_architecture_checks", {})
    if len(targeted) != 34 or not all(bool(v) for v in targeted.values()):
        raise ValueError("Release freeze requires all 34 targeted regression checks to pass in the full development run.")

    expected_meta = {
        "judge_config_version": "0.27.0",
        "evaluation_dataset_version": "3.1.0",
        "facet_spec_version": "0.9.7",
        "rubric_version": "0.1.0",
    }
    for key, expected in expected_meta.items():
        if str(metadata.get(key)) != expected:
            raise ValueError(f"Development metadata mismatch {key}: {metadata.get(key)!r} != {expected!r}")

    stability_spec = load_json(STABILITY_MANIFEST)
    expectations = stability_spec["expectations"]
    required_cases = set(expectations)
    stability: list[dict[str, object]] = []
    seen: set[Path] = set()
    score_sequences = {case_id: [] for case_id in expectations}

    for raw in args.stability_run:
        run = resolve(raw)
        if run in seen:
            raise ValueError(f"Duplicate stability run directory: {run}")
        seen.add(run)
        results = run / "judge_results.csv"
        run_meta = run / "run_metadata.json"
        if not results.exists() or not run_meta.exists():
            raise FileNotFoundError(run)
        df = pd.read_csv(results, dtype={"case_id": str}, encoding="utf-8")
        if len(df) != 7 or set(df["case_id"].astype(str)) != required_cases:
            raise ValueError(f"Stability run {run} must contain exactly the 7 registered cases.")
        scores = dict(zip(df["case_id"].astype(str), df["judge_score"].astype(int)))
        for case_id, spec in expectations.items():
            score = int(scores[case_id])
            minimum = int(spec["judge_min"])
            maximum = int(spec["judge_max"])
            if not minimum <= score <= maximum:
                raise ValueError(f"Stability failure in {run}: {case_id}={score}, expected {minimum}-{maximum}")
            score_sequences[case_id].append(score)
        smeta = load_json(run_meta)
        for key, expected in expected_meta.items():
            if str(smeta.get(key)) != expected:
                raise ValueError(f"Stability metadata mismatch in {run} for {key}: {smeta.get(key)!r} != {expected!r}")
        stability.append({
            "run_directory": str(run),
            "judge_results_sha256": sha256(results),
            "run_metadata_sha256": sha256(run_meta),
            "scores": {case_id: int(scores[case_id]) for case_id in expectations},
        })

    # This candidate exhibited zero observed score variation across the three
    # pre-freeze stability runs. Preserve that stronger observed property in the
    # freeze evidence instead of merely checking the allowed score ranges.
    unstable = {case_id: seq for case_id, seq in score_sequences.items() if len(set(seq)) != 1}
    if unstable:
        raise ValueError(f"Observed r7 stability runs are not identical across repetitions: {unstable}")

    manifest = {
        "release_candidate_id": "semantic_relevance_v0.27.0-r7",
        "status": "FROZEN_FOR_ONE_TIME_FINAL_HOLDOUT",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit_sha": git_sha(),
        "judge_version": "0.27.0",
        "facet_spec_version": "0.9.7",
        "rubric_version": "0.1.0",
        "development_dataset_version": "3.1.0",
        "artifacts": artifacts,
        "development_evidence": {
            "run_directory": str(dev),
            "development_summary_sha256": sha256(summary_file),
            "judge_results_sha256": sha256(results_file),
            "run_metadata_sha256": sha256(metadata_file),
            "development_dataset_sha256": dev_dataset_sha,
            "cases": 150,
            "all_reference_checks_pass": True,
            "all_34_targeted_checks_pass": True,
            "metrics": {
                "exact_agreement": summary["exact_agreement"],
                "within_one_agreement": summary["within_one_agreement"],
                "mean_absolute_difference": summary["mean_absolute_difference"],
                "linear_weighted_cohens_kappa": summary["linear_weighted_cohens_kappa"],
                "quadratic_weighted_cohens_kappa": summary["quadratic_weighted_cohens_kappa"],
                "binary_relevance_0_vs_positive": summary["binary_relevance_0_vs_positive"],
                "error_counts": summary["error_counts"],
            },
        },
        "stability_evidence": {
            "required_fresh_runs": 3,
            "cases_per_run": 7,
            "runs": stability,
            "observed_score_sequences": score_sequences,
            "identical_across_three_runs": True,
        },
        "independent_final_holdout": holdout,
        "final_holdout_protocol": {
            "expected_cases": 30,
            "one_time_only": True,
            "targeted_fresh_runs_forbidden": True,
            "resume_same_run_allowed_for_operational_recovery": True,
            "report_result_even_if_release_checks_fail": True,
            "no_relabel_or_retune_after_result": True,
            "pre_registered_release_checks": RELEASE_CHECKS,
        },
    }

    RELEASES.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("v0.27.0 r7 release candidate frozen.")
    print(f"Manifest: {OUT}")
    print(f"Manifest SHA-256: {sha256(OUT)}")
    print("PASS  150-case development evidence")
    print("PASS  all 34 targeted regression checks")
    print("PASS  three fresh 7-case stability runs; identical score sequences")
    print("PASS  final holdout hash pinned and still LOCKED_DO_NOT_RUN")
    print("PASS  final holdout judge_run_count = 0")
    print("NO JUDGE CALLS WERE MADE BY THIS FREEZE COMMAND.")


if __name__ == "__main__":
    main()
