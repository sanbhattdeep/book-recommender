"""Freeze semantic relevance v0.28.0 r4 as a release candidate.

This command performs NO judge calls. It validates the exact completed
180-case consumed-development regression and the three fresh 13-case stability
runs, pins the semantic artifacts by SHA-256, and writes a release-candidate
evidence manifest under evals/releases/.

The v0.27 final holdout is already consumed and is treated only as development
regression evidence. This freeze happens *before* any new independent v0.28
unseen evidence is created or consumed.
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
OUT = RELEASES / "semantic_relevance_v0.28.0_r4_release_candidate.json"

DEV_DATASET = D / "semantic_relevance_v0.28_development.v4.0.0.csv"
DEV_DATASET_MANIFEST = D / "semantic_relevance_v0.28_development_manifest.v1.0.0.json"
V027_FINAL_HOLDOUT = D / "semantic_relevance_final_holdout.v3.0.0.DO_NOT_RUN_YET.csv"
V027_FINAL_LOCK = D / "semantic_relevance_final_holdout_lock.v3.0.0.json"
STABILITY_MANIFEST = D / "semantic_relevance_v0.28_r4_stability_manifest.v1.1.0.json"
REGRESSION_MANIFEST = D / "semantic_relevance_v0.28_regression_manifest.v1.3.0.json"
PACKAGE_MANIFEST = REPO_ROOT / "v0.28.0_r4_package_manifest.json"

EXPECTED_DEV_DATASET_SHA = "fdf9ccbd8d131fa80c1e994681802b7f99050f497a60e503188c3e9ae7a3b83b"
EXPECTED_V027_FINAL_SHA = "cce944bd48b102452c024134560b7b87866b85f5fa1bc1194775138b8976eb2e"

EXPECTED_HASHES = {
    "evals/semantic_relevance_facet_judge.py": "2ad6fb76876c3d892de3299a7867d6d7bff8b8456975796b310e3969b9b09c1b",
    "evals/semantic_relevance_facet_scoring.py": "8447a75aee29d1ebf34147e625a7fe8c1f8ccf4b409029e917ec4c47285e8808",
    "evals/facets/semantic_relevance/semantic_relevance_query_facets.v0.9.9.json": "378fe3d381703a07b22256b4b5e26e578d2336391cc3e4d861591bdbe314aafe",
    "evals/judge_configs/semantic_relevance_judge.v0.28.0.json": "82e75d61e0b1c02bc126edd40749b480a5643a966c05725802688b98dc78c64b",
    "evals/datasets/semantic_relevance_v0.28_regression_manifest.v1.3.0.json": "28aa38338e3130a23e6b891b707e0a3ba886805d62845f5e91cea87744bd6973",
    "evals/datasets/semantic_relevance_v0.28_r4_stability_manifest.v1.1.0.json": "9a87ed55667a6dbcc25df286aea983eaeb2c57edcf53c2f7b2c0b248d0d83bab",
    "evals/rubrics/semantic_relevance/semantic_relevance_rubric.v0.1.0.json": "658fe8ef49f1b73e4d6d6bdb143ab4d43a040ed7ea347fc2d4a41a3051ee7d2e",
}

REFERENCE_CHECKS = {
    "within_one_ge_0_95",
    "exact_ge_0_50",
    "quadratic_kappa_ge_0_80",
    "linear_kappa_ge_0_60",
    "absolute_difference_ge_2_le_1_case",
    "binary_precision_ge_0_90",
    "binary_recall_ge_0_80",
    "deterministic_cue_severe_false_positives_eq_0",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def resolve(raw: str) -> Path:
    path = Path(raw)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def git_sha() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
            capture_output=True, text=True, check=True,
        )
        value = result.stdout.strip()
        return value or None
    except Exception:
        return None


def verify_consumed_v027_holdout() -> dict:
    for path in (V027_FINAL_HOLDOUT, V027_FINAL_LOCK):
        if not path.exists():
            raise FileNotFoundError(path)
    actual_sha = sha256(V027_FINAL_HOLDOUT)
    if actual_sha != EXPECTED_V027_FINAL_SHA:
        raise ValueError(
            f"Consumed v0.27 final-holdout hash mismatch: {actual_sha} != {EXPECTED_V027_FINAL_SHA}"
        )
    lock = load_json(V027_FINAL_LOCK)
    if lock.get("status") != "FINAL_HOLDOUT_COMPLETED":
        raise ValueError("v0.27 final holdout must remain FINAL_HOLDOUT_COMPLETED.")
    if int(lock.get("judge_run_count", -1)) != 1:
        raise ValueError("v0.27 final holdout judge_run_count must remain exactly 1.")
    return {
        "status": lock.get("status"),
        "judge_run_count": int(lock.get("judge_run_count")),
        "sha256": actual_sha,
        "role": "consumed_development_regression_evidence_only",
    }


def verify_metadata(metadata: dict, context: str) -> None:
    expected = {
        "judge_config_version": "0.28.0",
        "evaluation_dataset_version": "4.0.0",
        "facet_spec_version": "0.9.9",
        "rubric_version": "0.1.0",
    }
    for key, value in expected.items():
        actual = str(metadata.get(key))
        if actual != value:
            raise ValueError(f"{context} metadata mismatch {key}: {actual!r} != {value!r}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--development-run", required=True)
    parser.add_argument("--stability-run", action="append", required=True)
    args = parser.parse_args()

    if len(args.stability_run) != 3:
        parser.error("Exactly three --stability-run directories are required.")
    if OUT.exists():
        raise RuntimeError(f"Release candidate manifest already exists: {OUT}")

    artifacts: dict[str, dict[str, object]] = {}
    for rel, expected in EXPECTED_HASHES.items():
        path = REPO_ROOT / rel
        if not path.exists():
            raise FileNotFoundError(path)
        actual = sha256(path)
        if actual != expected:
            raise ValueError(f"Frozen r4 artifact hash mismatch for {rel}: {actual} != {expected}")
        artifacts[rel] = {"sha256": actual, "size_bytes": path.stat().st_size}

    if not PACKAGE_MANIFEST.exists():
        raise FileNotFoundError(PACKAGE_MANIFEST)
    package_manifest = load_json(PACKAGE_MANIFEST)
    if package_manifest.get("version") != "0.28.0-r4":
        raise ValueError("Unexpected r4 package manifest version.")
    if bool(package_manifest.get("new_independent_v0_28_evidence_exists")):
        raise ValueError("Package manifest unexpectedly says independent v0.28 evidence already exists.")

    for path in (DEV_DATASET, DEV_DATASET_MANIFEST, STABILITY_MANIFEST, REGRESSION_MANIFEST):
        if not path.exists():
            raise FileNotFoundError(path)
    dev_dataset_sha = sha256(DEV_DATASET)
    if dev_dataset_sha != EXPECTED_DEV_DATASET_SHA:
        raise ValueError(
            f"180-case consumed-development dataset hash mismatch: {dev_dataset_sha} != {EXPECTED_DEV_DATASET_SHA}"
        )
    dataset_manifest = load_json(DEV_DATASET_MANIFEST)
    if int(dataset_manifest.get("case_count", 0)) != 180:
        raise ValueError("Development dataset manifest must record exactly 180 cases.")
    if dataset_manifest.get("independent_v0_28_evidence") is not None:
        raise ValueError("Development dataset manifest must not contain independent v0.28 evidence.")
    consumed_v027 = verify_consumed_v027_holdout()

    dev = resolve(args.development_run)
    summary_file = dev / "development_summary.v0_28_r4.json"
    results_file = dev / "judge_results.csv"
    metadata_file = dev / "run_metadata.json"
    for path in (summary_file, results_file, metadata_file):
        if not path.exists():
            raise FileNotFoundError(path)

    summary = load_json(summary_file)
    metadata = load_json(metadata_file)
    verify_metadata(metadata, "development")

    dev_results = pd.read_csv(results_file, dtype={"case_id": str}, encoding="utf-8-sig")
    if len(dev_results) != 180 or dev_results["case_id"].nunique() != 180:
        raise ValueError("Release freeze requires a complete 180-case development run.")
    if not bool(summary.get("development_regression_gate_pass")):
        raise ValueError("r4 development regression gate did not pass.")

    prior_checks = summary.get("prior150_reference_checks", {})
    if set(prior_checks) != REFERENCE_CHECKS or not all(bool(v) for v in prior_checks.values()):
        raise ValueError("Release freeze requires all eight frozen 150-case reference checks to pass.")

    targeted = summary.get("targeted_checks", {})
    if len(targeted) != 41 or not all(bool(v) for v in targeted.values()):
        raise ValueError("Release freeze requires all 41 r4 targeted checks to pass in the full 180-case run.")
    if not bool(summary.get("consumed30_no_new_severe_regressions")):
        raise ValueError("Consumed v0.27 holdout subset introduced a new severe regression.")

    stability_spec = load_json(STABILITY_MANIFEST)
    expectations = stability_spec["expectations"]
    required_cases = set(expectations)
    if len(required_cases) != 13:
        raise ValueError("r4 stability manifest must contain exactly 13 cases.")

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
        for path in (results, run_meta):
            if not path.exists():
                raise FileNotFoundError(path)
        df = pd.read_csv(results, dtype={"case_id": str}, encoding="utf-8-sig")
        if len(df) != 13 or set(df["case_id"].astype(str)) != required_cases:
            raise ValueError(f"Stability run {run} must contain exactly the 13 registered cases.")
        smeta = load_json(run_meta)
        verify_metadata(smeta, f"stability run {run}")
        scores = dict(zip(df["case_id"].astype(str), df["judge_score"].astype(int)))
        for case_id, spec in expectations.items():
            score = int(scores[case_id])
            minimum = int(spec["judge_min"])
            maximum = int(spec["judge_max"])
            if not minimum <= score <= maximum:
                raise ValueError(
                    f"Stability failure in {run}: {case_id}={score}, expected {minimum}-{maximum}"
                )
            score_sequences[case_id].append(score)
        stability.append({
            "run_directory": str(run),
            "judge_results_sha256": sha256(results),
            "run_metadata_sha256": sha256(run_meta),
            "scores": {case_id: int(scores[case_id]) for case_id in expectations},
        })

    # The accepted r4 evidence showed zero observed variation on all 13 cases.
    unstable = {case_id: seq for case_id, seq in score_sequences.items() if len(set(seq)) != 1}
    if unstable:
        raise ValueError(f"Observed r4 stability evidence is not identical across repetitions: {unstable}")

    manifest = {
        "release_candidate_id": "semantic_relevance_v0.28.0-r4",
        "status": "FROZEN_BEFORE_NEW_UNSEEN_V0_28_EVIDENCE",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit_sha": git_sha(),
        "judge_version": "0.28.0",
        "facet_spec_version": "0.9.9",
        "rubric_version": "0.1.0",
        "development_dataset_version": "4.0.0",
        "methodology_note": (
            "The candidate is frozen after passing targeted regression, the complete 180-case consumed-development gate, "
            "and three fresh 13-case stability runs. The 180 cases are all consumed development evidence. The v0.27 "
            "final holdout was consumed once and is not independent evidence for v0.28. Any v0.28 generalization claim "
            "must use newly sampled and blind-labelled evidence created only after this freeze."
        ),
        "artifacts": artifacts,
        "package_manifest": {
            "path": str(PACKAGE_MANIFEST.relative_to(REPO_ROOT)),
            "sha256": sha256(PACKAGE_MANIFEST),
        },
        "development_evidence": {
            "run_directory": str(dev),
            "development_summary_sha256": sha256(summary_file),
            "judge_results_sha256": sha256(results_file),
            "run_metadata_sha256": sha256(metadata_file),
            "development_dataset_sha256": dev_dataset_sha,
            "cases": 180,
            "all_41_targeted_checks_pass": True,
            "all_8_frozen_150_reference_checks_pass": True,
            "consumed30_no_new_severe_regressions": True,
            "metrics": {
                "overall_180": summary.get("overall_180"),
                "prior_v0_27_development_150": summary.get("prior_v0_27_development_150"),
                "consumed_v0_27_final_holdout_30": summary.get("consumed_v0_27_final_holdout_30"),
            },
        },
        "stability_evidence": {
            "required_fresh_runs": 3,
            "cases_per_run": 13,
            "runs": stability,
            "observed_score_sequences": score_sequences,
            "identical_across_three_runs": True,
        },
        "consumed_v0_27_final_holdout": consumed_v027,
        "independent_v0_28_evidence": {
            "status_at_freeze": "NOT_YET_CREATED_OR_CONSUMED",
            "required_for_generalization_claim": True,
            "must_be_newly_sampled_after_freeze": True,
            "must_be_blind_labelled_before_judge_execution": True,
            "must_not_be_used_for_retuning_this_candidate": True,
        },
    }

    RELEASES.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("v0.28.0 r4 release candidate frozen.")
    print(f"Manifest: {OUT}")
    print(f"Manifest SHA-256: {sha256(OUT)}")
    print("PASS  complete 180-case consumed-development evidence")
    print("PASS  all 41 targeted regression checks in the full run")
    print("PASS  all eight frozen 150-case reference checks")
    print("PASS  consumed 30-case no-new-severe-regression guard")
    print("PASS  three fresh 13-case stability runs; identical observed score sequences")
    print("PASS  v0.27 final holdout remains consumed exactly once")
    print("PASS  no independent v0.28 evidence is part of this freeze")
    print("NO JUDGE CALLS WERE MADE BY THIS FREEZE COMMAND.")


if __name__ == "__main__":
    main()
