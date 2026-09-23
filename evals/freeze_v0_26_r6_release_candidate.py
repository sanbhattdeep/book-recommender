"""Freeze v0.26.0 r6 as the release candidate for the one-time final holdout.

This command must be run BEFORE any final-holdout judge call. It validates the
completed 90-case development evidence and the three r6 Q03/Q07 stability runs,
then records hashes of the exact frozen judge artifacts.
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
RELEASES = E / "releases"
OUT = RELEASES / "semantic_relevance_v0.26.0_r6_release_candidate.json"
FINALITY_MARKER = E / "runs" / "semantic_relevance_v0_26_final_holdout" / "FINAL_HOLDOUT_STARTED.json"

EXPECTED_HASHES = {
    "evals/semantic_relevance_facet_judge.py": "3f1f54bd7dba9fd08ddaafa55684c2f8bba8c7c888571996258cce57baeb3614",
    "evals/semantic_relevance_facet_scoring.py": "8447a75aee29d1ebf34147e625a7fe8c1f8ccf4b409029e917ec4c47285e8808",
    "evals/facets/semantic_relevance/semantic_relevance_query_facets.v0.9.3.json": "1821180bdbd9284db67d9bea661e69049713c29051b1ba5ea297d41690735c01",
    "evals/rubrics/semantic_relevance/semantic_relevance_rubric.v0.1.0.json": "658fe8ef49f1b73e4d6d6bdb143ab4d43a040ed7ea347fc2d4a41a3051ee7d2e",
    "evals/judge_configs/semantic_relevance_judge.v0.26.0.json": "49b3baff0c5816571845220372a43a7891a1d1b52cdef126fac1906b3cc377d1",
    "evals/datasets/semantic_relevance_label_revisions.v1.1.0.json": "51db1682bde0351afaa81e52fe8805e91aa3b4aafe73ebebed53eb3aa2759930",
    "evals/datasets/semantic_relevance_v0.26_regression_manifest.v6.2.0.json": "0303c36f7dc916b69eaaab8c875a675dbe42f7bb5b3be1e58e2f2279114a6c6d",
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
    p = Path(raw)
    if not p.is_absolute(): p = REPO_ROOT / p
    return p.resolve()

def git_sha() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    except Exception:
        return None

def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--development-run", required=True)
    ap.add_argument("--stability-run", action="append", required=True)
    args = ap.parse_args()

    if len(args.stability_run) != 3:
        raise ValueError("Exactly three --stability-run directories are required.")
    if FINALITY_MARKER.exists():
        raise RuntimeError("Final holdout has already been started; refusing to create or replace the release freeze manifest.")
    if OUT.exists():
        raise RuntimeError(f"Release candidate manifest already exists: {OUT}")

    artifacts = {}
    for rel, expected in EXPECTED_HASHES.items():
        p = REPO_ROOT / rel
        if not p.exists(): raise FileNotFoundError(p)
        actual = sha256(p)
        if actual != expected:
            raise ValueError(f"Frozen r6 artifact hash mismatch for {rel}: {actual} != {expected}")
        artifacts[rel] = {"sha256": actual, "size_bytes": p.stat().st_size}

    dev = resolve(args.development_run)
    summary_file = dev / "development_summary.json"
    results_file = dev / "judge_results.csv"
    metadata_file = dev / "run_metadata.json"
    for p in [summary_file, results_file, metadata_file]:
        if not p.exists(): raise FileNotFoundError(p)
    summary = load_json(summary_file)
    meta = load_json(metadata_file)

    if int(summary.get("cases_compared", 0)) != 90:
        raise ValueError("Release freeze requires a complete 90-case development analysis.")
    if not bool(summary.get("all_pre_registered_checks_pass")):
        raise ValueError("Development reference checks did not all pass.")
    targeted = summary.get("targeted_architecture_checks", {})
    if len(targeted) != 21 or not all(bool(x) for x in targeted.values()):
        raise ValueError("Release freeze requires all 21 targeted regression checks to pass in the full development run.")
    expected_meta = {
        "judge_config_version": "0.26.0",
        "evaluation_dataset_version": "2.2.0",
        "facet_spec_version": "0.9.3",
        "rubric_version": "0.1.0",
    }
    for k, v in expected_meta.items():
        if str(meta.get(k)) != v:
            raise ValueError(f"Development metadata mismatch {k}: {meta.get(k)!r} != {v!r}")

    stability = []
    for raw in args.stability_run:
        run = resolve(raw)
        rf = run / "judge_results.csv"
        mf = run / "run_metadata.json"
        if not rf.exists() or not mf.exists(): raise FileNotFoundError(run)
        df = pd.read_csv(rf, dtype={"case_id": str}, encoding="utf-8")
        scores = dict(zip(df["case_id"].astype(str), df["judge_score"].astype(int)))
        if scores.get("U_Q07_T02", -1) < 3:
            raise ValueError(f"Stability failure in {run}: U_Q07_T02={scores.get('U_Q07_T02')}")
        if scores.get("U_Q03_T02") != 2:
            raise ValueError(f"Stability failure in {run}: U_Q03_T02={scores.get('U_Q03_T02')}")
        stability.append({
            "run_directory": str(run),
            "judge_results_sha256": sha256(rf),
            "run_metadata_sha256": sha256(mf),
            "U_Q07_T02": int(scores["U_Q07_T02"]),
            "U_Q03_T02": int(scores["U_Q03_T02"]),
        })

    dev_dataset = REPO_ROOT / "evals/datasets/semantic_relevance_v0.26_development.v2.2.0.csv"
    if not dev_dataset.exists(): raise FileNotFoundError(dev_dataset)

    manifest = {
        "release_candidate_id": "semantic_relevance_v0.26.0-r6",
        "status": "FROZEN_FOR_ONE_TIME_FINAL_HOLDOUT",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit_sha": git_sha(),
        "judge_version": "0.26.0",
        "facet_spec_version": "0.9.3",
        "rubric_version": "0.1.0",
        "development_dataset_version": "2.2.0",
        "artifacts": artifacts,
        "development_evidence": {
            "run_directory": str(dev),
            "development_summary_sha256": sha256(summary_file),
            "judge_results_sha256": sha256(results_file),
            "run_metadata_sha256": sha256(metadata_file),
            "development_dataset_sha256": sha256(dev_dataset),
            "cases": 90,
            "all_reference_checks_pass": True,
            "all_21_targeted_checks_pass": True,
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
        "stability_evidence": stability,
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
    print("v0.26.0 r6 release candidate frozen.")
    print(f"Manifest: {OUT}")
    print(f"Manifest SHA-256: {sha256(OUT)}")
    print("Final holdout may now be started exactly once with run_judge_final_holdout.py.")

if __name__ == "__main__":
    main()
