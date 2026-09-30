"""Analyze three fresh-run stability spot-checks for semantic relevance v0.28.0 r8.

All selected cases are consumed development evidence. This is a stability gate,
not independent generalization evidence.

Gate:
1. Exactly three distinct fresh run directories.
2. Every selected case appears exactly once in every run.
3. Every score is inside its preregistered range.
4. For each case, max(score)-min(score) across the three runs <= the
   preregistered max_observed_score_span (currently 1).
5. Provenance identities are exactly r8 / dataset v5.0.0 / facet spec 0.9.13.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

HERE = Path(__file__).resolve().parent
MANIFEST = HERE / "datasets" / "semantic_relevance_v0.28_r8_stability_manifest.v1.0.0.json"

EXPECTED_DATASET_VERSION = "5.0.0"
EXPECTED_JUDGE_CONFIG_VERSION = "0.28.0-r8"
EXPECTED_FACET_SPEC_VERSION = "0.9.13"


def _read_metadata(run_dir: Path) -> dict:
    path = run_dir / "run_metadata.json"
    if not path.exists():
        raise FileNotFoundError(f"missing {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help="Fresh r8 consumed-development run directory. Supply exactly three times.",
    )
    args = parser.parse_args()

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    required_runs = int(manifest["required_fresh_runs"])
    max_span = int(manifest["max_observed_score_span"])
    expectations = manifest["expectations"]

    if len(args.run) != required_runs:
        parser.error(f"r8 stability gate requires exactly {required_runs} fresh --run directories")

    overall_ok = True
    seen: set[Path] = set()
    score_history: dict[str, list[int]] = {case_id: [] for case_id in expectations}

    print("v0.28.0 r8 stability spot-check")
    print("--------------------------------")
    print(f"Fresh runs required:       {required_runs}")
    print(f"Cases per run:             {len(expectations)}")
    print(f"Maximum score span/case:   {max_span}")
    print("Evidence role:             consumed development only")
    print("Independent final holdout: LOCKED / NOT EXECUTED")
    print()

    for index, raw_dir in enumerate(args.run, start=1):
        run_dir = Path(raw_dir).resolve()
        if run_dir in seen:
            print(f"FAIL  run {index}: duplicate run directory {run_dir}")
            overall_ok = False
            continue
        seen.add(run_dir)

        results_path = run_dir / "judge_results.csv"
        if not results_path.exists():
            print(f"FAIL  run {index}: missing {results_path}")
            overall_ok = False
            continue

        try:
            metadata = _read_metadata(run_dir)
        except Exception as exc:
            print(f"FAIL  run {index}: metadata error: {exc}")
            overall_ok = False
            continue

        provenance_checks = {
            "evaluation_dataset_version": EXPECTED_DATASET_VERSION,
            "judge_config_version": EXPECTED_JUDGE_CONFIG_VERSION,
            "facet_spec_version": EXPECTED_FACET_SPEC_VERSION,
        }
        for key, expected in provenance_checks.items():
            actual = str(metadata.get(key, ""))
            if actual != expected:
                print(f"FAIL  run {index}: {key}={actual!r}, expected {expected!r}")
                overall_ok = False

        df = pd.read_csv(results_path, dtype={"case_id": str})
        expected_ids = set(expectations)
        actual_ids = set(df["case_id"].astype(str))
        unexpected = sorted(actual_ids - expected_ids)
        if unexpected:
            print(f"FAIL  run {index}: unexpected case IDs in stability run: {unexpected}")
            overall_ok = False

        print(f"Run {index}: {run_dir}")
        for case_id, spec in expectations.items():
            rows = df.loc[df["case_id"] == case_id]
            if rows.empty:
                print(f"  FAIL  {case_id}: missing")
                overall_ok = False
                continue
            if len(rows) != 1:
                print(f"  FAIL  {case_id}: expected one row, found {len(rows)}")
                overall_ok = False
                continue

            score = int(rows.iloc[0]["judge_score"])
            score_history[case_id].append(score)
            minimum = int(spec["judge_min"])
            maximum = int(spec["judge_max"])
            ok = minimum <= score <= maximum
            status = "PASS" if ok else "FAIL"
            expected = str(minimum) if minimum == maximum else f"{minimum}-{maximum}"
            print(f"  {status}  {case_id}: judge={score} expected={expected} role={spec['role']}")
            overall_ok &= ok

    print()
    print("Observed score sequences")
    print("------------------------")
    identical_count = 0
    for case_id, spec in expectations.items():
        values = score_history[case_id]
        rendered = ",".join(str(value) for value in values) if values else "missing"
        if len(values) != required_runs:
            print(f"FAIL  {case_id}: sequence={rendered}; expected {required_runs} observations")
            overall_ok = False
            continue

        unique = len(set(values))
        span = max(values) - min(values)
        span_ok = span <= max_span
        if unique == 1:
            identical_count += 1
        print(
            f"{'PASS' if span_ok else 'FAIL'}  {case_id}: {rendered} "
            f"unique={unique} span={span} max_span={max_span}"
        )
        overall_ok &= span_ok

    print()
    print(f"Identical across all three runs: {identical_count}/{len(expectations)}")
    print("Stability spot-check:", "PASS" if overall_ok else "FAIL")
    print("NOTE: bounded score variation within the preregistered range and max-span gate is allowed.")
    print("NOTE: this remains consumed development evidence, not independent evidence.")
    return 0 if overall_ok else 1


if __name__ == "__main__":
    sys.exit(main())
