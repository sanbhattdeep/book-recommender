"""Check three fresh-run stability for the v0.27.0 r7 localized boundaries."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

HERE = Path(__file__).resolve().parent
MANIFEST = HERE / "datasets" / "semantic_relevance_v0.27_r7_stability_manifest.v1.0.0.json"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help="Fresh v0.27 development run directory. Supply exactly three times.",
    )
    args = parser.parse_args()

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    required_runs = int(manifest["required_fresh_runs"])
    expectations = manifest["expectations"]

    if len(args.run) != required_runs:
        parser.error(f"r7 stability gate requires exactly {required_runs} fresh --run directories")

    overall_ok = True
    seen: set[Path] = set()
    print("v0.27.0 r7 stability spot-check")
    print("--------------------------------")

    score_history: dict[str, list[int]] = {case_id: [] for case_id in expectations}

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

        df = pd.read_csv(results_path, dtype={"case_id": str})
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
    for case_id in expectations:
        values = score_history[case_id]
        rendered = ",".join(str(value) for value in values) if values else "missing"
        print(f"{case_id}: {rendered}")

    print()
    print("Stability spot-check:", "PASS" if overall_ok else "FAIL")
    return 0 if overall_ok else 1


if __name__ == "__main__":
    sys.exit(main())
