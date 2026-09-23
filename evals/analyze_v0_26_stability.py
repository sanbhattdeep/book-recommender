"""Check repeated fresh-run stability for the two v0.26.0 r6 semantic positives."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

EXPECTATIONS = {
    "U_Q07_T02": (3, 4),
    "U_Q03_T02": (2, 2),
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help="Fresh v0.26 development run directory. Supply exactly three times.",
    )
    args = parser.parse_args()

    if len(args.run) != 3:
        parser.error("r6 stability gate requires exactly three fresh --run directories")

    overall_ok = True
    seen = set()
    print("v0.26.0 r6 stability spot-check")
    print("-------------------------------")

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

        df = pd.read_csv(results_path)
        print(f"Run {index}: {run_dir}")
        for case_id, (minimum, maximum) in EXPECTATIONS.items():
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
            ok = minimum <= score <= maximum
            status = "PASS" if ok else "FAIL"
            expected = f"{minimum}" if minimum == maximum else f"{minimum}-{maximum}"
            print(f"  {status}  {case_id}: judge={score} expected={expected}")
            overall_ok &= ok

    print()
    print("Stability spot-check:", "PASS" if overall_ok else "FAIL")
    return 0 if overall_ok else 1


if __name__ == "__main__":
    sys.exit(main())
