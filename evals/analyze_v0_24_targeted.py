"""Check the preregistered v0.24.0 targeted regression on a partial run."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = REPO_ROOT / "evals" / "datasets" / "semantic_relevance_v0.24_regression_manifest.v3.0.0.json"
DATASET = REPO_ROOT / "evals" / "datasets" / "semantic_relevance_v0.24_development.v2.1.0.csv"


def resolve(raw: str) -> Path:
    p=Path(raw)
    if not p.is_absolute(): p=REPO_ROOT/p
    return p


def main() -> None:
    ap=argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    args=ap.parse_args()
    run=resolve(args.run)
    results=pd.read_csv(run/"judge_results.csv", dtype={"case_id":str})
    gold=pd.read_csv(DATASET, dtype={"case_id":str})
    manifest=json.loads(MANIFEST.read_text(encoding="utf-8"))
    merged=gold[["case_id","human_score","title"]].merge(results,on="case_id",how="right",validate="one_to_one")
    expectations=manifest["targeted_expectations"]
    required=set(expectations)
    present=set(merged.case_id.astype(str))
    missing=sorted(required-present)
    if missing:
        raise ValueError(f"Targeted run is missing required cases: {missing}")

    print("v0.24.0 targeted regression")
    print("-------------------------")
    all_pass=True
    for cid, exp in expectations.items():
        row=merged.loc[merged.case_id==cid].iloc[0]
        score=int(row.judge_score)
        checks=[]
        if "judge_min" in exp: checks.append(score>=int(exp["judge_min"]))
        if "judge_max" in exp: checks.append(score<=int(exp["judge_max"]))
        if "polarity_guard_min" in exp:
            checks.append(int(row.get("deterministic_cue_polarity_blocked_count",0) or 0)>=int(exp["polarity_guard_min"]))
        if "direct_cue_max" in exp:
            checks.append(int(row.get("deterministic_direct_cue_count",0) or 0)<=int(exp["direct_cue_max"]))
        passed=all(checks)
        all_pass &= passed
        print(f"{'PASS' if passed else 'FAIL'}  {cid}: human={int(row.human_score)} judge={score}  {row.title}")
    print()
    print(f"Targeted regression: {'PASS' if all_pass else 'FAIL'}")

if __name__ == "__main__": main()
