"""Diagnostic-only view of the frozen v0.26 final run after post-holdout re-adjudication.

This does NOT replace the frozen v0.26 final-holdout result. It exists only to
quantify the baseline that v0.27 development starts from after four human labels
were re-adjudicated for future development use.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import pandas as pd

EVALS = Path(__file__).resolve().parent
DATASETS = EVALS / "datasets"
HOLDOUT = DATASETS / "semantic_relevance_fresh_holdout.v2.0.0.DO_NOT_RUN_YET.csv"
REVISIONS = DATASETS / "semantic_relevance_post_holdout_label_revisions.v1.0.0.json"


def args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True, type=Path, help="Frozen v0.26 final-holdout run directory")
    return p.parse_args()


def main() -> None:
    a = args()
    results_path = a.run / "judge_results.csv"
    if not results_path.exists():
        raise FileNotFoundError(results_path)
    if not HOLDOUT.exists() or not REVISIONS.exists():
        raise FileNotFoundError("Required holdout/revision input missing")

    gold = pd.read_csv(HOLDOUT, dtype={"case_id": str}, keep_default_na=False)
    results = pd.read_csv(results_path, dtype={"case_id": str}, keep_default_na=False)
    if len(gold) != 30:
        raise ValueError(f"Expected 30 holdout rows, got {len(gold)}")

    rev = json.loads(REVISIONS.read_text(encoding="utf-8"))
    revised = gold.copy()
    for r in rev["revisions"]:
        cid = str(r["case_id"])
        mask = revised["case_id"].eq(cid)
        if int(mask.sum()) != 1:
            raise ValueError(f"Revision case missing or duplicated: {cid}")
        old = int(revised.loc[mask, "human_score"].iloc[0])
        if old != int(r["previous_human_score"]):
            raise ValueError(f"Unexpected original label for {cid}: {old}")
        revised.loc[mask, "human_score"] = int(r["new_human_score"])

    m = revised[["case_id", "query_id", "title", "human_score"]].merge(
        results[["case_id", "judge_score"]], on="case_id", how="inner", validate="one_to_one"
    )
    if len(m) != 30:
        raise ValueError(f"Expected 30 joined cases, got {len(m)}")
    m["human_score"] = m["human_score"].astype(int)
    m["judge_score"] = m["judge_score"].astype(int)
    m["signed_difference"] = m["judge_score"] - m["human_score"]
    m["absolute_difference"] = m["signed_difference"].abs()

    exact = (m["absolute_difference"] == 0).mean()
    within = (m["absolute_difference"] <= 1).mean()
    mae = m["absolute_difference"].mean()
    severe = int((m["absolute_difference"] >= 2).sum())
    over = int((m["signed_difference"] >= 2).sum())
    under = int((m["signed_difference"] <= -2).sum())
    false_zero = int(((m["human_score"] > 0) & (m["judge_score"] == 0)).sum())
    fp = int(((m["human_score"] == 0) & (m["judge_score"] > 0)).sum())
    tp = int(((m["human_score"] > 0) & (m["judge_score"] > 0)).sum())
    fn = false_zero
    tn = int(((m["human_score"] == 0) & (m["judge_score"] == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    print("v0.27 post-adjudication diagnostic baseline")
    print("-------------------------------------------")
    print("NOTE: This does NOT replace the frozen v0.26 final-holdout result.")
    print(f"Cases compared:            {len(m)}/30")
    print(f"Exact agreement:           {exact:.1%}")
    print(f"Within ±1 agreement:       {within:.1%}")
    print(f"Mean absolute difference:  {mae:.3f}")
    print()
    print("Binary relevance (0 vs >0)")
    print("--------------------------")
    print(f"TP={tp} FP={fp} FN={fn} TN={tn}")
    print(f"Precision={precision:.1%} Recall={recall:.1%} F1={f1:.1%}")
    print()
    print("Error counts")
    print("------------")
    print(f"False zeros:               {false_zero}")
    print(f"False-positive relevance:  {fp}")
    print(f"|difference| >= 2:         {severe}")
    print(f"Over-promotions >= 2:      {over}")
    print(f"Under-promotions >= 2:     {under}")
    print()
    print("Remaining severe disagreements")
    print("--------------------------------")
    print(m.loc[m["absolute_difference"] >= 2].sort_values(
        ["absolute_difference", "case_id"], ascending=[False, True]
    ).to_string(index=False))


if __name__ == "__main__":
    main()
