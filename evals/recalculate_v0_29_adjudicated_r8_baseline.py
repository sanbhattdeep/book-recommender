"""Recalculate the frozen r8 baseline after v0.29 human adjudication.

NO judge calls are made. Existing canonical r8 judge outputs are joined to the
new v6.1.0 human labels so we can see the post-adjudication development
baseline before any semantic repair.
"""
from __future__ import annotations

import json
from pathlib import Path
import pandas as pd

R = Path(__file__).resolve().parents[1]
E = R / "evals"
D = E / "datasets"
BASE = E / "baselines" / "semantic_relevance_v0_29"

DATASET = D / "semantic_relevance_v0.29_development.v6.1.0.csv"
FULL_STATE = E / "runs" / "semantic_relevance_v0_28_r8_development" / "semantic_relevance_v0.28_r8_full210_state.v1.0.0.json"
FINAL_LOCK = D / "semantic_relevance_final_holdout_lock.v4.0.0.json"

OUT_CSV = BASE / "r8_baseline_240.adjudicated.v1.0.0.csv"
OUT_JSON = BASE / "r8_baseline_240.adjudicated_summary.v1.0.0.json"

def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))

def require(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)
    print("PASS ", msg)

def weighted_kappa(human: pd.Series, judge: pd.Series, quadratic: bool) -> float:
    labels = [0,1,2,3,4]
    n = len(human)
    pos = {v:i for i,v in enumerate(labels)}
    obs = [[0.0]*5 for _ in range(5)]
    hc = [0.0]*5
    jc = [0.0]*5
    for h,j in zip(human.astype(int), judge.astype(int)):
        i,k = pos[int(h)],pos[int(j)]
        obs[i][k] += 1
        hc[i] += 1
        jc[k] += 1
    od = ed = 0.0
    for i in range(5):
        for j in range(5):
            d = abs(i-j)/4
            w = d*d if quadratic else d
            od += w*(obs[i][j]/n)
            ed += w*((hc[i]/n)*(jc[j]/n))
    return 1.0 if ed == 0 and od == 0 else (0.0 if ed == 0 else 1 - od/ed)

def main() -> None:
    print("v0.29.0 post-adjudication r8 baseline")
    print("--------------------------------------")
    print("JUDGE EXECUTION: DISABLED")

    ds = pd.read_csv(DATASET, dtype={"case_id": str}, encoding="utf-8-sig")
    require(len(ds) == 240 and ds["case_id"].is_unique,
            "adjudicated dataset contains 240 unique cases")

    full_state = load_json(FULL_STATE)
    full_run = Path(str(full_state["run_directory"]))
    full = pd.read_csv(full_run / "judge_results.csv", dtype={"case_id": str}, encoding="utf-8")
    require(len(full) == 210 and full["case_id"].is_unique,
            "canonical r8 full run contains 210 unique judge rows")

    lock = load_json(FINAL_LOCK)
    require(lock.get("status") == "FINAL_HOLDOUT_COMPLETED"
            and int(lock.get("judge_run_count", -1)) == 1,
            "canonical r8 final holdout remains consumed exactly once")
    final_run = Path(str(lock["run_directory"]))
    final = pd.read_csv(final_run / "judge_results.csv", dtype={"case_id": str}, encoding="utf-8")
    require(len(final) == 30 and final["case_id"].is_unique,
            "canonical r8 final run contains 30 unique judge rows")

    scores = pd.concat([full, final], ignore_index=True)
    require(len(scores) == 240 and scores["case_id"].is_unique,
            "existing r8 judge evidence reconstructs 240 unique scores")

    merged = ds[
        ["case_id","query_id","query","title","authors","description",
         "human_score","human_reason","development_source"]
    ].merge(scores, on="case_id", how="inner", validate="one_to_one")
    require(len(merged) == 240, "all 240 adjudicated cases matched to existing r8 scores")

    merged["human_score"] = merged["human_score"].astype(int)
    merged["judge_score"] = merged["judge_score"].astype(int)
    merged["signed_difference"] = merged["judge_score"] - merged["human_score"]
    merged["absolute_difference"] = merged["signed_difference"].abs()
    merged["exact_match"] = merged["absolute_difference"] == 0
    merged["within_one"] = merged["absolute_difference"] <= 1

    h,j = merged["human_score"], merged["judge_score"]
    hp,jp = h>0,j>0
    tp = int((hp & jp).sum()); fp = int((~hp & jp).sum())
    fn = int((hp & ~jp).sum()); tn = int((~hp & ~jp).sum())
    precision = tp/(tp+fp) if tp+fp else 0.0
    recall = tp/(tp+fn) if tp+fn else 0.0
    f1 = 2*precision*recall/(precision+recall) if precision+recall else 0.0

    severe = merged.loc[merged["absolute_difference"] >= 2,
                        ["case_id","query_id","title","human_score","judge_score",
                         "signed_difference","absolute_difference"]].sort_values("case_id")

    # After the three approved revisions, only two of the former five final
    # severe cases should remain severe. Q09 remains a repair target but is now
    # within one point (human=2, judge=3).
    final_severe = set(severe.loc[severe["case_id"].str.startswith("U4_"), "case_id"])
    require(final_severe == {"U4_Q03_T02","U4_Q04_T02"},
            "post-adjudication U4 severe cases are exactly Q03_T02 and Q04_T02")

    q09 = merged.loc[merged["case_id"] == "U4_Q09_T04"].iloc[0]
    require(int(q09["human_score"]) == 2 and int(q09["judge_score"]) == 3
            and int(q09["absolute_difference"]) == 1,
            "U4_Q09_T04 remains a one-point over-promotion repair target")

    summary = {
        "baseline": "frozen_v0.28.0_r8_on_v0.29_adjudicated_labels",
        "cases": 240,
        "exact_agreement": float(merged["exact_match"].mean()),
        "within_one_agreement": float(merged["within_one"].mean()),
        "mean_absolute_difference": float(merged["absolute_difference"].mean()),
        "linear_weighted_kappa": weighted_kappa(h,j,False),
        "quadratic_weighted_kappa": weighted_kappa(h,j,True),
        "binary": {
            "tp": tp,"fp": fp,"fn": fn,"tn": tn,
            "precision": precision,"recall": recall,"f1": f1,
        },
        "severe_count": int(len(severe)),
        "severe_case_ids": severe["case_id"].astype(str).tolist(),
        "semantic_repair_targets": [
            "U4_Q03_T02","U4_Q04_T02","U4_Q09_T04"
        ],
        "judge_calls_made": False,
    }

    BASE.mkdir(parents=True, exist_ok=True)
    merged.to_csv(OUT_CSV, index=False, encoding="utf-8")
    OUT_JSON.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print()
    print("Frozen r8 baseline after adjudication")
    print("-------------------------------------")
    print(f"Cases:                     240")
    print(f"Exact agreement:           {summary['exact_agreement']:.1%}")
    print(f"Within ±1 agreement:       {summary['within_one_agreement']:.1%}")
    print(f"MAE:                       {summary['mean_absolute_difference']:.3f}")
    print(f"Linear weighted kappa:     {summary['linear_weighted_kappa']:.3f}")
    print(f"Quadratic weighted kappa:  {summary['quadratic_weighted_kappa']:.3f}")
    print(f"Precision:                 {precision:.1%}")
    print(f"Recall:                    {recall:.1%}")
    print(f"F1:                        {f1:.1%}")
    print(f"Severe disagreements:      {len(severe)}")
    print("Severe IDs:                " + ", ".join(severe["case_id"].astype(str)))
    print("Semantic repair targets:   U4_Q03_T02, U4_Q04_T02, U4_Q09_T04")
    print("NO JUDGE CALLS WERE MADE.")

if __name__ == "__main__":
    main()
