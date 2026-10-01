"""Reconstruct the frozen r8 baseline over the 240 consumed v0.29 cases.

NO judge calls. Uses:
- canonical r8 full-210 results; and
- the one-time r8 final-holdout 30 results.

Also writes a dedicated five-case severe-review artifact for v0.29 adjudication.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import pandas as pd

R = Path(__file__).resolve().parents[1]
E = R / "evals"
D = E / "datasets"
REL = E / "releases"
RUNS = E / "runs"

DATASET = D / "semantic_relevance_v0.29_development.v6.0.0.csv"
FULL_STATE = E / "runs" / "semantic_relevance_v0_28_r8_development" / "semantic_relevance_v0.28_r8_full210_state.v1.0.0.json"
FINAL_LOCK = D / "semantic_relevance_final_holdout_lock.v4.0.0.json"
CLOSEOUT = REL / "semantic_relevance_v0.28.0_r8_final_evaluation_closeout.json"

BASELINE_DIR = E / "baselines" / "semantic_relevance_v0_29"
BASELINE_CSV = BASELINE_DIR / "r8_baseline_240.v1.0.0.csv"
SUMMARY = BASELINE_DIR / "r8_baseline_240_summary.v1.0.0.json"
SEVERE_REVIEW = D / "semantic_relevance_v0.29_severe_review.v1.0.0.csv"
SEVERE_MANIFEST = D / "semantic_relevance_v0.29_severe_review_manifest.v1.0.0.json"

EXPECTED_SEVERE = {"U4_Q03_T01","U4_Q03_T02","U4_Q04_T02","U4_Q06_T01","U4_Q09_T04"}

def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()

def load(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8-sig"))

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
    print("v0.29.0 r8 baseline reconstruction")
    print("----------------------------------")
    print("JUDGE EXECUTION: DISABLED")

    for p in [DATASET,FULL_STATE,FINAL_LOCK,CLOSEOUT]:
        require(p.exists(), f"required input exists: {p.relative_to(R)}")

    ds = pd.read_csv(DATASET, dtype={"case_id": str}, encoding="utf-8-sig")
    require(len(ds) == 240 and ds["case_id"].is_unique,
            "v0.29 dataset contains 240 unique cases")

    full_state = load(FULL_STATE)
    full_run = Path(str(full_state["run_directory"]))
    require(full_run.exists(), "canonical r8 full-210 run exists")
    full_results = pd.read_csv(full_run / "judge_results.csv", dtype={"case_id": str}, encoding="utf-8")
    require(len(full_results) == 210 and full_results["case_id"].is_unique,
            "canonical r8 full run contains 210 unique rows")

    lock = load(FINAL_LOCK)
    require(lock.get("status") == "FINAL_HOLDOUT_COMPLETED",
            "final-holdout lifecycle remains completed")
    require(int(lock.get("judge_run_count", -1)) == 1,
            "final holdout remains consumed exactly once")
    final_run = Path(str(lock["run_directory"]))
    require(final_run.exists(), "canonical r8 final-holdout run exists")
    final_results = pd.read_csv(final_run / "judge_results.csv", dtype={"case_id": str}, encoding="utf-8")
    require(len(final_results) == 30 and final_results["case_id"].is_unique,
            "canonical r8 final run contains 30 unique rows")

    overlap = set(full_results["case_id"]) & set(final_results["case_id"])
    require(not overlap, "210 and final-30 judge result IDs do not overlap")

    scores = pd.concat([full_results, final_results], ignore_index=True)
    require(len(scores) == 240 and scores["case_id"].is_unique,
            "reconstructed r8 baseline contains 240 unique judge rows")

    # Keep all judge result columns while joining frozen human/provenance data.
    base_cols = [
        "case_id","query_id","query","title","authors","description",
        "human_score","human_reason","development_source"
    ]
    for c in base_cols:
        if c not in ds.columns:
            ds[c] = ""

    merged = ds[base_cols].merge(scores, on="case_id", how="inner", validate="one_to_one")
    require(len(merged) == 240, "all 240 consumed cases have r8 baseline scores")

    merged["human_score"] = merged["human_score"].astype(int)
    merged["judge_score"] = merged["judge_score"].astype(int)
    merged["signed_difference"] = merged["judge_score"] - merged["human_score"]
    merged["absolute_difference"] = merged["signed_difference"].abs()
    merged["exact_match"] = merged["absolute_difference"] == 0
    merged["within_one"] = merged["absolute_difference"] <= 1

    h = merged["human_score"]
    j = merged["judge_score"]
    hp = h > 0
    jp = j > 0
    tp = int((hp & jp).sum()); fp = int((~hp & jp).sum())
    fn = int((hp & ~jp).sum()); tn = int((~hp & ~jp).sum())
    precision = tp/(tp+fp) if tp+fp else 0.0
    recall = tp/(tp+fn) if tp+fn else 0.0
    f1 = 2*precision*recall/(precision+recall) if precision+recall else 0.0

    summary = {
        "baseline": "frozen_v0.28.0_r8_reconstructed_without_judge_calls",
        "dataset": str(DATASET.relative_to(R)),
        "dataset_sha256": sha(DATASET),
        "cases": 240,
        "exact_agreement": float(merged["exact_match"].mean()),
        "within_one_agreement": float(merged["within_one"].mean()),
        "mean_absolute_difference": float(merged["absolute_difference"].mean()),
        "linear_weighted_kappa": weighted_kappa(h,j,False),
        "quadratic_weighted_kappa": weighted_kappa(h,j,True),
        "binary": {
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": precision, "recall": recall, "f1": f1,
        },
        "severe_count": int((merged["absolute_difference"] >= 2).sum()),
        "severe_case_ids": sorted(
            merged.loc[merged["absolute_difference"] >= 2, "case_id"].astype(str).tolist()
        ),
        "evidence_role": "consumed_baseline_for_v0.29_development",
    }

    BASELINE_DIR.mkdir(parents=True, exist_ok=True)
    merged.to_csv(BASELINE_CSV, index=False, encoding="utf-8")
    SUMMARY.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    severe = merged[merged["case_id"].isin(EXPECTED_SEVERE)].copy()
    require(len(severe) == 5, "all five r8 final-holdout severe cases present")
    require(set(severe["case_id"]) == EXPECTED_SEVERE,
            "severe-review IDs match frozen r8 final closeout")
    require((severe["absolute_difference"] >= 2).all(),
            "all five review targets are severe under frozen r8 baseline")

    review_cols = [
        "case_id","query_id","query","title","authors","description",
        "human_score","human_reason","judge_score",
        "signed_difference","absolute_difference","development_source"
    ]
    severe[review_cols].sort_values("case_id").to_csv(
        SEVERE_REVIEW, index=False, encoding="utf-8"
    )

    review_manifest = {
        "schema_version": "1.0.0",
        "lineage": "semantic_relevance_v0.29.0",
        "purpose": "human adjudication before any v0.29 semantic tuning",
        "case_count": 5,
        "case_ids": sorted(EXPECTED_SEVERE),
        "labels_changed": False,
        "judge_calls_made": False,
        "source": "consumed v0.28 r8 independent final-holdout evidence",
        "instruction": (
            "Review each case against the frozen rubric/facet semantics. Classify as "
            "judge defect, label/spec tension, or ambiguity before proposing repairs."
        ),
        "review_file": str(SEVERE_REVIEW.relative_to(R)),
        "review_file_sha256": sha(SEVERE_REVIEW),
    }
    SEVERE_MANIFEST.write_text(json.dumps(review_manifest, indent=2) + "\n", encoding="utf-8")

    print("r8 baseline over v0.29 development")
    print("---------------------------------")
    print(f"Cases:                     240")
    print(f"Exact agreement:           {summary['exact_agreement']:.1%}")
    print(f"Within ±1 agreement:       {summary['within_one_agreement']:.1%}")
    print(f"MAE:                       {summary['mean_absolute_difference']:.3f}")
    print(f"Linear weighted kappa:     {summary['linear_weighted_kappa']:.3f}")
    print(f"Quadratic weighted kappa:  {summary['quadratic_weighted_kappa']:.3f}")
    print(f"Precision:                 {precision:.1%}")
    print(f"Recall:                    {recall:.1%}")
    print(f"F1:                        {f1:.1%}")
    print(f"Severe disagreements:      {summary['severe_count']}")
    print("Severe final-review IDs:   " + ", ".join(sorted(EXPECTED_SEVERE)))
    print("Review artifact:", SEVERE_REVIEW)
    print("NOTE: baseline reconstructed from existing evidence; no judge calls made.")

if __name__ == "__main__":
    main()
