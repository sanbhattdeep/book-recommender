"""Analyze the canonical v0.28.0 r8 full 210-case consumed-development run.

Evidence status
---------------
All 210 cases are consumed development evidence:
  * 150 cases from the established v0.27 development set.
  * 30 cases from the consumed v0.27 final holdout.
  * 30 cases from the consumed one-time r4 independent validation.

The separate 30-case U4 final holdout is NOT part of this dataset and must
remain LOCKED_DO_NOT_RUN with judge_run_count=0.

Gate
----
The full-run gate requires:
  1. every preregistered r6 targeted regression expectation to pass;
  2. all eight frozen reference thresholds on the original 150 cases;
  3. no NEW severe error in the consumed v0.27 final-30 outside the four
     severe cases historically observed in the frozen v0.27 r7 final run;
  4. no NEW severe error in the consumed r4 validation-30 outside the five
     severe cases that motivated r5-r8.

The consumed 30+30 subgroup metrics and overall 210 metrics are diagnostic
only. They are not independent generalization evidence.

r8 specifically adds U2_Q11_T02 to the targeted regression gate after r6
exposed it as a new severe Q11 under-score and r7 failed to repair it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET = REPO_ROOT / "evals/datasets/semantic_relevance_v0.28_development.v5.0.0.csv"
REGRESSION_MANIFEST = REPO_ROOT / "evals/datasets/semantic_relevance_v0.28_regression_manifest.v1.7.0.json"
FINAL_HOLDOUT = REPO_ROOT / "evals/datasets/semantic_relevance_final_holdout.v4.0.0.DO_NOT_RUN_YET.csv"
FINAL_HOLDOUT_LOCK = REPO_ROOT / "evals/datasets/semantic_relevance_final_holdout_lock.v4.0.0.json"

EXPECTED_DATASET_SHA = "21e53ee575fab0f99e69ffabee705f8afcc843878a33d640d6ada795be1ea520"
EXPECTED_FINAL_HOLDOUT_SHA = "0c7b55065629b6c1fbdb3f0d530cf8bdc8e15011b0d213ab4be6259b95a43e85"

EXPECTED_METADATA = {
    "judge_config_version": "0.28.0-r8",
    "evaluation_dataset_version": "5.0.0",
    "evaluation_dataset_role": "post_r4_independent_validation_development",
    "evaluation_scope": "consumed_180_plus_r4_independent_validation_30",
    "facet_spec_version": "0.9.13",
    "rubric_version": "0.1.0",
}

# Severe cases observed in the frozen v0.27 r7 final-holdout result.
KNOWN_V027_FINAL_SEVERE = {
    "U3_Q11_T02",
    "U3_Q03_T01",
    "U3_Q10_T01",
    "U3_Q11_T04",
}

# Severe disagreements observed in the one-time r4 independent validation.
# These cases are now consumed development targets. U4_Q12_T05 was later
# human-adjudicated from 0 -> 1 for r5+ development, but remains listed here
# as historical provenance.
KNOWN_R4_VALIDATION_SEVERE = {
    "U4_Q02_T05",
    "U4_Q06_T02",
    "U4_Q10_T02",
    "U4_Q11_T03",
    "U4_Q12_T05",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resolve(raw: str) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else REPO_ROOT / path


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def safe_rate(n: int, d: int) -> float:
    return n / d if d else 0.0


def weighted_kappa(human: pd.Series, judge: pd.Series, mode: str) -> float:
    labels = [0, 1, 2, 3, 4]
    n = len(human)
    if n == 0:
        raise ValueError("Cannot compute kappa on zero cases.")
    pos = {v: i for i, v in enumerate(labels)}
    k = len(labels)
    observed = [[0.0] * k for _ in range(k)]
    hc = [0.0] * k
    jc = [0.0] * k
    for h, j in zip(human.tolist(), judge.tolist()):
        hi, ji = pos[int(h)], pos[int(j)]
        observed[hi][ji] += 1.0
        hc[hi] += 1.0
        jc[ji] += 1.0
    ow = ew = 0.0
    for i in range(k):
        for j in range(k):
            dist = abs(i - j) / (k - 1)
            weight = dist if mode == "linear" else dist * dist
            ow += weight * (observed[i][j] / n)
            ew += weight * ((hc[i] / n) * (jc[j] / n))
    if ew == 0:
        return 1.0 if ow == 0 else 0.0
    return 1.0 - (ow / ew)


def add_differences(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["human_score"] = out["human_score"].astype(int)
    out["judge_score"] = out["judge_score"].astype(int)
    out["signed_difference"] = out["judge_score"] - out["human_score"]
    out["absolute_difference"] = out["signed_difference"].abs()
    out["exact_match"] = out["absolute_difference"] == 0
    out["within_one"] = out["absolute_difference"] <= 1
    return out


def metrics(frame: pd.DataFrame) -> dict:
    human = frame["human_score"]
    judge = frame["judge_score"]
    hp, jp = human > 0, judge > 0
    tp = int((hp & jp).sum())
    fp = int((~hp & jp).sum())
    fn = int((hp & ~jp).sum())
    tn = int((~hp & ~jp).sum())
    precision = safe_rate(tp, tp + fp)
    recall = safe_rate(tp, tp + fn)
    f1 = safe_rate(2 * precision * recall, precision + recall)

    cue = pd.to_numeric(
        frame.get(
            "deterministic_direct_cue_count",
            pd.Series([0] * len(frame), index=frame.index),
        ),
        errors="coerce",
    ).fillna(0).astype(int)
    cue_severe_fp = int(((cue > 0) & (human == 0) & (judge >= 2)).sum())

    return {
        "cases": int(len(frame)),
        "exact_agreement": float(frame["exact_match"].mean()),
        "within_one_agreement": float(frame["within_one"].mean()),
        "mean_absolute_difference": float(frame["absolute_difference"].mean()),
        "linear_weighted_kappa": float(weighted_kappa(human, judge, "linear")),
        "quadratic_weighted_kappa": float(weighted_kappa(human, judge, "quadratic")),
        "binary": {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        },
        "errors": {
            "false_zeros": int(((human > 0) & (judge == 0)).sum()),
            "false_positive_relevance": int(((human == 0) & (judge > 0)).sum()),
            "absolute_difference_ge_2": int((frame["absolute_difference"] >= 2).sum()),
            "over_promotions_ge_2": int((frame["signed_difference"] >= 2).sum()),
            "under_promotions_ge_2": int((frame["signed_difference"] <= -2).sum()),
            "deterministic_cue_severe_false_positives": cue_severe_fp,
        },
    }


def reference_checks(m: dict) -> dict[str, bool]:
    """Frozen development thresholds applied only to the original 150."""
    return {
        "within_one_ge_0_95": m["within_one_agreement"] >= 0.95,
        "exact_ge_0_50": m["exact_agreement"] >= 0.50,
        "quadratic_kappa_ge_0_80": m["quadratic_weighted_kappa"] >= 0.80,
        "linear_kappa_ge_0_60": m["linear_weighted_kappa"] >= 0.60,
        "absolute_difference_ge_2_le_1_case": m["errors"]["absolute_difference_ge_2"] <= 1,
        "binary_precision_ge_0_90": m["binary"]["precision"] >= 0.90,
        "binary_recall_ge_0_80": m["binary"]["recall"] >= 0.80,
        "deterministic_cue_severe_false_positives_eq_0": (
            m["errors"]["deterministic_cue_severe_false_positives"] == 0
        ),
    }


def print_metrics(title: str, m: dict) -> None:
    print(title)
    print("-" * len(title))
    print(f"Cases compared:            {m['cases']}")
    print(f"Exact agreement:           {m['exact_agreement']:.1%}")
    print(f"Within ±1 agreement:       {m['within_one_agreement']:.1%}")
    print(f"Mean absolute difference:  {m['mean_absolute_difference']:.3f}")
    print(f"Linear weighted kappa:     {m['linear_weighted_kappa']:.3f}")
    print(f"Quadratic weighted kappa:  {m['quadratic_weighted_kappa']:.3f}")
    b = m["binary"]
    print(f"Binary: TP={b['tp']} FP={b['fp']} FN={b['fn']} TN={b['tn']}")
    print(f"Precision={b['precision']:.1%} Recall={b['recall']:.1%} F1={b['f1']:.1%}")
    e = m["errors"]
    print(
        "Errors: false_zero={false_zeros} false_positive={false_positive_relevance} "
        "|diff|>=2={absolute_difference_ge_2} over>=2={over_promotions_ge_2} "
        "under>=2={under_promotions_ge_2}".format(**e)
    )
    print()


def assert_final_holdout_untouched() -> None:
    if not FINAL_HOLDOUT.exists() or not FINAL_HOLDOUT_LOCK.exists():
        raise FileNotFoundError("Independent final holdout or lock is missing.")
    if sha256(FINAL_HOLDOUT) != EXPECTED_FINAL_HOLDOUT_SHA:
        raise ValueError("Independent final-holdout dataset hash changed.")
    lock = load_json(FINAL_HOLDOUT_LOCK)
    if lock.get("status") != "LOCKED_DO_NOT_RUN":
        raise ValueError(f"Final holdout status changed: {lock.get('status')!r}")
    if int(lock.get("judge_run_count", -1)) != 0:
        raise ValueError("Final holdout judge_run_count must remain 0.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    args = ap.parse_args()
    run = resolve(args.run)

    if sha256(DATASET) != EXPECTED_DATASET_SHA:
        raise ValueError("r8 development dataset SHA changed.")
    assert_final_holdout_untouched()

    metadata = load_json(run / "run_metadata.json")
    mismatches = [
        f"{k}: run={metadata.get(k)!r}, expected={v!r}"
        for k, v in EXPECTED_METADATA.items()
        if str(metadata.get(k)) != str(v)
    ]
    if mismatches:
        raise ValueError("Unexpected r8 run metadata:\n- " + "\n- ".join(mismatches))

    gold = pd.read_csv(DATASET, dtype={"case_id": str, "isbn13": str}, encoding="utf-8-sig")
    judged = pd.read_csv(run / "judge_results.csv", dtype={"case_id": str}, encoding="utf-8")
    if len(gold) != 210 or gold["case_id"].nunique() != 210:
        raise ValueError(f"Expected 210 unique gold cases, found {len(gold)}.")
    if judged["case_id"].duplicated().any():
        raise ValueError("Judge results contain duplicate case_id values.")

    cols = [
        "case_id", "query_id", "query", "title",
        "human_score", "human_reason", "development_source",
    ]
    comparison = gold[cols].merge(judged, on="case_id", how="inner", validate="one_to_one")
    if len(comparison) != 210:
        missing = sorted(set(gold["case_id"]) - set(judged["case_id"]))
        raise ValueError(f"Development run incomplete: {len(comparison)}/210; missing={missing}")
    comparison = add_differences(comparison)

    u4_mask = comparison["case_id"].astype(str).str.startswith("U4_")
    old_final_mask = (
        comparison["development_source"].fillna("").astype(str)
        == "consumed_v0_27_final_holdout_30"
    )
    prior150 = comparison.loc[~u4_mask & ~old_final_mask].copy()
    consumed_v027_final30 = comparison.loc[old_final_mask].copy()
    consumed_r4_validation30 = comparison.loc[u4_mask].copy()

    if len(prior150) != 150:
        raise ValueError(f"Expected original development subset=150; found {len(prior150)}")
    if len(consumed_v027_final30) != 30:
        raise ValueError(f"Expected consumed v0.27 final subset=30; found {len(consumed_v027_final30)}")
    if len(consumed_r4_validation30) != 30:
        raise ValueError(f"Expected consumed r4 validation subset=30; found {len(consumed_r4_validation30)}")

    overall_m = metrics(comparison)
    prior_m = metrics(prior150)
    old_final_m = metrics(consumed_v027_final30)
    r4_validation_m = metrics(consumed_r4_validation30)

    manifest = load_json(REGRESSION_MANIFEST)
    expectations = manifest["targeted_expectations"]
    if len(expectations) != 52:
        raise ValueError(f"Expected 52 gated regression expectations; found {len(expectations)}")

    targeted_checks: dict[str, bool] = {}
    targeted_observations: dict[str, dict] = {}
    for case_id, exp in expectations.items():
        row = comparison.loc[comparison["case_id"] == case_id]
        if row.empty:
            targeted_checks[case_id] = False
            targeted_observations[case_id] = {"missing": True}
            continue
        score = int(row.iloc[0]["judge_score"])
        ok = True
        if "judge_min" in exp:
            ok = ok and score >= int(exp["judge_min"])
        if "judge_max" in exp:
            ok = ok and score <= int(exp["judge_max"])
        targeted_checks[case_id] = ok
        targeted_observations[case_id] = {
            "human_score": int(row.iloc[0]["human_score"]),
            "judge_score": score,
            "passed": bool(ok),
        }

    prior_checks = reference_checks(prior_m)

    severe_old_final = set(
        consumed_v027_final30.loc[
            consumed_v027_final30["absolute_difference"] >= 2, "case_id"
        ].astype(str)
    )
    new_severe_old_final = sorted(severe_old_final - KNOWN_V027_FINAL_SEVERE)
    old_final_guard = len(new_severe_old_final) == 0

    severe_r4_validation = set(
        consumed_r4_validation30.loc[
            consumed_r4_validation30["absolute_difference"] >= 2, "case_id"
        ].astype(str)
    )
    new_severe_r4_validation = sorted(
        severe_r4_validation - KNOWN_R4_VALIDATION_SEVERE
    )
    r4_validation_guard = len(new_severe_r4_validation) == 0

    gate_pass = (
        all(targeted_checks.values())
        and all(prior_checks.values())
        and old_final_guard
        and r4_validation_guard
    )

    largest = comparison.sort_values(
        ["absolute_difference", "case_id"], ascending=[False, True]
    )[[
        "case_id", "query_id", "title", "development_source",
        "human_score", "judge_score", "signed_difference", "absolute_difference",
    ]]

    summary = {
        "analysis_scope": "v0.28.0_r8_consumed_development_210",
        "methodology_note": (
            "All 210 cases are consumed development evidence. The original 150-case "
            "subset retains the frozen development reference gate. The consumed v0.27 "
            "final-30 and consumed r4 validation-30 are diagnostic except for no-new-"
            "severe-regression guards. The separate 30-case U4 final holdout remains "
            "locked/unseen; no independent generalization claim is made by this run."
        ),
        "overall_210": overall_m,
        "original_development_150": prior_m,
        "consumed_v0_27_final_30": old_final_m,
        "consumed_r4_validation_30": r4_validation_m,
        "prior150_reference_checks": prior_checks,
        "targeted_checks": targeted_checks,
        "targeted_observations": targeted_observations,
        "consumed_v027_final_severe_case_ids": sorted(severe_old_final),
        "consumed_v027_final_new_severe_case_ids": new_severe_old_final,
        "consumed_v027_final_no_new_severe_regressions": old_final_guard,
        "consumed_r4_validation_severe_case_ids": sorted(severe_r4_validation),
        "consumed_r4_validation_new_severe_case_ids": new_severe_r4_validation,
        "consumed_r4_validation_no_new_severe_regressions": r4_validation_guard,
        "development_regression_gate_pass": gate_pass,
        "run_metadata": metadata,
    }

    comparison.to_csv(
        run / "human_vs_judge.v0_28_r8_development.csv",
        index=False,
        encoding="utf-8",
    )
    largest.to_csv(
        run / "largest_disagreements.v0_28_r8_development.csv",
        index=False,
        encoding="utf-8",
    )
    (run / "development_summary.v0_28_r8.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print_metrics("Overall 210-case consumed-development summary", overall_m)
    print_metrics("Original development subset (150 cases)", prior_m)
    print_metrics("Consumed v0.27 final-holdout subset (30; diagnostic)", old_final_m)
    print_metrics("Consumed r4 independent-validation subset (30; diagnostic)", r4_validation_m)

    print("Targeted v0.28 r8 regression checks")
    print("------------------------------------")
    for case_id, ok in targeted_checks.items():
        print(f"{'PASS' if ok else 'FAIL':4}  {case_id}")
    print()

    print("Frozen 150-case development reference checks")
    print("--------------------------------------------")
    for name, ok in prior_checks.items():
        print(f"{'PASS' if ok else 'FAIL':4}  {name}")
    print()

    print("Consumed v0.27 final-30 severe-regression guard")
    print("------------------------------------------------")
    print(f"Severe case IDs now: {sorted(severe_old_final)}")
    print(f"New severe IDs vs frozen v0.27 result: {new_severe_old_final}")
    print(f"{'PASS' if old_final_guard else 'FAIL'}  no_new_severe_v0_27_final_regressions")
    print()

    print("Consumed r4 validation-30 severe-regression guard")
    print("--------------------------------------------------")
    print(f"Severe case IDs now: {sorted(severe_r4_validation)}")
    print(f"New severe IDs outside historical r4 severe set: {new_severe_r4_validation}")
    print(f"{'PASS' if r4_validation_guard else 'FAIL'}  no_new_severe_r4_validation_regressions")
    print()

    print("v0.28.0 r8 development regression gate: " + ("PASS" if gate_pass else "FAIL"))
    print("NOTE: all 210 cases are consumed development evidence.")
    print("NOTE: independent final holdout remains LOCKED_DO_NOT_RUN and was not executed.")
    print()
    print("Largest disagreements")
    print("---------------------")
    print(largest.head(25).to_string(index=False))

    # Final check after writing all diagnostic artifacts.
    assert_final_holdout_untouched()

    if not gate_pass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
