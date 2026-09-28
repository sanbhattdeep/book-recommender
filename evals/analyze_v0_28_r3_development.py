"""Analyze a complete v0.28.0 r3 180-case consumed-development run.

Methodology
-----------
The 180 cases are all consumed development evidence:
  * 150 cases from the previously established v0.27 development set.
  * 30 cases from the now-consumed v0.27 final holdout.

The release-style regression gate is intentionally anchored to the original
150-case subset plus the 36 explicit targeted contracts.  The consumed
30-case subgroup and the overall 180-case metrics are reported diagnostically;
they are not treated as new independent evidence.

A separate guard rejects *new* severe (absolute error >= 2) failures in the
consumed 30-case subgroup outside the four severe disagreements already
observed in the frozen v0.27 r7 final-holdout result.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET = REPO_ROOT / "evals/datasets/semantic_relevance_v0.28_development.v4.0.0.csv"
REGRESSION_MANIFEST = REPO_ROOT / "evals/datasets/semantic_relevance_v0.28_regression_manifest.v1.2.0.json"

EXPECTED_METADATA = {
    "judge_config_version": "0.28.0",
    "evaluation_dataset_version": "4.0.0",
    "evaluation_dataset_role": "post_v0_27_final_holdout_development",
    "evaluation_scope": "consumed_v0_27_development_and_final_holdout",
    "facet_spec_version": "0.9.9",
}

# These are the four severe disagreements in the frozen v0.27 r7 final run.
# They are now consumed diagnostics, not unseen evidence.  v0.28 must not
# introduce severe failures on any *other* consumed-holdout case.
KNOWN_V027_FINAL_SEVERE = {
    "U3_Q11_T02",
    "U3_Q03_T01",
    "U3_Q10_T01",
    "U3_Q11_T04",
}


def resolve(raw: str) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else REPO_ROOT / path


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


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
        frame.get("deterministic_direct_cue_count", pd.Series([0] * len(frame), index=frame.index)),
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
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": precision, "recall": recall, "f1": f1,
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
    """Frozen v0.27 development reference thresholds for the original 150."""
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    args = ap.parse_args()
    run = resolve(args.run)
    metadata = load_json(run / "run_metadata.json")

    mismatches = [
        f"{k}: run={metadata.get(k)!r}, expected={v!r}"
        for k, v in EXPECTED_METADATA.items()
        if str(metadata.get(k)) != str(v)
    ]
    if mismatches:
        raise ValueError("Unexpected run metadata:\n- " + "\n- ".join(mismatches))

    gold = pd.read_csv(DATASET, dtype={"case_id": str, "isbn13": str}, encoding="utf-8")
    judged = pd.read_csv(run / "judge_results.csv", dtype={"case_id": str}, encoding="utf-8")
    if len(gold) != 180 or gold["case_id"].nunique() != 180:
        raise ValueError(f"Expected 180 unique gold cases, found {len(gold)}.")
    if judged["case_id"].duplicated().any():
        raise ValueError("Judge results contain duplicate case_id values.")

    cols = ["case_id", "query_id", "query", "title", "human_score", "human_reason", "development_source"]
    comparison = gold[cols].merge(judged, on="case_id", how="inner", validate="one_to_one")
    if len(comparison) != 180:
        missing = sorted(set(gold["case_id"]) - set(judged["case_id"]))
        raise ValueError(f"Development run incomplete: {len(comparison)}/180; missing={missing}")
    comparison = add_differences(comparison)

    final_mask = comparison["development_source"].astype(str) == "consumed_v0_27_final_holdout_30"
    prior150 = comparison.loc[~final_mask].copy()
    consumed30 = comparison.loc[final_mask].copy()
    if len(prior150) != 150 or len(consumed30) != 30:
        raise ValueError(
            f"Unexpected source split: prior150={len(prior150)}, consumed30={len(consumed30)}"
        )

    overall_m = metrics(comparison)
    prior_m = metrics(prior150)
    consumed_m = metrics(consumed30)

    manifest = load_json(REGRESSION_MANIFEST)
    expectations = manifest["targeted_expectations"]
    targeted_checks: dict[str, bool] = {}
    for case_id, exp in expectations.items():
        row = comparison.loc[comparison["case_id"] == case_id]
        if row.empty:
            targeted_checks[case_id] = False
            continue
        score = int(row.iloc[0]["judge_score"])
        ok = True
        if "judge_min" in exp:
            ok = ok and score >= int(exp["judge_min"])
        if "judge_max" in exp:
            ok = ok and score <= int(exp["judge_max"])
        targeted_checks[case_id] = ok

    prior_checks = reference_checks(prior_m)

    severe_consumed = set(
        consumed30.loc[consumed30["absolute_difference"] >= 2, "case_id"].astype(str)
    )
    new_severe_consumed = sorted(severe_consumed - KNOWN_V027_FINAL_SEVERE)
    consumed_guard = len(new_severe_consumed) == 0

    gate_pass = all(targeted_checks.values()) and all(prior_checks.values()) and consumed_guard

    largest = comparison.sort_values(
        ["absolute_difference", "case_id"], ascending=[False, True]
    )[[
        "case_id", "query_id", "title", "development_source",
        "human_score", "judge_score", "signed_difference", "absolute_difference",
    ]]

    summary = {
        "analysis_scope": "v0.28.0_r3_consumed_development_180",
        "methodology_note": (
            "All 180 cases are consumed development evidence. The original 150-case subset "
            "retains the frozen development reference gate. The consumed v0.27 final-holdout "
            "subset is diagnostic only except for a no-new-severe-regression guard. No v0.28 "
            "generalization claim is made without a newly sampled unseen pool."
        ),
        "overall_180": overall_m,
        "prior_v0_27_development_150": prior_m,
        "consumed_v0_27_final_holdout_30": consumed_m,
        "prior150_reference_checks": prior_checks,
        "targeted_checks": targeted_checks,
        "consumed30_severe_case_ids": sorted(severe_consumed),
        "consumed30_new_severe_case_ids": new_severe_consumed,
        "consumed30_no_new_severe_regressions": consumed_guard,
        "development_regression_gate_pass": gate_pass,
        "run_metadata": metadata,
    }

    comparison.to_csv(run / "human_vs_judge.v0_28_r3_development.csv", index=False, encoding="utf-8")
    largest.to_csv(run / "largest_disagreements.v0_28_r3_development.csv", index=False, encoding="utf-8")
    (run / "development_summary.v0_28_r3.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    print_metrics("Overall 180-case consumed-development summary", overall_m)
    print_metrics("Original v0.27 development subset (150 cases)", prior_m)
    print_metrics("Consumed v0.27 final-holdout subset (30 cases; diagnostic)", consumed_m)

    print("Targeted v0.28 r3 regression checks")
    print("------------------------------------")
    for case_id, ok in targeted_checks.items():
        print(f"{'PASS' if ok else 'FAIL':4}  {case_id}")
    print()

    print("Frozen 150-case development reference checks")
    print("--------------------------------------------")
    for name, ok in prior_checks.items():
        print(f"{'PASS' if ok else 'FAIL':4}  {name}")
    print()

    print("Consumed 30-case severe-regression guard")
    print("-----------------------------------------")
    print(f"Severe case IDs now: {sorted(severe_consumed)}")
    print(f"New severe case IDs vs frozen v0.27 result: {new_severe_consumed}")
    print(f"{'PASS' if consumed_guard else 'FAIL'}  no_new_severe_consumed_holdout_regressions")
    print()

    print("v0.28.0 r3 development regression gate: " + ("PASS" if gate_pass else "FAIL"))
    print("NOTE: this is consumed development evidence only; no independent v0.28 holdout exists yet.")
    print()
    print("Largest disagreements")
    print("---------------------")
    print(largest.head(20).to_string(index=False))

    if not gate_pass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
