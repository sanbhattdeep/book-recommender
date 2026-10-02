"""Analyze the canonical v0.29.0-r3 full 240-case consumed-development run.

Evidence status
---------------
All 240 cases are consumed development evidence:
  * 150 original development cases;
  * 30 consumed v0.26 final-holdout cases;
  * 30 consumed r4 independent-validation cases;
  * 30 consumed v0.28 final-holdout cases, adjudicated for v0.29 development.

No part of this 240-case run is independent v0.29 evaluation evidence.

Gate
----
The development regression gate preserves the established methodology:
  1. all 58 preregistered v0.29-r3 targeted expectations must pass;
  2. all eight frozen reference thresholds must pass on the original 150 cases;
  3. no NEW severe error may appear in the consumed v0.26 final-30 outside
     the severe cases historically observed there;
  4. no NEW severe error may appear in the consumed r4 validation-30 outside
     its historical severe set;
  5. no NEW severe error may appear in the consumed v0.28 final-30 outside
     the five severe cases observed in the one-time r8 final evaluation.

The overall 240 metrics and all three 30-case subgroup metrics are diagnostic.
They must not be represented as independent generalization evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET = (
    REPO_ROOT
    / "evals/datasets/semantic_relevance_v0.29_development.v6.1.0.csv"
)
REGRESSION_MANIFEST = (
    REPO_ROOT
    / "evals/datasets/semantic_relevance_v0.29_regression_manifest.v2.2.0.json"
)
FINAL_HOLDOUT_LOCK = (
    REPO_ROOT
    / "evals/datasets/semantic_relevance_final_holdout_lock.v4.0.0.json"
)
R8_CLOSEOUT = (
    REPO_ROOT
    / "evals/releases/semantic_relevance_v0.28.0_r8_final_evaluation_closeout.json"
)
EVIDENCE_BOUNDARY = (
    REPO_ROOT
    / "evals/datasets/semantic_relevance_v0.29_evidence_boundary.v1.0.0.json"
)

EXPECTED_DATASET_SHA = (
    "cfcc465497079944fec6da17f14bbe5f3350d9385ea9ddc646610b7751d72da7"
)
EXPECTED_R8_CLOSEOUT_SHA = (
    "24f90865503f4572d54e168d7f2cca33647a7c9a3c294bfb1ab727fd544cb2de"
)

EXPECTED_METADATA = {
    "judge_config_version": "0.29.0-r3",
    "evaluation_dataset_version": "6.1.0",
    "evaluation_dataset_role": "v0_29_consumed_development_after_r8_closeout",
    "evaluation_scope": "consumed_v0_28_development_210_plus_r8_final_30_adjudicated",
    "facet_spec_version": "0.10.1",
    "rubric_version": "0.1.0",
}

# Historical severe sets are provenance allowances only. A case in one of
# these sets is not automatically acceptable now; it merely does not count as
# a NEW severe regression for that consumed subgroup.
KNOWN_V026_FINAL_SEVERE = {
    "U3_Q11_T02",
    "U3_Q03_T01",
    "U3_Q10_T01",
    "U3_Q11_T04",
}

KNOWN_R4_VALIDATION_SEVERE = {
    "U4_Q02_T05",
    "U4_Q06_T02",
    "U4_Q10_T02",
    "U4_Q11_T03",
    "U4_Q12_T05",
}

KNOWN_V028_FINAL_SEVERE = {
    "U4_Q03_T01",
    "U4_Q03_T02",
    "U4_Q04_T02",
    "U4_Q06_T01",
    "U4_Q09_T04",
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
    """Frozen thresholds applied only to the original 150-case reference set."""
    return {
        "within_one_ge_0_95": m["within_one_agreement"] >= 0.95,
        "exact_ge_0_50": m["exact_agreement"] >= 0.50,
        "quadratic_kappa_ge_0_80": m["quadratic_weighted_kappa"] >= 0.80,
        "linear_kappa_ge_0_60": m["linear_weighted_kappa"] >= 0.60,
        "absolute_difference_ge_2_le_1_case": (
            m["errors"]["absolute_difference_ge_2"] <= 1
        ),
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
    print(
        f"Precision={b['precision']:.1%} "
        f"Recall={b['recall']:.1%} F1={b['f1']:.1%}"
    )
    e = m["errors"]
    print(
        "Errors: false_zero={false_zeros} "
        "false_positive={false_positive_relevance} "
        "|diff|>=2={absolute_difference_ge_2} "
        "over>=2={over_promotions_ge_2} "
        "under>=2={under_promotions_ge_2}".format(**e)
    )
    print()


def assert_consumed_evidence_lifecycle() -> None:
    if sha256(DATASET) != EXPECTED_DATASET_SHA:
        raise ValueError("v0.29 adjudicated development dataset SHA changed.")
    if sha256(R8_CLOSEOUT) != EXPECTED_R8_CLOSEOUT_SHA:
        raise ValueError("v0.28 r8 closeout hash changed.")

    lock = load_json(FINAL_HOLDOUT_LOCK)
    if lock.get("status") != "FINAL_HOLDOUT_COMPLETED":
        raise ValueError(
            f"v0.28 final-holdout status changed: {lock.get('status')!r}"
        )
    if int(lock.get("judge_run_count", -1)) != 1:
        raise ValueError("v0.28 final holdout must remain consumed exactly once.")

    closeout = load_json(R8_CLOSEOUT)
    if closeout.get("closeout_status") != "EVALUATED_NOT_FINAL_QUALIFIED":
        raise ValueError(
            "Unexpected r8 closeout status: "
            f"{closeout.get('closeout_status')!r}"
        )
    if closeout.get("final_holdout_decision") != "FINAL_HOLDOUT_REVIEW":
        raise ValueError(
            "Unexpected r8 final-holdout decision: "
            f"{closeout.get('final_holdout_decision')!r}"
        )
    closeout_boundary = closeout.get(
        "evidence_boundary_for_next_lineage", {}
    )
    if (
        closeout_boundary.get("next_candidate_lineage")
        != "semantic_relevance_v0.29.0"
    ):
        raise ValueError(
            "r8 closeout does not authorize semantic_relevance_v0.29.0."
        )
    if (
        closeout_boundary.get(
            "r8_final_holdout_may_be_reused_as_independent_evidence"
        )
        is not False
    ):
        raise ValueError(
            "r8 independent-evidence boundary changed unexpectedly."
        )

    boundary = load_json(EVIDENCE_BOUNDARY)
    if boundary.get("new_independent_evidence_required") is not True:
        raise ValueError(
            "v0.29 evidence boundary no longer requires new independent evidence."
        )


def severe_ids(frame: pd.DataFrame) -> set[str]:
    return set(
        frame.loc[frame["absolute_difference"] >= 2, "case_id"].astype(str)
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    args = ap.parse_args()
    run = resolve(args.run)

    assert_consumed_evidence_lifecycle()

    metadata_path = run / "run_metadata.json"
    results_path = run / "judge_results.csv"
    if not metadata_path.exists() or not results_path.exists():
        raise FileNotFoundError(
            "Run must contain run_metadata.json and judge_results.csv."
        )

    metadata = load_json(metadata_path)
    mismatches = [
        f"{k}: run={metadata.get(k)!r}, expected={v!r}"
        for k, v in EXPECTED_METADATA.items()
        if str(metadata.get(k)) != str(v)
    ]
    if mismatches:
        raise ValueError(
            "Unexpected v0.29-r3 run metadata:\n- " + "\n- ".join(mismatches)
        )

    gold = pd.read_csv(
        DATASET,
        dtype={"case_id": str, "isbn13": str},
        encoding="utf-8-sig",
    )
    judged = pd.read_csv(
        results_path,
        dtype={"case_id": str},
        encoding="utf-8",
    )

    if len(gold) != 240 or gold["case_id"].nunique() != 240:
        raise ValueError(f"Expected 240 unique gold cases, found {len(gold)}.")
    if judged["case_id"].duplicated().any():
        raise ValueError("Judge results contain duplicate case_id values.")

    cols = [
        "case_id",
        "query_id",
        "query",
        "title",
        "human_score",
        "human_reason",
        "development_source",
    ]
    comparison = gold[cols].merge(
        judged,
        on="case_id",
        how="inner",
        validate="one_to_one",
    )
    if len(comparison) != 240:
        missing = sorted(set(gold["case_id"]) - set(judged["case_id"]))
        raise ValueError(
            f"Development run incomplete: {len(comparison)}/240; missing={missing}"
        )
    comparison = add_differences(comparison)

    source = comparison["development_source"].fillna("").astype(str)
    case_id = comparison["case_id"].astype(str)

    v026_final_mask = source == "consumed_v0_26_final_holdout_30"
    v028_final_mask = source == "consumed_v0_28_final_holdout_30"
    r4_validation_mask = case_id.str.startswith("U4_") & ~v028_final_mask
    prior150_mask = ~(v026_final_mask | r4_validation_mask | v028_final_mask)

    prior150 = comparison.loc[prior150_mask].copy()
    consumed_v026_final30 = comparison.loc[v026_final_mask].copy()
    consumed_r4_validation30 = comparison.loc[r4_validation_mask].copy()
    consumed_v028_final30 = comparison.loc[v028_final_mask].copy()

    counts = {
        "original_development_150": len(prior150),
        "consumed_v0_26_final_30": len(consumed_v026_final30),
        "consumed_r4_validation_30": len(consumed_r4_validation30),
        "consumed_v0_28_final_30": len(consumed_v028_final30),
    }
    expected_counts = {
        "original_development_150": 150,
        "consumed_v0_26_final_30": 30,
        "consumed_r4_validation_30": 30,
        "consumed_v0_28_final_30": 30,
    }
    if counts != expected_counts:
        raise ValueError(
            f"Unexpected v0.29 development partition counts: {counts}"
        )

    overall_m = metrics(comparison)
    prior_m = metrics(prior150)
    v026_final_m = metrics(consumed_v026_final30)
    r4_validation_m = metrics(consumed_r4_validation30)
    v028_final_m = metrics(consumed_v028_final30)

    manifest = load_json(REGRESSION_MANIFEST)
    if str(manifest.get("version")) != "2.2.0":
        raise ValueError(
            f"Unexpected regression manifest version: {manifest.get('version')!r}"
        )

    expectations = manifest["targeted_expectations"]
    if len(expectations) != 58:
        raise ValueError(
            f"Expected 58 gated regression expectations; found {len(expectations)}"
        )

    targeted_checks: dict[str, bool] = {}
    targeted_observations: dict[str, dict] = {}
    for cid, exp in expectations.items():
        row = comparison.loc[comparison["case_id"] == cid]
        if row.empty:
            targeted_checks[cid] = False
            targeted_observations[cid] = {"missing": True}
            continue
        score = int(row.iloc[0]["judge_score"])
        ok = True
        if "judge_min" in exp:
            ok = ok and score >= int(exp["judge_min"])
        if "judge_max" in exp:
            ok = ok and score <= int(exp["judge_max"])
        targeted_checks[cid] = bool(ok)
        targeted_observations[cid] = {
            "human_score": int(row.iloc[0]["human_score"]),
            "judge_score": score,
            "passed": bool(ok),
        }

    diagnostics = {}
    for cid, spec in manifest.get("diagnostic_only_cases", {}).items():
        row = comparison.loc[comparison["case_id"] == cid]
        diagnostics[cid] = {
            "human_score": (
                int(row.iloc[0]["human_score"]) if not row.empty else None
            ),
            "judge_score": (
                int(row.iloc[0]["judge_score"]) if not row.empty else None
            ),
            "reason": spec.get("reason"),
        }

    prior_checks = reference_checks(prior_m)

    sev_v026 = severe_ids(consumed_v026_final30)
    new_v026 = sorted(sev_v026 - KNOWN_V026_FINAL_SEVERE)
    guard_v026 = len(new_v026) == 0

    sev_r4 = severe_ids(consumed_r4_validation30)
    new_r4 = sorted(sev_r4 - KNOWN_R4_VALIDATION_SEVERE)
    guard_r4 = len(new_r4) == 0

    sev_v028 = severe_ids(consumed_v028_final30)
    new_v028 = sorted(sev_v028 - KNOWN_V028_FINAL_SEVERE)
    guard_v028 = len(new_v028) == 0

    gate_pass = (
        all(targeted_checks.values())
        and all(prior_checks.values())
        and guard_v026
        and guard_r4
        and guard_v028
    )

    largest = comparison.sort_values(
        ["absolute_difference", "case_id"],
        ascending=[False, True],
    )[
        [
            "case_id",
            "query_id",
            "title",
            "development_source",
            "human_score",
            "judge_score",
            "signed_difference",
            "absolute_difference",
        ]
    ]

    summary = {
        "analysis_scope": "v0.29.0_r3_consumed_development_240",
        "methodology_note": (
            "All 240 cases are consumed development evidence. The original "
            "150-case subset retains the frozen quantitative reference gate. "
            "The three consumed 30-case subgroups are diagnostic except for "
            "no-new-severe-regression guards. A fresh independent v0.29 "
            "evaluation set is still required after candidate freeze."
        ),
        "overall_240": overall_m,
        "original_development_150": prior_m,
        "consumed_v0_26_final_30": v026_final_m,
        "consumed_r4_validation_30": r4_validation_m,
        "consumed_v0_28_final_30": v028_final_m,
        "prior150_reference_checks": prior_checks,
        "targeted_checks": targeted_checks,
        "targeted_observations": targeted_observations,
        "diagnostic_only_cases": diagnostics,
        "consumed_v026_final_severe_case_ids": sorted(sev_v026),
        "consumed_v026_final_new_severe_case_ids": new_v026,
        "consumed_v026_final_no_new_severe_regressions": guard_v026,
        "consumed_r4_validation_severe_case_ids": sorted(sev_r4),
        "consumed_r4_validation_new_severe_case_ids": new_r4,
        "consumed_r4_validation_no_new_severe_regressions": guard_r4,
        "consumed_v028_final_severe_case_ids": sorted(sev_v028),
        "consumed_v028_final_new_severe_case_ids": new_v028,
        "consumed_v028_final_no_new_severe_regressions": guard_v028,
        "development_regression_gate_pass": gate_pass,
        "run_metadata": metadata,
    }

    comparison.to_csv(
        run / "human_vs_judge.v0_29_r3_development.csv",
        index=False,
        encoding="utf-8",
    )
    largest.to_csv(
        run / "largest_disagreements.v0_29_r3_development.csv",
        index=False,
        encoding="utf-8",
    )
    (run / "development_summary.v0_29_r3.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print_metrics("Overall 240-case consumed-development summary", overall_m)
    print_metrics("Original development subset (150 cases)", prior_m)
    print_metrics(
        "Consumed v0.26 final-holdout subset (30; diagnostic)",
        v026_final_m,
    )
    print_metrics(
        "Consumed r4 independent-validation subset (30; diagnostic)",
        r4_validation_m,
    )
    print_metrics(
        "Consumed v0.28 final-holdout subset (30; diagnostic)",
        v028_final_m,
    )

    print("Targeted v0.29 r3 regression checks")
    print("------------------------------------")
    for cid, ok in targeted_checks.items():
        print(f"{'PASS' if ok else 'FAIL':4}  {cid}")
    print()

    print("Adjudication-only diagnostics (not gate criteria)")
    print("------------------------------------------------")
    for cid, d in diagnostics.items():
        print(
            f"INFO  {cid}: human={d['human_score']} judge={d['judge_score']} "
            f"-- {d['reason']}"
        )
    print()

    print("Frozen 150-case development reference checks")
    print("--------------------------------------------")
    for name, ok in prior_checks.items():
        print(f"{'PASS' if ok else 'FAIL':4}  {name}")
    print()

    guard_rows = [
        (
            "Consumed v0.26 final-30 severe-regression guard",
            sev_v026,
            new_v026,
            guard_v026,
        ),
        (
            "Consumed r4 validation-30 severe-regression guard",
            sev_r4,
            new_r4,
            guard_r4,
        ),
        (
            "Consumed v0.28 final-30 severe-regression guard",
            sev_v028,
            new_v028,
            guard_v028,
        ),
    ]
    for title, severe_now, new_severe, passed in guard_rows:
        print(title)
        print("-" * len(title))
        print(f"Severe case IDs now: {sorted(severe_now)}")
        print(f"New severe IDs outside historical set: {new_severe}")
        print(
            f"{'PASS' if passed else 'FAIL'}  "
            "no_new_severe_regressions"
        )
        print()

    print(
        "v0.29.0 r3 development regression gate: "
        + ("PASS" if gate_pass else "FAIL")
    )
    print("NOTE: all 240 cases are consumed development evidence.")
    print(
        "NOTE: fresh independent v0.29 evaluation evidence is still required "
        "after candidate freeze."
    )
    print()
    print("Largest disagreements")
    print("---------------------")
    print(largest.head(30).to_string(index=False))

    # Re-check lifecycle after writing artifacts.
    assert_consumed_evidence_lifecycle()

    if not gate_pass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
