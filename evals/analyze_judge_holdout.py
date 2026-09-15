"""
Analyze the frozen 30-case final unseen holdout without scikit-learn/scipy.

This is a drop-in replacement for evals/analyze_judge_holdout.py when
Windows Application Control blocks SciPy native DLLs.

Usage:
    uv run python evals/analyze_judge_holdout.py `
      --run evals/runs/semantic_relevance_final_holdout/<RUN_ID>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]

EXPECTED_JUDGE_CONFIG_VERSION = "0.15.0"
EXPECTED_EVALUATION_DATASET_VERSION = "1.0.0"
EXPECTED_EVALUATION_DATASET_ROLE = "final_holdout"
EXPECTED_EVALUATION_SCOPE = "unseen_final_holdout"


def resolve_run_dir(raw: str) -> Path:
    path = Path(raw)
    if not path.is_absolute():
        path = REPO_ROOT / path
    if not path.exists():
        raise FileNotFoundError(f"Run directory not found: {path}")
    return path


def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_dataset(metadata: dict) -> Path:
    raw = metadata.get("evaluation_dataset_file")
    if not raw:
        raise ValueError(
            "run_metadata.json does not contain evaluation_dataset_file."
        )
    path = Path(str(raw))
    if not path.is_absolute():
        path = REPO_ROOT / path
    if not path.exists():
        raise FileNotFoundError(
            f"Final-holdout dataset recorded in metadata was not found: {path}"
        )
    return path


def safe_rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def confusion_matrix_no_sklearn(
    human: pd.Series,
    judge: pd.Series,
    labels: list[int],
) -> list[list[int]]:
    """Rows = human score, columns = judge score."""
    pos = {label: i for i, label in enumerate(labels)}
    matrix = [[0 for _ in labels] for _ in labels]

    for h, j in zip(human.tolist(), judge.tolist()):
        matrix[pos[int(h)]][pos[int(j)]] += 1

    return matrix


def weighted_cohen_kappa(
    human: pd.Series,
    judge: pd.Series,
    labels: list[int],
    mode: str,
) -> float:
    """
    Weighted Cohen's kappa using disagreement weights.

    Equivalent to sklearn's linear/quadratic weighting for the fixed
    ordinal score set 0..4.
    """
    if mode not in {"linear", "quadratic"}:
        raise ValueError("mode must be 'linear' or 'quadratic'")

    n = len(human)
    if n == 0:
        raise ValueError("Cannot compute kappa with zero cases.")

    pos = {label: i for i, label in enumerate(labels)}
    k = len(labels)
    max_distance = k - 1

    observed = [[0.0 for _ in labels] for _ in labels]
    human_counts = [0.0 for _ in labels]
    judge_counts = [0.0 for _ in labels]

    for h, j in zip(human.tolist(), judge.tolist()):
        hi = pos[int(h)]
        ji = pos[int(j)]
        observed[hi][ji] += 1.0
        human_counts[hi] += 1.0
        judge_counts[ji] += 1.0

    observed_weighted = 0.0
    expected_weighted = 0.0

    for i in range(k):
        for j in range(k):
            distance = abs(i - j)

            if mode == "linear":
                weight = distance / max_distance
            else:
                weight = (distance ** 2) / (max_distance ** 2)

            observed_prob = observed[i][j] / n
            expected_prob = (
                (human_counts[i] / n)
                * (judge_counts[j] / n)
            )

            observed_weighted += weight * observed_prob
            expected_weighted += weight * expected_prob

    if expected_weighted == 0:
        return 1.0 if observed_weighted == 0 else 0.0

    return 1.0 - (observed_weighted / expected_weighted)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze frozen v0.15 final unseen holdout run."
    )
    parser.add_argument("--run", required=True)
    args = parser.parse_args()

    run_dir = resolve_run_dir(args.run)
    metadata = load_json(run_dir / "run_metadata.json")

    expected = {
        "judge_config_version": EXPECTED_JUDGE_CONFIG_VERSION,
        "evaluation_dataset_version": EXPECTED_EVALUATION_DATASET_VERSION,
        "evaluation_dataset_role": EXPECTED_EVALUATION_DATASET_ROLE,
        "evaluation_scope": EXPECTED_EVALUATION_SCOPE,
    }

    mismatches = []
    for key, value in expected.items():
        actual = metadata.get(key)
        if str(actual) != str(value):
            mismatches.append(
                f"{key}: run={actual!r}, expected={value!r}"
            )

    if mismatches:
        raise ValueError(
            "Run is not the expected frozen v0.15 final-holdout run:\n- "
            + "\n- ".join(mismatches)
        )

    dataset_file = resolve_dataset(metadata)
    results_file = run_dir / "judge_results.csv"

    if not results_file.exists():
        raise FileNotFoundError(f"Judge results not found: {results_file}")

    gold = pd.read_csv(
        dataset_file,
        dtype={
            "case_id": str,
            "query_id": str,
            "isbn13": str,
        },
        encoding="utf-8",
    )

    judged = pd.read_csv(
        results_file,
        dtype={"case_id": str},
        encoding="utf-8",
    )

    if len(gold) != 30:
        raise ValueError(
            f"Expected 30 final-holdout gold cases, found {len(gold)}."
        )

    if gold["case_id"].duplicated().any():
        raise ValueError("Gold dataset contains duplicate case_id values.")

    if judged["case_id"].duplicated().any():
        raise ValueError("Judge results contain duplicate case_id values.")

    comparison = gold[
        [
            "case_id",
            "query_id",
            "query",
            "title",
            "human_score",
            "human_reason",
        ]
    ].merge(
        judged,
        on="case_id",
        how="inner",
        validate="one_to_one",
    )

    if len(comparison) != 30:
        missing = sorted(
            set(gold["case_id"])
            - set(judged["case_id"])
        )
        raise ValueError(
            f"Final-holdout run is incomplete: compared {len(comparison)}/30. "
            f"Missing judge results: {missing}"
        )

    comparison["human_score"] = comparison["human_score"].astype(int)
    comparison["judge_score"] = comparison["judge_score"].astype(int)

    comparison["signed_difference"] = (
        comparison["judge_score"]
        - comparison["human_score"]
    )
    comparison["absolute_difference"] = (
        comparison["signed_difference"].abs()
    )
    comparison["exact_match"] = (
        comparison["human_score"]
        == comparison["judge_score"]
    )
    comparison["within_one"] = (
        comparison["absolute_difference"] <= 1
    )

    human = comparison["human_score"]
    judge = comparison["judge_score"]
    labels = [0, 1, 2, 3, 4]

    exact = float(comparison["exact_match"].mean())
    within_one = float(comparison["within_one"].mean())
    mae = float(comparison["absolute_difference"].mean())

    linear_kappa = weighted_cohen_kappa(
        human,
        judge,
        labels,
        "linear",
    )
    quadratic_kappa = weighted_cohen_kappa(
        human,
        judge,
        labels,
        "quadratic",
    )

    human_positive = human > 0
    judge_positive = judge > 0

    tp = int((human_positive & judge_positive).sum())
    fp = int((~human_positive & judge_positive).sum())
    fn = int((human_positive & ~judge_positive).sum())
    tn = int((~human_positive & ~judge_positive).sum())

    precision = safe_rate(tp, tp + fp)
    recall = safe_rate(tp, tp + fn)
    f1 = safe_rate(2 * precision * recall, precision + recall)

    severe = int((comparison["absolute_difference"] >= 2).sum())
    over2 = int((comparison["signed_difference"] >= 2).sum())
    under2 = int((comparison["signed_difference"] <= -2).sum())
    false_zero = int(((human > 0) & (judge == 0)).sum())
    false_positive = int(((human == 0) & (judge > 0)).sum())

    matrix = pd.DataFrame(
        confusion_matrix_no_sklearn(
            human,
            judge,
            labels,
        ),
        index=[f"human_{x}" for x in labels],
        columns=[f"judge_{x}" for x in labels],
    )

    agreement = (
        comparison
        .groupby("human_score", as_index=False)
        .agg(
            cases=("case_id", "count"),
            exact_agreement=("exact_match", "mean"),
            within_one_agreement=("within_one", "mean"),
            mean_absolute_difference=("absolute_difference", "mean"),
        )
    )

    human_dist = {
        str(k): int(v)
        for k, v
        in human.value_counts().sort_index().items()
    }
    judge_dist = {
        str(k): int(v)
        for k, v
        in judge.value_counts().sort_index().items()
    }

    generation_mode_counts = {}
    if "semantic_generation_mode" in comparison.columns:
        generation_mode_counts = {
            str(k): int(v)
            for k, v
            in comparison["semantic_generation_mode"]
            .fillna("<none>")
            .value_counts()
            .items()
        }

    clear_rule_counts = {}
    if "clear_rule_applied" in comparison.columns:
        clear_rule_counts = {
            str(k): int(v)
            for k, v
            in comparison["clear_rule_applied"]
            .fillna("<none>")
            .replace("", "<none>")
            .value_counts()
            .items()
        }

    cue_severe_false_positives = 0
    cue_summary = {}

    if "deterministic_direct_cue_count" in comparison.columns:
        cue_count = pd.to_numeric(
            comparison["deterministic_direct_cue_count"],
            errors="coerce",
        ).fillna(0).astype(int)

        comparison["has_deterministic_direct_cue"] = cue_count > 0

        cue_rows = comparison[
            comparison["has_deterministic_direct_cue"]
        ]
        no_cue_rows = comparison[
            ~comparison["has_deterministic_direct_cue"]
        ]

        def group_stats(frame: pd.DataFrame) -> dict:
            if frame.empty:
                return {
                    "cases": 0,
                    "exact_agreement": None,
                    "within_one_agreement": None,
                    "mean_absolute_difference": None,
                    "false_positive_relevance": 0,
                    "absolute_difference_ge_2": 0,
                }

            return {
                "cases": int(len(frame)),
                "exact_agreement": float(
                    frame["exact_match"].mean()
                ),
                "within_one_agreement": float(
                    frame["within_one"].mean()
                ),
                "mean_absolute_difference": float(
                    frame["absolute_difference"].mean()
                ),
                "false_positive_relevance": int(
                    (
                        (frame["human_score"] == 0)
                        & (frame["judge_score"] > 0)
                    ).sum()
                ),
                "absolute_difference_ge_2": int(
                    (frame["absolute_difference"] >= 2).sum()
                ),
            }

        cue_summary = {
            "cue_present": group_stats(cue_rows),
            "cue_absent": group_stats(no_cue_rows),
        }

        cue_severe_false_positives = int(
            (
                comparison["has_deterministic_direct_cue"]
                & (comparison["human_score"] == 0)
                & (comparison["judge_score"] >= 2)
            ).sum()
        )

    checks = {
        "within_one_ge_0_95": within_one >= 0.95,
        "exact_ge_0_50": exact >= 0.50,
        "quadratic_kappa_ge_0_80": quadratic_kappa >= 0.80,
        "linear_kappa_ge_0_60": linear_kappa >= 0.60,
        "absolute_difference_ge_2_le_1_case": severe <= 1,
        "binary_precision_ge_0_90": precision >= 0.90,
        "binary_recall_ge_0_80": recall >= 0.80,
        "deterministic_cue_severe_false_positives_eq_0": (
            cue_severe_false_positives == 0
        ),
    }

    transport_overrides = metadata.get("transport_overrides", [])
    if not isinstance(transport_overrides, list):
        transport_overrides = [transport_overrides]
    transport_override_case_ids = [
        str(item.get("case_id"))
        for item in transport_overrides
        if isinstance(item, dict) and item.get("case_id") is not None
    ]

    # These historical thresholds are shown only for comparability with the
    # development/validation decision. The final holdout is not a tuning gate.
    finality_note = (
        "This is the one-time final holdout result for frozen Judge v0.15.0. "
        "Reference thresholds are descriptive only; do not tune or relabel "
        "against this holdout."
    )

    summary = {
        "finality_note": finality_note,
        "analysis_scope": EXPECTED_EVALUATION_SCOPE,
        "evaluation_dataset_version": (
            EXPECTED_EVALUATION_DATASET_VERSION
        ),
        "cases_compared": int(len(comparison)),
        "exact_agreement": exact,
        "within_one_agreement": within_one,
        "mean_absolute_difference": mae,
        "linear_weighted_cohens_kappa": linear_kappa,
        "quadratic_weighted_cohens_kappa": quadratic_kappa,
        "max_absolute_difference": int(
            comparison["absolute_difference"].max()
        ),
        "human_score_distribution": human_dist,
        "judge_score_distribution": judge_dist,
        "binary_relevance_0_vs_positive": {
            "true_positive": tp,
            "false_positive": fp,
            "false_negative": fn,
            "true_negative": tn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        },
        "error_counts": {
            "false_zeros": false_zero,
            "false_positive_relevance": false_positive,
            "absolute_difference_ge_2": severe,
            "over_promotions_by_2_or_more": over2,
            "under_promotions_by_2_or_more": under2,
        },
        "semantic_generation_mode_counts": generation_mode_counts,
        "clear_rule_counts": clear_rule_counts,
        "deterministic_cue_analysis": cue_summary,
        "reference_threshold_checks": checks,
        "all_reference_thresholds_met": bool(
            all(checks.values())
        ),
        "transport_recovery": {
            "override_count": len(transport_override_case_ids),
            "case_ids": transport_override_case_ids,
        },
        "run_metadata": metadata,
    }

    by_query = (
        comparison
        .groupby(["query_id", "query"], as_index=False)
        .agg(
            cases=("case_id", "count"),
            exact_agreement=("exact_match", "mean"),
            within_one_agreement=("within_one", "mean"),
            mean_absolute_difference=("absolute_difference", "mean"),
            severe_disagreements=("absolute_difference", lambda s: int((s >= 2).sum())),
        )
    )

    largest = comparison.sort_values(
        ["absolute_difference", "case_id"],
        ascending=[False, True],
    )[
        [
            "case_id",
            "query_id",
            "title",
            "human_score",
            "judge_score",
            "signed_difference",
            "absolute_difference",
        ]
    ]

    comparison.to_csv(
        run_dir / "human_vs_judge.final_holdout.csv",
        index=False,
        encoding="utf-8",
    )
    matrix.to_csv(
        run_dir / "confusion_matrix.final_holdout.csv",
        encoding="utf-8",
    )
    agreement.to_csv(
        run_dir / "agreement_by_human_score.final_holdout.csv",
        index=False,
        encoding="utf-8",
    )
    by_query.to_csv(
        run_dir / "agreement_by_query.final_holdout.csv",
        index=False,
        encoding="utf-8",
    )
    largest.to_csv(
        run_dir / "largest_disagreements.final_holdout.csv",
        index=False,
        encoding="utf-8",
    )
    (
        run_dir
        / "final_holdout_summary.json"
    ).write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    print("Final unseen holdout summary")
    print("-------------------------")
    print(f"Cases compared:            {len(comparison)}/30")
    print(f"Exact agreement:           {exact:.1%}")
    print(f"Within ±1 agreement:       {within_one:.1%}")
    print(f"Mean absolute difference:  {mae:.3f}")
    print(f"Linear weighted kappa:     {linear_kappa:.3f}")
    print(f"Quadratic weighted kappa:  {quadratic_kappa:.3f}")
    print()
    print("Binary relevance (0 vs >0)")
    print("--------------------------")
    print(f"TP={tp} FP={fp} FN={fn} TN={tn}")
    print(
        f"Precision={precision:.1%} "
        f"Recall={recall:.1%} "
        f"F1={f1:.1%}"
    )
    print()
    print("Error counts")
    print("------------")
    print(f"False zeros:               {false_zero}")
    print(f"False-positive relevance:  {false_positive}")
    print(f"|difference| >= 2:         {severe}")
    print(f"Over-promotions >= 2:      {over2}")
    print(f"Under-promotions >= 2:     {under2}")
    print()
    print("Transport recovery")
    print("------------------")
    print(f"Direct-transport overrides: {len(transport_override_case_ids)}")
    if transport_override_case_ids:
        print("Cases: " + ", ".join(transport_override_case_ids))
    print()
    print("Validation-era reference thresholds (descriptive only)")
    print("------------------------------------------------------")
    for name, passed in checks.items():
        print(f"{'PASS' if passed else 'FAIL':4}  {name}")
    print()
    print(
        "Reference-threshold comparison: "
        + ("ALL MET" if all(checks.values()) else "SOME NOT MET")
    )
    print()
    print("Finality note")
    print("-------------")
    print(finality_note)
    print()
    print("Largest disagreements")
    print("---------------------")
    print(largest.head(15).to_string(index=False))


if __name__ == "__main__":
    main()
