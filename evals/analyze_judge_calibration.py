"""
Analyze agreement between human gold labels and semantic-relevance judge scores.

Run from the repository root:

    uv run python evals/analyze_judge_calibration.py `
      --run evals/runs/semantic_relevance/<RUN_ID>

Optional subset analysis:

    uv run python evals/analyze_judge_calibration.py `
      --run evals/runs/semantic_relevance/<RUN_ID> `
      --exclude-query-id Q01

Important methodology note
--------------------------
Q01-Q12 have all now been inspected and used during judge development.

Therefore the current 60-case calibration dataset is DEVELOPMENT/CALIBRATION
data, not an untouched holdout.

This analyzer deliberately calls an exclusion-based analysis a "subset" rather
than a "holdout" so generated artifacts do not accidentally make a
generalization claim that the data no longer supports.

Source-of-truth rule
--------------------
The analyzer no longer hard-codes the calibration dataset version.

Instead it reads `run_metadata.json` first and resolves the exact dataset
version recorded by the runner:

    calibration_dataset_version -> evals/datasets/
        semantic_relevance_calibration.v<version>.csv

This prevents the earlier provenance bug where a v0.2.0 judge run was analyzed
against v0.1.0 candidate metadata.

Primary metrics
---------------
- Exact agreement
- Within ±1 agreement
- Linear weighted Cohen's kappa

Secondary metrics / diagnostics
-------------------------------
- Quadratic weighted Cohen's kappa
- Confusion matrix
- Agreement by human score
- Human and judge score distributions
- Binary relevance precision / recall / F1 for 0 vs >0
- False-zero cases
- False-positive relevance cases
- Severe ordinal disagreements (absolute difference >= 2)
- Semantic structured-output mode counts
- Agreement by semantic generation mode
- Clear-score rule usage and agreement

The additional v0.7 diagnostics help distinguish:
- semantic facet-classification failures,
- deterministic aggregation failures,
- and structured-output fallback behavior.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.metrics import (
    cohen_kappa_score,
    confusion_matrix,
    precision_recall_fscore_support,
)


# =============================================================================
# Repository paths
# =============================================================================

REPO_ROOT = Path(__file__).resolve().parents[1]
EVALS_DIR = REPO_ROOT / "evals"
DATASETS_DIR = EVALS_DIR / "datasets"


# =============================================================================
# Generic helpers
# =============================================================================

def load_json(
    path: Path,
) -> dict[str, Any]:
    """
    Load one UTF-8 JSON object and fail with the concrete path if missing.
    """

    if not path.exists():
        raise FileNotFoundError(
            f"Required JSON file not found: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def resolve_run_dir(
    run_argument: str,
) -> Path:
    """
    Resolve --run as either an absolute path or repository-relative path.
    """

    run_dir = Path(
        run_argument
    )

    if not run_dir.is_absolute():
        run_dir = (
            REPO_ROOT
            / run_dir
        )

    if not run_dir.exists():
        raise FileNotFoundError(
            f"Run directory not found: {run_dir}"
        )

    return run_dir


def dataset_version_from_metadata(
    metadata: dict[str, Any],
) -> str:
    """
    Resolve dataset version from run metadata.

    `calibration_dataset_version` is the canonical key used by the current
    runner. `dataset_version` is accepted as a compatibility fallback for older
    runs.
    """

    version = (
        metadata.get(
            "calibration_dataset_version"
        )
        or metadata.get(
            "dataset_version"
        )
    )

    if not version:
        raise ValueError(
            "run_metadata.json does not contain "
            "'calibration_dataset_version' or 'dataset_version'."
        )

    return str(
        version
    )


def dataset_file_for_version(
    dataset_version: str,
) -> Path:
    """
    Construct the exact versioned calibration dataset path.
    """

    return (
        DATASETS_DIR
        / f"semantic_relevance_calibration.v{dataset_version}.csv"
    )


# =============================================================================
# Input / provenance validation
# =============================================================================

def validate_analysis_inputs(
    gold: pd.DataFrame,
    judged: pd.DataFrame,
    metadata: dict[str, Any],
    dataset_version: str,
) -> None:
    """
    Validate gold labels, judge results, and run provenance before joining.

    The goal is to fail loudly rather than produce numerically plausible
    metrics from mismatched artifacts.
    """

    required_gold_columns = {
        "case_id",
        "query_id",
        "query",
        "title",
        "human_score",
        "human_reason",
    }

    required_judge_columns = {
        "case_id",
        "judge_score",
        "judge_reason",
    }

    missing_gold = (
        required_gold_columns
        - set(
            gold.columns
        )
    )

    missing_judge = (
        required_judge_columns
        - set(
            judged.columns
        )
    )

    if missing_gold:
        raise ValueError(
            "Gold dataset is missing columns: "
            f"{sorted(missing_gold)}"
        )

    if missing_judge:
        raise ValueError(
            "Judge results are missing columns: "
            f"{sorted(missing_judge)}"
        )

    if gold[
        "case_id"
    ].duplicated().any():

        duplicates = gold.loc[
            gold[
                "case_id"
            ].duplicated(
                keep=False
            ),
            "case_id",
        ].tolist()

        raise ValueError(
            "Gold dataset contains duplicate case_id values: "
            f"{duplicates}"
        )

    if judged[
        "case_id"
    ].duplicated().any():

        duplicates = judged.loc[
            judged[
                "case_id"
            ].duplicated(
                keep=False
            ),
            "case_id",
        ].tolist()

        raise ValueError(
            "Judge results contain duplicate case_id values: "
            f"{duplicates}"
        )

    if not gold[
        "human_score"
    ].isin(
        [0, 1, 2, 3, 4]
    ).all():
        raise ValueError(
            "Gold human_score contains values outside the 0-4 rubric scale."
        )

    if not judged[
        "judge_score"
    ].isin(
        [0, 1, 2, 3, 4]
    ).all():
        raise ValueError(
            "Judge score contains values outside the 0-4 rubric scale."
        )

    # -------------------------------------------------------------------------
    # Dataset provenance recorded inside the CSV, when available.
    # -------------------------------------------------------------------------
    if "dataset_version" in gold.columns:

        declared_versions = set(
            gold[
                "dataset_version"
            ].astype(
                str
            )
        )

        if declared_versions != {
            dataset_version
        }:
            raise ValueError(
                "Gold dataset rows declare dataset versions "
                f"{sorted(declared_versions)}, but run metadata requires "
                f"{dataset_version!r}."
            )

    # -------------------------------------------------------------------------
    # Rubric provenance, when both run metadata and dataset expose it.
    # -------------------------------------------------------------------------
    metadata_rubric_version = metadata.get(
        "rubric_version"
    )

    if (
        metadata_rubric_version
        and "rubric_version" in gold.columns
    ):

        declared_rubric_versions = set(
            gold[
                "rubric_version"
            ].astype(
                str
            )
        )

        if declared_rubric_versions != {
            str(
                metadata_rubric_version
            )
        }:
            raise ValueError(
                "Gold dataset rubric_version does not match run metadata. "
                f"dataset={sorted(declared_rubric_versions)}, "
                f"run={metadata_rubric_version!r}"
            )


# =============================================================================
# Analysis scoping
# =============================================================================

def apply_exclusions(
    gold: pd.DataFrame,
    exclude_query_ids: list[str],
) -> pd.DataFrame:
    """
    Remove requested query IDs from analysis.

    This is a generic subset mechanism. It must not be interpreted as creating
    a new holdout after those cases have already been inspected during judge
    development.
    """

    if not exclude_query_ids:
        return gold.copy()

    available_query_ids = set(
        gold[
            "query_id"
        ].astype(
            str
        )
    )

    unknown_query_ids = (
        set(
            exclude_query_ids
        )
        - available_query_ids
    )

    if unknown_query_ids:
        raise ValueError(
            "Requested excluded query IDs were not found in the dataset: "
            f"{sorted(unknown_query_ids)}"
        )

    scoped_gold = gold[
        ~gold[
            "query_id"
        ]
        .astype(
            str
        )
        .isin(
            exclude_query_ids
        )
    ].copy()

    if scoped_gold.empty:
        raise ValueError(
            "All calibration cases were excluded. "
            "At least one case must remain for analysis."
        )

    return scoped_gold


# =============================================================================
# Comparison construction
# =============================================================================

def build_comparison(
    gold: pd.DataFrame,
    judged: pd.DataFrame,
) -> pd.DataFrame:
    """
    Join human labels to judge results by case_id and derive error fields.

    All judge diagnostic columns are retained. This is especially useful for
    v0.7+, where facet assessments, clear-score rules, and semantic generation
    modes explain how the final score was produced.
    """

    gold_columns = [
        "case_id",
        "query_id",
        "query",
        "title",
        "human_score",
        "human_reason",
    ]

    # Avoid duplicate gold/provenance columns if a future judge result happens
    # to persist them as diagnostics.
    judge_columns = [
        column
        for column in judged.columns
        if (
            column == "case_id"
            or column not in gold_columns
        )
    ]

    comparison = gold[
        gold_columns
    ].merge(
        judged[
            judge_columns
        ],
        on="case_id",
        how="inner",
        validate="one_to_one",
    )

    comparison[
        "human_score"
    ] = comparison[
        "human_score"
    ].astype(
        int
    )

    comparison[
        "judge_score"
    ] = comparison[
        "judge_score"
    ].astype(
        int
    )

    comparison[
        "signed_difference"
    ] = (
        comparison[
            "judge_score"
        ]
        - comparison[
            "human_score"
        ]
    )

    comparison[
        "absolute_difference"
    ] = comparison[
        "signed_difference"
    ].abs()

    comparison[
        "exact_match"
    ] = (
        comparison[
            "human_score"
        ]
        == comparison[
            "judge_score"
        ]
    )

    comparison[
        "within_one"
    ] = (
        comparison[
            "absolute_difference"
        ]
        <= 1
    )

    comparison[
        "human_relevant"
    ] = (
        comparison[
            "human_score"
        ]
        > 0
    )

    comparison[
        "judge_relevant"
    ] = (
        comparison[
            "judge_score"
        ]
        > 0
    )

    return comparison


# =============================================================================
# Metric helpers
# =============================================================================

def score_distribution(
    series: pd.Series,
) -> dict[str, int]:
    """
    Return all five score buckets, including buckets with zero observations.
    """

    counts = series.value_counts()

    return {
        str(
            score
        ): int(
            counts.get(
                score,
                0,
            )
        )
        for score in range(
            5
        )
    }


def agreement_group_summary(
    comparison: pd.DataFrame,
    group_column: str,
) -> pd.DataFrame:
    """
    Generic agreement breakdown for a diagnostic categorical column.
    """

    if group_column not in comparison.columns:
        return pd.DataFrame()

    working = comparison.copy()

    working[
        group_column
    ] = working[
        group_column
    ].fillna(
        "<none>"
    ).astype(
        str
    )

    return (
        working
        .groupby(
            group_column,
            as_index=False,
            dropna=False,
        )
        .agg(
            cases=(
                "case_id",
                "count",
            ),
            exact_agreement=(
                "exact_match",
                "mean",
            ),
            within_one_agreement=(
                "within_one",
                "mean",
            ),
            mean_absolute_difference=(
                "absolute_difference",
                "mean",
            ),
        )
        .sort_values(
            [
                "cases",
                group_column,
            ],
            ascending=[
                False,
                True,
            ],
        )
    )


# =============================================================================
# Output naming
# =============================================================================

def build_output_suffix(
    exclude_query_ids: list[str],
) -> str:
    """
    Produce stable filenames for full or subset analysis.
    """

    if not exclude_query_ids:
        return "all"

    safe_ids = "_".join(
        sorted(
            exclude_query_ids
        )
    )

    return (
        f"exclude_{safe_ids}"
    )


# =============================================================================
# Main analysis
# =============================================================================

def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Compare semantic-relevance judge scores against frozen human "
            "calibration labels using the artifact versions recorded by the run."
        )
    )

    parser.add_argument(
        "--run",
        required=True,
        help=(
            "Run directory created by run_judge_calibration.py. "
            "Example: "
            "evals/runs/semantic_relevance/20260910T045820Z"
        ),
    )

    parser.add_argument(
        "--exclude-query-id",
        action="append",
        default=[],
        dest="exclude_query_ids",
        help=(
            "Exclude a query ID from analysis. May be supplied multiple times. "
            "This creates a diagnostic subset, not a new untouched holdout."
        ),
    )

    args = parser.parse_args()

    exclude_query_ids = [
        str(
            query_id
        )
        for query_id
        in args.exclude_query_ids
    ]

    # =========================================================================
    # 1. Resolve run + metadata FIRST.
    #
    # Metadata owns the dataset version used by the run.
    # =========================================================================

    run_dir = resolve_run_dir(
        args.run
    )

    results_file = (
        run_dir
        / "judge_results.csv"
    )

    metadata_file = (
        run_dir
        / "run_metadata.json"
    )

    if not results_file.exists():
        raise FileNotFoundError(
            f"Judge results not found: {results_file}"
        )

    if not metadata_file.exists():
        raise FileNotFoundError(
            "run_metadata.json is required for provenance-safe analysis: "
            f"{metadata_file}"
        )

    metadata = load_json(
        metadata_file
    )

    dataset_version = dataset_version_from_metadata(
        metadata
    )

    dataset_file = dataset_file_for_version(
        dataset_version
    )

    if not dataset_file.exists():
        raise FileNotFoundError(
            "Gold dataset recorded by run metadata was not found: "
            f"{dataset_file}"
        )

    # =========================================================================
    # 2. Load exact gold dataset + blind judge results.
    # =========================================================================

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
        dtype={
            "case_id": str,
        },
        encoding="utf-8",
    )

    validate_analysis_inputs(
        gold=gold,
        judged=judged,
        metadata=metadata,
        dataset_version=dataset_version,
    )

    # =========================================================================
    # 3. Apply optional diagnostic exclusions.
    # =========================================================================

    scoped_gold = apply_exclusions(
        gold=gold,
        exclude_query_ids=exclude_query_ids,
    )

    # =========================================================================
    # 4. Join human and judge data.
    # =========================================================================

    comparison = build_comparison(
        gold=scoped_gold,
        judged=judged,
    )

    scoped_gold_cases = len(
        scoped_gold
    )

    compared_cases = len(
        comparison
    )

    if compared_cases == 0:
        raise ValueError(
            "No judged cases matched the selected analysis scope."
        )

    # =========================================================================
    # 5. Primary ordinal agreement metrics.
    # =========================================================================

    exact_agreement = float(
        comparison[
            "exact_match"
        ].mean()
    )

    within_one_agreement = float(
        comparison[
            "within_one"
        ].mean()
    )

    y_true = comparison[
        "human_score"
    ]

    y_pred = comparison[
        "judge_score"
    ]

    linear_weighted_kappa = float(
        cohen_kappa_score(
            y_true,
            y_pred,
            labels=[
                0,
                1,
                2,
                3,
                4,
            ],
            weights="linear",
        )
    )

    quadratic_weighted_kappa = float(
        cohen_kappa_score(
            y_true,
            y_pred,
            labels=[
                0,
                1,
                2,
                3,
                4,
            ],
            weights="quadratic",
        )
    )

    # =========================================================================
    # 6. Confusion matrix.
    # =========================================================================

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=[
            0,
            1,
            2,
            3,
            4,
        ],
    )

    confusion_df = pd.DataFrame(
        matrix,
        index=[
            f"human_{score}"
            for score
            in range(
                5
            )
        ],
        columns=[
            f"judge_{score}"
            for score
            in range(
                5
            )
        ],
    )

    # =========================================================================
    # 7. Agreement by human score.
    # =========================================================================

    agreement_by_human_score = (
        comparison
        .groupby(
            "human_score",
            as_index=False,
        )
        .agg(
            cases=(
                "case_id",
                "count",
            ),
            exact_agreement=(
                "exact_match",
                "mean",
            ),
            within_one_agreement=(
                "within_one",
                "mean",
            ),
            mean_absolute_difference=(
                "absolute_difference",
                "mean",
            ),
        )
    )

    # =========================================================================
    # 8. Binary 0-vs->0 diagnostic.
    #
    # This is NOT a replacement for the ordinal metrics. It tells us whether a
    # judge that is poorly calibrated on 0-4 might still be useful as a binary
    # relevance detector.
    # =========================================================================

    precision, recall, f1, _ = precision_recall_fscore_support(
        comparison[
            "human_relevant"
        ].astype(
            int
        ),
        comparison[
            "judge_relevant"
        ].astype(
            int
        ),
        average="binary",
        zero_division=0,
    )

    binary_tp = int(
        (
            comparison[
                "human_relevant"
            ]
            & comparison[
                "judge_relevant"
            ]
        ).sum()
    )

    binary_fp = int(
        (
            ~comparison[
                "human_relevant"
            ]
            & comparison[
                "judge_relevant"
            ]
        ).sum()
    )

    binary_fn = int(
        (
            comparison[
                "human_relevant"
            ]
            & ~comparison[
                "judge_relevant"
            ]
        ).sum()
    )

    binary_tn = int(
        (
            ~comparison[
                "human_relevant"
            ]
            & ~comparison[
                "judge_relevant"
            ]
        ).sum()
    )

    # =========================================================================
    # 9. v0.7 structured-output / aggregation diagnostics.
    # =========================================================================

    agreement_by_generation_mode = agreement_group_summary(
        comparison,
        "semantic_generation_mode",
    )

    agreement_by_clear_rule = agreement_group_summary(
        comparison,
        "clear_rule_applied",
    )

    generation_mode_counts = (
        comparison[
            "semantic_generation_mode"
        ]
        .fillna(
            "<not-recorded>"
        )
        .astype(
            str
        )
        .value_counts()
        .to_dict()
        if "semantic_generation_mode"
        in comparison.columns
        else {}
    )

    clear_rule_counts = (
        comparison[
            "clear_rule_applied"
        ]
        .fillna(
            "<none>"
        )
        .astype(
            str
        )
        .value_counts()
        .to_dict()
        if "clear_rule_applied"
        in comparison.columns
        else {}
    )

    # =========================================================================
    # 10. Error slices for manual review.
    # =========================================================================

    false_zeros = comparison[
        (
            comparison[
                "human_score"
            ]
            > 0
        )
        & (
            comparison[
                "judge_score"
            ]
            == 0
        )
    ].copy()

    false_positive_relevance = comparison[
        (
            comparison[
                "human_score"
            ]
            == 0
        )
        & (
            comparison[
                "judge_score"
            ]
            > 0
        )
    ].copy()

    severe_disagreements = comparison[
        comparison[
            "absolute_difference"
        ]
        >= 2
    ].copy()

    over_promotions = comparison[
        comparison[
            "signed_difference"
        ]
        >= 2
    ].copy()

    under_promotions = comparison[
        comparison[
            "signed_difference"
        ]
        <= -2
    ].copy()

    # =========================================================================
    # 11. Summary JSON.
    # =========================================================================

    analysis_scope = (
        "development_subset"
        if exclude_query_ids
        else "development_all_cases"
    )

    summary = {
        "analysis_scope": (
            analysis_scope
        ),
        "methodology_note": (
            "Q01-Q12 have all been inspected/used during judge development; "
            "this analysis is development/calibration evidence, not an unseen "
            "holdout generalization result."
        ),
        "excluded_query_ids": (
            sorted(
                exclude_query_ids
            )
        ),
        "dataset_version_from_run_metadata": (
            dataset_version
        ),
        "gold_dataset_file": (
            str(
                dataset_file
                .relative_to(
                    REPO_ROOT
                )
            )
        ),
        "gold_cases_in_scope": (
            scoped_gold_cases
        ),
        "cases_compared": (
            compared_cases
        ),
        "is_complete_for_scope": (
            compared_cases
            == scoped_gold_cases
        ),
        "exact_agreement": (
            exact_agreement
        ),
        "within_one_agreement": (
            within_one_agreement
        ),
        "linear_weighted_cohens_kappa": (
            linear_weighted_kappa
        ),
        "quadratic_weighted_cohens_kappa": (
            quadratic_weighted_kappa
        ),
        "max_absolute_difference": int(
            comparison[
                "absolute_difference"
            ].max()
        ),
        "human_score_distribution": (
            score_distribution(
                comparison[
                    "human_score"
                ]
            )
        ),
        "judge_score_distribution": (
            score_distribution(
                comparison[
                    "judge_score"
                ]
            )
        ),
        "binary_relevance_0_vs_positive": {
            "true_positive": (
                binary_tp
            ),
            "false_positive": (
                binary_fp
            ),
            "false_negative": (
                binary_fn
            ),
            "true_negative": (
                binary_tn
            ),
            "precision": float(
                precision
            ),
            "recall": float(
                recall
            ),
            "f1": float(
                f1
            ),
        },
        "error_counts": {
            "false_zeros": int(
                len(
                    false_zeros
                )
            ),
            "false_positive_relevance": int(
                len(
                    false_positive_relevance
                )
            ),
            "absolute_difference_ge_2": int(
                len(
                    severe_disagreements
                )
            ),
            "over_promotions_by_2_or_more": int(
                len(
                    over_promotions
                )
            ),
            "under_promotions_by_2_or_more": int(
                len(
                    under_promotions
                )
            ),
        },
        "semantic_generation_mode_counts": {
            str(
                key
            ): int(
                value
            )
            for key, value
            in generation_mode_counts.items()
        },
        "clear_rule_counts": {
            str(
                key
            ): int(
                value
            )
            for key, value
            in clear_rule_counts.items()
        },
        "run_metadata": (
            metadata
        ),
    }

    # =========================================================================
    # 12. Write analysis artifacts.
    # =========================================================================

    output_suffix = build_output_suffix(
        exclude_query_ids
    )

    comparison_file = (
        run_dir
        / f"human_vs_judge.{output_suffix}.csv"
    )

    confusion_file = (
        run_dir
        / f"confusion_matrix.{output_suffix}.csv"
    )

    agreement_by_score_file = (
        run_dir
        / f"agreement_by_human_score.{output_suffix}.csv"
    )

    summary_file = (
        run_dir
        / f"calibration_summary.{output_suffix}.json"
    )

    false_zeros_file = (
        run_dir
        / f"false_zeros.{output_suffix}.csv"
    )

    false_positive_file = (
        run_dir
        / f"false_positive_relevance.{output_suffix}.csv"
    )

    severe_file = (
        run_dir
        / f"severe_disagreements.{output_suffix}.csv"
    )

    generation_mode_file = (
        run_dir
        / f"agreement_by_generation_mode.{output_suffix}.csv"
    )

    clear_rule_file = (
        run_dir
        / f"agreement_by_clear_rule.{output_suffix}.csv"
    )

    comparison.sort_values(
        [
            "absolute_difference",
            "case_id",
        ],
        ascending=[
            False,
            True,
        ],
    ).to_csv(
        comparison_file,
        index=False,
        encoding="utf-8",
    )

    confusion_df.to_csv(
        confusion_file,
        encoding="utf-8",
    )

    agreement_by_human_score.to_csv(
        agreement_by_score_file,
        index=False,
        encoding="utf-8",
    )

    false_zeros.sort_values(
        [
            "absolute_difference",
            "case_id",
        ],
        ascending=[
            False,
            True,
        ],
    ).to_csv(
        false_zeros_file,
        index=False,
        encoding="utf-8",
    )

    false_positive_relevance.sort_values(
        [
            "absolute_difference",
            "case_id",
        ],
        ascending=[
            False,
            True,
        ],
    ).to_csv(
        false_positive_file,
        index=False,
        encoding="utf-8",
    )

    severe_disagreements.sort_values(
        [
            "absolute_difference",
            "case_id",
        ],
        ascending=[
            False,
            True,
        ],
    ).to_csv(
        severe_file,
        index=False,
        encoding="utf-8",
    )

    if not agreement_by_generation_mode.empty:
        agreement_by_generation_mode.to_csv(
            generation_mode_file,
            index=False,
            encoding="utf-8",
        )

    if not agreement_by_clear_rule.empty:
        agreement_by_clear_rule.to_csv(
            clear_rule_file,
            index=False,
            encoding="utf-8",
        )

    summary_file.write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    # =========================================================================
    # 13. Console report.
    # =========================================================================

    print(
        "Calibration summary"
    )

    print(
        "-------------------"
    )

    print(
        f"Analysis scope:            {analysis_scope}"
    )

    print(
        f"Dataset version:           {dataset_version}"
    )

    print(
        "Excluded query IDs:        "
        f"{sorted(exclude_query_ids) if exclude_query_ids else 'None'}"
    )

    print(
        f"Cases compared:            "
        f"{compared_cases}/{scoped_gold_cases}"
    )

    print(
        f"Exact agreement:           "
        f"{exact_agreement:.1%}"
    )

    print(
        f"Within ±1 agreement:       "
        f"{within_one_agreement:.1%}"
    )

    print(
        f"Linear weighted kappa:     "
        f"{linear_weighted_kappa:.3f}"
    )

    print(
        f"Quadratic weighted kappa:  "
        f"{quadratic_weighted_kappa:.3f}"
    )

    print()

    print(
        "Score distributions"
    )

    print(
        "-------------------"
    )

    print(
        "Human: "
        f"{summary['human_score_distribution']}"
    )

    print(
        "Judge: "
        f"{summary['judge_score_distribution']}"
    )

    print()

    print(
        "Binary relevance (0 vs >0)"
    )

    print(
        "--------------------------"
    )

    print(
        f"TP={binary_tp} FP={binary_fp} "
        f"FN={binary_fn} TN={binary_tn}"
    )

    print(
        f"Precision={precision:.1%} "
        f"Recall={recall:.1%} "
        f"F1={f1:.1%}"
    )

    print()

    print(
        "Error counts"
    )

    print(
        "------------"
    )

    print(
        f"False zeros:               {len(false_zeros)}"
    )

    print(
        f"False-positive relevance:  {len(false_positive_relevance)}"
    )

    print(
        f"|difference| >= 2:         {len(severe_disagreements)}"
    )

    print(
        f"Over-promotions >= 2:      {len(over_promotions)}"
    )

    print(
        f"Under-promotions >= 2:     {len(under_promotions)}"
    )

    print()

    print(
        "Confusion matrix"
    )

    print(
        "----------------"
    )

    print(
        confusion_df.to_string()
    )

    print()

    print(
        "Agreement by human score"
    )

    print(
        "------------------------"
    )

    print(
        agreement_by_human_score.to_string(
            index=False
        )
    )

    print()

    if generation_mode_counts:

        print(
            "Semantic generation modes"
        )

        print(
            "-------------------------"
        )

        print(
            generation_mode_counts
        )

        if not agreement_by_generation_mode.empty:
            print()
            print(
                agreement_by_generation_mode.to_string(
                    index=False
                )
            )

        print()

    if clear_rule_counts:

        print(
            "Clear-score rule usage"
        )

        print(
            "----------------------"
        )

        print(
            clear_rule_counts
        )

        if not agreement_by_clear_rule.empty:
            print()
            print(
                agreement_by_clear_rule.to_string(
                    index=False
                )
            )

        print()

    print(
        "Largest disagreements"
    )

    print(
        "---------------------"
    )

    disagreements = comparison[
        comparison[
            "absolute_difference"
        ]
        > 0
    ].sort_values(
        [
            "absolute_difference",
            "case_id",
        ],
        ascending=[
            False,
            True,
        ],
    )

    if disagreements.empty:
        print(
            "None"
        )

    else:
        print(
            disagreements[
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
            .head(
                20
            )
            .to_string(
                index=False
            )
        )

    print()

    print(
        f"Detailed comparison:       {comparison_file}"
    )

    print(
        f"Confusion matrix:          {confusion_file}"
    )

    print(
        f"Agreement by score:        {agreement_by_score_file}"
    )

    print(
        f"False-zero cases:          {false_zeros_file}"
    )

    print(
        f"False-positive relevance:  {false_positive_file}"
    )

    print(
        f"Severe disagreements:      {severe_file}"
    )

    print(
        f"Summary:                   {summary_file}"
    )


if __name__ == "__main__":
    main()
