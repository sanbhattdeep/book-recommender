"""Create a formatted Excel labeling workbook from the blind CSV.

This script intentionally reads a dedicated human-labeling score guide rather
than assuming the canonical rubric JSON contains a ``scores`` object.

Recommended command:
  uv run --with xlsxwriter python evals/prepare_fresh_unseen_labeling_workbook.py
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from fresh_unseen_contract import REPO_ROOT

BLIND = (
    REPO_ROOT
    / "evals"
    / "datasets"
    / "semantic_relevance_fresh_unseen_pool.v2.0.0.BLIND_LABELING.csv"
)

OUT = (
    REPO_ROOT
    / "evals"
    / "datasets"
    / "semantic_relevance_fresh_unseen_pool.v2.0.0.BLIND_LABELING.xlsx"
)

SCORE_GUIDE_FILE = (
    REPO_ROOT
    / "evals"
    / "fresh_unseen"
    / "human_labeling_score_guide.v0.1.0.json"
)

EXPECTED_RUBRIC_VERSION = "0.1.0"


def load_score_guide() -> dict:
    if not SCORE_GUIDE_FILE.exists():
        raise FileNotFoundError(
            "Human-labeling score guide not found: "
            f"{SCORE_GUIDE_FILE}"
        )

    guide = json.loads(
        SCORE_GUIDE_FILE.read_text(encoding="utf-8")
    )

    if guide.get("rubric_version") != EXPECTED_RUBRIC_VERSION:
        raise ValueError(
            "Human-labeling score guide rubric_version mismatch: "
            f"{guide.get('rubric_version')!r}; "
            f"expected {EXPECTED_RUBRIC_VERSION!r}."
        )

    scores = guide.get("scores")

    if not isinstance(scores, dict):
        raise ValueError(
            "Human-labeling score guide is missing the 'scores' object."
        )

    expected = {"0", "1", "2", "3", "4"}

    if set(scores) != expected:
        raise ValueError(
            "Human-labeling score guide must define exactly scores 0-4."
        )

    for score in expected:
        item = scores[score]

        if not isinstance(item, dict):
            raise ValueError(
                f"Score {score} entry must be an object."
            )

        if not str(item.get("label", "")).strip():
            raise ValueError(
                f"Score {score} is missing a label."
            )

        if not str(item.get("definition", "")).strip():
            raise ValueError(
                f"Score {score} is missing a definition."
            )

    return guide


def main() -> None:
    try:
        import xlsxwriter  # noqa: F401
    except ImportError as error:
        raise RuntimeError(
            "xlsxwriter is required. Run with: "
            "uv run --with xlsxwriter python "
            "evals/prepare_fresh_unseen_labeling_workbook.py"
        ) from error

    if not BLIND.exists():
        raise FileNotFoundError(
            "Blind labeling CSV not found. Create it first with:\n"
            "  uv run python "
            "evals/prepare_fresh_unseen_blind_labeling_csv.py\n"
            f"Expected: {BLIND}"
        )

    df = pd.read_csv(
        BLIND,
        encoding="utf-8",
        keep_default_na=False,
    )

    required_columns = [
        "blind_label_id",
        "query",
        "title",
        "authors",
        "description",
        "human_score",
        "human_reason",
        "review_status",
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            "Blind labeling CSV is missing required columns: "
            f"{missing}"
        )

    guide = load_score_guide()

    with pd.ExcelWriter(
        OUT,
        engine="xlsxwriter",
    ) as writer:
        df.to_excel(
            writer,
            sheet_name="Labeling",
            index=False,
        )

        workbook = writer.book
        worksheet = writer.sheets["Labeling"]

        header = workbook.add_format(
            {
                "bold": True,
                "bg_color": "#1F4E78",
                "font_color": "white",
                "border": 1,
                "valign": "top",
            }
        )

        wrap = workbook.add_format(
            {
                "text_wrap": True,
                "valign": "top",
                "border": 1,
            }
        )

        score_format = workbook.add_format(
            {
                "align": "center",
                "valign": "top",
                "border": 1,
            }
        )

        for column_index, name in enumerate(df.columns):
            worksheet.write(
                0,
                column_index,
                name,
                header,
            )

        widths = {
            "blind_label_id": 12,
            "query": 42,
            "title": 34,
            "authors": 28,
            "description": 80,
            "human_score": 13,
            "human_reason": 70,
            "review_status": 16,
        }

        for column_index, name in enumerate(df.columns):
            worksheet.set_column(
                column_index,
                column_index,
                widths.get(name, 20),
                score_format
                if name == "human_score"
                else wrap,
            )

        worksheet.freeze_panes(1, 0)

        worksheet.autofilter(
            0,
            0,
            len(df),
            len(df.columns) - 1,
        )

        score_column = df.columns.get_loc("human_score")
        status_column = df.columns.get_loc("review_status")

        worksheet.data_validation(
            1,
            score_column,
            len(df),
            score_column,
            {
                "validate": "list",
                "source": [0, 1, 2, 3, 4],
            },
        )

        worksheet.data_validation(
            1,
            status_column,
            len(df),
            status_column,
            {
                "validate": "list",
                "source": [
                    "UNLABELLED",
                    "LABELLED",
                ],
            },
        )

        instructions = workbook.add_worksheet(
            "Instructions"
        )

        title_format = workbook.add_format(
            {
                "bold": True,
                "font_size": 16,
                "bg_color": "#D9EAF7",
            }
        )

        subheading = workbook.add_format(
            {
                "bold": True,
                "font_size": 12,
            }
        )

        text_format = workbook.add_format(
            {
                "text_wrap": True,
                "valign": "top",
            }
        )

        instructions.set_column(
            "A:A",
            24,
        )

        instructions.set_column(
            "B:B",
            110,
        )

        instructions.merge_range(
            "A1:B1",
            "Blind semantic-relevance labeling",
            title_format,
        )

        instruction_rows = [
            (
                "Goal",
                "Judge each query-book pair independently from the supplied description only.",
            ),
            (
                "Evidence",
                guide["evidence_rule"],
            ),
            (
                "Tie-break",
                guide["tie_break_rule"],
            ),
            (
                "Blindness",
                "Do not use retrieval rank, candidate source, ISBN, old judge outputs, "
                "or old human labels. Opaque B### IDs intentionally hide sampling identity.",
            ),
            (
                "Completion",
                "Every row requires human_score 0-4, a non-empty human_reason, "
                "and review_status=LABELLED.",
            ),
        ]

        row = 2

        for key, value in instruction_rows:
            instructions.write(
                row,
                0,
                key,
                subheading,
            )
            instructions.write(
                row,
                1,
                value,
                text_format,
            )
            row += 2

        instructions.write(
            row,
            0,
            "Score",
            subheading,
        )
        instructions.write(
            row,
            1,
            "Definition",
            subheading,
        )
        row += 1

        for score in [4, 3, 2, 1, 0]:
            item = guide["scores"][str(score)]

            instructions.write(
                row,
                0,
                f"{score} — {item['label']}",
                subheading,
            )

            instructions.write(
                row,
                1,
                item["definition"],
                text_format,
            )

            row += 1

    print(
        "Wrote blind labeling workbook: "
        f"{OUT}"
    )


if __name__ == "__main__":
    main()
