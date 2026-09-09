"""
Build the human-labelled calibration candidate dataset for the
Semantic Recommendation Relevance judge.

Run from repository root:

    uv run python evals/build_calibration_candidates.py

The script deliberately DOES NOT assign human relevance labels.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import random
from pathlib import Path


# ------------------------------------------------------------------
# Versioning
# ------------------------------------------------------------------

DATASET_VERSION = "0.1.0"
RUBRIC_VERSION = "0.1.0"


# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]

EVALS_DIR = REPO_ROOT / "evals"
DATASETS_DIR = EVALS_DIR / "datasets"
RUBRICS_DIR = EVALS_DIR / "rubrics" / "semantic_relevance"

QUERY_FILE = DATASETS_DIR / "calibration_queries.json"

RUBRIC_FILE = (
    RUBRICS_DIR
    / f"semantic_relevance_rubric.v{RUBRIC_VERSION}.json"
)

OUTPUT_FILE = (
    DATASETS_DIR
    / f"semantic_relevance_calibration.v{DATASET_VERSION}.csv"
)


# ------------------------------------------------------------------
# Candidate sampling configuration
# ------------------------------------------------------------------

RANKS_TO_SAMPLE = [1, 5, 20, 50]
RANDOM_SEED = 42


def load_app_module():
    app_path = REPO_ROOT / "gradio-dashboard.py"

    spec = importlib.util.spec_from_file_location(
        "book_recommender_app",
        app_path,
    )

    module = importlib.util.module_from_spec(spec)

    assert spec.loader is not None

    spec.loader.exec_module(module)

    return module


def isbn_from_document(page_content: str) -> int:
    return int(
        page_content
        .strip('"')
        .split()[0]
    )


def row_for_isbn(books, isbn: int):

    matches = books[
        books["isbn13"] == isbn
    ]

    if matches.empty:
        return None

    return matches.iloc[0]


def validate_inputs():

    if not QUERY_FILE.exists():
        raise FileNotFoundError(
            f"Calibration query file not found:\n{QUERY_FILE}"
        )

    if not RUBRIC_FILE.exists():
        raise FileNotFoundError(
            f"Rubric version {RUBRIC_VERSION} not found:\n"
            f"{RUBRIC_FILE}"
        )

    if OUTPUT_FILE.exists():
        raise FileExistsError(
            f"Calibration dataset already exists:\n"
            f"{OUTPUT_FILE}\n\n"
            "Do not overwrite a versioned calibration dataset. "
            "Increment DATASET_VERSION instead."
        )


def main():

    validate_inputs()

    app = load_app_module()

    with QUERY_FILE.open(
        "r",
        encoding="utf-8",
    ) as file:

        queries = json.load(file)

    rng = random.Random(
        RANDOM_SEED
    )

    records = []

    for q in queries:

        query_id = q["query_id"]
        query = q["query"]
        query_slice = q["slice"]

        # Query Chroma directly so candidate selection uses the
        # actual semantic ranking before any DataFrame reordering.
        recs = app.db_books.similarity_search(
            query,
            k=max(RANKS_TO_SAMPLE),
        )

        retrieved_isbns = [
            isbn_from_document(
                rec.page_content
            )
            for rec in recs
        ]

        # ----------------------------------------------------------
        # Sample selected semantic ranks
        # ----------------------------------------------------------

        for rank in RANKS_TO_SAMPLE:

            if rank > len(retrieved_isbns):
                continue

            isbn = retrieved_isbns[
                rank - 1
            ]

            row = row_for_isbn(
                app.books,
                isbn,
            )

            if row is None:
                continue

            records.append(
                {
                    "case_id":
                        f"{query_id}_R{rank:02d}",

                    "query_id":
                        query_id,

                    "query":
                        query,

                    "query_slice":
                        query_slice,

                    "candidate_source":
                        "semantic_retrieval",

                    "retrieval_rank":
                        rank,

                    "isbn13":
                        int(row["isbn13"]),

                    "title":
                        row["title"],

                    "authors":
                        row["authors"],

                    "description":
                        row["description"],

                    "dataset_version":
                        DATASET_VERSION,

                    "rubric_version":
                        RUBRIC_VERSION,

                    "human_score":
                        "",

                    "human_reason":
                        "",

                    "review_status":
                        "UNLABELLED",
                }
            )

        # ----------------------------------------------------------
        # Add one deterministic random negative candidate
        # ----------------------------------------------------------

        excluded = set(
            retrieved_isbns
        )

        possible_indices = [
            idx
            for idx, isbn
            in app.books["isbn13"].items()
            if int(isbn) not in excluded
        ]

        if possible_indices:

            random_idx = rng.choice(
                possible_indices
            )

            row = app.books.loc[
                random_idx
            ]

            records.append(
                {
                    "case_id":
                        f"{query_id}_NEG",

                    "query_id":
                        query_id,

                    "query":
                        query,

                    "query_slice":
                        query_slice,

                    "candidate_source":
                        "random_negative_candidate",

                    "retrieval_rank":
                        "",

                    "isbn13":
                        int(row["isbn13"]),

                    "title":
                        row["title"],

                    "authors":
                        row["authors"],

                    "description":
                        row["description"],

                    "dataset_version":
                        DATASET_VERSION,

                    "rubric_version":
                        RUBRIC_VERSION,

                    "human_score":
                        "",

                    "human_reason":
                        "",

                    "review_status":
                        "UNLABELLED",
                }
            )

    fieldnames = [
        "case_id",
        "query_id",
        "query",
        "query_slice",
        "candidate_source",
        "retrieval_rank",
        "isbn13",
        "title",
        "authors",
        "description",
        "dataset_version",
        "rubric_version",
        "human_score",
        "human_reason",
        "review_status",
    ]

    DATASETS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_FILE.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(records)

    print(
        f"Wrote {len(records)} calibration candidates"
    )

    print(
        f"Dataset version : {DATASET_VERSION}"
    )

    print(
        f"Rubric version  : {RUBRIC_VERSION}"
    )

    print(
        f"Rubric          : {RUBRIC_FILE}"
    )

    print(
        f"Output          : {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()