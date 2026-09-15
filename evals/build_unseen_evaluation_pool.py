"""
Build a 60-case UNSEEN human-labeling pool for semantic relevance validation.

Run from the repository root:

    uv run python evals/build_unseen_evaluation_pool.py

Purpose
-------
Create 60 query-book pairs that are completely unseen relative to the existing
development/calibration dataset:

    12 frozen queries
    x (4 semantic-retrieval candidates + 1 random candidate)
    = 60 cases

The script deliberately:
- uses the application's existing raw Chroma similarity ranking;
- excludes every ISBN present in the development dataset;
- prevents duplicate ISBNs across the new 60-case pool;
- writes blank human_score / human_reason fields;
- NEVER runs the LLM judge.

Sampling targets
----------------
For each query the script aims for raw retrieval ranks:

    2, 10, 30, 70

If a target rank points to a development-set ISBN or to a book already selected
elsewhere in this new pool, the script scans forward to the next eligible raw
retrieval result and records that book's ACTUAL raw retrieval rank.

One deterministic random corpus book is then added for each query.

Important
---------
Do not run the judge on this pool before all 60 human labels are complete and
the validation/final-holdout split has been frozen.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import random
import re
from pathlib import Path

# =============================================================================
# Configuration
# =============================================================================

REPO_ROOT = Path(__file__).resolve().parents[1]
EVALS_DIR = REPO_ROOT / "evals"
DATASETS_DIR = EVALS_DIR / "datasets"

POOL_VERSION = "1.0.0"
RUBRIC_VERSION = "0.1.0"

TARGET_RANKS = [2, 10, 30, 70]
SEARCH_DEPTH = 250
RANDOM_SEED = 20260911

OUTPUT_FILE = (
    DATASETS_DIR
    / f"semantic_relevance_unseen_pool.v{POOL_VERSION}.csv"
)

# Prefer the post-adjudication development dataset if present. v0.4.0 contains
# the same candidate identities, so it is a safe fallback for duplicate
# exclusion if v0.5.0 has not yet been written locally.
DEVELOPMENT_DATASET_CANDIDATES = [
    DATASETS_DIR / "semantic_relevance_calibration.v0.5.0.csv",
    DATASETS_DIR / "semantic_relevance_calibration.v0.4.0.csv",
    DATASETS_DIR / "semantic_relevance_calibration.v0.3.0.csv",
]

QUERY_FILE_CANDIDATES = [
    DATASETS_DIR / "calibration_queries.json",
    EVALS_DIR / "calibration_queries.json",
]


# =============================================================================
# Helpers
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the 60-case unseen semantic-relevance human-labeling pool."
        )
    )

    parser.add_argument(
        "--development-dataset",
        type=Path,
        default=None,
        help=(
            "Optional explicit path to the existing development/calibration CSV. "
            "Used only to exclude previously seen ISBNs."
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_FILE,
        help=f"Output CSV path. Default: {OUTPUT_FILE}",
    )

    return parser.parse_args()


def resolve_existing_path(
    explicit: Path | None,
    candidates: list[Path],
    label: str,
) -> Path:
    if explicit is not None:
        path = explicit
        if not path.is_absolute():
            path = REPO_ROOT / path
        if not path.exists():
            raise FileNotFoundError(
                f"{label} not found: {path}"
            )
        return path

    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError(
        f"Could not locate {label}. Tried:\n"
        + "\n".join(f"  - {path}" for path in candidates)
    )


def load_app_module():
    app_path = REPO_ROOT / "gradio-dashboard.py"

    if not app_path.exists():
        raise FileNotFoundError(
            f"Application module not found: {app_path}"
        )

    spec = importlib.util.spec_from_file_location(
        "book_recommender_app",
        app_path,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"Unable to load application module from {app_path}"
        )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    for required_attr in ["db_books", "books"]:
        if not hasattr(module, required_attr):
            raise AttributeError(
                f"gradio-dashboard.py does not expose app.{required_attr}"
            )

    return module


def canonical_isbn(value) -> str:
    """
    Convert catalog / CSV ISBN values into stable 13-digit strings.
    """
    raw = str(value).strip()

    # Handle values such as 9781234567890.0 without scientific notation loss.
    if re.fullmatch(r"\d+\.0", raw):
        raw = raw[:-2]

    if not re.fullmatch(r"\d{13}", raw):
        raise ValueError(
            f"Expected a 13-digit ISBN, got {value!r}"
        )

    return raw


def isbn_from_document(page_content: str) -> str:
    """
    Chroma documents are stored as '<isbn13> <description>'.
    """
    raw = page_content.strip().strip('"').split()[0]
    return canonical_isbn(raw)


def row_for_isbn(books, isbn: str):
    isbn_series = books["isbn13"].map(canonical_isbn)
    matches = books[isbn_series == isbn]

    if matches.empty:
        return None

    return matches.iloc[0]


def load_seen_isbns(dataset_path: Path) -> set[str]:
    import pandas as pd

    development = pd.read_csv(
        dataset_path,
        dtype={"isbn13": str},
        encoding="utf-8",
    )

    required = {"case_id", "isbn13"}

    missing = required - set(development.columns)
    if missing:
        raise ValueError(
            f"Development dataset missing columns: {sorted(missing)}"
        )

    seen = {
        canonical_isbn(value)
        for value in development["isbn13"].dropna()
    }

    if not seen:
        raise ValueError(
            "Development dataset contained no usable ISBNs."
        )

    return seen


def select_ranked_candidates(
    retrieved_isbns: list[str],
    seen_development_isbns: set[str],
    globally_selected_isbns: set[str],
) -> list[tuple[int, int, str]]:
    """
    Returns:
        [(target_rank, actual_rank, isbn), ...]

    Search starts at each target rank and scans forward until an eligible
    unseen/global-unique ISBN is found.
    """
    selected: list[tuple[int, int, str]] = []
    locally_selected: set[str] = set()

    for target_rank in TARGET_RANKS:
        chosen = None

        # target_rank is 1-based, Python index is 0-based.
        for index in range(
            target_rank - 1,
            len(retrieved_isbns),
        ):
            isbn = retrieved_isbns[index]

            if isbn in seen_development_isbns:
                continue

            if isbn in globally_selected_isbns:
                continue

            if isbn in locally_selected:
                continue

            chosen = (
                target_rank,
                index + 1,
                isbn,
            )
            break

        if chosen is None:
            raise RuntimeError(
                "Unable to find an eligible unseen candidate "
                f"at or after target rank {target_rank}. "
                f"Increase SEARCH_DEPTH (currently {SEARCH_DEPTH})."
            )

        selected.append(chosen)
        locally_selected.add(chosen[2])

    return selected


def build_record(
    *,
    case_id: str,
    query_id: str,
    query: str,
    query_slice: str,
    candidate_source: str,
    sampling_target_rank: int | str,
    retrieval_rank: int | str,
    row,
) -> dict:
    return {
        "case_id": case_id,
        "query_id": query_id,
        "query": query,
        "query_slice": query_slice,
        "candidate_source": candidate_source,
        "sampling_target_rank": sampling_target_rank,
        "retrieval_rank": retrieval_rank,
        "isbn13": canonical_isbn(row["isbn13"]),
        "title": row["title"],
        "authors": row["authors"],
        "description": row["description"],
        "dataset_version": POOL_VERSION,
        "rubric_version": RUBRIC_VERSION,
        "human_score": "",
        "human_reason": "",
        "review_status": "UNLABELLED",
    }


def validate_records(
    records: list[dict],
    seen_development_isbns: set[str],
) -> None:
    if len(records) != 60:
        raise ValueError(
            f"Expected exactly 60 unseen cases, got {len(records)}."
        )

    case_ids = [record["case_id"] for record in records]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError(
            "Duplicate case IDs detected in unseen pool."
        )

    pool_isbns = [record["isbn13"] for record in records]
    if len(pool_isbns) != len(set(pool_isbns)):
        raise ValueError(
            "Duplicate ISBNs detected in unseen pool."
        )

    overlap = set(pool_isbns) & seen_development_isbns
    if overlap:
        raise ValueError(
            "Unseen pool overlaps development candidates: "
            f"{sorted(overlap)}"
        )

    query_counts: dict[str, int] = {}

    for record in records:
        query_counts[record["query_id"]] = (
            query_counts.get(record["query_id"], 0)
            + 1
        )

        if record["human_score"] != "":
            raise ValueError(
                f"{record['case_id']} unexpectedly contains a human score."
            )

        if record["human_reason"] != "":
            raise ValueError(
                f"{record['case_id']} unexpectedly contains a human reason."
            )

        if record["review_status"] != "UNLABELLED":
            raise ValueError(
                f"{record['case_id']} must start UNLABELLED."
            )

    bad_counts = {
        query_id: count
        for query_id, count in query_counts.items()
        if count != 5
    }

    if bad_counts:
        raise ValueError(
            "Each query must contribute exactly five cases. "
            f"Bad counts: {bad_counts}"
        )


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    args = parse_args()

    development_dataset = resolve_existing_path(
        explicit=args.development_dataset,
        candidates=DEVELOPMENT_DATASET_CANDIDATES,
        label="development dataset",
    )

    query_file = resolve_existing_path(
        explicit=None,
        candidates=QUERY_FILE_CANDIDATES,
        label="calibration query file",
    )

    print(f"Development ISBN exclusion source: {development_dataset}")
    print(f"Query source:                       {query_file}")
    print()

    seen_development_isbns = load_seen_isbns(
        development_dataset
    )

    print(
        "Development candidate ISBNs excluded: "
        f"{len(seen_development_isbns)}"
    )

    with query_file.open(
        "r",
        encoding="utf-8",
    ) as handle:
        queries = json.load(handle)

    if len(queries) != 12:
        raise ValueError(
            f"Expected 12 frozen queries, got {len(queries)}."
        )

    app = load_app_module()

    rng = random.Random(
        RANDOM_SEED
    )

    records: list[dict] = []
    globally_selected_isbns: set[str] = set()

    for q in queries:
        query_id = str(q["query_id"])
        query = str(q["query"])
        query_slice = str(q["slice"])

        # Raw Chroma ranking only. This deliberately avoids any post-retrieval
        # DataFrame filtering or tone/category reordering.
        recs = app.db_books.similarity_search(
            query,
            k=SEARCH_DEPTH,
        )

        retrieved_isbns = [
            isbn_from_document(
                rec.page_content
            )
            for rec in recs
        ]

        ranked_candidates = select_ranked_candidates(
            retrieved_isbns=retrieved_isbns,
            seen_development_isbns=seen_development_isbns,
            globally_selected_isbns=globally_selected_isbns,
        )

        for target_rank, actual_rank, isbn in ranked_candidates:
            row = row_for_isbn(
                app.books,
                isbn,
            )

            if row is None:
                raise RuntimeError(
                    f"ISBN {isbn} from Chroma ranking is missing "
                    "from app.books."
                )

            records.append(
                build_record(
                    case_id=(
                        f"U_{query_id}_T{target_rank:02d}"
                    ),
                    query_id=query_id,
                    query=query,
                    query_slice=query_slice,
                    candidate_source="semantic_retrieval",
                    sampling_target_rank=target_rank,
                    retrieval_rank=actual_rank,
                    row=row,
                )
            )

            globally_selected_isbns.add(
                isbn
            )

        # Random negative:
        # - cannot be a development ISBN;
        # - cannot already be in the new pool;
        # - cannot be among this query's retrieved top SEARCH_DEPTH, to reduce
        #   the chance that the "random" negative is actually a high-ranked hit.
        excluded_for_random = (
            seen_development_isbns
            | globally_selected_isbns
            | set(retrieved_isbns)
        )

        possible_indices = [
            idx
            for idx, value
            in app.books["isbn13"].items()
            if (
                canonical_isbn(value)
                not in excluded_for_random
            )
        ]

        if not possible_indices:
            raise RuntimeError(
                f"No eligible random candidate remains for {query_id}."
            )

        random_idx = rng.choice(
            possible_indices
        )

        random_row = app.books.loc[
            random_idx
        ]

        random_isbn = canonical_isbn(
            random_row["isbn13"]
        )

        records.append(
            build_record(
                case_id=f"U_{query_id}_NEG",
                query_id=query_id,
                query=query,
                query_slice=query_slice,
                candidate_source="random_negative_candidate",
                sampling_target_rank="",
                retrieval_rank="",
                row=random_row,
            )
        )

        globally_selected_isbns.add(
            random_isbn
        )

    validate_records(
        records=records,
        seen_development_isbns=seen_development_isbns,
    )

    output_file = args.output

    if not output_file.is_absolute():
        output_file = REPO_ROOT / output_file

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "case_id",
        "query_id",
        "query",
        "query_slice",
        "candidate_source",
        "sampling_target_rank",
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

    with output_file.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(
            records
        )

    print()
    print(
        f"Wrote {len(records)} unseen cases to:"
    )
    print(
        f"  {output_file}"
    )
    print()
    print(
        "Validation:"
    )
    print(
        "  12 queries x 5 cases = 60"
    )
    print(
        "  duplicate ISBNs in new pool = 0"
    )
    print(
        "  development ISBN overlap = 0"
    )
    print(
        "  all human labels blank = yes"
    )
    print()
    print(
        "IMPORTANT: Human-label all 60 before running the judge "
        "or splitting validation/final holdout."
    )


if __name__ == "__main__":
    main()
