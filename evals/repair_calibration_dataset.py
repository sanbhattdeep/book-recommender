"""
Repair the original human-labelled semantic relevance calibration dataset.

This script is specifically designed for the original labelled CSV that was
accidentally damaged when opened/saved through spreadsheet software:

1. ISBN-13 values were converted to scientific notation, losing exact digits.
2. The file contains mixed UTF-8 and Windows-1252 text bytes.
3. Human labels/reasons belong to the ORIGINAL candidate books and must not be
   merged onto newly retrieved candidates.

Important
---------
This script DOES NOT rerun semantic retrieval.

Instead, it:
- decodes the original labelled CSV,
- loads the repository's canonical `app.books` DataFrame,
- finds the original book by title/authors/description,
- restores the exact ISBN-13 from the canonical book catalog,
- replaces book text fields with their canonical UTF-8 values,
- preserves the original human labels,
- writes a new immutable dataset version.

Expected repository layout
--------------------------
    <repo>/
      gradio-dashboard.py
      evals/
        datasets/
          semantic_relevance_calibration.labelled-temp.csv
        repair_calibration_dataset.py

Typical usage
-------------
    uv run python evals/repair_calibration_dataset.py

Optional explicit paths:

    uv run python evals/repair_calibration_dataset.py `
      --input evals/datasets/semantic_relevance_calibration.labelled-temp.csv `
      --output evals/datasets/semantic_relevance_calibration.v0.2.0.csv
"""

from __future__ import annotations

import argparse
import importlib.util
import io
import re
import unicodedata
from pathlib import Path

import pandas as pd


# =============================================================================
# Versioning
# =============================================================================

SOURCE_DATASET_VERSION = "0.1.0"
OUTPUT_DATASET_VERSION = "0.2.0"
RUBRIC_VERSION = "0.1.0"


# =============================================================================
# Repository paths
# =============================================================================

REPO_ROOT = Path(__file__).resolve().parents[1]
EVALS_DIR = REPO_ROOT / "evals"
DATASETS_DIR = EVALS_DIR / "datasets"

DEFAULT_INPUT = (
    DATASETS_DIR
    / "semantic_relevance_calibration.labelled-temp.csv"
)

DEFAULT_OUTPUT = (
    DATASETS_DIR
    / f"semantic_relevance_calibration.v{OUTPUT_DATASET_VERSION}.csv"
)

DEFAULT_REPORT = (
    DATASETS_DIR
    / f"semantic_relevance_calibration.v{OUTPUT_DATASET_VERSION}.repair_report.csv"
)


# =============================================================================
# Mixed-encoding recovery
# =============================================================================

def decode_mixed_utf8_windows1252(raw: bytes) -> str:
    """
    Decode a file that is mostly UTF-8 but contains occasional standalone
    Windows-1252 bytes such as 0x93 / 0x94 for curly quotes.

    Why not simply use latin-1?
    --------------------------
    Latin-1 will read every byte, but valid UTF-8 text such as:

        Anaïs
        Húrin
        em dashes
        curly quotes already encoded as UTF-8

    becomes mojibake such as:

        AnaÃ¯s
        HÃºrin

    Why not simply use cp1252?
    --------------------------
    The file also contains valid UTF-8 byte sequences. Some bytes inside those
    sequences are undefined as standalone cp1252 characters.

    Strategy
    --------
    Decode valid UTF-8 spans normally. Only when the UTF-8 decoder encounters
    an invalid byte/span do we decode that bad span as Windows-1252.
    """

    pieces: list[str] = []
    offset = 0

    while offset < len(raw):
        try:
            pieces.append(
                raw[offset:].decode("utf-8")
            )
            break

        except UnicodeDecodeError as error:
            # Decode the valid UTF-8 prefix first.
            valid_end = offset + error.start

            if valid_end > offset:
                pieces.append(
                    raw[offset:valid_end].decode("utf-8")
                )

            bad_start = offset + error.start
            bad_end = offset + error.end
            bad_bytes = raw[bad_start:bad_end]

            # The observed damaged file contains Windows-1252 punctuation in
            # these invalid spans. "replace" protects against any unexpected
            # undefined byte while still making the failure visible.
            pieces.append(
                bad_bytes.decode(
                    "cp1252",
                    errors="replace",
                )
            )

            offset = bad_end

    return "".join(pieces)


def load_damaged_labelled_csv(
    path: Path,
) -> pd.DataFrame:
    """
    Read the damaged original labelled CSV without allowing pandas/Excel-style
    numeric inference to further alter ISBN values.
    """

    raw = path.read_bytes()

    text = decode_mixed_utf8_windows1252(
        raw
    )

    return pd.read_csv(
        io.StringIO(text),
        dtype=str,
        keep_default_na=False,
    )


# =============================================================================
# Canonical book catalog
# =============================================================================

def load_app_module():
    """
    Load gradio-dashboard.py to obtain the same canonical `books` DataFrame
    used by the recommender.

    No similarity search is performed by this repair script.
    """

    app_path = (
        REPO_ROOT
        / "gradio-dashboard.py"
    )

    if not app_path.exists():
        raise FileNotFoundError(
            f"Could not find {app_path}"
        )

    spec = importlib.util.spec_from_file_location(
        "book_recommender_app",
        app_path,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"Unable to load module from {app_path}"
        )

    module = importlib.util.module_from_spec(
        spec
    )

    spec.loader.exec_module(
        module
    )

    if not hasattr(
        module,
        "books",
    ):
        raise AttributeError(
            "gradio-dashboard.py does not expose app.books"
        )

    return module


# =============================================================================
# Matching normalization
# =============================================================================

_TRANSLATION_TABLE = str.maketrans(
    {
        "’": "'",
        "‘": "'",
        "“": '"',
        "”": '"',
        "–": "-",
        "—": "-",
        "\u00a0": " ",
    }
)


def normalize_for_match(
    value,
) -> str:
    """
    Normalize text only for deterministic candidate matching.

    We do NOT write this normalized text to the repaired dataset.
    Canonical catalog text is written instead.
    """

    if value is None:
        return ""

    text = str(value)

    if text.lower() == "nan":
        return ""

    text = unicodedata.normalize(
        "NFKC",
        text,
    )

    text = text.translate(
        _TRANSLATION_TABLE
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip().casefold()


def prepare_catalog(
    books: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add normalized helper columns to a copy of the canonical catalog.
    """

    required = {
        "isbn13",
        "title",
        "authors",
        "description",
    }

    missing = (
        required
        - set(books.columns)
    )

    if missing:
        raise ValueError(
            "Canonical books DataFrame is missing columns: "
            f"{sorted(missing)}"
        )

    catalog = books.copy()

    catalog["_title_norm"] = (
        catalog["title"]
        .map(normalize_for_match)
    )

    catalog["_authors_norm"] = (
        catalog["authors"]
        .map(normalize_for_match)
    )

    catalog["_description_norm"] = (
        catalog["description"]
        .map(normalize_for_match)
    )

    return catalog


# =============================================================================
# Original-candidate recovery
# =============================================================================

def find_original_book(
    row: pd.Series,
    catalog: pd.DataFrame,
) -> tuple[pd.Series | None, str, int]:
    """
    Recover one original book from the canonical catalog.

    Matching order is intentionally conservative:

    1. title + authors + description
    2. title + description
    3. title + authors
    4. unique title only

    A candidate is accepted only when the selected strategy yields exactly
    one catalog row.

    Returns
    -------
    (matched_row, strategy, candidate_count)

    matched_row is None when a unique match could not be established.
    """

    title = normalize_for_match(
        row["title"]
    )

    authors = normalize_for_match(
        row["authors"]
    )

    description = normalize_for_match(
        row["description"]
    )

    strategies = [
        (
            "title+authors+description",
            (
                (catalog["_title_norm"] == title)
                & (catalog["_authors_norm"] == authors)
                & (catalog["_description_norm"] == description)
            ),
        ),
        (
            "title+description",
            (
                (catalog["_title_norm"] == title)
                & (catalog["_description_norm"] == description)
            ),
        ),
        (
            "title+authors",
            (
                (catalog["_title_norm"] == title)
                & (catalog["_authors_norm"] == authors)
            ),
        ),
        (
            "unique-title",
            (
                catalog["_title_norm"] == title
            ),
        ),
    ]

    for strategy, mask in strategies:
        matches = catalog.loc[
            mask
        ]

        if len(matches) == 1:
            return (
                matches.iloc[0],
                strategy,
                1,
            )

        # Do not immediately fail on an ambiguous stricter strategy because a
        # later strategy cannot become more specific. In practice ambiguity at
        # a stricter level is already suspicious, so report it immediately.
        if len(matches) > 1:
            return (
                None,
                strategy,
                len(matches),
            )

    return (
        None,
        "no-match",
        0,
    )


def canonical_isbn13(
    value,
) -> str:
    """
    Convert a canonical catalog ISBN to a 13-digit string.

    The catalog, not the damaged CSV, is the source of truth.
    """

    if pd.isna(value):
        raise ValueError(
            "Canonical catalog ISBN is missing."
        )

    # Most existing repository datasets hold ISBN as integer/float-compatible
    # values. Handle both safely.
    if isinstance(
        value,
        (int,),
    ):
        isbn = str(
            value
        )

    else:
        raw = str(
            value
        ).strip()

        if re.fullmatch(
            r"\d{13}",
            raw,
        ):
            isbn = raw

        elif re.fullmatch(
            r"\d+\.0",
            raw,
        ):
            isbn = raw[:-2]

        else:
            # A canonical catalog should not contain scientific notation, but
            # allow numeric conversion as a final compatibility path.
            try:
                isbn = str(
                    int(
                        float(raw)
                    )
                )
            except ValueError as error:
                raise ValueError(
                    f"Cannot convert canonical ISBN value {value!r}"
                ) from error

    if not re.fullmatch(
        r"\d{13}",
        isbn,
    ):
        raise ValueError(
            f"Recovered ISBN is not 13 digits: {isbn!r}"
        )

    return isbn


# =============================================================================
# Dataset validation
# =============================================================================

def validate_source_dataset(
    source: pd.DataFrame,
) -> None:
    """
    Validate the labelled source before attempting repair.
    """

    required = {
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
    }

    missing = (
        required
        - set(source.columns)
    )

    if missing:
        raise ValueError(
            "Source dataset is missing columns: "
            f"{sorted(missing)}"
        )

    if source["case_id"].duplicated().any():
        duplicate_ids = source.loc[
            source["case_id"].duplicated(
                keep=False
            ),
            "case_id",
        ].tolist()

        raise ValueError(
            f"Duplicate case IDs: {duplicate_ids}"
        )

    if set(
        source["dataset_version"]
    ) != {SOURCE_DATASET_VERSION}:
        raise ValueError(
            "Source rows do not all declare dataset version "
            f"{SOURCE_DATASET_VERSION}."
        )

    if set(
        source["rubric_version"]
    ) != {RUBRIC_VERSION}:
        raise ValueError(
            "Source rows do not all declare rubric version "
            f"{RUBRIC_VERSION}."
        )

    scores = pd.to_numeric(
        source["human_score"],
        errors="coerce",
    )

    if scores.isna().any():
        raise ValueError(
            "Source contains missing/non-numeric human_score values."
        )

    if not scores.isin(
        [0, 1, 2, 3, 4]
    ).all():
        raise ValueError(
            "Source human_score contains values outside 0-4."
        )


def validate_repaired_dataset(
    repaired: pd.DataFrame,
) -> None:
    """
    Final fail-fast validation before the v0.2.0 file is written.
    """

    if repaired["case_id"].duplicated().any():
        raise ValueError(
            "Repaired dataset contains duplicate case IDs."
        )

    if set(
        repaired["dataset_version"]
    ) != {OUTPUT_DATASET_VERSION}:
        raise ValueError(
            "Repaired dataset version is inconsistent."
        )

    if set(
        repaired["rubric_version"]
    ) != {RUBRIC_VERSION}:
        raise ValueError(
            "Repaired rubric version is inconsistent."
        )

    bad_isbn = ~repaired[
        "isbn13"
    ].astype(str).str.fullmatch(
        r"\d{13}"
    )

    if bad_isbn.any():
        raise ValueError(
            "One or more repaired ISBN values are not 13-digit strings."
        )

    if repaired["isbn13"].astype(str).str.contains(
        r"[Ee]\+",
        regex=True,
    ).any():
        raise ValueError(
            "Scientific-notation ISBN values remain in repaired dataset."
        )

    if repaired[
        [
            "case_id",
            "query",
            "title",
            "description",
            "human_score",
        ]
    ].replace(
        "",
        pd.NA,
    ).isna().any().any():
        raise ValueError(
            "Required repaired fields contain blank values."
        )


# =============================================================================
# Main repair operation
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Repair exact ISBNs and mixed encoding in the original "
            "human-labelled semantic relevance calibration dataset "
            "without rerunning retrieval."
        )
    )

    parser.add_argument(
        "--input",
        default=str(
            DEFAULT_INPUT
        ),
        help=(
            "Original damaged labelled CSV. "
            f"Default: {DEFAULT_INPUT}"
        ),
    )

    parser.add_argument(
        "--output",
        default=str(
            DEFAULT_OUTPUT
        ),
        help=(
            "Repaired versioned dataset output. "
            f"Default: {DEFAULT_OUTPUT}"
        ),
    )

    parser.add_argument(
        "--report",
        default=str(
            DEFAULT_REPORT
        ),
        help=(
            "Repair audit report output. "
            f"Default: {DEFAULT_REPORT}"
        ),
    )

    args = parser.parse_args()

    input_path = Path(
        args.input
    )

    output_path = Path(
        args.output
    )

    report_path = Path(
        args.report
    )

    if not input_path.is_absolute():
        input_path = (
            REPO_ROOT
            / input_path
        )

    if not output_path.is_absolute():
        output_path = (
            REPO_ROOT
            / output_path
        )

    if not report_path.is_absolute():
        report_path = (
            REPO_ROOT
            / report_path
        )

    if not input_path.exists():
        raise FileNotFoundError(
            f"Source dataset not found: {input_path}"
        )

    source = load_damaged_labelled_csv(
        input_path
    )

    validate_source_dataset(
        source
    )

    app = load_app_module()

    catalog = prepare_catalog(
        app.books
    )

    repaired_records: list[dict] = []
    report_records: list[dict] = []
    failures: list[dict] = []

    for _, source_row in source.iterrows():

        (
            catalog_row,
            match_strategy,
            candidate_count,
        ) = find_original_book(
            row=source_row,
            catalog=catalog,
        )

        if catalog_row is None:
            failure = {
                "case_id": source_row[
                    "case_id"
                ],
                "title": source_row[
                    "title"
                ],
                "authors": source_row[
                    "authors"
                ],
                "damaged_isbn13": source_row[
                    "isbn13"
                ],
                "match_strategy": (
                    match_strategy
                ),
                "candidate_count": (
                    candidate_count
                ),
                "status": "FAILED",
            }

            failures.append(
                failure
            )

            report_records.append(
                failure
            )

            continue

        recovered_isbn = canonical_isbn13(
            catalog_row[
                "isbn13"
            ]
        )

        # Preserve evaluation-case identity and HUMAN annotations from the
        # original labelled file.
        #
        # Replace book metadata with canonical catalog values so corrupted text
        # encoding in title/authors/description is also repaired.
        repaired_records.append(
            {
                "case_id": source_row[
                    "case_id"
                ],
                "query_id": source_row[
                    "query_id"
                ],
                "query": source_row[
                    "query"
                ],
                "query_slice": source_row[
                    "query_slice"
                ],
                "candidate_source": source_row[
                    "candidate_source"
                ],
                "retrieval_rank": source_row[
                    "retrieval_rank"
                ],
                "isbn13": recovered_isbn,
                "title": catalog_row[
                    "title"
                ],
                "authors": catalog_row[
                    "authors"
                ],
                "description": catalog_row[
                    "description"
                ],
                "dataset_version": (
                    OUTPUT_DATASET_VERSION
                ),
                "rubric_version": source_row[
                    "rubric_version"
                ],
                "human_score": int(
                    source_row[
                        "human_score"
                    ]
                ),
                "human_reason": source_row[
                    "human_reason"
                ],
                "review_status": source_row[
                    "review_status"
                ],
            }
        )

        report_records.append(
            {
                "case_id": source_row[
                    "case_id"
                ],
                "title": catalog_row[
                    "title"
                ],
                "authors": catalog_row[
                    "authors"
                ],
                "damaged_isbn13": source_row[
                    "isbn13"
                ],
                "recovered_isbn13": (
                    recovered_isbn
                ),
                "match_strategy": (
                    match_strategy
                ),
                "candidate_count": 1,
                "status": "RECOVERED",
            }
        )

    # Always write the audit report, including failures, before stopping.
    report_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    pd.DataFrame(
        report_records
    ).to_csv(
        report_path,
        index=False,
        encoding="utf-8",
    )

    if failures:
        print()
        print(
            "Dataset repair FAILED."
        )
        print(
            f"{len(failures)} source candidates could not be "
            "uniquely matched to the canonical book catalog."
        )
        print()

        print(
            pd.DataFrame(
                failures
            ).to_string(
                index=False
            )
        )

        print()
        print(
            f"Audit report written to: {report_path}"
        )

        print()
        print(
            "No repaired calibration dataset was written. "
            "Resolve these rows manually instead of guessing ISBNs."
        )

        raise SystemExit(
            1
        )

    repaired = pd.DataFrame(
        repaired_records,
        columns=[
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
        ],
    )

    # Preserve original source row order.
    repaired_order = {
        case_id: position
        for position, case_id
        in enumerate(
            source["case_id"]
        )
    }

    repaired["_source_order"] = (
        repaired["case_id"]
        .map(
            repaired_order
        )
    )

    repaired = (
        repaired
        .sort_values(
            "_source_order"
        )
        .drop(
            columns=[
                "_source_order"
            ]
        )
        .reset_index(
            drop=True
        )
    )

    validate_repaired_dataset(
        repaired
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    repaired.to_csv(
        output_path,
        index=False,
        encoding="utf-8",
    )

    print()
    print(
        "Dataset repair completed successfully."
    )
    print(
        f"Rows recovered: {len(repaired)}"
    )
    print(
        f"Dataset version: {OUTPUT_DATASET_VERSION}"
    )
    print(
        f"Repaired dataset: {output_path}"
    )
    print(
        f"Audit report:     {report_path}"
    )
    print()
    print(
        "No semantic retrieval was rerun."
    )
    print(
        "Original case IDs, retrieval ranks, human scores, and human reasons "
        "were preserved."
    )
    print(
        "Book ISBN/title/authors/description were recovered from the "
        "canonical repository catalog."
    )


if __name__ == "__main__":
    main()
