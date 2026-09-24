"""Build the populated 60-case v0.27 unseen human-labeling pool.

Run from the repository root:

    uv run python evals/build_v0_27_unseen_evaluation_pool.py

Purpose
-------
Create 60 fresh query-book pairs for independent validation of the frozen
Semantic Relevance Judge v0.27.0 r5 candidate:

    12 frozen queries
    x (4 raw semantic-retrieval candidates + 1 random candidate)
    = 60 cases

This is a retrieval/data-preparation script only. It NEVER calls the semantic
relevance judge and it NEVER creates validation/final-holdout model output.

Sampling policy
---------------
For every frozen query, request the raw Chroma ranking used by the application
and target raw ranks 2, 10, 30 and 70. If the book at a target rank has already
been consumed by development or selected elsewhere in this new pool, scan
forward to the next eligible result and record the ACTUAL raw retrieval rank.

The fifth candidate is a deterministic random corpus book that:
- was not consumed by ANY prior judge calibration/development evidence;
- is not already selected anywhere in the new pool; and
- is not in that query's retrieved top SEARCH_DEPTH.

Freshness
---------
Freshness is defined against ALL historically consumed judge-calibration evidence, not just the current consolidated development set. Candidate identity is protected by BOTH:
- canonical ISBN-13 when available; and
- normalized title + authors.

This prevents accidental reuse when one source has a damaged/missing ISBN.

Important
---------
Human labels remain blank. After this script completes, use the generated
Excel workbook for blind human labeling. Do not run the judge before all 60
human labels are complete and the deterministic 30/30 split is frozen.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import random
import re
import unicodedata
from pathlib import Path
from typing import Any

import pandas as pd


# =============================================================================
# Frozen configuration
# =============================================================================

REPO_ROOT = Path(__file__).resolve().parents[1]
EVALS_DIR = REPO_ROOT / "evals"
DATASETS_DIR = EVALS_DIR / "datasets"

POOL_VERSION = "3.0.0"
RUBRIC_VERSION = "0.1.0"
FACET_SPEC_VERSION = "0.9.5"

TARGET_RANKS = [2, 10, 30, 70]
SEARCH_DEPTH = 250
RANDOM_SEED = 20260924

OUTPUT_FILE = DATASETS_DIR / "semantic_relevance_unseen_pool.v3.0.0.csv"
DEVELOPMENT_DATASET = DATASETS_DIR / "semantic_relevance_v0.27_development.v3.0.0.csv"

# The original 60-case calibration set was used directly to tune the judge in
# early versions. It is historically consumed even though later consolidated
# development datasets do not necessarily contain those exact candidates.
# Require the latest canonical copy when available; older immutable versions are
# accepted only as a fallback because they preserve the same candidate identities.
ORIGINAL_CALIBRATION_DATASET_CANDIDATES = [
    DATASETS_DIR / "semantic_relevance_calibration.v0.5.0.csv",
    DATASETS_DIR / "semantic_relevance_calibration.v0.4.0.csv",
    DATASETS_DIR / "semantic_relevance_calibration.v0.3.0.csv",
]
FACET_SPEC = (
    EVALS_DIR
    / "facets"
    / "semantic_relevance"
    / "semantic_relevance_query_facets.v0.9.5.json"
)
FREEZE_MANIFEST = (
    EVALS_DIR
    / "releases"
    / "semantic_relevance_v0.27.0_r5_development_candidate.json"
)
EXPECTED_FREEZE_SHA256 = (
    "3de6ee3a6274a5d066f17ff24bc965b05dfb8d60b785c0210d45d743538d84e5"
)

# The existing r1 package ships a blank 60-row skeleton at OUTPUT_FILE.
# It is safe to replace ONLY while it is still truly blank/unreviewed.
SKELETON_REQUIRED_COLUMNS = {
    "case_id",
    "query_id",
    "query",
    "candidate_source",
    "isbn13",
    "title",
    "authors",
    "description",
    "human_score",
    "human_reason",
    "review_status",
}


# =============================================================================
# Generic helpers
# =============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the populated 60-case v0.27 unseen evaluation pool."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_FILE,
        help=f"Output CSV. Default: {OUTPUT_FILE}",
    )
    parser.add_argument(
        "--search-depth",
        type=int,
        default=SEARCH_DEPTH,
        help=f"Raw Chroma retrieval depth. Default: {SEARCH_DEPTH}",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalize_text(value: object) -> str:
    if value is None:
        return ""
    text = unicodedata.normalize("NFKC", str(value))
    text = text.replace("’", "'").replace("‘", "'")
    text = text.replace("“", '"').replace("”", '"')
    text = text.replace("–", "-").replace("—", "-")
    text = re.sub(r"\s+", " ", text).strip().casefold()
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def title_author_key(title: object, authors: object) -> str:
    return f"{normalize_text(title)}|{normalize_text(authors)}"


def canonical_isbn(value: object) -> str:
    """Return a stable ISBN-13 string or '' for a genuinely blank value."""
    if value is None:
        return ""
    raw = str(value).strip()
    if not raw or raw.lower() == "nan":
        return ""
    if re.fullmatch(r"\d+\.0", raw):
        raw = raw[:-2]
    if not re.fullmatch(r"\d{13}", raw):
        raise ValueError(f"Expected a 13-digit ISBN, got {value!r}")
    return raw


def isbn_from_document(page_content: str) -> str:
    """Application Chroma documents are '<isbn13> <description>'."""
    raw = page_content.strip().strip('"').split()[0]
    isbn = canonical_isbn(raw)
    if not isbn:
        raise ValueError(f"Retrieved document does not begin with ISBN-13: {page_content[:80]!r}")
    return isbn


def nonblank(value: object) -> bool:
    return bool(str(value if value is not None else "").strip())


# =============================================================================
# Frozen-input loading and validation
# =============================================================================


def validate_frozen_candidate() -> None:
    if not FREEZE_MANIFEST.exists():
        raise FileNotFoundError(f"Frozen r5 manifest not found: {FREEZE_MANIFEST}")
    actual = sha256(FREEZE_MANIFEST)
    if actual != EXPECTED_FREEZE_SHA256:
        raise ValueError(
            "Frozen v0.27 r5 development-candidate manifest hash mismatch. "
            f"Expected={EXPECTED_FREEZE_SHA256} Actual={actual}. "
            "Do not build a new pool against a changed candidate."
        )


def load_frozen_queries() -> list[dict[str, str]]:
    if not FACET_SPEC.exists():
        raise FileNotFoundError(f"Facet spec not found: {FACET_SPEC}")
    payload = json.loads(FACET_SPEC.read_text(encoding="utf-8-sig"))
    raw_queries = payload.get("queries")
    if not isinstance(raw_queries, list) or len(raw_queries) != 12:
        raise ValueError("Facet spec must contain exactly 12 frozen queries.")

    queries: list[dict[str, str]] = []
    for raw in raw_queries:
        qid = str(raw.get("query_id", "")).strip()
        query = str(raw.get("query", "")).strip()
        if not re.fullmatch(r"Q(?:0[1-9]|1[0-2])", qid):
            raise ValueError(f"Unexpected query_id in facet spec: {qid!r}")
        if not query:
            raise ValueError(f"Frozen query text is blank for {qid}.")
        queries.append({"query_id": qid, "query": query})

    expected = {f"Q{i:02d}" for i in range(1, 13)}
    if {q["query_id"] for q in queries} != expected:
        raise ValueError("Facet spec query IDs must cover Q01-Q12 exactly.")
    return sorted(queries, key=lambda q: q["query_id"])


def resolve_original_calibration_dataset() -> Path:
    for path in ORIGINAL_CALIBRATION_DATASET_CANDIDATES:
        if path.exists():
            return path
    raise FileNotFoundError(
        "Original consumed calibration dataset not found. Tried:\n"
        + "\n".join(f"  - {path}" for path in ORIGINAL_CALIBRATION_DATASET_CANDIDATES)
    )


def load_identity_source(path: Path, *, expected_rows: int, label: str) -> tuple[set[str], set[str], int]:
    frame = pd.read_csv(
        path,
        dtype=str,
        keep_default_na=False,
        encoding="utf-8-sig",
    )
    required = {"case_id", "isbn13", "title", "authors"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{label} missing columns: {sorted(missing)}")
    if len(frame) != expected_rows:
        raise ValueError(
            f"Expected {label} to contain {expected_rows} rows; found {len(frame)}."
        )

    isbns = {
        isbn
        for isbn in (canonical_isbn(v) for v in frame["isbn13"])
        if isbn
    }
    keys = {
        title_author_key(title, authors)
        for title, authors in zip(frame["title"], frame["authors"])
        if title_author_key(title, authors) != "|"
    }
    if not keys:
        raise ValueError(f"{label} contained no candidate identities.")
    return isbns, keys, len(frame)


def load_seen_identities() -> tuple[set[str], set[str], list[dict[str, object]]]:
    if not DEVELOPMENT_DATASET.exists():
        raise FileNotFoundError(
            f"Consumed v0.27 development dataset not found: {DEVELOPMENT_DATASET}"
        )

    original_calibration = resolve_original_calibration_dataset()

    dev_isbns, dev_keys, dev_rows = load_identity_source(
        DEVELOPMENT_DATASET,
        expected_rows=120,
        label="consumed v0.27 development dataset",
    )
    cal_isbns, cal_keys, cal_rows = load_identity_source(
        original_calibration,
        expected_rows=60,
        label="original consumed calibration dataset",
    )

    seen_isbns = dev_isbns | cal_isbns
    seen_keys = dev_keys | cal_keys

    sources: list[dict[str, object]] = [
        {
            "label": "v0.27 consolidated development",
            "path": DEVELOPMENT_DATASET,
            "rows": dev_rows,
            "isbn_identities": len(dev_isbns),
            "title_author_identities": len(dev_keys),
        },
        {
            "label": "original calibration",
            "path": original_calibration,
            "rows": cal_rows,
            "isbn_identities": len(cal_isbns),
            "title_author_identities": len(cal_keys),
        },
    ]

    # The union must be strictly larger than the current 120-case development
    # set. If not, the historical calibration source is not adding any protection
    # and provenance should be investigated before calling a new pool "unseen".
    if len(seen_keys) <= len(dev_keys):
        raise ValueError(
            "Historical calibration exclusion did not enlarge the consumed "
            "title+author identity set. Refusing to claim an unseen pool."
        )

    return seen_isbns, seen_keys, sources


# =============================================================================
# Application access
# =============================================================================


def load_app_module():
    """Load the same db_books/books objects used by gradio-dashboard.py."""
    app_path = REPO_ROOT / "gradio-dashboard.py"
    if not app_path.exists():
        raise FileNotFoundError(f"Application module not found: {app_path}")

    spec = importlib.util.spec_from_file_location("book_recommender_app", app_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load application module from {app_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    for attr in ("db_books", "books"):
        if not hasattr(module, attr):
            raise AttributeError(f"gradio-dashboard.py does not expose app.{attr}")

    books = module.books
    required = {"isbn13", "title", "authors", "description"}
    missing = required - set(books.columns)
    if missing:
        raise ValueError(f"app.books missing required columns: {sorted(missing)}")

    return module


def prepare_catalog(books: pd.DataFrame) -> pd.DataFrame:
    catalog = books.copy()
    catalog["_isbn"] = catalog["isbn13"].map(canonical_isbn)
    catalog["_identity_key"] = catalog.apply(
        lambda r: title_author_key(r["title"], r["authors"]), axis=1
    )

    # Candidate evidence must be usable for blind human/judge evaluation.
    usable = (
        catalog["_isbn"].str.len().eq(13)
        & catalog["title"].map(nonblank)
        & catalog["authors"].map(nonblank)
        & catalog["description"].map(nonblank)
        & catalog["_identity_key"].ne("|")
    )
    catalog = catalog.loc[usable].copy()

    if catalog["_isbn"].duplicated().any():
        dupes = catalog.loc[catalog["_isbn"].duplicated(keep=False), "_isbn"].tolist()
        raise ValueError(f"Canonical app.books contains duplicate usable ISBNs: {dupes[:10]}")
    return catalog


def row_for_isbn(catalog: pd.DataFrame, isbn: str) -> pd.Series | None:
    matches = catalog.loc[catalog["_isbn"].eq(isbn)]
    if matches.empty:
        return None
    if len(matches) != 1:
        raise ValueError(f"ISBN {isbn} resolves to {len(matches)} catalog rows.")
    return matches.iloc[0]


# =============================================================================
# Deterministic candidate selection
# =============================================================================


def is_eligible(
    row: pd.Series | None,
    *,
    seen_isbns: set[str],
    seen_keys: set[str],
    selected_isbns: set[str],
    selected_keys: set[str],
) -> bool:
    if row is None:
        return False
    isbn = str(row["_isbn"])
    key = str(row["_identity_key"])
    return (
        isbn not in seen_isbns
        and key not in seen_keys
        and isbn not in selected_isbns
        and key not in selected_keys
    )


def select_ranked_candidates(
    retrieved_isbns: list[str],
    catalog: pd.DataFrame,
    *,
    seen_isbns: set[str],
    seen_keys: set[str],
    selected_isbns: set[str],
    selected_keys: set[str],
) -> list[tuple[int, int, pd.Series]]:
    selected: list[tuple[int, int, pd.Series]] = []
    local_isbns: set[str] = set()
    local_keys: set[str] = set()

    for target_rank in TARGET_RANKS:
        chosen: tuple[int, int, pd.Series] | None = None
        for index in range(target_rank - 1, len(retrieved_isbns)):
            isbn = retrieved_isbns[index]
            row = row_for_isbn(catalog, isbn)
            if row is None:
                continue
            row_isbn = str(row["_isbn"])
            row_key = str(row["_identity_key"])
            if row_isbn in local_isbns or row_key in local_keys:
                continue
            if not is_eligible(
                row,
                seen_isbns=seen_isbns,
                seen_keys=seen_keys,
                selected_isbns=selected_isbns,
                selected_keys=selected_keys,
            ):
                continue
            chosen = (target_rank, index + 1, row)
            break

        if chosen is None:
            raise RuntimeError(
                f"Could not find an eligible candidate at/after raw rank {target_rank}. "
                "Increase --search-depth if needed."
            )
        selected.append(chosen)
        local_isbns.add(str(chosen[2]["_isbn"]))
        local_keys.add(str(chosen[2]["_identity_key"]))

    return selected


def select_random_candidate(
    catalog: pd.DataFrame,
    *,
    retrieved_isbns: set[str],
    seen_isbns: set[str],
    seen_keys: set[str],
    selected_isbns: set[str],
    selected_keys: set[str],
    rng: random.Random,
) -> pd.Series:
    eligible_indices: list[Any] = []
    for idx, row in catalog.iterrows():
        isbn = str(row["_isbn"])
        if isbn in retrieved_isbns:
            continue
        if is_eligible(
            row,
            seen_isbns=seen_isbns,
            seen_keys=seen_keys,
            selected_isbns=selected_isbns,
            selected_keys=selected_keys,
        ):
            eligible_indices.append(idx)

    if not eligible_indices:
        raise RuntimeError("No eligible deterministic-random corpus candidate remains.")
    return catalog.loc[rng.choice(eligible_indices)]


def build_record(
    *,
    case_id: str,
    query_id: str,
    query: str,
    candidate_source: str,
    sampling_target_rank: int | str,
    retrieval_rank: int | str,
    row: pd.Series,
) -> dict[str, object]:
    return {
        "case_id": case_id,
        "query_id": query_id,
        "query": query,
        "query_slice": "unseen_pool_v3",
        "candidate_source": candidate_source,
        "sampling_target_rank": sampling_target_rank,
        "retrieval_rank": retrieval_rank,
        "isbn13": str(row["_isbn"]),
        "title": str(row["title"]),
        "authors": str(row["authors"]),
        "description": str(row["description"]),
        "dataset_version": POOL_VERSION,
        "rubric_version": RUBRIC_VERSION,
        "human_score": "",
        "human_reason": "",
        "review_status": "UNREVIEWED",
    }


# =============================================================================
# Output safety and final validation
# =============================================================================


def existing_output_is_blank_skeleton(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        existing = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    except Exception:
        return False
    if len(existing) != 60 or not SKELETON_REQUIRED_COLUMNS.issubset(existing.columns):
        return False

    blank_candidate_cols = ["candidate_source", "isbn13", "title", "authors", "description"]
    if any(existing[c].astype(str).str.strip().ne("").any() for c in blank_candidate_cols):
        return False
    if existing["human_score"].astype(str).str.strip().ne("").any():
        return False
    if existing["human_reason"].astype(str).str.strip().ne("").any():
        return False
    allowed_status = existing["review_status"].astype(str).str.upper().isin(["UNREVIEWED", "UNLABELLED", ""])
    return bool(allowed_status.all())


def validate_records(
    records: list[dict[str, object]],
    *,
    seen_isbns: set[str],
    seen_keys: set[str],
) -> None:
    if len(records) != 60:
        raise ValueError(f"Expected exactly 60 unseen cases, got {len(records)}.")

    expected_ids = {f"U3_Q{q:02d}_T{i:02d}" for q in range(1, 13) for i in range(1, 6)}
    actual_ids = {str(r["case_id"]) for r in records}
    if actual_ids != expected_ids:
        raise ValueError("Generated case IDs do not match U3_Qxx_T01..T05 contract.")

    pool_isbns = [str(r["isbn13"]) for r in records]
    pool_keys = [title_author_key(r["title"], r["authors"]) for r in records]
    if len(pool_isbns) != len(set(pool_isbns)):
        raise ValueError("Duplicate ISBNs detected in new unseen pool.")
    if len(pool_keys) != len(set(pool_keys)):
        raise ValueError("Duplicate normalized title+authors detected in new unseen pool.")
    if set(pool_isbns) & seen_isbns:
        raise ValueError("Generated pool overlaps consumed development ISBNs.")
    if set(pool_keys) & seen_keys:
        raise ValueError("Generated pool overlaps consumed development title+authors identities.")

    counts: dict[str, int] = {}
    source_counts: dict[str, int] = {}
    for record in records:
        qid = str(record["query_id"])
        counts[qid] = counts.get(qid, 0) + 1
        source = str(record["candidate_source"])
        source_counts[source] = source_counts.get(source, 0) + 1

        for col in ("isbn13", "title", "authors", "description"):
            if not str(record[col]).strip():
                raise ValueError(f"{record['case_id']} has blank {col}.")
        if str(record["human_score"]).strip() or str(record["human_reason"]).strip():
            raise ValueError(f"{record['case_id']} unexpectedly contains human labels.")
        if record["review_status"] != "UNREVIEWED":
            raise ValueError(f"{record['case_id']} must start UNREVIEWED.")

    if any(counts.get(f"Q{i:02d}") != 5 for i in range(1, 13)):
        raise ValueError(f"Each query must contribute exactly five cases; counts={counts}")
    if source_counts.get("semantic_retrieval") != 48:
        raise ValueError(f"Expected 48 semantic-retrieval cases; sources={source_counts}")
    if source_counts.get("random_negative_candidate") != 12:
        raise ValueError(f"Expected 12 random candidates; sources={source_counts}")


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    args = parse_args()
    if args.search_depth < max(TARGET_RANKS):
        raise ValueError(
            f"--search-depth must be >= {max(TARGET_RANKS)}; got {args.search_depth}."
        )

    validate_frozen_candidate()
    queries = load_frozen_queries()
    seen_isbns, seen_keys, consumed_sources = load_seen_identities()

    output = args.output if args.output.is_absolute() else REPO_ROOT / args.output
    if output.exists() and not existing_output_is_blank_skeleton(output):
        raise FileExistsError(
            f"Refusing to overwrite nonblank unseen-pool file: {output}. "
            "If this is an earlier generated/labelled pool, preserve it rather than rebuilding in place."
        )

    print("v0.27 unseen-pool retrieval builder")
    print("----------------------------------")
    print(f"Frozen candidate manifest: {FREEZE_MANIFEST}")
    print(f"Frozen manifest SHA-256:   {sha256(FREEZE_MANIFEST)}")
    print("Consumed identity sources:")
    for source in consumed_sources:
        print(
            f"  - {source['label']}: rows={source['rows']} "
            f"isbn={source['isbn_identities']} title+author={source['title_author_identities']} "
            f"path={source['path']}"
        )
    print(f"Excluded UNION ISBN identities:  {len(seen_isbns)}")
    print(f"Excluded UNION title+author keys: {len(seen_keys)}")
    print(f"Target raw ranks/query:     {TARGET_RANKS}")
    print(f"Search depth:               {args.search_depth}")
    print(f"Random seed:                {RANDOM_SEED}")
    print()

    app = load_app_module()
    catalog = prepare_catalog(app.books)
    rng = random.Random(RANDOM_SEED)

    records: list[dict[str, object]] = []
    selected_isbns: set[str] = set()
    selected_keys: set[str] = set()

    for q in queries:
        qid = q["query_id"]
        query = q["query"]
        print(f"Retrieving {qid}: {query}")

        recs = app.db_books.similarity_search(query, k=args.search_depth)
        retrieved_isbns = [isbn_from_document(rec.page_content) for rec in recs]
        if len(retrieved_isbns) < max(TARGET_RANKS):
            raise RuntimeError(
                f"{qid}: only {len(retrieved_isbns)} raw results returned; need at least {max(TARGET_RANKS)}."
            )

        ranked = select_ranked_candidates(
            retrieved_isbns,
            catalog,
            seen_isbns=seen_isbns,
            seen_keys=seen_keys,
            selected_isbns=selected_isbns,
            selected_keys=selected_keys,
        )

        # Stable slot mapping: T01/T02/T03/T04 correspond to target ranks
        # 2/10/30/70. T05 is the deterministic random candidate.
        for slot, (target_rank, actual_rank, row) in enumerate(ranked, start=1):
            record = build_record(
                case_id=f"U3_{qid}_T{slot:02d}",
                query_id=qid,
                query=query,
                candidate_source="semantic_retrieval",
                sampling_target_rank=target_rank,
                retrieval_rank=actual_rank,
                row=row,
            )
            records.append(record)
            selected_isbns.add(str(row["_isbn"]))
            selected_keys.add(str(row["_identity_key"]))
            print(
                f"  T{slot:02d}: target={target_rank:>2} actual={actual_rank:>3} "
                f"isbn={row['_isbn']} title={row['title']}"
            )

        random_row = select_random_candidate(
            catalog,
            retrieved_isbns=set(retrieved_isbns),
            seen_isbns=seen_isbns,
            seen_keys=seen_keys,
            selected_isbns=selected_isbns,
            selected_keys=selected_keys,
            rng=rng,
        )
        records.append(
            build_record(
                case_id=f"U3_{qid}_T05",
                query_id=qid,
                query=query,
                candidate_source="random_negative_candidate",
                sampling_target_rank="",
                retrieval_rank="",
                row=random_row,
            )
        )
        selected_isbns.add(str(random_row["_isbn"]))
        selected_keys.add(str(random_row["_identity_key"]))
        print(f"  T05: random              isbn={random_row['_isbn']} title={random_row['title']}")
        print()

    validate_records(records, seen_isbns=seen_isbns, seen_keys=seen_keys)

    output.parent.mkdir(parents=True, exist_ok=True)
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
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    print("v0.27 unseen-pool retrieval build complete")
    print("------------------------------------------")
    print("PASS  60/60 populated candidates")
    print("PASS  12 queries x 5 cases")
    print("PASS  48 raw semantic-retrieval + 12 deterministic-random candidates")
    print("PASS  no duplicate ISBN/title+author identities inside new pool")
    print("PASS  no overlap with ANY consumed calibration/development candidate identities")
    print("PASS  human labels remain blank; review_status=UNREVIEWED")
    print(f"Pool:        {output}")
    print(f"Pool SHA-256: {sha256(output)}")
    print()
    print("NEXT: export the blind-labeling workbook. Do NOT run the judge.")


if __name__ == "__main__":
    main()
