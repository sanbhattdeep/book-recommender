"""Validate the fully human-labelled 60-case v0.27 unseen pool before splitting.

This script NEVER calls the judge. It validates:
- 60 complete human-labelled cases;
- frozen Q01-Q12 query fidelity;
- candidate uniqueness inside the pool;
- no overlap with the current 120-case consolidated development set; and
- no overlap with the original 60-case calibration history that was also used
  during judge development.
"""
from __future__ import annotations

from pathlib import Path
import hashlib
import json
import re
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
EVALS = REPO_ROOT / "evals"
DATASETS = EVALS / "datasets"
POOL = DATASETS / "semantic_relevance_unseen_pool.v3.0.0.csv"
DEV = DATASETS / "semantic_relevance_v0.27_development.v3.0.0.csv"
FACETS = EVALS / "facets" / "semantic_relevance" / "semantic_relevance_query_facets.v0.9.5.json"
FREEZE = EVALS / "releases" / "semantic_relevance_v0.27.0_r5_development_candidate.json"
EXPECTED_FREEZE_SHA = "3de6ee3a6274a5d066f17ff24bc965b05dfb8d60b785c0210d45d743538d84e5"

CALIBRATION_CANDIDATES = [
    DATASETS / "semantic_relevance_calibration.v0.5.0.csv",
    DATASETS / "semantic_relevance_calibration.v0.4.0.csv",
    DATASETS / "semantic_relevance_calibration.v0.3.0.csv",
]

REQUIRED = {
    "case_id", "query_id", "query", "query_slice", "candidate_source",
    "sampling_target_rank", "retrieval_rank", "isbn13", "title", "authors",
    "description", "dataset_version", "rubric_version", "human_score",
    "human_reason", "review_status",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def norm(value: object) -> str:
    s = str(value or "").strip().lower()
    return re.sub(r"[^a-z0-9]+", " ", s).strip()


def title_author_key(title: object, authors: object) -> str:
    return f"{norm(title)}|{norm(authors)}"


def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")


def resolve_original_calibration_dataset() -> Path:
    for path in CALIBRATION_CANDIDATES:
        if path.exists():
            return path
    raise FileNotFoundError(
        "Original consumed calibration dataset not found. Tried:\n"
        + "\n".join(f"  - {p}" for p in CALIBRATION_CANDIDATES)
    )


def identity_sets(df: pd.DataFrame) -> tuple[set[str], set[str]]:
    isbns = {str(x).strip() for x in df["isbn13"] if str(x).strip()}
    keys = {
        title_author_key(t, a)
        for t, a in zip(df["title"], df["authors"])
        if norm(t) or norm(a)
    }
    return isbns, keys


def fail(msg: str) -> None:
    raise ValueError(msg)


def main() -> None:
    calibration = resolve_original_calibration_dataset()
    for p in (POOL, DEV, calibration, FACETS, FREEZE):
        if not p.exists():
            raise FileNotFoundError(f"Required file not found: {p}")

    if sha256(FREEZE) != EXPECTED_FREEZE_SHA:
        fail(
            "Frozen v0.27 r5 development-candidate manifest hash mismatch. "
            "Do not split against a changed candidate."
        )

    pool = load_csv(POOL)
    dev = load_csv(DEV)
    old = load_csv(calibration)
    facets = json.loads(FACETS.read_text(encoding="utf-8-sig"))
    query_map = {str(q["query_id"]): str(q["query"]) for q in facets.get("queries", [])}

    missing = REQUIRED - set(pool.columns)
    if missing:
        fail(f"Unseen pool missing required columns: {sorted(missing)}")
    if len(pool) != 60:
        fail(f"Unseen pool must contain exactly 60 rows; found {len(pool)}.")
    if pool["case_id"].duplicated().any():
        fail("Unseen pool contains duplicate case_id values.")
    if set(pool["query_id"]) != {f"Q{i:02d}" for i in range(1, 13)}:
        fail("Unseen pool must cover Q01-Q12 exactly.")

    counts = pool.groupby("query_id").size().to_dict()
    if any(counts.get(f"Q{i:02d}") != 5 for i in range(1, 13)):
        fail(f"Each query must have exactly 5 cases; counts={counts}")

    expected_case_ids = {
        f"U3_Q{q:02d}_T{i:02d}"
        for q in range(1, 13)
        for i in range(1, 6)
    }
    if set(pool["case_id"]) != expected_case_ids:
        fail("case_id values must be the frozen U3_Qxx_T01..T05 IDs.")

    for row in pool.itertuples(index=False):
        expected_query = query_map.get(row.query_id)
        if expected_query is None or row.query != expected_query:
            fail(f"Frozen query mismatch for {row.case_id}: {row.query!r}")

    if set(pool["dataset_version"]) != {"3.0.0"}:
        fail("All pool rows must declare dataset_version=3.0.0.")
    if set(pool["rubric_version"]) != {"0.1.0"}:
        fail("All pool rows must declare rubric_version=0.1.0.")

    for col in ("candidate_source", "title", "authors", "description", "human_reason"):
        bad = pool[col].astype(str).str.strip().eq("")
        if bad.any():
            fail(f"Column {col} has {int(bad.sum())} blank values.")

    scores = pd.to_numeric(pool["human_score"], errors="coerce")
    if scores.isna().any() or not scores.isin([0, 1, 2, 3, 4]).all():
        fail("human_score must be complete and contain only 0..4 before splitting.")
    if not pool["review_status"].astype(str).str.upper().eq("LABELLED").all():
        fail("Every review_status must be LABELLED before splitting.")

    nonblank_isbn = pool.loc[pool["isbn13"].str.strip().ne(""), "isbn13"].str.strip()
    if nonblank_isbn.duplicated().any():
        fail(
            "Duplicate nonblank ISBNs inside unseen pool: "
            f"{nonblank_isbn[nonblank_isbn.duplicated(keep=False)].tolist()}"
        )
    pool_keys = {
        title_author_key(t, a)
        for t, a in zip(pool["title"], pool["authors"])
    }
    if len(pool_keys) != 60:
        fail("Duplicate normalized title+authors identities inside unseen pool.")

    dev_isbn, dev_keys = identity_sets(dev)
    old_isbn, old_keys = identity_sets(old)
    consumed_isbn = dev_isbn | old_isbn
    consumed_keys = dev_keys | old_keys

    new_isbn = {x.strip() for x in pool["isbn13"].astype(str) if x.strip()}
    isbn_overlap = sorted(consumed_isbn & new_isbn)
    if isbn_overlap:
        fail(f"ISBN overlap with consumed calibration/development history: {isbn_overlap}")

    key_overlap = sorted(consumed_keys & pool_keys)
    if key_overlap:
        fail(
            "Normalized title+authors overlap with consumed calibration/development "
            f"history: {key_overlap[:10]}"
        )

    score_distribution = scores.astype(int).value_counts().sort_index().to_dict()

    print("v0.27 unseen-pool authoring validation")
    print("------------------------------------")
    print("PASS  60/60 rows present")
    print("PASS  Q01-Q12 coverage; 5 cases/query")
    print("PASS  frozen query text matches facet spec v0.9.5")
    print("PASS  60/60 human labels and reasons complete")
    print("PASS  60/60 review_status=LABELLED")
    print("PASS  no duplicate candidate identities inside new pool")
    print("PASS  no overlap with ANY consumed calibration/development identity")
    print("PASS  frozen v0.27 r5 candidate manifest hash matches")
    print(f"Human score distribution: {score_distribution}")
    print(f"Original calibration identity source: {calibration}")
    print(f"Pool SHA-256: {sha256(POOL)}")
    print("AUTHORING VALIDATION: PASS")


if __name__ == "__main__":
    main()
