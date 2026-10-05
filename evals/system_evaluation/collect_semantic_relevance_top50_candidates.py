from __future__ import annotations

import ast
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
EVALS = REPO_ROOT / "evals"

CONTRACT_FILE = EVALS / "system_evaluation/semantic_relevance_system_eval.v1.0.0.json"
V1_INPUT_LOCK_FILE = EVALS / "system_evaluation/semantic_relevance_system_eval_input_lock.v1.0.0.json"
V2_RECOMMENDER_LOCK_FILE = EVALS / "system_evaluation/semantic_recommender_v2_lock.json"
QUERY_FILE = EVALS / "datasets/semantic_relevance_system_eval_queries.v1.0.0.csv"
RECOMMENDER_FILE = REPO_ROOT / "gradio-dashboard.py"

V1_RUN_DIR = EVALS / "runs/semantic_relevance_system_eval_v1/20261002T152856Z_recommendations"
V2_RUN_DIR = EVALS / "runs/semantic_relevance_system_eval_v2/20261004T140218Z_recommendations"
V1_RECOMMENDATIONS_FILE = V1_RUN_DIR / "recommendations.csv"
V2_RECOMMENDATIONS_FILE = V2_RUN_DIR / "recommendations.csv"
V1_COLLECTION_LOCK_FILE = V1_RUN_DIR / "recommendation_collection_lock.json"
V2_COLLECTION_LOCK_FILE = V2_RUN_DIR / "recommendation_collection_lock.json"

RUN_ROOT = EVALS / "runs/semantic_relevance_top50_diagnostic_v1"

EXPECTED_RECOMMENDER_FUNCTION = "retrieve_semantic_recommedations"
EXPECTED_V2_RECOMMENDER_SHA256 = "cdd0458ff23bd669185a8c882ceb132438d037af1a29f4dd60bd7fef9c405b60"
EXPECTED_QUERY_COUNT = 12
EXPECTED_TOP_K = 50
EXPECTED_V2_FINAL_K = 10
EXPECTED_ROWS = EXPECTED_QUERY_COUNT * EXPECTED_TOP_K


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8-sig"))


def git_sha() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def canonical_isbn(value: Any) -> str:
    """Normalize ISBN identity without inventing digits.

    The project stores ISBN13 values in a mix of numeric/string representations.
    This helper only removes formatting artifacts such as whitespace, quotes,
    and a terminal '.0' introduced by numeric CSV/DataFrame coercion.
    """

    if pd.isna(value):
        raise ValueError("ISBN13 is missing.")

    text = str(value).strip().strip('"').strip("'")
    if re.fullmatch(r"\d+\.0", text):
        text = text[:-2]
    if not text:
        raise ValueError("ISBN13 is empty after normalization.")
    return text


def parse_isbn_from_retrieval_document(page_content: str) -> str:
    text = str(page_content)
    tokens = text.strip('"').split()
    if not tokens:
        raise ValueError("Vector-store document has empty page_content.")
    return canonical_isbn(tokens[0])


def check_hash(label: str, path: Path, expected: str) -> str:
    if not path.exists():
        raise FileNotFoundError(path)
    actual = sha256(path)
    if actual != expected:
        raise ValueError(
            f"{label} hash changed: expected={expected} actual={actual} path={path}"
        )
    print(f"PASS  {label} hash unchanged")
    return actual


def verify_frozen_inputs() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Verify the frozen benchmark inputs and v2 recommender source.

    Important: this diagnostic intentionally does NOT require current Git HEAD
    to equal the historical v2 freeze commit. Later analysis/report commits are
    allowed. The recommender source itself must still match the frozen v2 hash.
    """

    contract = load_json(CONTRACT_FILE)
    v1_lock = load_json(V1_INPUT_LOCK_FILE)
    v2_lock = load_json(V2_RECOMMENDER_LOCK_FILE)

    if v1_lock.get("status") != "FROZEN_BEFORE_SYSTEM_EVAL_EXECUTION":
        raise ValueError("The original system-evaluation input lock is not frozen.")

    if contract.get("version") != v1_lock.get("evaluation_version"):
        raise ValueError("Contract version differs from original frozen input lock.")

    check_hash("frozen query file", QUERY_FILE, str(v1_lock["query_file_sha256"]))
    check_hash("frozen contract", CONTRACT_FILE, str(v1_lock["contract_file_sha256"]))
    check_hash("frozen v2 recommender", RECOMMENDER_FILE, EXPECTED_V2_RECOMMENDER_SHA256)

    # Cross-check the v2 lock when its schema exposes a source hash. The
    # hard-pinned expected source hash above remains authoritative for this
    # diagnostic package because older lock schemas may use different key names.
    candidate_hash_keys = [
        "recommender_sha256",
        "source_sha256",
        "recommender_source_sha256",
    ]
    exposed_hashes = [
        str(v2_lock[key])
        for key in candidate_hash_keys
        if key in v2_lock and v2_lock[key]
    ]
    if exposed_hashes and EXPECTED_V2_RECOMMENDER_SHA256 not in exposed_hashes:
        raise ValueError(
            "v2 recommender lock exposes a source hash that disagrees with the "
            f"expected frozen v2 source: {exposed_hashes}"
        )
    print("PASS  v2 recommender lock present and compatible")

    benchmark = contract.get("benchmark", {})
    if int(benchmark.get("recommendations_per_query", -1)) != EXPECTED_V2_FINAL_K:
        raise ValueError("Frozen contract no longer specifies 10 recommendations/query.")

    return contract, v1_lock, v2_lock


def verify_frozen_recommendation_artifact(
    *,
    label: str,
    recommendations_file: Path,
    lock_file: Path,
) -> str:
    if not recommendations_file.exists():
        raise FileNotFoundError(recommendations_file)
    lock = load_json(lock_file)
    expected = lock.get("recommendations_sha256")
    actual = sha256(recommendations_file)
    if expected and actual != str(expected):
        raise ValueError(
            f"{label} recommendations changed after freeze: "
            f"expected={expected} actual={actual}"
        )
    print(f"PASS  {label} frozen recommendations unchanged")
    return actual


def load_backend_namespace_without_ui() -> dict[str, Any]:
    """Execute backend setup through the recommender function, not the Gradio UI."""

    source = RECOMMENDER_FILE.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(RECOMMENDER_FILE))

    function_index: int | None = None
    for index, node in enumerate(tree.body):
        if (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == EXPECTED_RECOMMENDER_FUNCTION
        ):
            function_index = index
            break

    if function_index is None:
        raise ValueError(
            f"Could not find {EXPECTED_RECOMMENDER_FUNCTION} in {RECOMMENDER_FILE}."
        )

    backend_tree = ast.Module(
        body=tree.body[: function_index + 1],
        type_ignores=[],
    )
    ast.fix_missing_locations(backend_tree)

    namespace: dict[str, Any] = {
        "__name__": "_semantic_relevance_top50_diagnostic_backend",
        "__file__": str(RECOMMENDER_FILE),
    }

    previous_cwd = Path.cwd()
    try:
        os.chdir(REPO_ROOT)
        exec(
            compile(backend_tree, filename=str(RECOMMENDER_FILE), mode="exec"),
            namespace,
            namespace,
        )
    finally:
        os.chdir(previous_cwd)

    if "db_books" not in namespace:
        raise ValueError("Backend namespace does not expose db_books.")
    if "books" not in namespace:
        raise ValueError("Backend namespace does not expose books metadata DataFrame.")
    if not isinstance(namespace["books"], pd.DataFrame):
        raise TypeError("Backend variable 'books' is not a pandas DataFrame.")

    return namespace


def load_frozen_top10(path: Path, *, label: str) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8")
    required = {"query_id", "rank", "isbn13", "title"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{label} recommendations missing columns: {missing}")

    frame = frame.copy()
    frame["rank"] = pd.to_numeric(frame["rank"], errors="raise").astype(int)
    frame["isbn_key"] = frame["isbn13"].map(canonical_isbn)

    counts = frame.groupby("query_id").size().to_dict()
    if len(counts) != EXPECTED_QUERY_COUNT or any(v != EXPECTED_V2_FINAL_K for v in counts.values()):
        raise ValueError(f"{label}: expected 12 x 10 recommendations; got {counts}")

    if frame.duplicated(["query_id", "isbn_key"]).any():
        raise ValueError(f"{label}: duplicate query-book pairs detected.")

    return frame


def build_books_lookup(books: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, list[int]]]:
    if "isbn13" not in books.columns:
        raise ValueError("books metadata is missing isbn13.")

    meta = books.copy().reset_index(drop=True)
    meta["_isbn_key"] = meta["isbn13"].map(canonical_isbn)

    lookup: dict[str, list[int]] = {}
    for idx, key in enumerate(meta["_isbn_key"]):
        lookup.setdefault(key, []).append(int(idx))
    return meta, lookup


def safe_text(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value)


def collect_query_top50(
    *,
    query_id: str,
    query: str,
    db_books: Any,
    books_meta: pd.DataFrame,
    books_lookup: dict[str, list[int]],
    v1: pd.DataFrame,
    v2: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    recs = db_books.similarity_search(query, k=EXPECTED_TOP_K)
    if len(recs) != EXPECTED_TOP_K:
        raise ValueError(
            f"{query_id}: expected exactly {EXPECTED_TOP_K} vector candidates; got {len(recs)}"
        )

    rows: list[dict[str, Any]] = []
    seen_isbns: set[str] = set()

    v1_q = v1[v1["query_id"] == query_id].sort_values("rank", kind="stable")
    v2_q = v2[v2["query_id"] == query_id].sort_values("rank", kind="stable")
    if len(v1_q) != EXPECTED_V2_FINAL_K or len(v2_q) != EXPECTED_V2_FINAL_K:
        raise ValueError(f"{query_id}: frozen v1/v2 recommendation rows are incomplete.")

    v1_rank_by_isbn = dict(zip(v1_q["isbn_key"], v1_q["rank"]))
    v2_rank_by_isbn = dict(zip(v2_q["isbn_key"], v2_q["rank"]))

    for vector_rank, rec in enumerate(recs, start=1):
        page_content = safe_text(getattr(rec, "page_content", ""))
        isbn_key = parse_isbn_from_retrieval_document(page_content)

        if isbn_key in seen_isbns:
            raise ValueError(
                f"{query_id}: duplicate ISBN {isbn_key} appears more than once in vector top 50."
            )
        seen_isbns.add(isbn_key)

        if isbn_key not in books_lookup:
            raise ValueError(
                f"{query_id}: vector candidate ISBN {isbn_key} is missing from books metadata."
            )
        matching_rows = books_lookup[isbn_key]
        if len(matching_rows) != 1:
            raise ValueError(
                f"{query_id}: vector candidate ISBN {isbn_key} maps to {len(matching_rows)} "
                "metadata rows; diagnostic refuses an ambiguous join."
            )

        meta_row = books_meta.iloc[matching_rows[0]]
        row: dict[str, Any] = {
            "query_id": query_id,
            "query": query,
            "vector_rank": vector_rank,
            "isbn13": isbn_key,
            "title": safe_text(meta_row.get("title", "")),
            "description": safe_text(meta_row.get("description", "")),
            "authors": safe_text(meta_row.get("authors", "")),
            "retrieval_document_text": page_content,
            "in_frozen_v2_top10": isbn_key in v2_rank_by_isbn,
            "frozen_v2_rank": v2_rank_by_isbn.get(isbn_key, ""),
            "in_frozen_v1_top10": isbn_key in v1_rank_by_isbn,
            "frozen_v1_rank": v1_rank_by_isbn.get(isbn_key, ""),
        }
        rows.append(row)

    frame = pd.DataFrame(rows)

    # Reproduce the exact frozen v2 final top 10 from vector ranks 1..10.
    new_top10 = frame.head(EXPECTED_V2_FINAL_K)["isbn13"].tolist()
    frozen_v2_top10 = v2_q["isbn_key"].tolist()
    if new_top10 != frozen_v2_top10:
        details = [
            {
                "rank": rank,
                "new": new,
                "frozen_v2": old,
                "match": new == old,
            }
            for rank, (new, old) in enumerate(zip(new_top10, frozen_v2_top10), start=1)
        ]
        raise ValueError(
            f"{query_id}: current vector ranks 1..10 do not reproduce frozen v2 top 10. "
            f"Candidate retrieval/environment drift suspected. Details={details}"
        )

    # Every v1 top-10 result historically came from the original top-50 pool.
    missing_v1 = sorted(set(v1_q["isbn_key"]) - set(frame["isbn13"]))
    if missing_v1:
        raise ValueError(
            f"{query_id}: frozen v1 top-10 candidates missing from current top 50: {missing_v1}. "
            "Candidate retrieval/environment drift suspected."
        )

    v1_vector_ranks = {
        isbn: int(frame.loc[frame["isbn13"] == isbn, "vector_rank"].iloc[0])
        for isbn in v1_q["isbn_key"]
    }

    check = {
        "query_id": query_id,
        "v2_top10_reproduced": True,
        "v1_top10_contained_in_top50": True,
        "candidate_count": len(frame),
        "unique_candidate_count": int(frame["isbn13"].nunique()),
        "v1_top10_vector_ranks": json.dumps(v1_vector_ranks, sort_keys=True),
    }
    return frame, check


def main() -> None:
    print("Semantic relevance top-50 diagnostic - Checkpoint A")
    print("---------------------------------------------------")
    print("Purpose: collect and freeze unchanged v2 vector top-50 candidates")
    print("Judge execution: DISABLED")
    print("Human relevance labels: NOT USED")
    print("Recommender modification: NONE")
    print()

    contract, v1_input_lock, v2_recommender_lock = verify_frozen_inputs()

    v1_recommendations_sha = verify_frozen_recommendation_artifact(
        label="v1",
        recommendations_file=V1_RECOMMENDATIONS_FILE,
        lock_file=V1_COLLECTION_LOCK_FILE,
    )
    v2_recommendations_sha = verify_frozen_recommendation_artifact(
        label="v2",
        recommendations_file=V2_RECOMMENDATIONS_FILE,
        lock_file=V2_COLLECTION_LOCK_FILE,
    )

    queries = pd.read_csv(QUERY_FILE, dtype=str, keep_default_na=False, encoding="utf-8")
    if len(queries) != EXPECTED_QUERY_COUNT:
        raise ValueError(f"Expected exactly {EXPECTED_QUERY_COUNT} frozen queries; got {len(queries)}")
    if queries["query_id"].duplicated().any():
        raise ValueError("Frozen query file contains duplicate query_id values.")
    if not {"query_id", "query"}.issubset(queries.columns):
        raise ValueError("Frozen query file must contain query_id and query columns.")

    v1 = load_frozen_top10(V1_RECOMMENDATIONS_FILE, label="v1")
    v2 = load_frozen_top10(V2_RECOMMENDATIONS_FILE, label="v2")

    print("Loading frozen v2 backend without Gradio UI...")
    namespace = load_backend_namespace_without_ui()
    db_books = namespace["db_books"]
    books_meta, books_lookup = build_books_lookup(namespace["books"])
    print("PASS  backend loaded; db_books and unique books metadata available")

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_candidates"
    run_dir = RUN_ROOT / run_id
    if run_dir.exists():
        raise FileExistsError(run_dir)
    run_dir.mkdir(parents=True)

    frames: list[pd.DataFrame] = []
    checks: list[dict[str, Any]] = []

    for index, row in queries.iterrows():
        query_id = str(row["query_id"])
        query = str(row["query"])
        print(f"{index + 1:2d}/{EXPECTED_QUERY_COUNT} {query_id}: {query}")

        frame, check = collect_query_top50(
            query_id=query_id,
            query=query,
            db_books=db_books,
            books_meta=books_meta,
            books_lookup=books_lookup,
            v1=v1,
            v2=v2,
        )
        frames.append(frame)
        checks.append(check)

        pd.concat(frames, ignore_index=True).to_csv(
            run_dir / "top50_candidates.partial.csv",
            index=False,
            encoding="utf-8",
        )

    candidates = pd.concat(frames, ignore_index=True)
    checks_df = pd.DataFrame(checks)

    if len(candidates) != EXPECTED_ROWS:
        raise ValueError(f"Expected {EXPECTED_ROWS} candidate rows; got {len(candidates)}")

    counts = candidates.groupby("query_id").size().to_dict()
    if len(counts) != EXPECTED_QUERY_COUNT or any(v != EXPECTED_TOP_K for v in counts.values()):
        raise ValueError(f"Expected 50 candidates for every query; got {counts}")

    if candidates.duplicated(["query_id", "isbn13"]).any():
        raise ValueError("Duplicate query-book pairs exist in the completed top-50 artifact.")

    if not checks_df["v2_top10_reproduced"].all():
        raise ValueError("At least one query failed v2 top-10 reproduction.")
    if not checks_df["v1_top10_contained_in_top50"].all():
        raise ValueError("At least one query failed v1 top-10 containment.")

    candidates_file = run_dir / "top50_candidates.csv"
    checks_file = run_dir / "top50_reproduction_checks.csv"
    candidates.to_csv(candidates_file, index=False, encoding="utf-8")
    checks_df.to_csv(checks_file, index=False, encoding="utf-8")

    partial_file = run_dir / "top50_candidates.partial.csv"
    if partial_file.exists():
        partial_file.unlink()

    metadata = {
        "diagnostic_id": "semantic_relevance_top50_diagnostic_v1",
        "stage": "CHECKPOINT_A_COLLECTION_COMPLETE",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_directory": str(run_dir),
        "current_git_commit_sha": git_sha(),
        "historical_v2_git_commit_enforced": False,
        "historical_v2_git_commit_note": (
            "Current HEAD may contain later analysis/report commits. Frozen recommender source hash is enforced instead."
        ),
        "recommender_file": str(RECOMMENDER_FILE.relative_to(REPO_ROOT)),
        "recommender_sha256": sha256(RECOMMENDER_FILE),
        "expected_v2_recommender_sha256": EXPECTED_V2_RECOMMENDER_SHA256,
        "v2_recommender_lock_sha256": sha256(V2_RECOMMENDER_LOCK_FILE),
        "v1_input_lock_sha256": sha256(V1_INPUT_LOCK_FILE),
        "contract_sha256": sha256(CONTRACT_FILE),
        "query_file_sha256": sha256(QUERY_FILE),
        "v1_recommendations_sha256": v1_recommendations_sha,
        "v2_recommendations_sha256": v2_recommendations_sha,
        "query_count": EXPECTED_QUERY_COUNT,
        "candidates_per_query": EXPECTED_TOP_K,
        "query_book_pairs": EXPECTED_ROWS,
        "category": "All",
        "tone": "All",
        "candidate_generation": "direct db_books.similarity_search(query, k=50)",
        "v2_top10_reproduction_passed_queries": int(checks_df["v2_top10_reproduced"].sum()),
        "v1_top10_containment_passed_queries": int(checks_df["v1_top10_contained_in_top50"].sum()),
        "top50_candidates_file": candidates_file.name,
        "top50_candidates_sha256": sha256(candidates_file),
        "reproduction_checks_file": checks_file.name,
        "reproduction_checks_sha256": sha256(checks_file),
        "judge_calls_performed": 0,
        "human_relevance_labels_used": False,
        "python": sys.version,
        "platform": platform.platform(),
        "selected_package_versions": {
            name: package_version(name)
            for name in [
                "pandas",
                "langchain",
                "langchain-community",
                "langchain-openai",
                "chromadb",
                "openai",
                "gradio",
            ]
        },
        "frozen_contract_version": contract.get("version"),
        "v1_input_lock_status": v1_input_lock.get("status"),
        "v2_recommender_lock_status": v2_recommender_lock.get("status"),
    }

    metadata_file = run_dir / "top50_collection_metadata.json"
    metadata_file.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    collection_lock = {
        "status": "TOP50_CANDIDATES_FROZEN_BEFORE_SCORE_REUSE",
        "diagnostic_id": metadata["diagnostic_id"],
        "run_directory": str(run_dir),
        "top50_candidates_sha256": sha256(candidates_file),
        "reproduction_checks_sha256": sha256(checks_file),
        "recommender_sha256": sha256(RECOMMENDER_FILE),
        "query_file_sha256": sha256(QUERY_FILE),
        "query_book_pairs": EXPECTED_ROWS,
        "queries": EXPECTED_QUERY_COUNT,
        "candidates_per_query": EXPECTED_TOP_K,
        "v2_top10_reproduction_passed": True,
        "v1_top10_containment_passed": True,
        "judge_calls_performed": 0,
        "human_relevance_labels_used": False,
        "next_stage": "CHECKPOINT_B_SCORE_REUSE_PREPARATION",
    }
    lock_file = run_dir / "top50_collection_lock.json"
    lock_file.write_text(json.dumps(collection_lock, indent=2) + "\n", encoding="utf-8")

    # Compact evidence summary useful before uploading full CSVs.
    print()
    print("CHECKPOINT A INTEGRITY")
    print("----------------------")
    print(f"Frozen v2 top-10 reproduction: {int(checks_df['v2_top10_reproduced'].sum())}/12 PASS")
    print(f"Frozen v1 top-10 containment:  {int(checks_df['v1_top10_contained_in_top50'].sum())}/12 PASS")
    print(f"Top-50 rows:                   {len(candidates)}/600")
    print(f"Unique query-book pairs:       {candidates[['query_id','isbn13']].drop_duplicates().shape[0]}/600")
    print()
    print("Known v1 top-10 books mapped to current vector ranks")
    print("----------------------------------------------------")
    for _, row in checks_df.iterrows():
        mapping = json.loads(row["v1_top10_vector_ranks"])
        ranks = sorted(mapping.values())
        deeper = sum(rank > 10 for rank in ranks)
        print(f"{row['query_id']}: ranks={ranks}  below_top10={deeper}/10")

    print()
    print("TOP-50 DIAGNOSTIC CHECKPOINT A: PASS")
    print(f"Run directory: {run_dir}")
    print(f"Candidates: {len(candidates)}")
    print(f"Top-50 artifact SHA-256: {sha256(candidates_file)}")
    print("Judge calls performed: 0")
    print("Human relevance labels used: NO")
    print("Next: review Checkpoint A before preparing score reuse / novel judge input.")


if __name__ == "__main__":
    main()
