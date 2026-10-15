from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
EVALS = REPO_ROOT / "evals"

QUERY_FILE = EVALS / "datasets/semantic_relevance_v3_holdout_queries.v0.1.0.csv"
FACET_FILE = EVALS / "facets/semantic_relevance/semantic_relevance_v3_holdout_query_facets.v0.1.0.json"
HOLDOUT_INPUT_LOCK = EVALS / "system_evaluation/semantic_relevance_v3_holdout_input_lock.v0.1.0.json"
SELECTED_CANDIDATE_LOCK = EVALS / "system_evaluation/semantic_relevance_v3_selected_candidate.lock.json"

RECOMMENDER_FILE = REPO_ROOT / "gradio-dashboard.py"

RUN_ROOT = EVALS / "runs/semantic_relevance_v3_holdout"

EXPECTED_RECOMMENDER_FUNCTION = "retrieve_semantic_recommedations"
EXPECTED_RECOMMENDER_SHA256 = "cdd0458ff23bd669185a8c882ceb132438d037af1a29f4dd60bd7fef9c405b60"
EXPECTED_QUERY_COUNT = 24
EXPECTED_TOP_K = 50
EXPECTED_ROWS = EXPECTED_QUERY_COUNT * EXPECTED_TOP_K
EXPECTED_QUERY_IDS = [f"H{i:02d}" for i in range(1, 25)]


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


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def canonical_isbn(value: Any) -> str:
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


def safe_text(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value)


def load_backend_namespace_without_ui() -> dict[str, Any]:
    """
    Execute the recommender backend setup only through the frozen retrieval
    function definition. The Gradio UI block is intentionally not executed.
    """

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
        "__name__": "_semantic_relevance_v3_holdout_retrieval_backend",
        "__file__": str(RECOMMENDER_FILE),
    }

    previous_cwd = Path.cwd()
    try:
        os.chdir(REPO_ROOT)
        exec(
            compile(
                backend_tree,
                filename=str(RECOMMENDER_FILE),
                mode="exec",
            ),
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


def build_books_lookup(
    books: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, list[int]]]:
    if "isbn13" not in books.columns:
        raise ValueError("books metadata is missing isbn13.")

    meta = books.copy().reset_index(drop=True)
    meta["_isbn_key"] = meta["isbn13"].map(canonical_isbn)

    lookup: dict[str, list[int]] = {}
    for idx, key in enumerate(meta["_isbn_key"]):
        lookup.setdefault(key, []).append(int(idx))

    return meta, lookup


def verify_frozen_boundary() -> tuple[pd.DataFrame, dict[str, Any]]:
    print("Verifying frozen holdout boundary...")

    require(
        sha256(RECOMMENDER_FILE) == EXPECTED_RECOMMENDER_SHA256,
        "Frozen v2 candidate-generator source changed.",
    )
    print("PASS  frozen v2 candidate-generator source hash unchanged")

    selected = load_json(SELECTED_CANDIDATE_LOCK)
    require(
        selected.get("status")
        == "V3_SELECTED_CANDIDATE_FROZEN_BEFORE_UNSEEN_HOLDOUT_AUTHORING",
        "Selected v3 candidate lock is not frozen.",
    )
    require(
        selected.get("selected_variant") == "v3b_rrf_hybrid",
        "Unexpected selected v3 variant.",
    )
    print("PASS  frozen selected v3b candidate lock verified")

    holdout_lock = load_json(HOLDOUT_INPUT_LOCK)
    require(
        holdout_lock.get("status")
        == "V3_UNSEEN_HOLDOUT_INPUTS_FROZEN_BEFORE_RETRIEVAL",
        "Holdout inputs are not frozen before retrieval.",
    )
    require(
        holdout_lock.get("selected_variant") == "v3b_rrf_hybrid",
        "Holdout lock references unexpected selected variant.",
    )
    require(
        holdout_lock.get("holdout_retrieval_performed_before_freeze") is False,
        "Holdout lock says retrieval occurred before freeze.",
    )

    require(
        sha256(QUERY_FILE) == holdout_lock.get("holdout_query_file_sha256"),
        "Holdout query file changed after input freeze.",
    )
    require(
        sha256(FACET_FILE) == holdout_lock.get("holdout_facet_file_sha256"),
        "Holdout facet file changed after input freeze.",
    )
    require(
        sha256(SELECTED_CANDIDATE_LOCK)
        == holdout_lock.get("selected_candidate_lock_sha256"),
        "Selected-candidate lock changed after holdout input freeze.",
    )
    print("PASS  frozen holdout query/facet/selected-candidate hashes unchanged")

    queries = pd.read_csv(
        QUERY_FILE,
        dtype=str,
        keep_default_na=False,
        encoding="utf-8",
    )
    required = {"query_id", "query", "stratum"}
    missing = required - set(queries.columns)
    if missing:
        raise ValueError(f"Holdout query file missing columns: {sorted(missing)}")

    require(len(queries) == EXPECTED_QUERY_COUNT, "Expected exactly 24 holdout queries.")
    require(queries["query_id"].tolist() == EXPECTED_QUERY_IDS, "Holdout query IDs/order changed.")
    require(not queries["query_id"].duplicated().any(), "Duplicate holdout query IDs.")
    require(not queries["query"].duplicated().any(), "Duplicate holdout query text.")

    facets = load_json(FACET_FILE)
    fqueries = facets.get("queries", [])
    require(len(fqueries) == EXPECTED_QUERY_COUNT, "Facet spec does not contain 24 queries.")
    require(
        [str(q["query_id"]) for q in fqueries] == EXPECTED_QUERY_IDS,
        "Facet-spec holdout query IDs/order changed.",
    )
    print("PASS  holdout query/facet structural checks")

    return queries, holdout_lock


def collect_query_top50(
    *,
    query_id: str,
    query: str,
    stratum: str,
    db_books: Any,
    books_meta: pd.DataFrame,
    books_lookup: dict[str, list[int]],
) -> pd.DataFrame:
    recs = db_books.similarity_search(query, k=EXPECTED_TOP_K)

    if len(recs) != EXPECTED_TOP_K:
        raise ValueError(
            f"{query_id}: expected {EXPECTED_TOP_K} vector candidates; got {len(recs)}"
        )

    rows: list[dict[str, Any]] = []
    seen_isbns: set[str] = set()

    for vector_rank, rec in enumerate(recs, start=1):
        page_content = safe_text(getattr(rec, "page_content", ""))
        isbn_key = parse_isbn_from_retrieval_document(page_content)

        if isbn_key in seen_isbns:
            raise ValueError(
                f"{query_id}: duplicate ISBN {isbn_key} in vector top50."
            )
        seen_isbns.add(isbn_key)

        if isbn_key not in books_lookup:
            raise ValueError(
                f"{query_id}: ISBN {isbn_key} missing from books metadata."
            )

        matching_rows = books_lookup[isbn_key]
        if len(matching_rows) != 1:
            raise ValueError(
                f"{query_id}: ISBN {isbn_key} maps to {len(matching_rows)} metadata rows."
            )

        meta = books_meta.iloc[matching_rows[0]]

        rows.append(
            {
                "query_id": query_id,
                "query": query,
                "stratum": stratum,
                "vector_rank": vector_rank,
                "isbn13": isbn_key,
                "title": safe_text(meta.get("title", "")),
                "description": safe_text(meta.get("description", "")),
                "authors": safe_text(meta.get("authors", "")),
                "retrieval_document_text": page_content,
            }
        )

    frame = pd.DataFrame(rows)

    require(len(frame) == EXPECTED_TOP_K, f"{query_id}: top50 row count mismatch.")
    require(
        frame["vector_rank"].tolist() == list(range(1, EXPECTED_TOP_K + 1)),
        f"{query_id}: vector ranks are not exactly 1..50.",
    )
    require(
        not frame["isbn13"].duplicated().any(),
        f"{query_id}: duplicate ISBNs after collection.",
    )

    return frame


def main() -> None:
    print("Semantic relevance v3 - unseen holdout retrieval collection")
    print("----------------------------------------------------------")
    print("Stage: frozen candidate generation only")
    print("Reranker inference: DISABLED")
    print("Judge labels loaded: NO")
    print("Human relevance labels loaded: NO")
    print("Judge calls: 0")
    print()

    queries, holdout_input_lock = verify_frozen_boundary()

    namespace = load_backend_namespace_without_ui()
    db_books = namespace["db_books"]
    books = namespace["books"]

    books_meta, books_lookup = build_books_lookup(books)

    all_frames: list[pd.DataFrame] = []

    print()
    print("Collecting frozen vector top50 for 24 unseen queries...")
    for row in queries.itertuples(index=False):
        qid = str(row.query_id)
        print(f"  {qid}: retrieving top50")
        frame = collect_query_top50(
            query_id=qid,
            query=str(row.query),
            stratum=str(row.stratum),
            db_books=db_books,
            books_meta=books_meta,
            books_lookup=books_lookup,
        )
        all_frames.append(frame)

    candidates = pd.concat(all_frames, ignore_index=True)

    require(len(candidates) == EXPECTED_ROWS, f"Expected {EXPECTED_ROWS} candidate rows.")
    require(
        not candidates.duplicated(["query_id", "isbn13"]).any(),
        "Duplicate query-book pairs detected.",
    )

    counts = candidates.groupby("query_id").size().to_dict()
    require(
        len(counts) == EXPECTED_QUERY_COUNT
        and all(value == EXPECTED_TOP_K for value in counts.values()),
        f"Expected 24x50 candidate rows; got {counts}",
    )

    # The v2 baseline is deterministic from the same pool: vector ranks 1..10.
    v2_top10 = candidates[candidates["vector_rank"] <= 10].copy()
    v2_top10 = v2_top10.sort_values(
        ["query_id", "vector_rank"],
        kind="stable",
    )
    v2_top10 = v2_top10.rename(columns={"vector_rank": "rank"})
    v2_top10["vector_rank"] = v2_top10["rank"].astype(int)

    baseline_counts = v2_top10.groupby("query_id").size().to_dict()
    require(
        len(baseline_counts) == EXPECTED_QUERY_COUNT
        and all(value == 10 for value in baseline_counts.values()),
        "v2 holdout baseline is not exactly 24x10.",
    )

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = RUN_ROOT / f"{timestamp}_retrieval"
    if run_dir.exists():
        raise FileExistsError(run_dir)
    run_dir.mkdir(parents=True)

    candidates_file = run_dir / "holdout_top50_candidates.csv"
    v2_file = run_dir / "v2_holdout_recommendations.csv"
    metadata_file = run_dir / "holdout_retrieval_metadata.json"
    lock_file = run_dir / "holdout_retrieval_lock.json"

    candidates.to_csv(candidates_file, index=False, encoding="utf-8")
    v2_top10[
        [
            "query_id",
            "query",
            "stratum",
            "rank",
            "vector_rank",
            "isbn13",
            "title",
            "description",
            "authors",
            "retrieval_document_text",
        ]
    ].to_csv(v2_file, index=False, encoding="utf-8")

    metadata = {
        "status": "V3_HOLDOUT_RETRIEVAL_COMPLETE",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": git_sha(),
        "candidate_generator_file": str(
            RECOMMENDER_FILE.relative_to(REPO_ROOT)
        ).replace("\\", "/"),
        "candidate_generator_sha256": sha256(RECOMMENDER_FILE),
        "holdout_input_lock_sha256": sha256(HOLDOUT_INPUT_LOCK),
        "selected_candidate_lock_sha256": sha256(SELECTED_CANDIDATE_LOCK),
        "query_file_sha256": sha256(QUERY_FILE),
        "facet_file_sha256": sha256(FACET_FILE),
        "query_count": EXPECTED_QUERY_COUNT,
        "candidate_depth": EXPECTED_TOP_K,
        "candidate_rows": EXPECTED_ROWS,
        "baseline_display_depth": 10,
        "v2_baseline_rows": len(v2_top10),
        "reranker_inference_performed": False,
        "judge_labels_loaded": False,
        "human_relevance_labels_loaded": False,
        "judge_calls_performed": 0,
        "outputs": {
            "holdout_top50_candidates.csv": sha256(candidates_file),
            "v2_holdout_recommendations.csv": sha256(v2_file),
        },
    }
    metadata_file.write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )

    retrieval_lock = {
        "status": "V3_HOLDOUT_RETRIEVAL_FROZEN_BEFORE_RERANKING",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "holdout_input_lock_sha256": sha256(HOLDOUT_INPUT_LOCK),
        "selected_candidate_lock_sha256": sha256(SELECTED_CANDIDATE_LOCK),
        "candidate_generator_sha256": sha256(RECOMMENDER_FILE),
        "holdout_top50_candidates_sha256": sha256(candidates_file),
        "v2_holdout_recommendations_sha256": sha256(v2_file),
        "holdout_retrieval_metadata_sha256": sha256(metadata_file),
        "query_count": EXPECTED_QUERY_COUNT,
        "candidate_depth": EXPECTED_TOP_K,
        "candidate_rows": EXPECTED_ROWS,
        "v2_baseline_rows": len(v2_top10),
        "reranker_inference_performed": False,
        "judge_execution_performed": False,
        "human_relevance_labels_used": False,
        "next_stage": "selected_v3b_reranking_on_frozen_holdout_top50",
    }
    lock_file.write_text(
        json.dumps(retrieval_lock, indent=2) + "\n",
        encoding="utf-8",
    )

    print()
    print("PASS  24x50 holdout candidate structure")
    print("PASS  v2 baseline reconstructed as vector ranks 1..10")
    print("PASS  retrieval artifacts frozen before reranking")
    print()
    print("V3 UNSEEN HOLDOUT RETRIEVAL: PASS")
    print(f"Run directory: {run_dir}")
    print(f"Top50 SHA-256: {sha256(candidates_file)}")
    print(f"v2 baseline SHA-256: {sha256(v2_file)}")
    print(f"Retrieval lock SHA-256: {sha256(lock_file)}")
    print("Reranker inference performed: NO")
    print("Judge calls performed: 0")
    print("Next: selected v3b reranking against this frozen top50 pool.")


if __name__ == "__main__":
    main()
