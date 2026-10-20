from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from semantic_relevance_v4a_hybrid import (
    BM25Config,
    BM25Index,
    build_hybrid_union,
    canonical_isbn,
    safe_text,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
EVALS = REPO_ROOT / "evals"
BASE = EVALS / "system_evaluation"

SPEC_JSON = BASE / "semantic_relevance_v4a_hybrid_retrieval_experiment.v0.1.0.json"
SPEC_MD = BASE / "semantic_relevance_v4a_hybrid_retrieval_experiment.v0.1.0.md"
SPEC_LOCK = BASE / "semantic_relevance_v4a_hybrid_retrieval_experiment.v0.1.0.lock.json"
IMPLEMENTATION = BASE / "semantic_relevance_v4a_hybrid.py"
IMPLEMENTATION_LOCK = BASE / "semantic_relevance_v4a_implementation.lock.json"

QUERY_FILE = EVALS / "datasets/semantic_relevance_v3_holdout_queries.v0.1.0.csv"
CATALOG_FILE = REPO_ROOT / "books_with_emotions.csv"

DENSE_RUN = EVALS / "runs/semantic_relevance_v3_holdout/20261007T124150Z_retrieval"
DENSE_TOP50 = DENSE_RUN / "holdout_top50_candidates.csv"
DENSE_RETRIEVAL_LOCK = DENSE_RUN / "holdout_retrieval_lock.json"

RUN_ROOT = EVALS / "runs/semantic_relevance_v4a_development"

EXPECTED_SPEC_JSON_SHA256 = "6aba9c8354f201ad41044dd82d4476612f81ebd27a5ce4f8bf4f327a8b3281b9"
EXPECTED_SPEC_MD_SHA256 = "5af31f1b986187b668b0d486d4209da198e88eb6ea03e3b12818ea8c04e33b90"
EXPECTED_SPEC_LOCK_SHA256 = "c9fee869cda6ccfd26c1ad0b2624cbcf693b9c896366b6baaa5c0b7b98371db2"
EXPECTED_IMPLEMENTATION_SHA256 = "6f31504e29078fffbab86ddb86ef26b02aaa22082d3bfdb6944792bf689f6db7"
EXPECTED_IMPLEMENTATION_LOCK_SHA256 = "b67d305ca5c5da7df82997f43129ad596c7c79eb085571bb4a83b8638f016775"
EXPECTED_QUERY_SHA256 = "75c06bb2c41155cee216339ab7c0421e214f7b3324fac5ec7414f3e2c5bb7b45"
EXPECTED_DENSE_TOP50_SHA256 = "5587b0400b1fcabb86bbbfcbc6deee741e418cc277e2b9a18ab0efa29e8d4acf"
EXPECTED_DENSE_RETRIEVAL_LOCK_SHA256 = "4c0553065f94314ded8cd4b5d8edd90cef3a812c910c69c0a5907950a2b503af"

EXPECTED_QUERY_IDS = [f"H{i:02d}" for i in range(1, 25)]
EXPECTED_QUERY_COUNT = 24
EXPECTED_DENSE_PER_QUERY = 50
EXPECTED_DENSE_ROWS = 1200


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


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


def validate_frozen_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    for label, path, expected in [
        ("v4a JSON spec", SPEC_JSON, EXPECTED_SPEC_JSON_SHA256),
        ("v4a Markdown spec", SPEC_MD, EXPECTED_SPEC_MD_SHA256),
        ("v4a spec lock", SPEC_LOCK, EXPECTED_SPEC_LOCK_SHA256),
        ("v4a implementation", IMPLEMENTATION, EXPECTED_IMPLEMENTATION_SHA256),
        ("v4a implementation lock", IMPLEMENTATION_LOCK, EXPECTED_IMPLEMENTATION_LOCK_SHA256),
        ("H01-H24 query file", QUERY_FILE, EXPECTED_QUERY_SHA256),
        ("frozen H01-H24 dense top50", DENSE_TOP50, EXPECTED_DENSE_TOP50_SHA256),
        ("frozen dense retrieval lock", DENSE_RETRIEVAL_LOCK, EXPECTED_DENSE_RETRIEVAL_LOCK_SHA256),
    ]:
        require(path.exists(), f"Missing {label}: {path}")
        actual = sha256(path)
        require(actual == expected, f"{label} hash mismatch: expected={expected} actual={actual}")
        print(f"PASS  {label} SHA-256 verified")

    spec = load_json(SPEC_JSON)
    implementation_lock = load_json(IMPLEMENTATION_LOCK)
    dense_lock = load_json(DENSE_RETRIEVAL_LOCK)

    require(spec.get("status") == "FROZEN_BEFORE_V4A_IMPLEMENTATION", "v4a spec is not frozen.")
    require(
        implementation_lock.get("status") == "V4A_IMPLEMENTATION_FROZEN_BEFORE_DEVELOPMENT_RETRIEVAL",
        "v4a implementation lock status is invalid.",
    )
    require(
        dense_lock.get("status") == "V3_HOLDOUT_RETRIEVAL_FROZEN_BEFORE_RERANKING",
        "Frozen dense retrieval lock status is invalid.",
    )
    require(
        dense_lock.get("holdout_top50_candidates_sha256") == EXPECTED_DENSE_TOP50_SHA256,
        "Dense retrieval lock disagrees with frozen top50 hash.",
    )

    queries = pd.read_csv(QUERY_FILE, dtype=str, keep_default_na=False, encoding="utf-8")
    require(set(queries.columns) >= {"query_id", "query", "stratum"}, "Query file columns changed.")
    require(len(queries) == EXPECTED_QUERY_COUNT, "Expected exactly 24 development queries.")
    require(queries["query_id"].tolist() == EXPECTED_QUERY_IDS, "H01-H24 query IDs/order changed.")
    require(not queries["query_id"].duplicated().any(), "Duplicate query IDs.")
    require(not queries["query"].duplicated().any(), "Duplicate query text.")

    dense = pd.read_csv(DENSE_TOP50, dtype=str, keep_default_na=False, encoding="utf-8")
    required_dense = {"query_id", "query", "stratum", "vector_rank", "isbn13", "title", "description"}
    missing = required_dense - set(dense.columns)
    require(not missing, f"Dense top50 missing columns: {sorted(missing)}")
    dense["vector_rank"] = pd.to_numeric(dense["vector_rank"], errors="raise").astype(int)
    require(len(dense) == EXPECTED_DENSE_ROWS, f"Expected 1200 dense rows, got {len(dense)}.")
    require(not dense.duplicated(["query_id", "isbn13"]).any(), "Duplicate dense query-book pairs.")
    counts = dense.groupby("query_id").size()
    require(len(counts) == 24 and (counts == 50).all(), "Frozen dense candidate structure is not 24x50.")
    for qid, group in dense.groupby("query_id", sort=True):
        require(sorted(group["vector_rank"].tolist()) == list(range(1, 51)), f"{qid}: dense ranks are not 1..50.")

    qmap = queries.set_index("query_id")[["query", "stratum"]].to_dict("index")
    for qid, group in dense.groupby("query_id", sort=True):
        require(qid in qmap, f"Dense query {qid} is not in frozen query file.")
        require(group["query"].nunique() == 1 and group["query"].iloc[0] == qmap[qid]["query"], f"{qid}: query text mismatch.")
        require(group["stratum"].nunique() == 1 and group["stratum"].iloc[0] == qmap[qid]["stratum"], f"{qid}: stratum mismatch.")

    return queries, dense


def load_and_validate_catalog(dense: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    require(CATALOG_FILE.exists(), f"Runtime catalog not found: {CATALOG_FILE}")
    catalog_sha = sha256(CATALOG_FILE)
    catalog = pd.read_csv(CATALOG_FILE, dtype=str, keep_default_na=False, encoding="utf-8")
    required = {"isbn13", "title", "description"}
    missing = required - set(catalog.columns)
    require(not missing, f"Runtime catalog missing columns: {sorted(missing)}")
    if "authors" not in catalog.columns:
        catalog["authors"] = ""

    catalog = catalog.copy().reset_index(drop=True)
    catalog["isbn13"] = catalog["isbn13"].map(canonical_isbn)
    require(not catalog["isbn13"].duplicated().any(), "Runtime catalog contains duplicate canonical ISBN13 values.")
    require(len(catalog) > 0, "Runtime catalog is empty.")

    lookup = catalog.set_index("isbn13", drop=False)
    dense_isbns = dense["isbn13"].map(canonical_isbn)
    missing_dense = sorted(set(dense_isbns) - set(lookup.index))
    require(not missing_dense, f"Frozen dense ISBNs missing from runtime catalog: {missing_dense[:10]}")

    # Guard against metadata drift on all 1,200 frozen dense candidates. BM25 sees
    # the full current catalog, so the complete catalog SHA is also captured below.
    mismatch_examples: list[str] = []
    for row in dense.itertuples(index=False):
        isbn = canonical_isbn(row.isbn13)
        current = lookup.loc[isbn]
        if safe_text(row.title) != safe_text(current["title"]) or safe_text(row.description) != safe_text(current["description"]):
            mismatch_examples.append(f"{row.query_id}:{isbn}")
            if len(mismatch_examples) >= 10:
                break
    require(
        not mismatch_examples,
        "Runtime catalog title/description drift detected for frozen dense candidates: " + ", ".join(mismatch_examples),
    )

    return catalog, catalog_sha


def main() -> None:
    print("Semantic relevance v4a - development hybrid retrieval")
    print("----------------------------------------------------")
    print("Development queries: H01-H24 (development-only for v4)")
    print("Dense retrieval rerun: NO; reuse frozen v3 H01-H24 top50")
    print("BM25 catalog indexing: ENABLED")
    print("BM25 retrieval: ENABLED")
    print("Hybrid RRF: ENABLED")
    print("Reranker inference: DISABLED")
    print("Judge labels loaded: NO")
    print("Judge calls: 0")
    print("Release decision: DISABLED")
    print()

    queries, dense = validate_frozen_inputs()
    print("PASS  frozen v4a spec/implementation/query/dense boundaries verified")

    catalog, catalog_sha = load_and_validate_catalog(dense)
    print(f"PASS  runtime catalog loaded and metadata drift checks passed ({len(catalog)} books)")
    print(f"INFO  runtime catalog SHA-256: {catalog_sha}")

    index = BM25Index(
        isbn13=catalog["isbn13"].tolist(),
        title=catalog["title"].tolist(),
        description=catalog["description"].tolist(),
        config=BM25Config(k1=1.5, b=0.75, candidate_depth=50),
    )
    print(f"PASS  deterministic BM25 index built: docs={index.n_docs} avgdl={index.avgdl:.6f} vocab={len(index.df)}")

    catalog_lookup = catalog.set_index("isbn13", drop=False)
    bm25_frames: list[pd.DataFrame] = []
    union_frames: list[pd.DataFrame] = []

    print()
    print("Collecting BM25 top50 and dense+BM25 hybrid unions...")
    for q in queries.itertuples(index=False):
        qid = str(q.query_id)
        query = str(q.query)
        stratum = str(q.stratum)

        bm25 = index.rank_query(query, top_k=50)
        require(len(bm25) <= 50, f"{qid}: BM25 returned >50 candidates.")
        require(not bm25["isbn13"].duplicated().any(), f"{qid}: duplicate BM25 ISBNs.")
        if len(bm25):
            require(bm25["bm25_rank"].tolist() == list(range(1, len(bm25) + 1)), f"{qid}: BM25 ranks are not contiguous.")
            require((bm25["bm25_score"] > 0).all(), f"{qid}: non-positive BM25 score returned.")

        bm25_out = bm25.copy()
        bm25_out.insert(0, "query_id", qid)
        bm25_out.insert(1, "query", query)
        bm25_out.insert(2, "stratum", stratum)
        if len(bm25_out):
            meta = catalog_lookup.loc[bm25_out["isbn13"].tolist()][["title", "description", "authors"]].reset_index(drop=True)
            bm25_out = pd.concat([bm25_out.reset_index(drop=True), meta], axis=1)
        else:
            bm25_out["title"] = ""
            bm25_out["description"] = ""
            bm25_out["authors"] = ""
        bm25_frames.append(bm25_out)

        dense_q = dense[dense["query_id"] == qid][["isbn13", "vector_rank"]].copy()
        dense_q["isbn13"] = dense_q["isbn13"].map(canonical_isbn)
        dense_q = dense_q.rename(columns={"vector_rank": "dense_rank"})

        union = build_hybrid_union(
            dense_q,
            bm25[["isbn13", "bm25_rank", "bm25_score"]],
            rrf_k=60.0,
        )
        require(len(union) >= 50 and len(union) <= 100, f"{qid}: hybrid union size outside [50,100]: {len(union)}")
        require(not union["isbn13"].duplicated().any(), f"{qid}: duplicate hybrid ISBNs.")
        require(sorted(union["hybrid_retrieval_rank"].tolist()) == list(range(1, len(union) + 1)), f"{qid}: hybrid ranks not contiguous.")
        require(set(dense_q["isbn13"]) <= set(union["isbn13"]), f"{qid}: dense candidates lost from union.")
        require(set(bm25["isbn13"]) <= set(union["isbn13"]), f"{qid}: BM25 candidates lost from union.")

        union["dense_rank"] = pd.to_numeric(union["dense_rank"], errors="coerce").astype("Int64")
        union["bm25_rank"] = pd.to_numeric(union["bm25_rank"], errors="coerce").astype("Int64")
        meta = catalog_lookup.loc[union["isbn13"].tolist()][["title", "description", "authors"]].reset_index(drop=True)
        union = pd.concat([union.reset_index(drop=True), meta], axis=1)
        union.insert(0, "query_id", qid)
        union.insert(1, "query", query)
        union.insert(2, "stratum", stratum)
        union_frames.append(union)

        source_counts = union["candidate_source"].value_counts().to_dict()
        print(
            f"  {qid}: BM25={len(bm25):2d} union={len(union):3d} "
            f"both={int(source_counts.get('both', 0)):2d} "
            f"dense_only={int(source_counts.get('dense_only', 0)):2d} "
            f"bm25_only={int(source_counts.get('bm25_only', 0)):2d}"
        )

    bm25_all = pd.concat(bm25_frames, ignore_index=True)
    union_all = pd.concat(union_frames, ignore_index=True)

    require(not bm25_all.duplicated(["query_id", "isbn13"]).any(), "Duplicate query-book pairs in BM25 output.")
    require(not union_all.duplicated(["query_id", "isbn13"]).any(), "Duplicate query-book pairs in hybrid union.")
    require(union_all["query_id"].nunique() == 24, "Hybrid union does not contain all 24 queries.")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = RUN_ROOT / f"{timestamp}_retrieval"
    if run_dir.exists():
        raise FileExistsError(run_dir)
    run_dir.mkdir(parents=True)

    bm25_file = run_dir / "v4a_dev_bm25_top50.csv"
    union_file = run_dir / "v4a_dev_hybrid_union.csv"
    metadata_file = run_dir / "v4a_dev_retrieval_metadata.json"
    lock_file = run_dir / "v4a_dev_retrieval.lock.json"

    bm25_all.to_csv(bm25_file, index=False, encoding="utf-8")
    union_all.sort_values(["query_id", "hybrid_retrieval_rank"], kind="stable").to_csv(
        union_file, index=False, encoding="utf-8"
    )

    bm25_counts = bm25_all.groupby("query_id").size().reindex(EXPECTED_QUERY_IDS, fill_value=0)
    union_counts = union_all.groupby("query_id").size().reindex(EXPECTED_QUERY_IDS, fill_value=0)
    source_counts = {str(k): int(v) for k, v in union_all["candidate_source"].value_counts().to_dict().items()}

    metadata = {
        "status": "V4A_DEVELOPMENT_RETRIEVAL_COMPLETE",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": git_sha(),
        "development_query_role": "H01-H24 are v4 development only; no release claim allowed",
        "query_file_sha256": EXPECTED_QUERY_SHA256,
        "frozen_dense_top50_sha256": EXPECTED_DENSE_TOP50_SHA256,
        "frozen_dense_retrieval_lock_sha256": EXPECTED_DENSE_RETRIEVAL_LOCK_SHA256,
        "v4a_spec_sha256": EXPECTED_SPEC_JSON_SHA256,
        "v4a_implementation_sha256": EXPECTED_IMPLEMENTATION_SHA256,
        "v4a_implementation_lock_sha256": EXPECTED_IMPLEMENTATION_LOCK_SHA256,
        "runtime_catalog_file": str(CATALOG_FILE.relative_to(REPO_ROOT)).replace("\\", "/"),
        "runtime_catalog_sha256": catalog_sha,
        "catalog_rows": len(catalog),
        "bm25_index": {
            "document_count": index.n_docs,
            "average_document_length": index.avgdl,
            "vocabulary_size": len(index.df),
            "k1": 1.5,
            "b": 0.75,
            "candidate_depth": 50,
            "fields": ["title", "description"],
        },
        "dense_branch": {
            "rerun": False,
            "candidate_depth": 50,
            "rows": 1200,
        },
        "bm25_branch": {
            "retrieval_executed": True,
            "rows": len(bm25_all),
            "min_candidates_per_query": int(bm25_counts.min()),
            "max_candidates_per_query": int(bm25_counts.max()),
            "mean_candidates_per_query": float(bm25_counts.mean()),
        },
        "hybrid_union": {
            "rows": len(union_all),
            "min_candidates_per_query": int(union_counts.min()),
            "max_candidates_per_query": int(union_counts.max()),
            "mean_candidates_per_query": float(union_counts.mean()),
            "source_counts": source_counts,
            "rrf_k": 60,
            "dense_weight": 1.0,
            "bm25_weight": 1.0,
            "truncated_before_reranker": False,
        },
        "execution_boundary": {
            "catalog_read": True,
            "bm25_index_built": True,
            "bm25_retrieval_executed": True,
            "dense_retrieval_rerun": False,
            "reranker_inference_executed": False,
            "judge_labels_loaded": False,
            "judge_calls_performed": 0,
            "human_relevance_labels_used": False,
            "release_decision_produced": False,
        },
        "outputs": {
            "v4a_dev_bm25_top50.csv": sha256(bm25_file),
            "v4a_dev_hybrid_union.csv": sha256(union_file),
        },
    }
    metadata_file.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    lock = {
        "status": "V4A_DEVELOPMENT_RETRIEVAL_FROZEN_BEFORE_RERANKING_OR_JUDGING",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_id": "semantic-relevance-hybrid-retrieval-v4a",
        "runtime_catalog_sha256": catalog_sha,
        "query_file_sha256": EXPECTED_QUERY_SHA256,
        "frozen_dense_top50_sha256": EXPECTED_DENSE_TOP50_SHA256,
        "v4a_implementation_sha256": EXPECTED_IMPLEMENTATION_SHA256,
        "v4a_implementation_lock_sha256": EXPECTED_IMPLEMENTATION_LOCK_SHA256,
        "bm25_top50_sha256": sha256(bm25_file),
        "hybrid_union_sha256": sha256(union_file),
        "retrieval_metadata_sha256": sha256(metadata_file),
        "query_count": 24,
        "bm25_candidate_depth": 50,
        "dense_candidate_depth": 50,
        "hybrid_rrf_k": 60,
        "hybrid_dense_weight": 1.0,
        "hybrid_bm25_weight": 1.0,
        "hybrid_union_rows": len(union_all),
        "hybrid_union_truncated_before_reranker": False,
        "dense_retrieval_rerun": False,
        "reranker_inference_performed": False,
        "judge_execution_performed": False,
        "human_relevance_labels_used": False,
        "release_decision_produced": False,
        "next_stage": "PREPARE_V4A_DEVELOPMENT_RERANKER_EXECUTION_ON_FROZEN_HYBRID_UNION",
    }
    lock_file.write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")

    print()
    print("PASS  BM25 development retrieval complete")
    print("PASS  dense50 + BM25-50 union built without pre-reranker truncation")
    print("PASS  development retrieval artifacts frozen before reranking/judging")
    print()
    print("V4A DEVELOPMENT RETRIEVAL: PASS")
    print(f"Run directory: {run_dir}")
    print(f"Catalog SHA-256: {catalog_sha}")
    print(f"BM25 top50 SHA-256: {sha256(bm25_file)}")
    print(f"Hybrid union SHA-256: {sha256(union_file)}")
    print(f"Retrieval lock SHA-256: {sha256(lock_file)}")
    print("Dense retrieval rerun: NO")
    print("Reranker inference performed: NO")
    print("Judge calls performed: 0")
    print("Release decision produced: NO")
    print("Next: prepare v4a development reranker execution on this frozen hybrid union.")


if __name__ == "__main__":
    main()
