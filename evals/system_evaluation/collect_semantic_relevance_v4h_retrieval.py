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

from semantic_relevance_v4a_hybrid import BM25Config, BM25Index, build_hybrid_union

REPO_ROOT = Path(__file__).resolve().parents[2]
SYS = REPO_ROOT / "evals/system_evaluation"
FACETS = REPO_ROOT / "evals/facets/semantic_relevance"
DEV_RUN = REPO_ROOT / "evals/runs/semantic_relevance_v4a_development/20261019T104415Z_retrieval"
RUN_ROOT = REPO_ROOT / "evals/runs/semantic_relevance_v4_holdout"

V4B_SPEC = SYS / "semantic_relevance_v4b_reranker_only_hybrid_experiment.v0.1.0.json"
V4B_DESIGN_LOCK = SYS / "semantic_relevance_v4b_reranker_only_hybrid_experiment.v0.1.0.lock.json"
V4B_SELECTION_LOCK = DEV_RUN / "v4b_dev_comparison_selection.lock.json"
V4H_QUERY_FILE = SYS / "semantic_relevance_v4_holdout_queries.v0.1.0.json"
V4H_FACET_FILE = FACETS / "semantic_relevance_v4_holdout_query_facets.v0.1.0.json"
V4H_AUTHORING_LOCK = SYS / "semantic_relevance_v4_holdout_authoring.lock.json"
RETRIEVAL_CONTRACT = SYS / "semantic_relevance_v4h_retrieval_contract.v0.1.0.json"

RECOMMENDER_FILE = REPO_ROOT / "gradio-dashboard.py"
HYBRID_IMPLEMENTATION = SYS / "semantic_relevance_v4a_hybrid.py"
CATALOG_FILE = REPO_ROOT / "books_with_emotions.csv"
TAGGED_FILE = REPO_ROOT / "tagged_description.txt"


EXPECTED = {
    "v4b_design_spec": "c72b5edbc330feae0620314fa5d5c6f90a55b356174d70a075ae77b47a8d0fca",
    "v4b_design_lock": "e60ca1032f7fa733f3e7a51ec87a229d7454645aa6109b5e557588af50383c6d",
    "v4b_selection_lock": "98ab3a62b978b922dc4b6e1bb031bdc6578a007a93de7ae84b20d36a80c9c9b4",
    "v4h_query": "eef2b5a0c7e1400adcc1d68edf7f26e16297f985fdb50f2fa26fa6ceff75f813",
    "v4h_facet": "dbdb7c5d5603e17164cf6ef880df912f6196663387885e1f5773e54b97d2b106",
    "v4h_authoring_lock": "14cd3a8a9f6bf22666dd4c74a150e85f82a53789ccaa5bd2870d606b66ef83e7",
    "catalog": "22244d277a643e56bf23a744d0d45eb2247db4a05792ec69ffd942c6d5fdbf6e",
    "tagged_description": "cb3f34f397be0d65af4f931bc24265b69b1ed151b0a76ade03885fbaf3c5e1a5"
}
EXPECTED_CONTRACT_SHA256 = "1bf245fdd3cec8baa4fd3e6ece3602ab2d72ee5fafd12e49b0bb68fb443dd6f8"
EXPECTED_GRADIO_BLOB = "89a769f7a752a517ebc7079e08c531e8df415bce"
EXPECTED_HYBRID_BLOB = "3c1a612dc4d33a7680dd1f93d40558cb094efa1b"
EXPECTED_GRADIO_CONTENT_SHA256 = "cdd0458ff23bd669185a8c882ceb132438d037af1a29f4dd60bd7fef9c405b60"
EXPECTED_HYBRID_CONTENT_SHA256 = "6f31504e29078fffbab86ddb86ef26b02aaa22082d3bfdb6944792bf689f6db7"
EXPECTED_QUERY_IDS = [f"V4H{i:02d}" for i in range(1, 25)]
EXPECTED_QUERY_COUNT = 24
EXPECTED_DEPTH = 50
EXPECTED_DENSE_ROWS = 1200

def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def lf_sha256(path: Path) -> str:
    b = path.read_bytes()
    return hashlib.sha256(b.replace(b"\r\n", b"\n").replace(b"\r", b"\n")).hexdigest()

def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)

def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8-sig"))

def canonical_isbn(value: Any) -> str:
    text = str(value).strip().strip('"').strip("'")
    if re.fullmatch(r"\d+\.0", text):
        text = text[:-2]
    if not text:
        raise ValueError("ISBN13 is empty after normalization.")
    return text

def verify_text_hash(path: Path, expected: str, label: str) -> None:
    require(path.exists(), f"Missing {label}: {path}")
    raw = sha256(path)
    if raw == expected:
        mode = "raw"
    else:
        normalized = lf_sha256(path)
        require(normalized == expected, f"{label} hash mismatch: expected={expected} raw={raw} lf={normalized}")
        mode = "canonical-LF"
    print(f"PASS  {label} {mode} SHA-256 verified")

def verify_raw_hash(path: Path, expected: str, label: str) -> None:
    require(path.exists(), f"Missing {label}: {path}")
    actual = sha256(path)
    require(actual == expected, f"{label} hash mismatch: expected={expected} actual={actual}")
    print(f"PASS  {label} raw SHA-256 verified")

def git_blob(repo_path: str) -> str:
    return subprocess.run(
        ["git", "rev-parse", f"HEAD:{repo_path}"],
        cwd=str(REPO_ROOT), capture_output=True, text=True, check=True
    ).stdout.strip()

def require_tracked_unmodified(repo_path: str, expected_blob: str, label: str) -> None:
    blob = git_blob(repo_path)
    require(blob == expected_blob, f"{label} tracked Git blob changed: expected={expected_blob} actual={blob}")
    diff = subprocess.run(["git", "diff", "--quiet", "--", repo_path], cwd=str(REPO_ROOT))
    require(diff.returncode == 0, f"{label} has working-tree modifications")
    print(f"PASS  {label} exact tracked Git blob verified and working tree unmodified")

def validate_query_and_freeze_boundary() -> list[dict[str, str]]:
    verify_text_hash(V4B_SPEC, EXPECTED["v4b_design_spec"], "V4B design spec")
    verify_raw_hash(V4B_DESIGN_LOCK, EXPECTED["v4b_design_lock"], "V4B design lock")
    verify_raw_hash(V4B_SELECTION_LOCK, EXPECTED["v4b_selection_lock"], "V4B development selection lock")
    verify_raw_hash(V4H_QUERY_FILE, EXPECTED["v4h_query"], "V4H query file")
    verify_raw_hash(V4H_FACET_FILE, EXPECTED["v4h_facet"], "V4H facet file")
    verify_raw_hash(V4H_AUTHORING_LOCK, EXPECTED["v4h_authoring_lock"], "V4H authoring lock")
    verify_raw_hash(RETRIEVAL_CONTRACT, EXPECTED_CONTRACT_SHA256, "V4H retrieval contract")

    require_tracked_unmodified("gradio-dashboard.py", EXPECTED_GRADIO_BLOB, "dense candidate-generator source")
    require_tracked_unmodified(
        "evals/system_evaluation/semantic_relevance_v4a_hybrid.py",
        EXPECTED_HYBRID_BLOB,
        "hybrid retrieval implementation",
    )
    # Also keep the historical byte-level hashes visible for provenance.
    verify_text_hash(RECOMMENDER_FILE, EXPECTED_GRADIO_CONTENT_SHA256, "dense candidate-generator content")
    verify_text_hash(HYBRID_IMPLEMENTATION, EXPECTED_HYBRID_CONTENT_SHA256, "hybrid retrieval implementation content")

    spec = load_json(V4B_SPEC)
    require(spec.get("status") == "PROPOSED_FOR_FREEZE_BEFORE_V4B_IMPLEMENTATION", "Unexpected V4B design-spec status")
    require(spec.get("new_unseen_holdout_protocol", {}).get("query_ids") == "V4H01-V4H24", "V4B unseen-holdout query IDs changed")
    require(spec.get("new_unseen_holdout_protocol", {}).get("query_and_facet_files_frozen_before_retrieval") is True,
            "V4B spec no longer requires query/facet freeze before retrieval")
    require(spec.get("final_v4b_ranking", {}).get("method") == "reranker_score_only", "V4B final ranking method changed")

    design_lock = load_json(V4B_DESIGN_LOCK)
    require(design_lock.get("status") == "V4B_DESIGN_FROZEN_BEFORE_IMPLEMENTATION", "Unexpected V4B design-lock status")
    require(design_lock.get("ranking_method") == "reranker_score_only", "V4B design lock ranking method changed")
    require(design_lock.get("non_tie_retrieval_rank_weight") is False, "V4B design lock gives retrieval rank non-tie weight")

    selection = load_json(V4B_SELECTION_LOCK)
    require(selection.get("status") == "V4B_DEVELOPMENT_COMPARISON_AND_SELECTION_FROZEN", "Unexpected V4B selection-lock status")
    require(selection.get("selected_candidate") == "v4b", "V4B is not the frozen selected development candidate")
    require(selection.get("requires_new_unseen_v4_holdout") is True, "Frozen selection does not require V4H holdout")
    require(selection.get("release_decision_produced") is False, "Development selection unexpectedly produced release decision")

    authoring = load_json(V4H_AUTHORING_LOCK)
    require(authoring.get("status") == "V4H_QUERY_AND_FACET_SPEC_FROZEN_BEFORE_RETRIEVAL", "Unexpected V4H authoring-lock status")
    require(authoring.get("query_file_sha256") == EXPECTED["v4h_query"], "Authoring lock query hash mismatch")
    require(authoring.get("facet_file_sha256") == EXPECTED["v4h_facet"], "Authoring lock facet hash mismatch")
    require(authoring.get("holdout_retrieval_performed") is False, "Authoring lock says V4H retrieval already occurred")
    require(authoring.get("judge_execution_performed") is False, "Authoring lock says V4H judge already ran")

    qdoc = load_json(V4H_QUERY_FILE)
    fdoc = load_json(V4H_FACET_FILE)
    queries = qdoc.get("queries", [])
    facets = fdoc.get("queries", [])
    require(len(queries) == EXPECTED_QUERY_COUNT and len(facets) == EXPECTED_QUERY_COUNT, "V4H query/facet count changed")
    require([q.get("query_id") for q in queries] == EXPECTED_QUERY_IDS, "V4H query IDs/order changed")
    require(
        [(q.get("query_id"), q.get("query"), q.get("stratum")) for q in queries]
        == [(q.get("query_id"), q.get("query"), q.get("stratum")) for q in facets],
        "V4H query and facet files disagree",
    )
    require(len({q.get("query") for q in queries}) == EXPECTED_QUERY_COUNT, "Duplicate V4H query text")
    print("PASS  frozen V4H query/facet boundary verified")
    return [
        {"query_id": str(q["query_id"]), "query": str(q["query"]), "stratum": str(q["stratum"])}
        for q in queries
    ]

def validate_runtime_data() -> tuple[pd.DataFrame, list[str]]:
    verify_raw_hash(CATALOG_FILE, EXPECTED["catalog"], "runtime catalog")
    verify_raw_hash(TAGGED_FILE, EXPECTED["tagged_description"], "tagged-description dense corpus")

    catalog = pd.read_csv(CATALOG_FILE, dtype=str, keep_default_na=False, encoding="utf-8")
    required = {"isbn13", "title", "description"}
    missing = required - set(catalog.columns)
    require(not missing, f"Runtime catalog missing columns: {sorted(missing)}")
    if "authors" not in catalog.columns:
        catalog["authors"] = ""
    catalog = catalog.copy().reset_index(drop=True)
    catalog["isbn13"] = catalog["isbn13"].map(canonical_isbn)
    require(len(catalog) == 5197, f"Expected 5197 catalog rows, got {len(catalog)}")
    require(not catalog["isbn13"].duplicated().any(), "Runtime catalog contains duplicate ISBN13 values")

    tagged_lines = [line for line in TAGGED_FILE.read_text(encoding="utf-8").splitlines() if line.strip()]
    require(len(tagged_lines) == 5197, f"Expected 5197 nonempty tagged-description lines, got {len(tagged_lines)}")
    tagged_isbns = [canonical_isbn(line.strip('"').split()[0]) for line in tagged_lines]
    require(len(set(tagged_isbns)) == 5197, "Tagged-description corpus contains duplicate ISBNs")
    require(set(tagged_isbns) == set(catalog["isbn13"]), "Catalog and tagged-description ISBN sets differ")
    print("PASS  runtime catalog/tagged-description structure and ISBN-set alignment verified")
    return catalog, tagged_lines


EXPECTED_RECOMMENDER_FUNCTION = "retrieve_semantic_recommedations"

def git_sha() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None

def load_backend_namespace_without_ui() -> dict[str, Any]:
    source = RECOMMENDER_FILE.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(RECOMMENDER_FILE))
    function_index = None
    for index, node in enumerate(tree.body):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == EXPECTED_RECOMMENDER_FUNCTION:
            function_index = index
            break
    require(function_index is not None, f"Could not find {EXPECTED_RECOMMENDER_FUNCTION} in {RECOMMENDER_FILE}")
    backend_tree = ast.Module(body=tree.body[: function_index + 1], type_ignores=[])
    ast.fix_missing_locations(backend_tree)
    namespace = {
        "__name__": "_semantic_relevance_v4h_retrieval_backend",
        "__file__": str(RECOMMENDER_FILE),
    }
    previous = Path.cwd()
    try:
        os.chdir(REPO_ROOT)
        exec(compile(backend_tree, filename=str(RECOMMENDER_FILE), mode="exec"), namespace, namespace)
    finally:
        os.chdir(previous)
    require("db_books" in namespace, "Backend namespace does not expose db_books")
    require("books" in namespace and isinstance(namespace["books"], pd.DataFrame),
            "Backend namespace does not expose books DataFrame")
    return namespace

def build_books_lookup(books: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, list[int]]]:
    require("isbn13" in books.columns, "books metadata missing isbn13")
    meta = books.copy().reset_index(drop=True)
    meta["_isbn_key"] = meta["isbn13"].map(canonical_isbn)
    lookup: dict[str, list[int]] = {}
    for idx, key in enumerate(meta["_isbn_key"]):
        lookup.setdefault(key, []).append(int(idx))
    return meta, lookup

def parse_isbn_from_document(page_content: str) -> str:
    tokens = str(page_content).strip('"').split()
    require(bool(tokens), "Vector-store document has empty page_content")
    return canonical_isbn(tokens[0])

def collect_dense_query(query_id: str, query: str, stratum: str, db_books: Any,
                        books_meta: pd.DataFrame, books_lookup: dict[str, list[int]]) -> pd.DataFrame:
    recs1 = db_books.similarity_search(query, k=EXPECTED_DEPTH)
    recs2 = db_books.similarity_search(query, k=EXPECTED_DEPTH)
    require(len(recs1) == EXPECTED_DEPTH and len(recs2) == EXPECTED_DEPTH,
            f"{query_id}: expected 50 dense candidates on both repeat calls")
    ids1 = [parse_isbn_from_document(r.page_content) for r in recs1]
    ids2 = [parse_isbn_from_document(r.page_content) for r in recs2]
    require(ids1 == ids2, f"{query_id}: repeated dense retrieval order differs")
    require(len(set(ids1)) == EXPECTED_DEPTH, f"{query_id}: duplicate ISBN in dense top50")

    rows = []
    for rank, rec in enumerate(recs1, 1):
        isbn = parse_isbn_from_document(rec.page_content)
        matching = books_lookup.get(isbn, [])
        require(len(matching) == 1, f"{query_id}: ISBN {isbn} maps to {len(matching)} metadata rows")
        meta = books_meta.iloc[matching[0]]
        rows.append({
            "query_id": query_id,
            "query": query,
            "stratum": stratum,
            "vector_rank": rank,
            "isbn13": isbn,
            "title": "" if pd.isna(meta.get("title", "")) else str(meta.get("title", "")),
            "description": "" if pd.isna(meta.get("description", "")) else str(meta.get("description", "")),
            "authors": "" if pd.isna(meta.get("authors", "")) else str(meta.get("authors", "")),
            "retrieval_document_text": str(rec.page_content),
        })
    frame = pd.DataFrame(rows)
    require(frame["vector_rank"].tolist() == list(range(1, 51)), f"{query_id}: vector ranks not 1..50")
    return frame

def main() -> None:
    print("Semantic relevance V4H - unseen final-holdout candidate retrieval")
    print("----------------------------------------------------------------")
    print("Dense retrieval: ENABLED")
    print("BM25 retrieval: ENABLED")
    print("Hybrid union: ENABLED")
    print("Dense repeat-order check: ENABLED")
    print("Reranker inference: DISABLED")
    print("Judge execution: DISABLED")
    print("Human relevance labels: NOT USED")
    print("Release metrics/decision: NOT CALCULATED")
    print()

    queries = validate_query_and_freeze_boundary()
    catalog, _ = validate_runtime_data()

    existing = sorted(RUN_ROOT.glob("*_retrieval")) if RUN_ROOT.exists() else []
    require(not existing, "A V4H retrieval run directory already exists; refuse to create another before review")

    namespace = load_backend_namespace_without_ui()
    db_books = namespace["db_books"]
    books_meta, books_lookup = build_books_lookup(namespace["books"])

    # The Gradio backend loads the same frozen catalog; enforce its ISBN universe too.
    require(len(books_meta) == 5197, f"Backend books row count changed: {len(books_meta)}")
    require(set(books_meta["_isbn_key"]) == set(catalog["isbn13"]), "Backend books ISBN universe differs from frozen catalog")

    catalog_lookup = catalog.set_index("isbn13", drop=False)
    bm25_index = BM25Index(
        isbn13=catalog["isbn13"].tolist(),
        title=catalog["title"].tolist(),
        description=catalog["description"].tolist(),
        config=BM25Config(k1=1.5, b=0.75, candidate_depth=50),
    )
    require(bm25_index.n_docs == 5197, "BM25 index document count changed")
    print(f"PASS  deterministic BM25 index built: docs={bm25_index.n_docs} avgdl={bm25_index.avgdl:.6f} vocab={len(bm25_index.df)}")

    dense_frames = []
    bm25_frames = []
    union_frames = []

    print()
    print("Collecting V4H dense50 + BM25-50 retrieval...")
    for q in queries:
        qid, query, stratum = q["query_id"], q["query"], q["stratum"]
        dense = collect_dense_query(qid, query, stratum, db_books, books_meta, books_lookup)
        dense_frames.append(dense)

        bm25 = bm25_index.rank_query(query, top_k=50)
        require(len(bm25) <= 50, f"{qid}: BM25 returned >50 rows")
        require(not bm25["isbn13"].duplicated().any(), f"{qid}: duplicate BM25 ISBNs")
        if len(bm25):
            require(bm25["bm25_rank"].tolist() == list(range(1, len(bm25) + 1)), f"{qid}: BM25 ranks not contiguous")
            require((bm25["bm25_score"] > 0).all(), f"{qid}: non-positive BM25 score returned")
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

        dense_for_union = dense[["isbn13", "vector_rank"]].rename(columns={"vector_rank": "dense_rank"}).copy()
        union = build_hybrid_union(
            dense_for_union,
            bm25[["isbn13", "bm25_rank", "bm25_score"]],
            rrf_k=60.0,
        )
        require(50 <= len(union) <= 100, f"{qid}: hybrid union size outside [50,100]: {len(union)}")
        require(not union["isbn13"].duplicated().any(), f"{qid}: duplicate hybrid ISBNs")
        require(union["hybrid_retrieval_rank"].tolist() == list(range(1, len(union) + 1)),
                f"{qid}: hybrid ranks not contiguous")
        require(set(dense_for_union["isbn13"]) <= set(union["isbn13"]), f"{qid}: dense candidates lost from union")
        require(set(bm25["isbn13"]) <= set(union["isbn13"]), f"{qid}: BM25 candidates lost from union")

        union["dense_rank"] = pd.to_numeric(union["dense_rank"], errors="coerce").astype("Int64")
        union["bm25_rank"] = pd.to_numeric(union["bm25_rank"], errors="coerce").astype("Int64")
        meta = catalog_lookup.loc[union["isbn13"].tolist()][["title", "description", "authors"]].reset_index(drop=True)
        union = pd.concat([union.reset_index(drop=True), meta], axis=1)
        union.insert(0, "query_id", qid)
        union.insert(1, "query", query)
        union.insert(2, "stratum", stratum)
        union_frames.append(union)

        sc = union["candidate_source"].value_counts().to_dict()
        print(
            f"  {qid}: dense=50 BM25={len(bm25):2d} union={len(union):3d} "
            f"both={int(sc.get('both',0)):2d} dense_only={int(sc.get('dense_only',0)):2d} "
            f"bm25_only={int(sc.get('bm25_only',0)):2d} repeat_dense=PASS"
        )

    dense_all = pd.concat(dense_frames, ignore_index=True)
    bm25_all = pd.concat(bm25_frames, ignore_index=True)
    union_all = pd.concat(union_frames, ignore_index=True)

    require(len(dense_all) == EXPECTED_DENSE_ROWS, f"Expected 1200 dense rows, got {len(dense_all)}")
    require(not dense_all.duplicated(["query_id","isbn13"]).any(), "Duplicate dense query-book pair")
    require(not bm25_all.duplicated(["query_id","isbn13"]).any(), "Duplicate BM25 query-book pair")
    require(not union_all.duplicated(["query_id","isbn13"]).any(), "Duplicate hybrid query-book pair")
    dense_counts = dense_all.groupby("query_id").size()
    require(len(dense_counts) == 24 and (dense_counts == 50).all(), "Dense artifact is not exact 24x50")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = RUN_ROOT / f"{timestamp}_retrieval"
    require(not run_dir.exists(), f"Run directory already exists: {run_dir}")
    run_dir.mkdir(parents=True)

    dense_file = run_dir / "v4h_dense_top50.csv"
    bm25_file = run_dir / "v4h_bm25_top50.csv"
    union_file = run_dir / "v4h_hybrid_union.csv"
    metadata_file = run_dir / "v4h_retrieval_metadata.json"
    lock_file = run_dir / "v4h_retrieval.lock.json"

    dense_all.sort_values(["query_id","vector_rank"], kind="stable").to_csv(dense_file, index=False, encoding="utf-8")
    bm25_all.sort_values(["query_id","bm25_rank"], kind="stable").to_csv(bm25_file, index=False, encoding="utf-8")
    union_all.sort_values(["query_id","hybrid_retrieval_rank"], kind="stable").to_csv(union_file, index=False, encoding="utf-8")

    bm25_counts = bm25_all.groupby("query_id").size().reindex(EXPECTED_QUERY_IDS, fill_value=0)
    union_counts = union_all.groupby("query_id").size().reindex(EXPECTED_QUERY_IDS, fill_value=0)
    source_counts = {str(k): int(v) for k,v in union_all["candidate_source"].value_counts().to_dict().items()}

    metadata = {
        "status": "V4H_FINAL_HOLDOUT_RETRIEVAL_COMPLETE",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": git_sha(),
        "query_set": "V4H01-V4H24",
        "query_count": 24,
        "development_selected_candidate": "v4b",
        "baseline": "v3b",
        "candidate": "v4b",
        "v4h_query_file_sha256": EXPECTED["v4h_query"],
        "v4h_facet_file_sha256": EXPECTED["v4h_facet"],
        "v4h_authoring_lock_sha256": EXPECTED["v4h_authoring_lock"],
        "v4b_development_selection_lock_sha256": EXPECTED["v4b_selection_lock"],
        "v4b_design_spec_sha256": EXPECTED["v4b_design_spec"],
        "v4b_design_lock_sha256": EXPECTED["v4b_design_lock"],
        "retrieval_contract_sha256": EXPECTED_CONTRACT_SHA256,
        "runtime_catalog_sha256": EXPECTED["catalog"],
        "tagged_description_sha256": EXPECTED["tagged_description"],
        "dense_candidate_generator_git_blob": EXPECTED_GRADIO_BLOB,
        "hybrid_retrieval_implementation_git_blob": EXPECTED_HYBRID_BLOB,
        "dense_branch": {
            "retrieval_executed": True,
            "candidate_depth": 50,
            "rows": len(dense_all),
            "repeat_query_order_identical_for_all_queries": True,
            "embedding_model": "nomic-embed-text",
        },
        "bm25_branch": {
            "retrieval_executed": True,
            "rows": len(bm25_all),
            "min_candidates_per_query": int(bm25_counts.min()),
            "max_candidates_per_query": int(bm25_counts.max()),
            "mean_candidates_per_query": float(bm25_counts.mean()),
            "k1": 1.5,
            "b": 0.75,
            "candidate_depth": 50,
        },
        "hybrid_union": {
            "rows": len(union_all),
            "min_candidates_per_query": int(union_counts.min()),
            "max_candidates_per_query": int(union_counts.max()),
            "mean_candidates_per_query": float(union_counts.mean()),
            "source_counts": source_counts,
            "rrf_k": 60.0,
            "dense_weight": 1.0,
            "bm25_weight": 1.0,
            "truncated_before_reranker": False,
        },
        "execution_boundary": {
            "reranker_inference_performed": False,
            "judge_execution_performed": False,
            "human_relevance_labels_used": False,
            "release_metrics_calculated": False,
            "release_decision_produced": False,
        },
        "outputs": {
            "v4h_dense_top50.csv": sha256(dense_file),
            "v4h_bm25_top50.csv": sha256(bm25_file),
            "v4h_hybrid_union.csv": sha256(union_file),
        },
    }
    metadata_file.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    lock = {
        "status": "V4H_RETRIEVAL_FROZEN_BEFORE_RERANKING_OR_JUDGING",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": git_sha(),
        "query_set": "V4H01-V4H24",
        "query_count": 24,
        "baseline": "v3b",
        "candidate": "v4b",
        "v4h_authoring_lock_sha256": EXPECTED["v4h_authoring_lock"],
        "v4b_development_selection_lock_sha256": EXPECTED["v4b_selection_lock"],
        "v4b_design_spec_sha256": EXPECTED["v4b_design_spec"],
        "v4b_design_lock_sha256": EXPECTED["v4b_design_lock"],
        "retrieval_contract_sha256": EXPECTED_CONTRACT_SHA256,
        "runtime_catalog_sha256": EXPECTED["catalog"],
        "tagged_description_sha256": EXPECTED["tagged_description"],
        "dense_candidate_generator_git_blob": EXPECTED_GRADIO_BLOB,
        "hybrid_retrieval_implementation_git_blob": EXPECTED_HYBRID_BLOB,
        "dense_top50_sha256": sha256(dense_file),
        "bm25_top50_sha256": sha256(bm25_file),
        "hybrid_union_sha256": sha256(union_file),
        "retrieval_metadata_sha256": sha256(metadata_file),
        "dense_candidate_depth": 50,
        "bm25_candidate_depth": 50,
        "hybrid_rrf_k": 60.0,
        "hybrid_union_rows": len(union_all),
        "hybrid_union_truncated_before_reranker": False,
        "dense_repeat_query_order_identical_for_all_queries": True,
        "reranker_inference_performed": False,
        "judge_execution_performed": False,
        "human_relevance_labels_used": False,
        "release_metrics_calculated": False,
        "release_decision_produced": False,
        "next_stage": "RERANK_FROZEN_DENSE50_FOR_V3B_AND_FROZEN_FULL_HYBRID_UNION_FOR_V4B",
    }
    lock_file.write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")

    print()
    print("PASS  exact 24x50 dense retrieval frozen for V3B baseline")
    print("PASS  BM25-50 retrieval frozen for V4B lexical branch")
    print("PASS  full dense50+BM25-50 V4B union frozen without pre-reranker truncation")
    print("PASS  dense repeated-query rank order identical for all 24 V4H queries")
    print()
    print("V4H FINAL HOLDOUT RETRIEVAL: PASS")
    print(f"Run directory: {run_dir}")
    print(f"Dense top50 SHA-256: {sha256(dense_file)}")
    print(f"BM25 top50 SHA-256: {sha256(bm25_file)}")
    print(f"Hybrid union SHA-256: {sha256(union_file)}")
    print(f"Retrieval lock SHA-256: {sha256(lock_file)}")
    print("Reranker inference performed: NO")
    print("Judge calls performed: 0")
    print("Release metrics calculated: NO")
    print("Next: rerank the frozen dense50 for V3B and the frozen full hybrid union for V4B.")

if __name__ == "__main__":
    main()
