from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
SYS = REPO_ROOT / "evals/system_evaluation"
FACETS = REPO_ROOT / "evals/facets/semantic_relevance"
RUN = REPO_ROOT / "evals/runs/semantic_relevance_v4a_development/20261019T104415Z_retrieval"
RUN_ROOT = REPO_ROOT / "evals/runs/semantic_relevance_v4_holdout"

V4B_SPEC = SYS / "semantic_relevance_v4b_reranker_only_hybrid_experiment.v0.1.0.json"
V4B_DESIGN_LOCK = SYS / "semantic_relevance_v4b_reranker_only_hybrid_experiment.v0.1.0.lock.json"
V4B_SELECTION_LOCK = RUN / "v4b_dev_comparison_selection.lock.json"
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


def main() -> None:
    print("Semantic relevance V4H - final holdout retrieval preflight")
    print("--------------------------------------------------------")
    print("Dense/BM25 retrieval: DISABLED")
    print("Hybrid-union generation: DISABLED")
    print("Reranker inference: DISABLED")
    print("Judge execution: DISABLED")
    print("Human relevance labels: NOT USED")
    print("Release metrics/decision: NOT CALCULATED")
    print()

    queries = validate_query_and_freeze_boundary()
    catalog, tagged_lines = validate_runtime_data()

    existing = sorted(RUN_ROOT.glob("*_retrieval")) if RUN_ROOT.exists() else []
    require(not existing, "A V4H retrieval run directory already exists; review it before any new retrieval: " + ", ".join(str(x) for x in existing))

    contract = load_json(RETRIEVAL_CONTRACT)
    require(contract.get("status") == "FROZEN_BY_PACKAGE_BEFORE_V4H_RETRIEVAL_EXECUTION", "Unexpected retrieval-contract status")
    require(contract.get("query_set") == "V4H01-V4H24", "Retrieval contract query set changed")
    require(contract.get("baseline") == "v3b" and contract.get("candidate") == "v4b", "Retrieval contract system identities changed")
    require(contract.get("dense_branch", {}).get("candidate_depth") == 50, "Dense depth changed")
    require(contract.get("bm25_branch", {}).get("candidate_depth") == 50, "BM25 depth changed")
    require(contract.get("bm25_branch", {}).get("k1") == 1.5, "BM25 k1 changed")
    require(contract.get("bm25_branch", {}).get("b") == 0.75, "BM25 b changed")
    require(contract.get("hybrid_union", {}).get("rrf_k") == 60.0, "Hybrid RRF k changed")
    require(contract.get("hybrid_union", {}).get("truncate_before_reranker") is False, "Hybrid union truncation changed")

    print()
    print("V4H FINAL HOLDOUT RETRIEVAL PREFLIGHT: PASS")
    print("Queries: V4H01-V4H24 (24)")
    print("Runtime catalog rows: 5197")
    print("Tagged-description rows: 5197")
    print(f"Runtime catalog SHA-256: {sha256(CATALOG_FILE)}")
    print(f"Tagged-description SHA-256: {sha256(TAGGED_FILE)}")
    print(f"Retrieval contract SHA-256: {sha256(RETRIEVAL_CONTRACT)}")
    print("Dense retrieval performed: NO")
    print("BM25 retrieval performed: NO")
    print("Reranker inference performed: NO")
    print("Judge calls performed: 0")
    print("Release metrics calculated: NO")
    print("Next: execute and freeze dense50 + BM25-50 full-union retrieval for V4H01-V4H24.")

if __name__ == "__main__":
    main()
