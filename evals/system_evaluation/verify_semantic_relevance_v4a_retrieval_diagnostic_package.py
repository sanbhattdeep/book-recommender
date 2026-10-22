from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[2]
BUILDER = HERE / "prepare_semantic_relevance_v4a_retrieval_diagnostic.py"
EXPECTED_BUILDER_SHA256 = "ad8345a7fc232b7dfc14f3197ad7d64002b6cb0a43b356aaec6692a088ac7cb1"

sys.path.insert(0, str(HERE))
import prepare_semantic_relevance_v4a_retrieval_diagnostic as prep  # noqa: E402


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def main() -> None:
    print("Semantic relevance v4a - 17-query retrieval diagnostic package preflight")
    print("-----------------------------------------------------------------------")
    print("Diagnostic input creation: DISABLED")
    print("Judge execution:           DISABLED")
    print("Model execution:           DISABLED")
    print("Development selection:     FROZEN / NOT REOPENED")
    print("Release analysis:          DISABLED")
    print()

    require(BUILDER.exists(), f"Missing builder: {BUILDER}")
    actual = sha256(BUILDER)
    require(actual == EXPECTED_BUILDER_SHA256, f"Builder hash mismatch: expected={EXPECTED_BUILDER_SHA256} actual={actual}")
    print("PASS  retrieval-diagnostic input builder SHA-256 verified")

    target_qids = prep.verify_frozen_inputs()

    union = pd.read_csv(prep.HYBRID_UNION, dtype=str, keep_default_na=False, encoding="utf-8")
    require(len(union) == 2211, f"Expected 2211 frozen hybrid-union rows, got {len(union)}")
    require(union["query_id"].nunique() == 24, "Frozen hybrid union is not H01-H24")
    diag = union[union["query_id"].astype(str).isin(target_qids)].copy()
    require(diag["query_id"].nunique() == 17, "Target diagnostic union is not 17 queries")
    require(set(diag["query_id"].astype(str)) == set(target_qids), "Diagnostic query IDs do not match frozen target set")
    dense = diag["dense_rank"].astype(str).str.strip().ne("")
    counts = diag.loc[dense].groupby("query_id").size()
    require((counts == 50).all(), f"Target queries are not exact dense50: {counts.to_dict()}")
    require(diag.loc[~dense, "candidate_source"].eq("bm25_only").all(), "A non-dense candidate is not bm25_only")
    print(f"PASS  frozen diagnostic structure verified: 17 queries / {len(diag)} hybrid-union pairs / 850 dense-top50 pairs")

    for path in [prep.OUT_INVENTORY, prep.OUT_NOVEL, prep.OUT_METADATA, prep.OUT_LOCK]:
        require(not path.exists(), f"Diagnostic output already exists; refusing preflight over existing artifact: {path}")

    print()
    print("V4A 17-QUERY RETRIEVAL DIAGNOSTIC PACKAGE PREFLIGHT: PASS")
    print("Diagnostic input created:  NO")
    print("Judge calls:               0")
    print("Model calls:               0")
    print("Human labels used:         NO")
    print("Development selection:     NOT REOPENED")
    print("Release analysis:          NO")
    print("Next: deterministically build and freeze the diagnostic score inventory and novel BM25-only judging input.")


if __name__ == "__main__":
    main()
