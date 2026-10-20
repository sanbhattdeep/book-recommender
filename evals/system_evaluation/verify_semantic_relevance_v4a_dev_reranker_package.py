from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
EVALS = REPO_ROOT / "evals"
BASE = EVALS / "system_evaluation"
RUN_DIR = EVALS / "runs/semantic_relevance_v4a_development/20261019T104415Z_retrieval"

EXPECTED = {
    "spec": "6aba9c8354f201ad41044dd82d4476612f81ebd27a5ce4f8bf4f327a8b3281b9",
    "implementation": "6f31504e29078fffbab86ddb86ef26b02aaa22082d3bfdb6944792bf689f6db7",
    "implementation_lock": "b67d305ca5c5da7df82997f43129ad596c7c79eb085571bb4a83b8638f016775",
    "selected_v3_lock": "87484c845fe332882a8ee97eba3824cdc69121f1e10b6614d4532d58fb82e053",
    "v3_reranker": "5e7a7d6b35c0674f32644e758218830d8a340a61588e2ff0b3f20582fdbb0f60",
    "retrieval_lock": "7587aa140c1cb26f8ac5addaf6873f3bc7ec05b6ec34775a6cbfe98d7ffa40d1",
    "hybrid_union": "71be13c297f6449e6a53c54c7b9db590f6f033d69a11f16b21377797d9390155",
    "runner": "fae1d2b519713667d47714b90ec4eaff483655166fb2d97cba679f988f5559da",
}

FILES = {
    "spec": BASE / "semantic_relevance_v4a_hybrid_retrieval_experiment.v0.1.0.json",
    "implementation": BASE / "semantic_relevance_v4a_hybrid.py",
    "implementation_lock": BASE / "semantic_relevance_v4a_implementation.lock.json",
    "selected_v3_lock": BASE / "semantic_relevance_v3_selected_candidate.lock.json",
    "v3_reranker": BASE / "semantic_relevance_v3_reranker.py",
    "retrieval_lock": RUN_DIR / "v4a_dev_retrieval.lock.json",
    "hybrid_union": RUN_DIR / "v4a_dev_hybrid_union.csv",
    "runner": BASE / "run_semantic_relevance_v4a_dev_reranker.py",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_lf_text_sha256(path: Path) -> str:
    """Hash text source after normalizing CRLF/CR line endings to LF."""
    data = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(data).hexdigest()


def req(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def main() -> None:
    print("Semantic relevance v4a - development reranker package preflight")
    print("--------------------------------------------------------------")
    print("Retrieval rerun:      DISABLED")
    print("Reranker inference:   DISABLED")
    print("Judge execution:      DISABLED")
    print("Release analysis:     DISABLED")
    print()

    for name, path in FILES.items():
        req(path.exists(), f"Missing {name}: {path}")
        if name == "v3_reranker":
            actual = canonical_lf_text_sha256(path)
            hash_mode = "canonical-LF text SHA-256"
        else:
            actual = sha256(path)
            hash_mode = "SHA-256"
        req(actual == EXPECTED[name], f"{name} hash mismatch: expected={EXPECTED[name]} actual={actual}")
        print(f"PASS  {name} {hash_mode} verified")

    lock = json.loads(FILES["retrieval_lock"].read_text(encoding="utf-8-sig"))
    req(lock.get("status") == "V4A_DEVELOPMENT_RETRIEVAL_FROZEN_BEFORE_RERANKING_OR_JUDGING", "Retrieval lock status changed.")
    req(lock.get("hybrid_union_sha256") == EXPECTED["hybrid_union"], "Retrieval lock union hash mismatch.")
    req(int(lock.get("hybrid_union_rows", -1)) == 2211, "Hybrid union row count changed.")
    req(lock.get("hybrid_union_truncated_before_reranker") is False, "Union truncation contract changed.")
    req(lock.get("reranker_inference_performed") is False, "Retrieval lock says reranker already ran.")
    req(lock.get("judge_execution_performed") is False, "Retrieval lock says judge already ran.")

    frame = pd.read_csv(FILES["hybrid_union"], dtype=str, keep_default_na=False, encoding="utf-8")
    req(len(frame) == 2211, f"Expected 2211 union rows, got {len(frame)}")
    req(frame["query_id"].nunique() == 24, "Expected 24 queries.")
    req(not frame.duplicated(["query_id", "isbn13"]).any(), "Duplicate hybrid pairs.")
    counts = frame.groupby("query_id").size()
    req(int(counts.min()) == 85 and int(counts.max()) == 98, "Per-query union sizes changed.")
    source_counts = frame["candidate_source"].value_counts().to_dict()
    req(int(source_counts.get("both", 0)) == 189, "Shared source count changed.")
    req(int(source_counts.get("dense_only", 0)) == 1011, "Dense-only count changed.")
    req(int(source_counts.get("bm25_only", 0)) == 1011, "BM25-only count changed.")

    print("PASS  frozen hybrid-union structure verified: 2211 rows / 24 queries")
    print("PASS  source composition verified: both=189 dense_only=1011 bm25_only=1011")
    print()
    print("V4A DEVELOPMENT RERANKER PACKAGE PREFLIGHT: PASS")
    print("Retrieval rerun:      NO")
    print("Reranker inference:   NO")
    print("Judge calls:           0")
    print("Release analysis:      NO")
    print("Next: execute frozen BGE reranker twice across the full hybrid union.")


if __name__ == "__main__":
    main()
