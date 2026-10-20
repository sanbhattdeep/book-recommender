from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from semantic_relevance_v3_reranker import (
    RerankerRuntimeConfig,
    candidate_text,
    load_cross_encoder,
    reranker_order_signature,
    score_query_candidate_pairs,
)
from semantic_relevance_v4a_hybrid import rank_v4a

REPO_ROOT = Path(__file__).resolve().parents[2]
EVALS = REPO_ROOT / "evals"
BASE = EVALS / "system_evaluation"
RUN_DIR = EVALS / "runs/semantic_relevance_v4a_development/20261019T104415Z_retrieval"

SPEC_JSON = BASE / "semantic_relevance_v4a_hybrid_retrieval_experiment.v0.1.0.json"
SPEC_MD = BASE / "semantic_relevance_v4a_hybrid_retrieval_experiment.v0.1.0.md"
SPEC_LOCK = BASE / "semantic_relevance_v4a_hybrid_retrieval_experiment.v0.1.0.lock.json"
IMPLEMENTATION = BASE / "semantic_relevance_v4a_hybrid.py"
IMPLEMENTATION_LOCK = BASE / "semantic_relevance_v4a_implementation.lock.json"
SELECTED_V3_LOCK = BASE / "semantic_relevance_v3_selected_candidate.lock.json"
V3_RERANKER = BASE / "semantic_relevance_v3_reranker.py"

RETRIEVAL_LOCK = RUN_DIR / "v4a_dev_retrieval.lock.json"
HYBRID_UNION = RUN_DIR / "v4a_dev_hybrid_union.csv"
BM25_TOP50 = RUN_DIR / "v4a_dev_bm25_top50.csv"

EXPECTED_SPEC_JSON_SHA256 = "6aba9c8354f201ad41044dd82d4476612f81ebd27a5ce4f8bf4f327a8b3281b9"
EXPECTED_SPEC_MD_SHA256 = "5af31f1b986187b668b0d486d4209da198e88eb6ea03e3b12818ea8c04e33b90"
EXPECTED_SPEC_LOCK_SHA256 = "c9fee869cda6ccfd26c1ad0b2624cbcf693b9c896366b6baaa5c0b7b98371db2"
EXPECTED_IMPLEMENTATION_SHA256 = "6f31504e29078fffbab86ddb86ef26b02aaa22082d3bfdb6944792bf689f6db7"
EXPECTED_IMPLEMENTATION_LOCK_SHA256 = "b67d305ca5c5da7df82997f43129ad596c7c79eb085571bb4a83b8638f016775"
EXPECTED_SELECTED_V3_LOCK_SHA256 = "87484c845fe332882a8ee97eba3824cdc69121f1e10b6614d4532d58fb82e053"
EXPECTED_V3_RERANKER_SOURCE_SHA256 = "5e7a7d6b35c0674f32644e758218830d8a340a61588e2ff0b3f20582fdbb0f60"
EXPECTED_RETRIEVAL_LOCK_SHA256 = "7587aa140c1cb26f8ac5addaf6873f3bc7ec05b6ec34775a6cbfe98d7ffa40d1"
EXPECTED_HYBRID_UNION_SHA256 = "71be13c297f6449e6a53c54c7b9db590f6f033d69a11f16b21377797d9390155"
EXPECTED_BM25_TOP50_SHA256 = "6e265a58c63680c51120f9f20a921303175a98f94a6f8861e876538f6cb11ce0"
EXPECTED_CATALOG_SHA256 = "22244d277a643e56bf23a744d0d45eb2247db4a05792ec69ffd942c6d5fdbf6e"
EXPECTED_MODEL_ID = "BAAI/bge-reranker-base"
EXPECTED_MODEL_REVISION = "2cfc18c9415c912f9d8155881c133215df768a70"
EXPECTED_QUERY_COUNT = 24
EXPECTED_UNION_ROWS = 2211
EXPECTED_SHARED_ROWS = 189
EXPECTED_DENSE_ONLY_ROWS = 1011
EXPECTED_BM25_ONLY_ROWS = 1011
EXPECTED_DISPLAY_ROWS = 240


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_lf_text_sha256(path: Path) -> str:
    """Hash text source after normalizing CRLF/CR line endings to LF.

    The frozen v3 reranker source hash was recorded from LF bytes in Git.
    Windows may check out the same source as CRLF. This normalization is used
    only for that frozen text-source verification boundary; artifact/data hashes
    remain byte-exact.
    """
    data = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(data).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8-sig"))


def validate_inputs() -> pd.DataFrame:
    for label, path, expected in [
        ("v4a JSON spec", SPEC_JSON, EXPECTED_SPEC_JSON_SHA256),
        ("v4a Markdown spec", SPEC_MD, EXPECTED_SPEC_MD_SHA256),
        ("v4a spec lock", SPEC_LOCK, EXPECTED_SPEC_LOCK_SHA256),
        ("v4a implementation", IMPLEMENTATION, EXPECTED_IMPLEMENTATION_SHA256),
        ("v4a implementation lock", IMPLEMENTATION_LOCK, EXPECTED_IMPLEMENTATION_LOCK_SHA256),
        ("selected v3 candidate lock", SELECTED_V3_LOCK, EXPECTED_SELECTED_V3_LOCK_SHA256),
        ("frozen v3 reranker source", V3_RERANKER, EXPECTED_V3_RERANKER_SOURCE_SHA256),
        ("v4a retrieval lock", RETRIEVAL_LOCK, EXPECTED_RETRIEVAL_LOCK_SHA256),
        ("v4a frozen hybrid union", HYBRID_UNION, EXPECTED_HYBRID_UNION_SHA256),
        ("v4a frozen BM25 top50", BM25_TOP50, EXPECTED_BM25_TOP50_SHA256),
    ]:
        require(path.exists(), f"Missing {label}: {path}")
        if path == V3_RERANKER:
            actual = canonical_lf_text_sha256(path)
            hash_mode = "canonical-LF text SHA-256"
        else:
            actual = sha256(path)
            hash_mode = "SHA-256"
        require(actual == expected, f"{label} hash mismatch: expected={expected} actual={actual}")
        print(f"PASS  {label} {hash_mode} verified")

    retrieval_lock = load_json(RETRIEVAL_LOCK)
    require(
        retrieval_lock.get("status") == "V4A_DEVELOPMENT_RETRIEVAL_FROZEN_BEFORE_RERANKING_OR_JUDGING",
        "v4a retrieval lock status is invalid.",
    )
    require(retrieval_lock.get("runtime_catalog_sha256") == EXPECTED_CATALOG_SHA256, "Catalog hash changed.")
    require(retrieval_lock.get("hybrid_union_sha256") == EXPECTED_HYBRID_UNION_SHA256, "Retrieval lock union hash mismatch.")
    require(retrieval_lock.get("bm25_top50_sha256") == EXPECTED_BM25_TOP50_SHA256, "Retrieval lock BM25 hash mismatch.")
    require(int(retrieval_lock.get("hybrid_union_rows", -1)) == EXPECTED_UNION_ROWS, "Hybrid union row count changed.")
    require(retrieval_lock.get("hybrid_union_truncated_before_reranker") is False, "Hybrid union was truncated.")
    require(retrieval_lock.get("dense_retrieval_rerun") is False, "Dense retrieval unexpectedly reran.")
    require(retrieval_lock.get("reranker_inference_performed") is False, "Reranker already ran according to retrieval lock.")
    require(retrieval_lock.get("judge_execution_performed") is False, "Judge already ran according to retrieval lock.")
    require(retrieval_lock.get("next_stage") == "PREPARE_V4A_DEVELOPMENT_RERANKER_EXECUTION_ON_FROZEN_HYBRID_UNION", "Retrieval lock next-stage contract changed.")

    selected = load_json(SELECTED_V3_LOCK)
    require(selected.get("selected_variant") == "v3b_rrf_hybrid", "Unexpected frozen v3 selected variant.")
    require(selected.get("model_id") == EXPECTED_MODEL_ID, "Frozen reranker model ID changed.")
    require(selected.get("model_revision") == EXPECTED_MODEL_REVISION, "Frozen reranker revision changed.")
    source_hashes = selected.get("source_hashes", {})
    require(source_hashes.get("semantic_relevance_v3_reranker.py") == EXPECTED_V3_RERANKER_SOURCE_SHA256, "Selected lock reranker source hash changed.")

    frame = pd.read_csv(HYBRID_UNION, dtype=str, keep_default_na=False, encoding="utf-8")
    required = {
        "query_id", "query", "stratum", "isbn13", "dense_rank", "bm25_rank",
        "bm25_score", "candidate_source", "hybrid_retrieval_rrf_score",
        "hybrid_retrieval_rank", "title", "description", "authors",
    }
    missing = required - set(frame.columns)
    require(not missing, f"Hybrid union missing columns: {sorted(missing)}")
    require(len(frame) == EXPECTED_UNION_ROWS, f"Expected {EXPECTED_UNION_ROWS} union rows, got {len(frame)}.")
    require(frame["query_id"].nunique() == EXPECTED_QUERY_COUNT, "Hybrid union does not contain 24 queries.")
    require(not frame.duplicated(["query_id", "isbn13"]).any(), "Duplicate query-book pairs in hybrid union.")

    frame["hybrid_retrieval_rank"] = pd.to_numeric(frame["hybrid_retrieval_rank"], errors="raise").astype(int)
    counts = frame.groupby("query_id").size()
    require(int(counts.min()) == 85 and int(counts.max()) == 98, f"Unexpected per-query union range: min={counts.min()} max={counts.max()}")
    for qid, group in frame.groupby("query_id", sort=True):
        require(
            sorted(group["hybrid_retrieval_rank"].tolist()) == list(range(1, len(group) + 1)),
            f"{qid}: hybrid retrieval ranks are not contiguous.",
        )

    source_counts = frame["candidate_source"].value_counts().to_dict()
    require(int(source_counts.get("both", 0)) == EXPECTED_SHARED_ROWS, "Shared-source row count changed.")
    require(int(source_counts.get("dense_only", 0)) == EXPECTED_DENSE_ONLY_ROWS, "Dense-only row count changed.")
    require(int(source_counts.get("bm25_only", 0)) == EXPECTED_BM25_ONLY_ROWS, "BM25-only row count changed.")

    return frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    args = parser.parse_args()

    print("Semantic relevance v4a - development reranker execution")
    print("------------------------------------------------------")
    print("Input: frozen dense50 + BM25-50 full hybrid union")
    print("Development role: H01-H24 only; no release claim")
    print("Retrieval rerun: NO")
    print("Reranker inference: ENABLED")
    print("Two inference passes: REQUIRED")
    print("Judge labels loaded: NO")
    print("Judge calls: 0")
    print("Release decision: DISABLED")
    print()

    frame = validate_inputs()
    print("PASS  frozen retrieval/reranker boundaries verified")
    print(f"PASS  frozen hybrid union structure verified: {len(frame)} rows")

    runtime = RerankerRuntimeConfig(
        model_id=EXPECTED_MODEL_ID,
        model_revision=EXPECTED_MODEL_REVISION,
        max_length_tokens=512,
        truncation="only_second",
        batch_size=int(args.batch_size),
        device=str(args.device),
    )
    tokenizer, model, device = load_cross_encoder(runtime)
    print(f"Model:    {runtime.model_id}")
    print(f"Revision: {runtime.model_revision}")
    print(f"Device:   {device}")
    print(f"Batch:    {runtime.batch_size}")

    working = frame.sort_values(["query_id", "hybrid_retrieval_rank"], kind="stable").copy()
    queries = working["query"].astype(str).tolist()
    texts = [candidate_text(t, d) for t, d in zip(working["title"], working["description"])]

    print()
    print("Scoring hybrid union pass 1/2...")
    scores_1 = score_query_candidate_pairs(
        queries=queries,
        candidates=texts,
        tokenizer=tokenizer,
        model=model,
        device=device,
        max_length_tokens=runtime.max_length_tokens,
        batch_size=runtime.batch_size,
    )
    print("Scoring hybrid union pass 2/2...")
    scores_2 = score_query_candidate_pairs(
        queries=queries,
        candidates=texts,
        tokenizer=tokenizer,
        model=model,
        device=device,
        max_length_tokens=runtime.max_length_tokens,
        batch_size=runtime.batch_size,
    )

    require(np.isfinite(scores_1).all(), "Pass 1 contains non-finite scores.")
    require(np.isfinite(scores_2).all(), "Pass 2 contains non-finite scores.")

    pass1 = working.copy()
    pass1["reranker_score"] = scores_1
    pass2 = working.copy()
    pass2["reranker_score"] = scores_2

    # Reuse the frozen v3 signature helper only to compare reranker ordering. It
    # expects vector_rank, so provide hybrid_retrieval_rank under that temporary
    # name. This does not calculate v3 ranking or alter v4a final ranking.
    sig1_input = pass1.rename(columns={"hybrid_retrieval_rank": "vector_rank"})
    sig2_input = pass2.rename(columns={"hybrid_retrieval_rank": "vector_rank"})
    signature_1 = reranker_order_signature(sig1_input)
    signature_2 = reranker_order_signature(sig2_input)
    require(signature_1 == signature_2, "Per-query reranker order changed between repeated inference passes.")

    max_abs_delta = float(np.max(np.abs(scores_1 - scores_2)))
    require(math.isfinite(max_abs_delta), "Repeated inference delta is non-finite.")
    print("PASS  repeated inference produced identical per-query reranker order")
    print(f"INFO  max absolute raw-score delta between passes: {max_abs_delta:.12g}")

    ranked = rank_v4a(pass1, rrf_k=60.0)
    require(len(ranked) == EXPECTED_UNION_ROWS, "Full v4a ranking row count changed.")
    require(not ranked.duplicated(["query_id", "isbn13"]).any(), "Duplicate pairs after v4a ranking.")

    for qid, group in ranked.groupby("query_id", sort=True):
        require(sorted(group["v4a_rank"].astype(int).tolist()) == list(range(1, len(group) + 1)), f"{qid}: v4a ranks not contiguous.")

    top10 = ranked[ranked["v4a_rank"].astype(int) <= 10].copy()
    top10 = top10.sort_values(["query_id", "v4a_rank"], kind="stable")
    top_counts = top10.groupby("query_id").size()
    require(len(top_counts) == 24 and (top_counts == 10).all(), "v4a displayed output is not 24x10.")
    require(len(top10) == EXPECTED_DISPLAY_ROWS, "Expected 240 displayed v4a rows.")

    scores_file = RUN_DIR / "v4a_dev_reranker_scores.csv"
    recommendations_file = RUN_DIR / "v4a_dev_recommendations.csv"
    metadata_file = RUN_DIR / "v4a_dev_reranker_metadata.json"
    lock_file = RUN_DIR / "v4a_dev_reranker.lock.json"
    if any(p.exists() for p in [scores_file, recommendations_file, metadata_file, lock_file]):
        raise FileExistsError("One or more v4a reranker artifacts already exist; refusing to overwrite.")

    ranked.sort_values(["query_id", "hybrid_retrieval_rank"], kind="stable").to_csv(
        scores_file, index=False, encoding="utf-8"
    )
    top10_out = top10.copy()
    top10_out["rank"] = top10_out["v4a_rank"].astype(int)
    top10_out.to_csv(recommendations_file, index=False, encoding="utf-8")

    source_top10 = {str(k): int(v) for k, v in top10["candidate_source"].value_counts().to_dict().items()}
    metadata = {
        "status": "V4A_DEVELOPMENT_RERANKER_COMPLETE",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "development_role": "H01-H24 v4 development only; no release claim allowed",
        "retrieval_run_name": RUN_DIR.name,
        "retrieval_lock_sha256": EXPECTED_RETRIEVAL_LOCK_SHA256,
        "hybrid_union_sha256": EXPECTED_HYBRID_UNION_SHA256,
        "v3_selected_candidate_lock_sha256": EXPECTED_SELECTED_V3_LOCK_SHA256,
        "v3_reranker_source_sha256": EXPECTED_V3_RERANKER_SOURCE_SHA256,
        "v4a_implementation_sha256": EXPECTED_IMPLEMENTATION_SHA256,
        "model_id": EXPECTED_MODEL_ID,
        "model_revision": EXPECTED_MODEL_REVISION,
        "candidate_rows": EXPECTED_UNION_ROWS,
        "display_rows": EXPECTED_DISPLAY_ROWS,
        "display_depth": 10,
        "final_rrf_k": 60,
        "hybrid_retrieval_weight": 1.0,
        "reranker_weight": 1.0,
        "top10_candidate_source_counts": source_top10,
        "repeated_inference_rank_order_identical": True,
        "max_absolute_score_delta_between_passes": max_abs_delta,
        "device": device,
        "batch_size": runtime.batch_size,
        "execution_boundary": {
            "retrieval_rerun": False,
            "reranker_inference_executed": True,
            "judge_labels_loaded": False,
            "judge_calls_performed": 0,
            "human_relevance_labels_used": False,
            "release_metrics_calculated": False,
            "release_decision_produced": False,
        },
        "outputs": {
            "v4a_dev_reranker_scores.csv": sha256(scores_file),
            "v4a_dev_recommendations.csv": sha256(recommendations_file),
        },
    }
    metadata_file.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    lock = {
        "status": "V4A_DEVELOPMENT_RERANKING_FROZEN_BEFORE_JUDGING",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_id": "semantic-relevance-hybrid-retrieval-v4a",
        "development_role": "H01-H24 v4 development only",
        "retrieval_lock_sha256": EXPECTED_RETRIEVAL_LOCK_SHA256,
        "hybrid_union_sha256": EXPECTED_HYBRID_UNION_SHA256,
        "model_id": EXPECTED_MODEL_ID,
        "model_revision": EXPECTED_MODEL_REVISION,
        "v3_reranker_source_sha256": EXPECTED_V3_RERANKER_SOURCE_SHA256,
        "v4a_implementation_sha256": EXPECTED_IMPLEMENTATION_SHA256,
        "v4a_dev_reranker_scores_sha256": sha256(scores_file),
        "v4a_dev_recommendations_sha256": sha256(recommendations_file),
        "v4a_dev_reranker_metadata_sha256": sha256(metadata_file),
        "candidate_rows": EXPECTED_UNION_ROWS,
        "recommendation_rows": EXPECTED_DISPLAY_ROWS,
        "query_count": EXPECTED_QUERY_COUNT,
        "recommendations_per_query": 10,
        "final_rrf_k": 60,
        "hybrid_retrieval_weight": 1.0,
        "reranker_weight": 1.0,
        "repeated_inference_rank_order_identical": True,
        "retrieval_rerun": False,
        "judge_execution_performed": False,
        "human_relevance_labels_used": False,
        "release_metrics_calculated": False,
        "release_decision_produced": False,
        "next_stage": "FREEZE_V4A_DEVELOPMENT_DISPLAY_UNION_BEFORE_JUDGING",
    }
    lock_file.write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")

    print()
    print("PASS  full hybrid union reranked without candidate truncation")
    print("PASS  v4a displayed recommendations are exactly 24x10")
    print("PASS  reranker artifacts frozen before judging")
    print()
    print("V4A DEVELOPMENT RERANKER: PASS")
    print(f"Run directory: {RUN_DIR}")
    print(f"Reranker scores SHA-256: {sha256(scores_file)}")
    print(f"v4a recommendations SHA-256: {sha256(recommendations_file)}")
    print(f"Reranker lock SHA-256: {sha256(lock_file)}")
    print(f"Top10 source counts: {source_top10}")
    print("Retrieval rerun: NO")
    print("Judge calls performed: 0")
    print("Release metrics calculated: NO")
    print("Release decision produced: NO")
    print("Next: freeze unique displayed v3b/v4a development union before judging new pairs.")


if __name__ == "__main__":
    main()
