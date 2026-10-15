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
    rank_v3b,
    reranker_order_signature,
    score_query_candidate_pairs,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
EVALS = REPO_ROOT / "evals"

SPEC_FILE = EVALS / "system_evaluation/semantic_relevance_v3_reranker_experiment.v0.1.0.json"
SPEC_LOCK_FILE = EVALS / "system_evaluation/semantic_relevance_v3_reranker_experiment.v0.1.0.lock.json"
SELECTED_LOCK_FILE = EVALS / "system_evaluation/semantic_relevance_v3_selected_candidate.lock.json"
HOLDOUT_INPUT_LOCK_FILE = EVALS / "system_evaluation/semantic_relevance_v3_holdout_input_lock.v0.1.0.json"

EXPECTED_SPEC_SHA256 = "5bb1f07f68db828b8db1a8d6d1c62831a5309651b7c62c91d04184bf41bb9261"
EXPECTED_SELECTED_VARIANT = "v3b_rrf_hybrid"
EXPECTED_MODEL_ID = "BAAI/bge-reranker-base"
EXPECTED_MODEL_REVISION = "2cfc18c9415c912f9d8155881c133215df768a70"

EXPECTED_RETRIEVAL_RUN_NAME = "20261007T124150Z_retrieval"
EXPECTED_TOP50_SHA256 = "5587b0400b1fcabb86bbbfcbc6deee741e418cc277e2b9a18ab0efa29e8d4acf"
EXPECTED_V2_BASELINE_SHA256 = "4e8969ee79341b04ada9f2d251a0e838ab790cc238ba51aa4af403ab4a1c2aaa"
EXPECTED_RETRIEVAL_LOCK_SHA256 = "4c0553065f94314ded8cd4b5d8edd90cef3a812c910c69c0a5907950a2b503af"

EXPECTED_QUERY_COUNT = 24
EXPECTED_CANDIDATES_PER_QUERY = 50
EXPECTED_ROWS = 1200
EXPECTED_OUTPUT_PER_QUERY = 10
EXPECTED_OUTPUT_ROWS = 240


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8-sig"))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate_frozen_inputs(retrieval_run: Path) -> tuple[dict[str, Any], dict[str, Any], pd.DataFrame]:
    if not retrieval_run.exists():
        raise FileNotFoundError(retrieval_run)

    require(
        retrieval_run.name == EXPECTED_RETRIEVAL_RUN_NAME,
        f"Unexpected retrieval run: {retrieval_run.name}",
    )

    top50_file = retrieval_run / "holdout_top50_candidates.csv"
    v2_file = retrieval_run / "v2_holdout_recommendations.csv"
    retrieval_lock_file = retrieval_run / "holdout_retrieval_lock.json"

    require(sha256(SPEC_FILE) == EXPECTED_SPEC_SHA256, "Reviewed v3 spec hash changed.")

    spec_lock = load_json(SPEC_LOCK_FILE)
    require(spec_lock.get("status") == "FROZEN_BEFORE_V3_DEV_EXECUTION", "Spec lock is not frozen.")
    require(spec_lock.get("spec_sha256") == EXPECTED_SPEC_SHA256, "Spec lock hash mismatch.")

    selected = load_json(SELECTED_LOCK_FILE)
    require(
        selected.get("status") == "V3_SELECTED_CANDIDATE_FROZEN_BEFORE_UNSEEN_HOLDOUT_AUTHORING",
        "Selected candidate lock status is invalid.",
    )
    require(selected.get("selected_variant") == EXPECTED_SELECTED_VARIANT, "Unexpected selected variant.")
    require(selected.get("model_id") == EXPECTED_MODEL_ID, "Selected model ID changed.")
    require(selected.get("model_revision") == EXPECTED_MODEL_REVISION, "Selected model revision changed.")

    policy = selected.get("ranking_policy", {})
    require(policy.get("method") == "reciprocal_rank_fusion", "Selected ranking policy changed.")
    require(int(policy.get("candidate_depth", -1)) == 50, "Candidate depth changed.")
    require(int(policy.get("display_depth", -1)) == 10, "Display depth changed.")
    require(float(policy.get("rrf_k", -1)) == 60.0, "RRF k changed.")
    require(float(policy.get("vector_weight", -1)) == 1.0, "Vector weight changed.")
    require(float(policy.get("reranker_weight", -1)) == 1.0, "Reranker weight changed.")

    holdout_input = load_json(HOLDOUT_INPUT_LOCK_FILE)
    require(
        holdout_input.get("status") == "V3_UNSEEN_HOLDOUT_INPUTS_FROZEN_BEFORE_RETRIEVAL",
        "Holdout inputs are not frozen.",
    )
    require(
        holdout_input.get("selected_variant") == EXPECTED_SELECTED_VARIANT,
        "Holdout input lock selected variant mismatch.",
    )

    require(sha256(top50_file) == EXPECTED_TOP50_SHA256, "Frozen holdout top50 changed.")
    require(sha256(v2_file) == EXPECTED_V2_BASELINE_SHA256, "Frozen v2 baseline changed.")
    require(
        sha256(retrieval_lock_file) == EXPECTED_RETRIEVAL_LOCK_SHA256,
        "Frozen holdout retrieval lock changed.",
    )

    retrieval_lock = load_json(retrieval_lock_file)
    require(
        retrieval_lock.get("status") == "V3_HOLDOUT_RETRIEVAL_FROZEN_BEFORE_RERANKING",
        "Retrieval lock status is invalid.",
    )
    require(
        retrieval_lock.get("holdout_top50_candidates_sha256") == EXPECTED_TOP50_SHA256,
        "Retrieval lock disagrees with top50 hash.",
    )
    require(
        retrieval_lock.get("v2_holdout_recommendations_sha256") == EXPECTED_V2_BASELINE_SHA256,
        "Retrieval lock disagrees with v2 baseline hash.",
    )
    require(
        retrieval_lock.get("next_stage") == "selected_v3b_reranking_on_frozen_holdout_top50",
        "Retrieval lock next-stage contract changed.",
    )
    require(
        retrieval_lock.get("reranker_inference_performed") is False,
        "Retrieval lock unexpectedly says reranker already ran.",
    )
    require(
        retrieval_lock.get("judge_execution_performed") is False,
        "Retrieval lock unexpectedly says judge already ran.",
    )

    frame = pd.read_csv(
        top50_file,
        dtype=str,
        keep_default_na=False,
        encoding="utf-8",
    )
    required = {
        "query_id",
        "query",
        "stratum",
        "vector_rank",
        "isbn13",
        "title",
        "description",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Frozen holdout top50 missing columns: {sorted(missing)}")

    frame["vector_rank"] = pd.to_numeric(frame["vector_rank"], errors="raise").astype(int)

    require(len(frame) == EXPECTED_ROWS, f"Expected {EXPECTED_ROWS} rows, got {len(frame)}.")
    require(
        not frame.duplicated(["query_id", "isbn13"]).any(),
        "Duplicate query-book pairs in frozen holdout top50.",
    )

    counts = frame.groupby("query_id").size()
    require(
        len(counts) == EXPECTED_QUERY_COUNT
        and (counts == EXPECTED_CANDIDATES_PER_QUERY).all(),
        f"Expected 24x50 candidate structure; got {counts.to_dict()}",
    )

    for qid, group in frame.groupby("query_id", sort=True):
        require(
            sorted(group["vector_rank"].tolist()) == list(range(1, 51)),
            f"{qid}: vector ranks are not exactly 1..50.",
        )

    return selected, retrieval_lock, frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--retrieval-run",
        required=True,
        help="Frozen holdout retrieval run directory.",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    args = parser.parse_args()

    retrieval_run = Path(args.retrieval_run)
    if not retrieval_run.is_absolute():
        retrieval_run = REPO_ROOT / retrieval_run

    print("Semantic relevance v3 - selected v3b unseen holdout reranking")
    print("------------------------------------------------------------")
    print("Input: frozen 1,200-row holdout top50 candidate pool")
    print("Selected variant: v3b_rrf_hybrid")
    print("Judge labels loaded: NO")
    print("Human relevance labels loaded: NO")
    print("Judge calls: 0")
    print("Two inference passes: REQUIRED for rank reproducibility")
    print()

    selected, retrieval_lock, frame = validate_frozen_inputs(retrieval_run)
    print("PASS  frozen spec/selected-candidate/holdout/retrieval locks verified")
    print("PASS  frozen top50 and v2 baseline hashes unchanged")
    print("PASS  24x50 candidate structure verified")

    runtime = RerankerRuntimeConfig(
        model_id=EXPECTED_MODEL_ID,
        model_revision=EXPECTED_MODEL_REVISION,
        max_length_tokens=512,
        truncation="only_second",
        batch_size=int(args.batch_size),
        device=str(args.device),
    )

    print()
    print(f"Model:    {runtime.model_id}")
    print(f"Revision: {runtime.model_revision}")
    print(f"Batch:    {runtime.batch_size}")

    tokenizer, model, device = load_cross_encoder(runtime)
    print(f"Device:   {device}")

    working = frame.sort_values(["query_id", "vector_rank"], kind="stable").copy()

    queries = working["query"].astype(str).tolist()
    texts = [
        candidate_text(title, description)
        for title, description in zip(
            working["title"],
            working["description"],
        )
    ]

    print()
    print("Scoring holdout pass 1/2...")
    scores_1 = score_query_candidate_pairs(
        queries=queries,
        candidates=texts,
        tokenizer=tokenizer,
        model=model,
        device=device,
        max_length_tokens=runtime.max_length_tokens,
        batch_size=runtime.batch_size,
    )

    print("Scoring holdout pass 2/2...")
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

    signature_1 = reranker_order_signature(pass1)
    signature_2 = reranker_order_signature(pass2)

    require(
        signature_1 == signature_2,
        "Per-query reranker order changed between repeated holdout inference passes.",
    )

    max_abs_delta = float(np.max(np.abs(scores_1 - scores_2)))
    require(
        math.isfinite(max_abs_delta),
        "Repeated inference delta is non-finite.",
    )

    print("PASS  repeated inference produced identical per-query reranker order")
    print(f"INFO  max absolute raw-score delta between passes: {max_abs_delta:.12g}")

    ranked = rank_v3b(pass1, rrf_k=60.0)
    require(
        len(ranked) == EXPECTED_ROWS,
        "Full v3b ranking row count changed.",
    )

    out_counts = ranked.groupby("query_id").size()
    require(
        len(out_counts) == EXPECTED_QUERY_COUNT and (out_counts == 50).all(),
        "v3b full ranking is not 24x50.",
    )

    top10 = ranked[ranked["v3b_rank"] <= 10].copy()
    top10 = top10.sort_values(["query_id", "v3b_rank"], kind="stable")

    top10_counts = top10.groupby("query_id").size()
    require(
        len(top10_counts) == EXPECTED_QUERY_COUNT
        and (top10_counts == EXPECTED_OUTPUT_PER_QUERY).all(),
        "v3b holdout output is not exactly 24x10.",
    )
    require(len(top10) == EXPECTED_OUTPUT_ROWS, "Expected exactly 240 v3b recommendations.")
    require(
        not top10.duplicated(["query_id", "isbn13"]).any(),
        "Duplicate query-book pairs in v3b top10.",
    )

    run_dir = retrieval_run

    scores_file = run_dir / "v3_holdout_reranker_scores.csv"
    recommendations_file = run_dir / "v3_holdout_recommendations.csv"
    metadata_file = run_dir / "v3_holdout_reranker_metadata.json"
    lock_file = run_dir / "v3_holdout_reranker_lock.json"

    if any(path.exists() for path in [scores_file, recommendations_file, metadata_file, lock_file]):
        raise FileExistsError(
            "One or more v3 holdout reranker artifacts already exist; "
            "refusing to overwrite an unseen-holdout execution."
        )

    score_cols = [
        "query_id",
        "query",
        "stratum",
        "vector_rank",
        "isbn13",
        "title",
        "description",
        "authors",
        "reranker_score",
        "reranker_rank",
        "rrf_score",
        "v3b_rank",
    ]
    ranked[score_cols].sort_values(
        ["query_id", "vector_rank"],
        kind="stable",
    ).to_csv(
        scores_file,
        index=False,
        encoding="utf-8",
    )

    top10_out = top10.copy()
    top10_out["rank"] = top10_out["v3b_rank"].astype(int)
    top10_out[
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
            "reranker_score",
            "reranker_rank",
            "rrf_score",
        ]
    ].to_csv(
        recommendations_file,
        index=False,
        encoding="utf-8",
    )

    metadata = {
        "status": "V3_HOLDOUT_RERANKING_COMPLETE",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "selected_variant": EXPECTED_SELECTED_VARIANT,
        "model_id": EXPECTED_MODEL_ID,
        "model_revision": EXPECTED_MODEL_REVISION,
        "candidate_depth": 50,
        "display_depth": 10,
        "rrf_k": 60,
        "vector_weight": 1.0,
        "reranker_weight": 1.0,
        "retrieval_run_name": retrieval_run.name,
        "holdout_top50_candidates_sha256": EXPECTED_TOP50_SHA256,
        "v2_holdout_recommendations_sha256": EXPECTED_V2_BASELINE_SHA256,
        "holdout_retrieval_lock_sha256": EXPECTED_RETRIEVAL_LOCK_SHA256,
        "selected_candidate_lock_sha256": sha256(SELECTED_LOCK_FILE),
        "holdout_input_lock_sha256": sha256(HOLDOUT_INPUT_LOCK_FILE),
        "judge_labels_loaded": False,
        "human_relevance_labels_loaded": False,
        "judge_calls_performed": 0,
        "repeated_inference_rank_order_identical": True,
        "max_absolute_score_delta_between_passes": max_abs_delta,
        "device": device,
        "batch_size": runtime.batch_size,
        "outputs": {
            "v3_holdout_reranker_scores.csv": sha256(scores_file),
            "v3_holdout_recommendations.csv": sha256(recommendations_file),
        },
    }
    metadata_file.write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )

    lock_payload = {
        "status": "V3_HOLDOUT_RERANKING_FROZEN_BEFORE_JUDGING",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "selected_variant": EXPECTED_SELECTED_VARIANT,
        "model_id": EXPECTED_MODEL_ID,
        "model_revision": EXPECTED_MODEL_REVISION,
        "rrf_k": 60,
        "vector_weight": 1.0,
        "reranker_weight": 1.0,
        "holdout_top50_candidates_sha256": EXPECTED_TOP50_SHA256,
        "v2_holdout_recommendations_sha256": EXPECTED_V2_BASELINE_SHA256,
        "holdout_retrieval_lock_sha256": EXPECTED_RETRIEVAL_LOCK_SHA256,
        "v3_holdout_reranker_scores_sha256": sha256(scores_file),
        "v3_holdout_recommendations_sha256": sha256(recommendations_file),
        "v3_holdout_reranker_metadata_sha256": sha256(metadata_file),
        "recommendation_rows": EXPECTED_OUTPUT_ROWS,
        "query_count": EXPECTED_QUERY_COUNT,
        "recommendations_per_query": EXPECTED_OUTPUT_PER_QUERY,
        "judge_execution_performed": False,
        "human_relevance_labels_used": False,
        "next_stage": "build_unique_v2_v3_displayed_pair_union_for_frozen_judging",
    }
    lock_file.write_text(
        json.dumps(lock_payload, indent=2) + "\n",
        encoding="utf-8",
    )

    print()
    print("PASS  v3b full ranking structure 24x50")
    print("PASS  v3b displayed recommendations 24x10")
    print("PASS  v3b artifacts frozen before judging")
    print()
    print("V3 UNSEEN HOLDOUT RERANKING: PASS")
    print(f"Run directory: {run_dir}")
    print(f"v3 scores SHA-256: {sha256(scores_file)}")
    print(f"v3 recommendations SHA-256: {sha256(recommendations_file)}")
    print(f"v3 reranker lock SHA-256: {sha256(lock_file)}")
    print("Judge labels loaded: NO")
    print("Judge calls performed: 0")
    print("Next: construct unique union of displayed v2/v3 pairs for frozen judging.")


if __name__ == "__main__":
    main()
