from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from semantic_relevance_v3_reranker import (
    RerankerRuntimeConfig,
    candidate_text,
    load_cross_encoder,
    rank_v3a,
    rank_v3b,
    reranker_order_signature,
    score_query_candidate_pairs,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
EVALS = REPO_ROOT / "evals"

SPEC_FILE = EVALS / "system_evaluation/semantic_relevance_v3_reranker_experiment.v0.1.0.json"
SPEC_LOCK_FILE = EVALS / "system_evaluation/semantic_relevance_v3_reranker_experiment.v0.1.0.lock.json"

TOP50_RUN = EVALS / "runs/semantic_relevance_top50_diagnostic_v1/20261005T113402Z_candidates"
CANDIDATES_FILE = TOP50_RUN / "top50_candidates.csv"

RUN_ROOT = EVALS / "runs/semantic_relevance_v3_dev"

EXPECTED_SPEC_SHA256 = "5bb1f07f68db828b8db1a8d6d1c62831a5309651b7c62c91d04184bf41bb9261"
EXPECTED_CANDIDATES_SHA256 = "679dc2a8a3607db8d98e4ee868b5d7fc6566d11cb235eebed24e4676240abbad"
EXPECTED_ROWS = 600
EXPECTED_QUERIES = 12
EXPECTED_PER_QUERY = 50


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
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


def validate_inputs() -> tuple[dict[str, Any], pd.DataFrame]:
    if sha256(SPEC_FILE) != EXPECTED_SPEC_SHA256:
        raise ValueError("Frozen v3 experiment spec hash mismatch.")
    lock = load_json(SPEC_LOCK_FILE)
    if lock.get("status") != "FROZEN_BEFORE_V3_DEV_EXECUTION":
        raise ValueError("v3 spec is not frozen.")
    if lock.get("spec_sha256") != EXPECTED_SPEC_SHA256:
        raise ValueError("v3 spec lock hash mismatch.")
    if int(lock.get("development_judge_calls_allowed", -1)) != 0:
        raise ValueError("Development judge calls must be zero.")

    actual_candidates_sha = sha256(CANDIDATES_FILE)
    if actual_candidates_sha != EXPECTED_CANDIDATES_SHA256:
        raise ValueError("Frozen top50 candidate artifact changed.")

    frame = pd.read_csv(
        CANDIDATES_FILE,
        dtype=str,
        keep_default_na=False,
        encoding="utf-8",
    )
    required = {"query_id", "query", "vector_rank", "isbn13", "title", "description"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"top50_candidates.csv missing columns: {sorted(missing)}")

    frame["vector_rank"] = pd.to_numeric(frame["vector_rank"], errors="raise").astype(int)

    if len(frame) != EXPECTED_ROWS:
        raise ValueError(f"Expected {EXPECTED_ROWS} candidates, got {len(frame)}.")
    if frame.duplicated(["query_id", "isbn13"]).any():
        raise ValueError("Duplicate query-book pairs in frozen candidates.")

    counts = frame.groupby("query_id").size()
    if len(counts) != EXPECTED_QUERIES or not (counts == EXPECTED_PER_QUERY).all():
        raise ValueError(f"Expected 12x50 candidates; got {counts.to_dict()}.")

    for qid, group in frame.groupby("query_id"):
        if sorted(group["vector_rank"].tolist()) != list(range(1, 51)):
            raise ValueError(f"{qid}: vector_rank is not exactly 1..50.")

    # Important leakage boundary: this execution script deliberately does not
    # read top50_judge_scores_complete.csv or any human relevance labels.
    return load_json(SPEC_FILE), frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    args = parser.parse_args()

    print("Semantic relevance v3 development reranker execution")
    print("----------------------------------------------------")
    print("Input: frozen Checkpoint A top50 candidates only")
    print("Judge labels loaded: NO")
    print("Human relevance labels loaded: NO")
    print("Judge calls: 0")
    print("Two inference passes: REQUIRED for rank reproducibility")
    print()

    spec, frame = validate_inputs()
    model_spec = spec["reranker_model"]

    runtime = RerankerRuntimeConfig(
        model_id=str(model_spec["model_id"]),
        model_revision=str(model_spec["model_revision"]),
        max_length_tokens=int(model_spec["max_length_tokens"]),
        truncation=str(model_spec["truncation"]),
        batch_size=int(args.batch_size),
        device=str(args.device),
    )

    print(f"Model:    {runtime.model_id}")
    print(f"Revision: {runtime.model_revision}")
    print(f"Batch:    {runtime.batch_size}")

    tokenizer, model, device = load_cross_encoder(runtime)
    print(f"Device:   {device}")

    working = frame.sort_values(["query_id", "vector_rank"], kind="stable").copy()
    queries = working["query"].astype(str).tolist()
    candidate_texts = [
        candidate_text(title, description)
        for title, description in zip(working["title"], working["description"])
    ]

    print()
    print("Scoring pass 1/2...")
    scores_1 = score_query_candidate_pairs(
        queries=queries,
        candidates=candidate_texts,
        tokenizer=tokenizer,
        model=model,
        device=device,
        max_length_tokens=runtime.max_length_tokens,
        batch_size=runtime.batch_size,
    )

    print("Scoring pass 2/2...")
    scores_2 = score_query_candidate_pairs(
        queries=queries,
        candidates=candidate_texts,
        tokenizer=tokenizer,
        model=model,
        device=device,
        max_length_tokens=runtime.max_length_tokens,
        batch_size=runtime.batch_size,
    )

    pass1 = working.copy()
    pass1["reranker_score"] = scores_1
    pass2 = working.copy()
    pass2["reranker_score"] = scores_2

    sig1 = reranker_order_signature(pass1)
    sig2 = reranker_order_signature(pass2)
    if sig1 != sig2:
        bad = [qid for qid in sig1 if sig1[qid] != sig2.get(qid)]
        raise ValueError(f"Reranker rank order changed between repeated inference passes: {bad}")

    max_abs_delta = float(np.max(np.abs(scores_1 - scores_2)))
    print("PASS  repeated inference produced identical per-query reranker order")
    print(f"INFO  max absolute raw-score delta between passes: {max_abs_delta:.12g}")

    v3a = rank_v3a(pass1)
    v3b = rank_v3b(pass1, rrf_k=60.0)

    score_cols = [
        "query_id", "query", "vector_rank", "isbn13", "title", "description",
        "reranker_score", "reranker_rank",
    ]
    scores_out = v3a[score_cols].copy()
    rrf_map = v3b[["query_id", "isbn13", "rrf_score", "v3b_rank"]].copy()
    scores_out = scores_out.merge(
        rrf_map,
        on=["query_id", "isbn13"],
        how="left",
        validate="one_to_one",
    )
    scores_out["v3a_rank"] = scores_out["reranker_rank"].astype(int)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = RUN_ROOT / f"{timestamp}_spec_v0_1_0"
    if run_dir.exists():
        raise FileExistsError(run_dir)
    run_dir.mkdir(parents=True)

    scores_file = run_dir / "v3_dev_reranker_scores.csv"
    v3a_file = run_dir / "v3_dev_recommendations_v3a.csv"
    v3b_file = run_dir / "v3_dev_recommendations_v3b.csv"
    metadata_file = run_dir / "v3_dev_reranker_run_metadata.json"

    scores_out.sort_values(["query_id", "vector_rank"], kind="stable").to_csv(
        scores_file, index=False, encoding="utf-8"
    )

    v3a_top10 = (
        v3a[v3a["v3a_rank"] <= 10]
        .sort_values(["query_id", "v3a_rank"], kind="stable")
        .copy()
    )
    v3a_top10.rename(columns={"v3a_rank": "rank"}, inplace=True)
    v3a_top10[
        ["query_id", "query", "rank", "vector_rank", "isbn13", "title", "description", "reranker_score"]
    ].to_csv(v3a_file, index=False, encoding="utf-8")

    v3b_top10 = (
        v3b[v3b["v3b_rank"] <= 10]
        .sort_values(["query_id", "v3b_rank"], kind="stable")
        .copy()
    )
    v3b_top10.rename(columns={"v3b_rank": "rank"}, inplace=True)
    v3b_top10[
        ["query_id", "query", "rank", "vector_rank", "isbn13", "title", "description",
         "reranker_score", "reranker_rank", "rrf_score"]
    ].to_csv(v3b_file, index=False, encoding="utf-8")

    metadata = {
        "status": "V3_DEV_RERANKER_EXECUTION_COMPLETE",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": git_sha(),
        "spec_sha256": sha256(SPEC_FILE),
        "spec_lock_sha256": sha256(SPEC_LOCK_FILE),
        "top50_candidates_sha256": sha256(CANDIDATES_FILE),
        "judge_labels_loaded": False,
        "human_relevance_labels_loaded": False,
        "judge_calls_performed": 0,
        "model_id": runtime.model_id,
        "model_revision": runtime.model_revision,
        "max_length_tokens": runtime.max_length_tokens,
        "truncation": runtime.truncation,
        "batch_size": runtime.batch_size,
        "device": device,
        "repeated_inference_rank_order_identical": True,
        "max_absolute_score_delta_between_passes": max_abs_delta,
        "python_version": platform.python_version(),
        "torch_version": package_version("torch"),
        "transformers_version": package_version("transformers"),
        "outputs": {
            "v3_dev_reranker_scores.csv": sha256(scores_file),
            "v3_dev_recommendations_v3a.csv": sha256(v3a_file),
            "v3_dev_recommendations_v3b.csv": sha256(v3b_file),
        },
    }
    metadata_file.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    print()
    print("V3 DEVELOPMENT RERANKER EXECUTION: PASS")
    print(f"Run directory: {run_dir}")
    print(f"Reranker scores SHA-256: {sha256(scores_file)}")
    print(f"v3a recommendations SHA-256: {sha256(v3a_file)}")
    print(f"v3b recommendations SHA-256: {sha256(v3b_file)}")
    print("Next: analyze against the frozen Checkpoint C score map; do not judge new cases.")


if __name__ == "__main__":
    main()
