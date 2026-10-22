from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from semantic_relevance_v4b_ranking import rank_v4b

REPO_ROOT = Path(__file__).resolve().parents[2]
EVALS = REPO_ROOT / "evals"
SYS = EVALS / "system_evaluation"
RUN = EVALS / "runs/semantic_relevance_v4a_development/20261019T104415Z_retrieval"

DESIGN_LOCK = SYS / "semantic_relevance_v4b_reranker_only_hybrid_experiment.v0.1.0.lock.json"
RANKING_SOURCE = SYS / "semantic_relevance_v4b_ranking.py"
IMPLEMENTATION_LOCK = SYS / "semantic_relevance_v4b_implementation.lock.json"
V4A_RERANKER_SCORES = RUN / "v4a_dev_reranker_scores.csv"
V4A_RERANKER_LOCK = RUN / "v4a_dev_reranker.lock.json"

OUT_FULL = RUN / "v4b_dev_ranked_full_union.csv"
OUT_RECS = RUN / "v4b_dev_recommendations.csv"
OUT_METADATA = RUN / "v4b_dev_ranking_metadata.json"
OUT_LOCK = RUN / "v4b_dev_ranking.lock.json"

EXPECTED = {
    "v4b_spec_json": "c72b5edbc330feae0620314fa5d5c6f90a55b356174d70a075ae77b47a8d0fca",
    "v4b_spec_md": "dae818b2519be70d49880e28642d76c414fa2a6ccda139242af5174984d29850",
    "v4b_design_lock": "e60ca1032f7fa733f3e7a51ec87a229d7454645aa6109b5e557588af50383c6d",
    "v4b_ranking_source": "5729b078cee1a2706a8c3d9bcaa9e8449292bc1f9f1a2816053b84758db7d677",
    "v4b_ranking_tests": "76593ff7471ed8e4beb61bcd349e3788bf97081d405decda0821f403b6c730f2",
    "v4b_implementation_lock": "f30a61e7ca7a5e07766c3f2baa3d4a65abcedb48cdedf23ee6d5c0e05b3171b0",
    "v4a_reranker_scores": "c94d4208257b2e17cc92f2a9f675c8db67a1ab675f945a0f7ddd96a362f34b49",
    "v4a_reranker_lock": "97a8cc18cf2e8c2cf5abec679f41de81546a945b6ae9b36ab06abd97d2508a22",
    "v4a_hybrid_union": "71be13c297f6449e6a53c54c7b9db590f6f033d69a11f16b21377797d9390155",
    "v4a_retrieval_lock": "7587aa140c1cb26f8ac5addaf6873f3bc7ec05b6ec34775a6cbfe98d7ffa40d1"
}
EXPECTED_QUERY_IDS = [f"H{i:02d}" for i in range(1,25)]
KNOWN_QUARANTINE_QUERY = "H14"
KNOWN_QUARANTINE_ISBN = "9781582972947"

def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))

def req(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)

def main() -> None:
    print("Semantic relevance V4B - deterministic development ranking")
    print("---------------------------------------------------------")
    print("Input: frozen V4A full-union reranker scores")
    print("Retrieval rerun: NO")
    print("Model inference: NO")
    print("Judge execution: NO")
    print("Relevance labels loaded: NO")
    print("Development selection: NOT CALCULATED")
    print("Release analysis: DISABLED")
    print()

    for path, expected, label in [
        (DESIGN_LOCK, EXPECTED["v4b_design_lock"], "V4B design lock"),
        (RANKING_SOURCE, EXPECTED["v4b_ranking_source"], "V4B ranking source"),
        (IMPLEMENTATION_LOCK, EXPECTED["v4b_implementation_lock"], "V4B implementation lock"),
        (V4A_RERANKER_SCORES, EXPECTED["v4a_reranker_scores"], "frozen V4A reranker scores"),
        (V4A_RERANKER_LOCK, EXPECTED["v4a_reranker_lock"], "V4A reranker lock"),
    ]:
        req(path.exists(), f"Missing {label}: {path}")
        actual = sha256(path)
        if path == RANKING_SOURCE and actual != expected:
            raw = path.read_bytes()
            actual_lf = hashlib.sha256(raw.replace(b"\r\n", b"\n").replace(b"\r", b"\n")).hexdigest()
            req(actual_lf == expected, f"{label} hash mismatch")
        else:
            req(actual == expected, f"{label} hash mismatch: expected={expected} actual={actual}")

    for out in [OUT_FULL, OUT_RECS, OUT_METADATA, OUT_LOCK]:
        req(not out.exists(), f"Refusing to overwrite existing V4B ranking artifact: {out}")

    frame = pd.read_csv(V4A_RERANKER_SCORES, dtype=str, keep_default_na=False, encoding="utf-8")
    req(len(frame) == 2211, f"Expected 2211 candidate rows, got {len(frame)}")
    req(frame["query_id"].nunique() == 24, "Expected 24 development queries")
    req(sorted(frame["query_id"].unique().tolist()) == EXPECTED_QUERY_IDS, "Development query IDs changed")
    req(not frame.duplicated(["query_id","isbn13"]).any(), "Duplicate query-book pair in input")

    ranked = rank_v4b(frame)
    req(len(ranked) == 2211, "V4B full-union ranking changed row count")
    req(not ranked.duplicated(["query_id","isbn13"]).any(), "Duplicate pair after V4B ranking")
    for qid, group in ranked.groupby("query_id", sort=True):
        ranks = sorted(group["v4b_rank"].astype(int).tolist())
        req(ranks == list(range(1, len(group)+1)), f"{qid} V4B ranks are not contiguous")

    top10 = ranked[ranked["v4b_rank"].astype(int) <= 10].copy()
    top10["rank"] = top10["v4b_rank"].astype(int)
    top10 = top10.sort_values(["query_id","rank"], kind="stable").reset_index(drop=True)
    req(len(top10) == 240, f"Expected 240 V4B recommendation rows, got {len(top10)}")
    counts = top10.groupby("query_id").size()
    req(len(counts) == 24 and (counts == 10).all(), "V4B recommendations are not exactly 24x10")

    quarantine_in_top10 = bool(
        (
            top10["query_id"].astype(str).eq(KNOWN_QUARANTINE_QUERY)
            & top10["isbn13"].astype(str).eq(KNOWN_QUARANTINE_ISBN)
        ).any()
    )

    source_counts = {
        str(k): int(v)
        for k, v in top10["candidate_source"].value_counts().to_dict().items()
    }

    # Write full deterministic rank order and displayed top10. This stage does
    # not read or attach any semantic-relevance labels.
    ranked.sort_values(["query_id","v4b_rank"], kind="stable").to_csv(
        OUT_FULL, index=False, encoding="utf-8"
    )
    top10.to_csv(OUT_RECS, index=False, encoding="utf-8")

    metadata = {
        "status": "V4B_DEVELOPMENT_RANKING_COMPLETE",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "development_role": "H01-H24 development only; no release claim",
        "v4b_design_lock_sha256": EXPECTED["v4b_design_lock"],
        "v4b_implementation_lock_sha256": EXPECTED["v4b_implementation_lock"],
        "v4b_ranking_source_sha256": EXPECTED["v4b_ranking_source"],
        "source_v4a_reranker_scores_sha256": EXPECTED["v4a_reranker_scores"],
        "source_v4a_reranker_lock_sha256": EXPECTED["v4a_reranker_lock"],
        "candidate_rows": 2211,
        "query_count": 24,
        "recommendation_rows": 240,
        "recommendations_per_query": 10,
        "ranking_method": "reranker_score_only",
        "exact_score_tie_breaks": ["hybrid_retrieval_rank ascending", "isbn13 ascending"],
        "top10_candidate_source_counts": source_counts,
        "known_h14_quarantine_candidate_in_v4b_top10": quarantine_in_top10,
        "relevance_labels_loaded": False,
        "retrieval_calls_performed": 0,
        "model_calls_performed": 0,
        "judge_calls_performed": 0,
        "human_relevance_labels_used": False,
        "development_selection_calculated": False,
        "release_metrics_calculated": False,
        "release_decision_produced": False,
        "outputs": {
            OUT_FULL.name: sha256(OUT_FULL),
            OUT_RECS.name: sha256(OUT_RECS),
        }
    }
    OUT_METADATA.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    lock = {
        "status": "V4B_DEVELOPMENT_RANKING_FROZEN_BEFORE_DISPLAY_UNION_OR_JUDGING",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "development_role": "H01-H24 development only",
        "v4b_design_lock_sha256": EXPECTED["v4b_design_lock"],
        "v4b_implementation_lock_sha256": EXPECTED["v4b_implementation_lock"],
        "v4b_ranking_source_sha256": EXPECTED["v4b_ranking_source"],
        "source_v4a_reranker_scores_sha256": EXPECTED["v4a_reranker_scores"],
        "source_v4a_reranker_lock_sha256": EXPECTED["v4a_reranker_lock"],
        "v4b_full_union_ranked_sha256": sha256(OUT_FULL),
        "v4b_recommendations_sha256": sha256(OUT_RECS),
        "v4b_ranking_metadata_sha256": sha256(OUT_METADATA),
        "candidate_rows": 2211,
        "recommendation_rows": 240,
        "query_count": 24,
        "recommendations_per_query": 10,
        "known_h14_quarantine_candidate_in_v4b_top10": quarantine_in_top10,
        "relevance_labels_loaded": False,
        "retrieval_calls_performed": 0,
        "model_calls_performed": 0,
        "judge_calls_performed": 0,
        "human_relevance_labels_used": False,
        "development_selection_calculated": False,
        "release_metrics_calculated": False,
        "release_decision_produced": False,
        "next_stage": "FREEZE_UNIQUE_V3B_V4B_DISPLAYED_PAIR_UNION_BEFORE_SCORE_REUSE_OR_JUDGING",
    }
    OUT_LOCK.write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")

    print("PASS  V4B full-union ranking derived deterministically from frozen scores")
    print("PASS  V4B displayed recommendations are exactly 24x10")
    print("PASS  V4B ranking artifacts frozen before score reuse/judging")
    print()
    print("V4B DEVELOPMENT RANKING: PASS")
    print(f"Full-union ranking SHA-256: {sha256(OUT_FULL)}")
    print(f"V4B recommendations SHA-256: {sha256(OUT_RECS)}")
    print(f"V4B ranking lock SHA-256: {sha256(OUT_LOCK)}")
    print(f"Top10 candidate-source counts: {source_counts}")
    print(f"Known H14 quarantine candidate in V4B top10: {quarantine_in_top10}")
    print("Relevance labels loaded: NO")
    print("Retrieval calls: 0")
    print("Model calls: 0")
    print("Judge calls: 0")
    print("Development selection calculated: NO")
    print("Release analysis: NO")
    print("Next: freeze unique v3b/v4b displayed-pair union before score reuse or judging.")

if __name__ == "__main__":
    main()
