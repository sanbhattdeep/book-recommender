from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
EVALS = REPO_ROOT / "evals"
SYS = EVALS / "system_evaluation"
RUN = EVALS / "runs/semantic_relevance_v4a_development/20261019T104415Z_retrieval"

SPEC_JSON = SYS / "semantic_relevance_v4b_reranker_only_hybrid_experiment.v0.1.0.json"
SPEC_MD = SYS / "semantic_relevance_v4b_reranker_only_hybrid_experiment.v0.1.0.md"
DESIGN_LOCK = SYS / "semantic_relevance_v4b_reranker_only_hybrid_experiment.v0.1.0.lock.json"
RANKING_SOURCE = SYS / "semantic_relevance_v4b_ranking.py"
RANKING_TESTS = SYS / "test_semantic_relevance_v4b_ranking.py"
IMPLEMENTATION_LOCK = SYS / "semantic_relevance_v4b_implementation.lock.json"

V4A_RERANKER_SCORES = RUN / "v4a_dev_reranker_scores.csv"
V4A_RERANKER_LOCK = RUN / "v4a_dev_reranker.lock.json"
V4A_HYBRID_UNION = RUN / "v4a_dev_hybrid_union.csv"
V4A_RETRIEVAL_LOCK = RUN / "v4a_dev_retrieval.lock.json"

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

def lf_sha256(path: Path) -> str:
    raw = path.read_bytes()
    return hashlib.sha256(raw.replace(b"\r\n", b"\n").replace(b"\r", b"\n")).hexdigest()

def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))

def req(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)

def verify_text(path: Path, expected: str, label: str) -> None:
    req(path.exists(), f"Missing {label}: {path}")
    raw = sha256(path)
    if raw == expected:
        mode = "raw"
    else:
        norm = lf_sha256(path)
        req(norm == expected, f"{label} hash mismatch: expected={expected} raw={raw} lf={norm}")
        mode = "canonical-LF"
    print(f"PASS  {label} {mode} SHA-256 verified")

def verify_raw(path: Path, expected: str, label: str) -> None:
    req(path.exists(), f"Missing {label}: {path}")
    actual = sha256(path)
    req(actual == expected, f"{label} hash mismatch: expected={expected} actual={actual}")
    print(f"PASS  {label} raw SHA-256 verified")

def main() -> None:
    print("Semantic relevance V4B - development ranking execution preflight")
    print("----------------------------------------------------------------")
    print("Recommendation generation: DISABLED")
    print("Retrieval execution: DISABLED")
    print("Model execution: DISABLED")
    print("Judge execution: DISABLED")
    print("Relevance labels loaded: NO")
    print("Development selection: NOT CALCULATED")
    print("Release analysis: DISABLED")
    print()

    for out in [OUT_FULL, OUT_RECS, OUT_METADATA, OUT_LOCK]:
        req(not out.exists(), f"Refusing to overwrite existing V4B ranking artifact: {out}")

    for path, key, label, text_mode in [
        (SPEC_JSON, "v4b_spec_json", "V4B JSON spec", True),
        (SPEC_MD, "v4b_spec_md", "V4B Markdown spec", True),
        (DESIGN_LOCK, "v4b_design_lock", "V4B design lock", False),
        (RANKING_SOURCE, "v4b_ranking_source", "V4B ranking source", True),
        (RANKING_TESTS, "v4b_ranking_tests", "V4B ranking tests", True),
        (IMPLEMENTATION_LOCK, "v4b_implementation_lock", "V4B implementation lock", False),
        (V4A_RERANKER_SCORES, "v4a_reranker_scores", "frozen V4A full-union reranker scores", False),
        (V4A_RERANKER_LOCK, "v4a_reranker_lock", "V4A reranker lock", False),
        (V4A_HYBRID_UNION, "v4a_hybrid_union", "frozen V4A hybrid union", False),
        (V4A_RETRIEVAL_LOCK, "v4a_retrieval_lock", "V4A retrieval lock", False),
    ]:
        (verify_text if text_mode else verify_raw)(path, EXPECTED[key], label)

    design = load(DESIGN_LOCK)
    req(design.get("status") == "V4B_DESIGN_FROZEN_BEFORE_IMPLEMENTATION", "Unexpected V4B design lock status")
    req(design.get("ranking_method") == "reranker_score_only", "Unexpected V4B ranking method")
    req(design.get("reuse_frozen_union") is True, "V4B design no longer reuses frozen union")
    req(design.get("reuse_frozen_reranker_scores") is True, "V4B design no longer reuses frozen reranker scores")

    impl = load(IMPLEMENTATION_LOCK)
    req(impl.get("status") == "V4B_IMPLEMENTATION_FROZEN_BEFORE_DEVELOPMENT_RANKING", "Unexpected V4B implementation lock status")
    req(impl.get("recommendations_generated") is False, "Implementation lock says recommendations already existed")
    req(impl.get("retrieval_calls_performed") == 0, "Implementation freeze retrieval-call boundary changed")
    req(impl.get("model_calls_performed") == 0, "Implementation freeze model-call boundary changed")
    req(impl.get("judge_calls_performed") == 0, "Implementation freeze judge-call boundary changed")

    rlock = load(V4A_RERANKER_LOCK)
    req(rlock.get("status") == "V4A_DEVELOPMENT_RERANKING_FROZEN_BEFORE_JUDGING", "Unexpected V4A reranker lock status")
    req(int(rlock.get("candidate_rows", -1)) == 2211, "Frozen V4A reranker candidate count changed")
    req(rlock.get("repeated_inference_rank_order_identical") is True, "Frozen reranker repeatability contract changed")

    frame = pd.read_csv(V4A_RERANKER_SCORES, dtype=str, keep_default_na=False, encoding="utf-8")
    required = {
        "query_id","query","stratum","isbn13","dense_rank","bm25_rank","bm25_score",
        "candidate_source","hybrid_retrieval_rrf_score","hybrid_retrieval_rank",
        "title","description","authors","reranker_score"
    }
    req(not (required - set(frame.columns)), f"Reranker score artifact missing columns: {sorted(required - set(frame.columns))}")
    req(len(frame) == 2211, f"Expected 2211 full-union rows, got {len(frame)}")
    req(frame["query_id"].nunique() == 24, "Expected 24 development queries")
    req(sorted(frame["query_id"].unique().tolist()) == EXPECTED_QUERY_IDS, "Development query IDs changed")
    req(not frame.duplicated(["query_id","isbn13"]).any(), "Duplicate query-book pair in frozen reranker scores")

    rr = pd.to_numeric(frame["reranker_score"], errors="raise").astype(float)
    req(np.isfinite(rr.to_numpy()).all(), "Frozen reranker scores contain non-finite values")
    hr = pd.to_numeric(frame["hybrid_retrieval_rank"], errors="raise").astype(int)
    req((hr >= 1).all(), "Hybrid retrieval rank outside valid range")

    for qid, group in frame.assign(_rank=hr).groupby("query_id", sort=True):
        ranks = sorted(group["_rank"].tolist())
        req(ranks == list(range(1, len(group)+1)), f"{qid} hybrid retrieval ranks are not contiguous")

    quarantine_rows = frame[
        frame["query_id"].astype(str).eq(KNOWN_QUARANTINE_QUERY)
        & frame["isbn13"].astype(str).eq(KNOWN_QUARANTINE_ISBN)
    ]
    req(len(quarantine_rows) == 1, "Known H14 quarantine candidate not uniquely present in frozen full union")

    print("PASS  frozen 2211-row reranker-score artifact structure verified")
    print("PASS  exact H01-H24 development query set verified")
    print("PASS  known H14 quarantine candidate identity present exactly once")
    print("PASS  zero-retrieval/model/judge execution boundary verified")
    print()
    print("V4B DEVELOPMENT RANKING PREFLIGHT: PASS")
    print("Candidate rows: 2211")
    print("Queries: 24")
    print("Recommendation rows to derive: 240")
    print("Relevance labels loaded: NO")
    print("Retrieval calls: 0")
    print("Model calls: 0")
    print("Judge calls: 0")
    print("Development selection calculated: NO")
    print("Release analysis: NO")
    print("Next: deterministically derive V4B full-union ranking and 24x10 recommendations.")

if __name__ == "__main__":
    main()
