from __future__ import annotations

import argparse
import hashlib, json, math, subprocess
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
from semantic_relevance_v4b_ranking import rank_v4b

REPO_ROOT = Path(__file__).resolve().parents[2]
SYS = REPO_ROOT / "evals/system_evaluation"
RUN = REPO_ROOT / "evals/runs/semantic_relevance_v4_holdout/20261022T103659Z_retrieval"

RETRIEVAL_CONTRACT = SYS / "semantic_relevance_v4h_retrieval_contract.v0.1.0.json"
RETRIEVAL_LOCK = RUN / "v4h_retrieval.lock.json"
DENSE_TOP50 = RUN / "v4h_dense_top50.csv"
BM25_TOP50 = RUN / "v4h_bm25_top50.csv"
HYBRID_UNION = RUN / "v4h_hybrid_union.csv"
V4H_AUTHORING_LOCK = SYS / "semantic_relevance_v4_holdout_authoring.lock.json"

V3_RERANKER_SOURCE = SYS / "semantic_relevance_v3_reranker.py"
V3_RERANKER_SPEC = SYS / "semantic_relevance_v3_reranker_experiment.v0.1.0.json"
V4B_RANKING_SOURCE = SYS / "semantic_relevance_v4b_ranking.py"
V4B_IMPLEMENTATION_LOCK = SYS / "semantic_relevance_v4b_implementation.lock.json"
V4B_DESIGN_LOCK = SYS / "semantic_relevance_v4b_reranker_only_hybrid_experiment.v0.1.0.lock.json"
RERANKING_CONTRACT = SYS / "semantic_relevance_v4h_reranking_contract.v0.1.0.json"


EXPECTED = {
    "retrieval_contract": "1bf245fdd3cec8baa4fd3e6ece3602ab2d72ee5fafd12e49b0bb68fb443dd6f8",
    "retrieval_lock": "e2902e1cc537e74a39b6a0c1d421c8db0d00637596c89a7889841ae242089884",
    "dense_top50": "40abebfe33af7352e9c82352daa3ca2143e666073257696fdf270aae9d01a8aa",
    "bm25_top50": "35f1096b50d2caf41bbdbdeab055bcca94bba279a84f2261eeb92f77f539418a",
    "hybrid_union": "97fd79621bacc4ec56d204bdc366ad1757c1f6329d014861a1bbef6999fdd311",
    "v3_reranker_source": "5e7a7d6b35c0674f32644e758218830d8a340a61588e2ff0b3f20582fdbb0f60",
    "v3_reranker_spec": "5bb1f07f68db828b8db1a8d6d1c62831a5309651b7c62c91d04184bf41bb9261",
    "v4b_ranking_source": "5729b078cee1a2706a8c3d9bcaa9e8449292bc1f9f1a2816053b84758db7d677",
    "v4b_implementation_lock": "f30a61e7ca7a5e07766c3f2baa3d4a65abcedb48cdedf23ee6d5c0e05b3171b0",
    "v4b_design_lock": "e60ca1032f7fa733f3e7a51ec87a229d7454645aa6109b5e557588af50383c6d",
    "v4h_authoring_lock": "14cd3a8a9f6bf22666dd4c74a150e85f82a53789ccaa5bd2870d606b66ef83e7"
}
EXPECTED_CONTRACT_SHA256 = "0cf17eda345d7abb906a8f5f4730fb5d356d1514401d603271ab7cf4707df471"
EXPECTED_V3_RERANKER_BLOB = "d0d216e9c49ea5357ab94d83a1a6fbc1b9baa353"
MODEL_ID = "BAAI/bge-reranker-base"
MODEL_REVISION = "2cfc18c9415c912f9d8155881c133215df768a70"
EXPECTED_QUERY_IDS = [f"V4H{i:02d}" for i in range(1, 25)]
EXPECTED_QUERY_COUNT = 24
EXPECTED_DENSE_ROWS = 1200

def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def lf_sha256(path: Path) -> str:
    raw = path.read_bytes()
    return hashlib.sha256(raw.replace(b"\r\n", b"\n").replace(b"\r", b"\n")).hexdigest()

def req(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)

def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8-sig"))

def verify_raw(path: Path, expected: str, label: str) -> None:
    req(path.exists(), f"Missing {label}: {path}")
    actual = sha256(path)
    req(actual == expected, f"{label} hash mismatch: expected={expected} actual={actual}")
    print(f"PASS  {label} raw SHA-256 verified")

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

def git_blob(repo_path: str) -> str:
    return subprocess.run(
        ["git","rev-parse",f"HEAD:{repo_path}"],
        cwd=str(REPO_ROOT), capture_output=True, text=True, check=True
    ).stdout.strip()

def verify_frozen_boundary() -> tuple[pd.DataFrame, pd.DataFrame]:
    verify_raw(RETRIEVAL_CONTRACT, EXPECTED["retrieval_contract"], "V4H retrieval contract")
    verify_raw(RETRIEVAL_LOCK, EXPECTED["retrieval_lock"], "V4H retrieval lock")
    verify_raw(DENSE_TOP50, EXPECTED["dense_top50"], "V4H dense top50")
    verify_raw(BM25_TOP50, EXPECTED["bm25_top50"], "V4H BM25 top50")
    verify_raw(HYBRID_UNION, EXPECTED["hybrid_union"], "V4H hybrid union")
    verify_raw(V4H_AUTHORING_LOCK, EXPECTED["v4h_authoring_lock"], "V4H authoring lock")
    verify_text(V3_RERANKER_SOURCE, EXPECTED["v3_reranker_source"], "V3 frozen reranker source")
    verify_text(V3_RERANKER_SPEC, EXPECTED["v3_reranker_spec"], "V3 reranker spec")
    verify_text(V4B_RANKING_SOURCE, EXPECTED["v4b_ranking_source"], "V4B ranking source")
    verify_raw(V4B_IMPLEMENTATION_LOCK, EXPECTED["v4b_implementation_lock"], "V4B implementation lock")
    verify_raw(V4B_DESIGN_LOCK, EXPECTED["v4b_design_lock"], "V4B design lock")
    verify_raw(RERANKING_CONTRACT, EXPECTED_CONTRACT_SHA256, "V4H reranking contract")

    blob = git_blob("evals/system_evaluation/semantic_relevance_v3_reranker.py")
    req(blob == EXPECTED_V3_RERANKER_BLOB,
        f"Tracked V3 reranker Git blob changed: expected={EXPECTED_V3_RERANKER_BLOB} actual={blob}")
    diff = subprocess.run(
        ["git","diff","--quiet","--","evals/system_evaluation/semantic_relevance_v3_reranker.py"],
        cwd=str(REPO_ROOT)
    )
    req(diff.returncode == 0, "V3 reranker source has working-tree modifications")
    print("PASS  V3 reranker exact tracked Git blob verified and working tree unmodified")

    rlock = load_json(RETRIEVAL_LOCK)
    req(rlock.get("status") == "V4H_RETRIEVAL_FROZEN_BEFORE_RERANKING_OR_JUDGING",
        "Unexpected V4H retrieval-lock status")
    req(rlock.get("query_set") == "V4H01-V4H24", "Retrieval lock query set changed")
    req(rlock.get("baseline") == "v3b" and rlock.get("candidate") == "v4b",
        "Retrieval lock system identities changed")
    req(rlock.get("dense_top50_sha256") == EXPECTED["dense_top50"], "Retrieval lock dense hash mismatch")
    req(rlock.get("bm25_top50_sha256") == EXPECTED["bm25_top50"], "Retrieval lock BM25 hash mismatch")
    req(rlock.get("hybrid_union_sha256") == EXPECTED["hybrid_union"], "Retrieval lock union hash mismatch")
    req(rlock.get("dense_repeat_query_order_identical_for_all_queries") is True,
        "Dense repeatability gate did not pass")
    req(rlock.get("reranker_inference_performed") is False, "Retrieval lock says reranker already ran")
    req(rlock.get("judge_execution_performed") is False, "Retrieval lock says judge already ran")
    req(rlock.get("next_stage") == "RERANK_FROZEN_DENSE50_FOR_V3B_AND_FROZEN_FULL_HYBRID_UNION_FOR_V4B",
        "Retrieval lock next-stage contract changed")

    spec = load_json(V3_RERANKER_SPEC)
    req(spec.get("reranker_model", {}).get("model_id") == MODEL_ID, "V3 reranker model ID changed")
    req(spec.get("reranker_model", {}).get("model_revision") == MODEL_REVISION, "V3 reranker model revision changed")
    req(int(spec.get("reranker_model", {}).get("max_length_tokens", -1)) == 512, "V3 reranker max length changed")

    v4b_lock = load_json(V4B_DESIGN_LOCK)
    req(v4b_lock.get("status") == "V4B_DESIGN_FROZEN_BEFORE_IMPLEMENTATION",
        "Unexpected V4B design lock status")
    req(v4b_lock.get("ranking_method") == "reranker_score_only", "V4B ranking method changed")
    req(v4b_lock.get("non_tie_retrieval_rank_weight") is False,
        "V4B design now gives retrieval rank non-tie weight")

    dense = pd.read_csv(DENSE_TOP50, dtype=str, keep_default_na=False, encoding="utf-8")
    union = pd.read_csv(HYBRID_UNION, dtype=str, keep_default_na=False, encoding="utf-8")

    dense_req = {"query_id","query","stratum","vector_rank","isbn13","title","description"}
    union_req = {"query_id","query","stratum","hybrid_retrieval_rank","dense_rank","isbn13","title","description"}
    req(not (dense_req-set(dense.columns)), f"Dense file missing columns: {sorted(dense_req-set(dense.columns))}")
    req(not (union_req-set(union.columns)), f"Union file missing columns: {sorted(union_req-set(union.columns))}")

    dense["vector_rank"] = pd.to_numeric(dense["vector_rank"], errors="raise").astype(int)
    union["hybrid_retrieval_rank"] = pd.to_numeric(union["hybrid_retrieval_rank"], errors="raise").astype(int)

    req(len(dense) == EXPECTED_DENSE_ROWS, f"Expected 1200 dense rows, got {len(dense)}")
    req(dense["query_id"].nunique() == 24 and union["query_id"].nunique() == 24,
        "Expected 24 queries in dense and union artifacts")
    req(sorted(dense["query_id"].unique()) == EXPECTED_QUERY_IDS, "Dense query IDs changed")
    req(sorted(union["query_id"].unique()) == EXPECTED_QUERY_IDS, "Union query IDs changed")
    req(not dense.duplicated(["query_id","isbn13"]).any(), "Duplicate dense query-book pair")
    req(not union.duplicated(["query_id","isbn13"]).any(), "Duplicate union query-book pair")

    for qid, dg in dense.groupby("query_id", sort=True):
        req(sorted(dg["vector_rank"].tolist()) == list(range(1,51)), f"{qid}: dense ranks not 1..50")
        ug = union[union["query_id"] == qid]
        req(50 <= len(ug) <= 100, f"{qid}: union size outside [50,100]")
        req(sorted(ug["hybrid_retrieval_rank"].tolist()) == list(range(1,len(ug)+1)),
            f"{qid}: union ranks not contiguous")
        req(set(dg["isbn13"].astype(str)) <= set(ug["isbn13"].astype(str)),
            f"{qid}: dense candidates missing from full union")

    print("PASS  frozen V4H dense and full-union structures verified")
    return dense, union


FULL_SCORES = RUN / "v4h_full_union_reranker_scores.csv"
V3_FULL = RUN / "v4h_v3b_ranked_dense50.csv"
V3_RECS = RUN / "v4h_v3b_recommendations.csv"
V4_FULL = RUN / "v4h_v4b_ranked_full_union.csv"
V4_RECS = RUN / "v4h_v4b_recommendations.csv"
METADATA = RUN / "v4h_reranking_metadata.json"
LOCK = RUN / "v4h_reranking.lock.json"

def full_union_signature(frame: pd.DataFrame) -> dict[str, tuple[str,...]]:
    signatures = {}
    for qid, group in frame.groupby("query_id", sort=True):
        g = group.copy()
        g["reranker_score"] = pd.to_numeric(g["reranker_score"], errors="raise").astype(float)
        g["hybrid_retrieval_rank"] = pd.to_numeric(g["hybrid_retrieval_rank"], errors="raise").astype(int)
        g["isbn13"] = g["isbn13"].astype(str)
        g = g.sort_values(
            ["reranker_score","hybrid_retrieval_rank","isbn13"],
            ascending=[False,True,True], kind="stable"
        )
        signatures[str(qid)] = tuple(g["isbn13"].tolist())
    return signatures

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", choices=["auto","cuda","cpu"], default="auto")
    args = parser.parse_args()

    print("Semantic relevance V4H - frozen V3B/V4B reranking")
    print("------------------------------------------------")
    print("Input: frozen V4H dense50 + full hybrid union")
    print("Retrieval calls: 0")
    print("Judge execution: DISABLED")
    print("Human relevance labels: NOT USED")
    print("Release metrics/decision: NOT CALCULATED")
    print("Two cross-encoder passes: REQUIRED")
    print()

    dense, union = verify_frozen_boundary()
    for out in [FULL_SCORES,V3_FULL,V3_RECS,V4_FULL,V4_RECS,METADATA,LOCK]:
        req(not out.exists(), f"Refusing to overwrite existing reranking artifact: {out}")

    runtime = RerankerRuntimeConfig(
        model_id=MODEL_ID,
        model_revision=MODEL_REVISION,
        max_length_tokens=512,
        truncation="only_second",
        batch_size=int(args.batch_size),
        device=str(args.device),
    )
    tokenizer, model, device = load_cross_encoder(runtime)
    print(f"Model:    {MODEL_ID}")
    print(f"Revision: {MODEL_REVISION}")
    print(f"Device:   {device}")
    print(f"Batch:    {runtime.batch_size}")

    working = union.sort_values(["query_id","hybrid_retrieval_rank"], kind="stable").copy()
    queries = working["query"].astype(str).tolist()
    texts = [candidate_text(t,d) for t,d in zip(working["title"],working["description"])]

    print()
    print("Scoring full frozen union pass 1/2...")
    scores1 = score_query_candidate_pairs(
        queries=queries, candidates=texts, tokenizer=tokenizer, model=model, device=device,
        max_length_tokens=512, batch_size=runtime.batch_size
    )
    print("Scoring full frozen union pass 2/2...")
    scores2 = score_query_candidate_pairs(
        queries=queries, candidates=texts, tokenizer=tokenizer, model=model, device=device,
        max_length_tokens=512, batch_size=runtime.batch_size
    )

    req(np.isfinite(scores1).all() and np.isfinite(scores2).all(), "Non-finite reranker score")
    p1 = working.copy(); p1["reranker_score"] = scores1
    p2 = working.copy(); p2["reranker_score"] = scores2

    sig_full1 = full_union_signature(p1)
    sig_full2 = full_union_signature(p2)
    req(sig_full1 == sig_full2, "Full-union reranker order changed between passes")

    dense_keys = set(zip(dense["query_id"].astype(str), dense["isbn13"].astype(str)))
    dense1 = p1[
        [(str(q),str(i)) in dense_keys for q,i in zip(p1["query_id"],p1["isbn13"])]
    ].copy()
    dense2 = p2[
        [(str(q),str(i)) in dense_keys for q,i in zip(p2["query_id"],p2["isbn13"])]
    ].copy()

    dense_rank_map = dense[["query_id","isbn13","vector_rank"]].copy()
    dense1 = dense1.drop(columns=["dense_rank"], errors="ignore").merge(
        dense_rank_map, on=["query_id","isbn13"], how="left", validate="one_to_one"
    )
    dense2 = dense2.drop(columns=["dense_rank"], errors="ignore").merge(
        dense_rank_map, on=["query_id","isbn13"], how="left", validate="one_to_one"
    )
    req(len(dense1) == EXPECTED_DENSE_ROWS and len(dense2) == EXPECTED_DENSE_ROWS,
        "Dense score subset is not exactly 1200 rows")
    req(dense1["vector_rank"].notna().all() and dense2["vector_rank"].notna().all(),
        "Dense vector rank missing after score reuse")

    sig_dense1 = reranker_order_signature(dense1)
    sig_dense2 = reranker_order_signature(dense2)
    req(sig_dense1 == sig_dense2, "Dense-only V3B reranker order changed between passes")

    max_delta = float(np.max(np.abs(scores1-scores2)))
    req(math.isfinite(max_delta), "Repeated-inference score delta is non-finite")
    print("PASS  repeated inference produced identical full-union rank order")
    print("PASS  repeated inference produced identical dense-only reranker order")
    print(f"INFO  max absolute raw-score delta between passes: {max_delta:.12g}")

    v3_ranked = rank_v3b(dense1, rrf_k=60.0)
    v4_ranked = rank_v4b(p1)

    req(len(v3_ranked) == 1200, "V3B ranked dense pool row count changed")
    req(len(v4_ranked) == len(union), "V4B ranked union row count changed")
    req(not v3_ranked.duplicated(["query_id","isbn13"]).any(), "Duplicate V3B ranked pair")
    req(not v4_ranked.duplicated(["query_id","isbn13"]).any(), "Duplicate V4B ranked pair")

    v3_top = v3_ranked[v3_ranked["v3b_rank"].astype(int) <= 10].copy()
    v4_top = v4_ranked[v4_ranked["v4b_rank"].astype(int) <= 10].copy()
    v3_top["rank"] = v3_top["v3b_rank"].astype(int)
    v4_top["rank"] = v4_top["v4b_rank"].astype(int)
    v3_top = v3_top.sort_values(["query_id","rank"], kind="stable")
    v4_top = v4_top.sort_values(["query_id","rank"], kind="stable")

    for label, top in [("V3B",v3_top),("V4B",v4_top)]:
        counts = top.groupby("query_id").size()
        req(len(top) == 240 and len(counts) == 24 and (counts == 10).all(),
            f"{label} recommendations are not exact 24x10")
        req(not top.duplicated(["query_id","isbn13"]).any(), f"Duplicate {label} top10 pair")

    # Persist pass-1 raw scores over the complete union, then deterministic system rankings.
    score_cols = [
        "query_id","query","stratum","hybrid_retrieval_rank","dense_rank","bm25_rank",
        "candidate_source","isbn13","title","description","authors","reranker_score"
    ]
    existing_score_cols = [c for c in score_cols if c in p1.columns]
    p1[existing_score_cols].sort_values(["query_id","hybrid_retrieval_rank"], kind="stable").to_csv(
        FULL_SCORES,index=False,encoding="utf-8"
    )
    v3_ranked.sort_values(["query_id","v3b_rank"],kind="stable").to_csv(V3_FULL,index=False,encoding="utf-8")
    v3_top.to_csv(V3_RECS,index=False,encoding="utf-8")
    v4_ranked.sort_values(["query_id","v4b_rank"],kind="stable").to_csv(V4_FULL,index=False,encoding="utf-8")
    v4_top.to_csv(V4_RECS,index=False,encoding="utf-8")

    source_counts = {str(k):int(v) for k,v in v4_top["candidate_source"].value_counts().to_dict().items()}

    metadata = {
        "status":"V4H_FINAL_HOLDOUT_RERANKING_COMPLETE",
        "completed_at_utc":datetime.now(timezone.utc).isoformat(),
        "query_set":"V4H01-V4H24",
        "query_count":24,
        "baseline":"v3b",
        "candidate":"v4b",
        "model_id":MODEL_ID,
        "model_revision":MODEL_REVISION,
        "device":device,
        "batch_size":runtime.batch_size,
        "max_length_tokens":512,
        "truncation":"only_second",
        "candidate_text_template":"Title: {title}\nDescription: {description}",
        "source_retrieval_lock_sha256":EXPECTED["retrieval_lock"],
        "source_dense_top50_sha256":EXPECTED["dense_top50"],
        "source_hybrid_union_sha256":EXPECTED["hybrid_union"],
        "score_reuse":"full-union cross-encoder scores reused for dense-only V3B subset",
        "repeated_inference_passes":2,
        "full_union_reranker_order_identical":True,
        "dense_subset_reranker_order_identical":True,
        "max_absolute_score_delta_between_passes":max_delta,
        "v3b":{
            "candidate_rows":1200,
            "ranking_method":"vector+reranker reciprocal-rank fusion",
            "rrf_k":60.0,
            "recommendation_rows":240,
        },
        "v4b":{
            "candidate_rows":len(v4_ranked),
            "ranking_method":"reranker_score_only",
            "exact_score_tie_breaks":["hybrid_retrieval_rank ascending","isbn13 ascending"],
            "recommendation_rows":240,
            "top10_candidate_source_counts":source_counts,
        },
        "execution_boundary":{
            "retrieval_calls_performed":0,
            "model_inference_passes":2,
            "judge_calls_performed":0,
            "human_relevance_labels_used":False,
            "release_metrics_calculated":False,
            "release_decision_produced":False,
        },
        "outputs":{
            FULL_SCORES.name:sha256(FULL_SCORES),
            V3_FULL.name:sha256(V3_FULL),
            V3_RECS.name:sha256(V3_RECS),
            V4_FULL.name:sha256(V4_FULL),
            V4_RECS.name:sha256(V4_RECS),
        }
    }
    METADATA.write_text(json.dumps(metadata,indent=2)+"\n",encoding="utf-8")

    lock = {
        "status":"V4H_V3B_V4B_RERANKING_FROZEN_BEFORE_DISPLAY_UNION_OR_JUDGING",
        "frozen_at_utc":datetime.now(timezone.utc).isoformat(),
        "query_set":"V4H01-V4H24",
        "query_count":24,
        "baseline":"v3b",
        "candidate":"v4b",
        "model_id":MODEL_ID,
        "model_revision":MODEL_REVISION,
        "reranking_contract_sha256":EXPECTED_CONTRACT_SHA256,
        "source_retrieval_lock_sha256":EXPECTED["retrieval_lock"],
        "source_dense_top50_sha256":EXPECTED["dense_top50"],
        "source_hybrid_union_sha256":EXPECTED["hybrid_union"],
        "full_union_reranker_scores_sha256":sha256(FULL_SCORES),
        "v3b_ranked_dense50_sha256":sha256(V3_FULL),
        "v3b_recommendations_sha256":sha256(V3_RECS),
        "v4b_ranked_full_union_sha256":sha256(V4_FULL),
        "v4b_recommendations_sha256":sha256(V4_RECS),
        "reranking_metadata_sha256":sha256(METADATA),
        "repeated_inference_passes":2,
        "full_union_reranker_order_identical":True,
        "dense_subset_reranker_order_identical":True,
        "retrieval_calls_performed":0,
        "judge_calls_performed":0,
        "human_relevance_labels_used":False,
        "release_metrics_calculated":False,
        "release_decision_produced":False,
        "next_stage":"FREEZE_UNIQUE_V3B_V4B_DISPLAYED_PAIR_UNION_BEFORE_JUDGING",
    }
    LOCK.write_text(json.dumps(lock,indent=2)+"\n",encoding="utf-8")

    print()
    print("PASS  frozen full-union cross-encoder scores captured")
    print("PASS  V3B dense-only ranking frozen as exact 24x10 displayed recommendations")
    print("PASS  V4B full-union ranking frozen as exact 24x10 displayed recommendations")
    print("PASS  reranking artifacts frozen before judging")
    print()
    print("V4H V3B/V4B RERANKING: PASS")
    print(f"Full-union scores SHA-256: {sha256(FULL_SCORES)}")
    print(f"V3B recommendations SHA-256: {sha256(V3_RECS)}")
    print(f"V4B recommendations SHA-256: {sha256(V4_RECS)}")
    print(f"Reranking lock SHA-256: {sha256(LOCK)}")
    print(f"V4B top10 candidate-source counts: {source_counts}")
    print("Retrieval calls performed: 0")
    print("Judge calls performed: 0")
    print("Release metrics calculated: NO")
    print("Next: freeze the unique V3B/V4B displayed-pair union before judging.")

if __name__ == "__main__":
    main()
