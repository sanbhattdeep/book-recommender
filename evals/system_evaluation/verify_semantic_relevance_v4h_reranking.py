from __future__ import annotations

import hashlib, json, subprocess
from pathlib import Path
from typing import Any
import pandas as pd

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


OUTPUTS = [
    RUN / "v4h_full_union_reranker_scores.csv",
    RUN / "v4h_v3b_ranked_dense50.csv",
    RUN / "v4h_v3b_recommendations.csv",
    RUN / "v4h_v4b_ranked_full_union.csv",
    RUN / "v4h_v4b_recommendations.csv",
    RUN / "v4h_reranking_metadata.json",
    RUN / "v4h_reranking.lock.json",
]

def main() -> None:
    print("Semantic relevance V4H - reranking preflight")
    print("-------------------------------------------")
    print("Cross-encoder inference: DISABLED")
    print("Retrieval calls: DISABLED")
    print("Judge execution: DISABLED")
    print("Human relevance labels: NOT USED")
    print("Release metrics/decision: NOT CALCULATED")
    print()

    dense, union = verify_frozen_boundary()
    for out in OUTPUTS:
        req(not out.exists(), f"Refusing preflight because reranking output already exists: {out}")

    contract = load_json(RERANKING_CONTRACT)
    req(contract.get("status") == "FROZEN_BY_PACKAGE_BEFORE_V4H_RERANKER_EXECUTION",
        "Unexpected reranking-contract status")
    req(contract.get("model", {}).get("id") == MODEL_ID, "Reranking contract model ID changed")
    req(contract.get("model", {}).get("revision") == MODEL_REVISION, "Reranking contract model revision changed")
    req(contract.get("model", {}).get("repeat_inference_passes") == 2,
        "Reranking repeat-inference contract changed")
    req(contract.get("v3b_ranking", {}).get("rrf_k") == 60.0, "V3B RRF k changed")
    req(contract.get("v4b_ranking", {}).get("method") == "reranker_score_only",
        "V4B final-ranking method changed")

    total_union = len(union)
    counts = union.groupby("query_id").size()
    print()
    print("V4H RERANKING PREFLIGHT: PASS")
    print(f"Dense rows for V3B: {len(dense)}")
    print(f"Full-union rows for V4B: {total_union}")
    print(f"V4B union size range: {int(counts.min())}..{int(counts.max())}")
    print(f"Model: {MODEL_ID}")
    print(f"Revision: {MODEL_REVISION}")
    print("Inference passes planned: 2")
    print("Cross-encoder inference performed: NO")
    print("Judge calls performed: 0")
    print("Release metrics calculated: NO")
    print("Next: score frozen full union twice, derive frozen V3B and V4B top10, and freeze before judging.")

if __name__ == "__main__":
    main()
