from __future__ import annotations

import gc
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SYS = ROOT / "evals/system_evaluation"
sys.path.insert(0, str(SYS))

import verify_semantic_relevance_v5_r1_k1_model_qualification as boundary
from semantic_relevance_v5_r1_retrieval import (
    R1DenseIndex, R1DenseRuntimeConfig, encode_texts, load_dense_encoder,
)
from semantic_relevance_v5_k1_reranker import (
    K1RuntimeConfig, candidate_text, load_reranker, rank_k1, score_query_candidate_pairs,
)

CONTRACT = boundary.CONTRACT
RESULTS = boundary.RESULTS
QUAL_LOCK = boundary.QUAL_LOCK
EXPECTED_VERIFY_SHA256 = "08d66428303f7a41062636642cb2c20a99ed3267476dc18e1f36bee7c19c9f43"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def req(cond: bool, msg: str) -> None:
    if not cond: raise ValueError(msg)


def snapshot_commit(repo_id: str, revision: str) -> tuple[str,str]:
    from huggingface_hub import snapshot_download
    path=Path(snapshot_download(repo_id=repo_id, revision=revision, local_files_only=True))
    commit=path.name
    req(commit==revision, f"Resolved snapshot commit mismatch for {repo_id}: expected={revision} actual={commit} path={path}")
    return str(path), commit


def rank_dense(ids: list[str], doc_embeddings: np.ndarray, query_embeddings: np.ndarray, qids: list[str]) -> tuple[dict[str,list[str]],dict[str,list[float]]]:
    index=R1DenseIndex(tuple(ids), doc_embeddings)
    signatures={}; scores={}
    for i,qid in enumerate(qids):
        frame=index.rank_query_embedding(query_embeddings[i], top_k=50)
        signatures[qid]=frame["isbn13"].astype(str).tolist()
        scores[qid]=[float(x) for x in frame["dense_score"].tolist()]
    return signatures,scores


def max_ranked_score_delta(a: dict[str,list[float]], b: dict[str,list[float]]) -> float:
    vals=[]
    for qid in a:
        req(len(a[qid])==len(b[qid]), f"Score vector length changed for {qid}")
        vals.extend(abs(x-y) for x,y in zip(a[qid],b[qid]))
    return max(vals) if vals else 0.0


def main() -> None:
    print("Semantic relevance V5 - R1/K1 pinned-model qualification")
    print("--------------------------------------------------------")
    print("Probe role: fixed synthetic non-V5D mechanical qualification only")
    print("Development selection: DISABLED")
    print("Judge execution: DISABLED")
    print("Release analysis: DISABLED")
    print()

    req(sha256(boundary.Path(__file__).with_name("verify_semantic_relevance_v5_r1_k1_model_qualification.py"))==EXPECTED_VERIFY_SHA256,
        "Qualification verifier source changed after package freeze")
    c=boundary.verify_boundary(require_outputs_absent=True)
    queries=c["probes"]["queries"]; docs=c["probes"]["documents"]
    qids=[q["probe_query_id"] for q in queries]
    qtexts=[q["text"] for q in queries]
    ids=[d["isbn13"] for d in docs]
    doc_texts=[f"Title: {d['title']}\nDescription: {d['description']}" for d in docs]

    import torch
    print()
    print("R1 qualification: loading pinned BAAI/bge-m3 revision...")
    r1_model,r1_device=load_dense_encoder(R1DenseRuntimeConfig(device="auto"))
    r1_snapshot,r1_commit=snapshot_commit("BAAI/bge-m3","5617a9f61b028005a4858fdac845db406aefb181")
    print(f"PASS  R1 snapshot resolved to pinned commit {r1_commit}")
    print(f"INFO  R1 device: {r1_device}")
    d1=encode_texts(r1_model,doc_texts); q1=encode_texts(r1_model,qtexts)
    d2=encode_texts(r1_model,doc_texts); q2=encode_texts(r1_model,qtexts)
    emb_delta=float(max(np.max(np.abs(d1-d2)),np.max(np.abs(q1-q2))))
    s1,sc1=rank_dense(ids,d1,q1,qids); s2,sc2=rank_dense(ids,d2,q2,qids)
    score_delta=max_ranked_score_delta(sc1,sc2)
    req(s1==s2, f"R1 dense rank signature changed across repeated passes: first={s1} second={s2}")
    req(emb_delta <= float(c["r1"]["max_embedding_abs_delta"]), f"R1 embedding repeatability delta exceeded tolerance: {emb_delta}")
    req(score_delta <= float(c["r1"]["max_similarity_score_abs_delta"]), f"R1 similarity repeatability delta exceeded tolerance: {score_delta}")
    print(f"PASS  R1 repeated rank signatures exact across {len(qids)} probes")
    print(f"PASS  R1 max embedding abs delta={emb_delta:.9g}")
    print(f"PASS  R1 max similarity-score abs delta={score_delta:.9g}")
    del r1_model,d1,d2,q1,q2
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    print()
    print("K1 qualification: loading pinned BAAI/bge-reranker-v2-m3 revision...")
    tok,k1_model,k1_device=load_reranker(K1RuntimeConfig(device="auto"))
    k1_snapshot,k1_commit=snapshot_commit("BAAI/bge-reranker-v2-m3","953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e")
    print(f"PASS  K1 snapshot resolved to pinned commit {k1_commit}")
    print(f"INFO  K1 device: {k1_device}")
    pair_queries=[]; pair_candidates=[]; rows=[]
    for q in queries:
        for rank,d in enumerate(docs,start=1):
            pair_queries.append(q["text"])
            pair_candidates.append(candidate_text(d["title"],d["description"]))
            rows.append({"query_id":q["probe_query_id"],"isbn13":d["isbn13"],"hybrid_retrieval_rank":rank})
    kscore1=score_query_candidate_pairs(queries=pair_queries,candidates=pair_candidates,tokenizer=tok,model=k1_model,device=k1_device)
    kscore2=score_query_candidate_pairs(queries=pair_queries,candidates=pair_candidates,tokenizer=tok,model=k1_model,device=k1_device)
    kdelta=float(np.max(np.abs(kscore1-kscore2)))
    f1=pd.DataFrame(rows); f1["reranker_score"]=kscore1
    f2=pd.DataFrame(rows); f2["reranker_score"]=kscore2
    rk1=rank_k1(f1); rk2=rank_k1(f2)
    sig1={qid:g.sort_values("k1_rank")["isbn13"].astype(str).tolist() for qid,g in rk1.groupby("query_id",sort=False)}
    sig2={qid:g.sort_values("k1_rank")["isbn13"].astype(str).tolist() for qid,g in rk2.groupby("query_id",sort=False)}
    req(sig1==sig2, f"K1 reranker rank signature changed across repeated passes: first={sig1} second={sig2}")
    req(kdelta <= float(c["k1"]["max_score_abs_delta"]), f"K1 score repeatability delta exceeded tolerance: {kdelta}")
    print(f"PASS  K1 repeated rank signatures exact across {len(qids)} probes")
    print(f"PASS  K1 max score abs delta={kdelta:.9g}")

    results={
        "status":"PASS",
        "qualified_at_utc":datetime.now(timezone.utc).isoformat(),
        "role":"fixed_synthetic_non_v5d_model_qualification_only",
        "contract_sha256":boundary.EXPECTED["contract"],
        "implementation_lock_sha256":boundary.EXPECTED["impl_lock"],
        "probe_query_count":len(queries),
        "probe_document_count":len(docs),
        "r1":{
            "model_id":"BAAI/bge-m3","model_revision":r1_commit,"snapshot_path":r1_snapshot,"device":r1_device,
            "max_embedding_abs_delta":emb_delta,"max_similarity_score_abs_delta":score_delta,"rank_signatures":s1,
        },
        "k1":{
            "model_id":"BAAI/bge-reranker-v2-m3","model_revision":k1_commit,"snapshot_path":k1_snapshot,"device":k1_device,
            "max_score_abs_delta":kdelta,"rank_signatures":sig1,
        },
        "environment":{
            "python":sys.version,"platform":platform.platform(),"torch":torch.__version__,
            "cuda_available":bool(torch.cuda.is_available()),"cuda_version":torch.version.cuda,
            "gpu_name":torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "semantic_quality_assertions_performed":False,
        "catalog_retrieval_performed":False,
        "bm25_retrieval_performed":False,
        "judge_calls_performed":0,
        "v5d_authored_or_executed":False,
        "development_selection_calculated":False,
        "release_metrics_calculated":False,
    }
    RESULTS.write_text(json.dumps(results,indent=2)+"\n",encoding="utf-8")
    lock={
        "status":"V5_R1_K1_PINNED_MODELS_QUALIFIED_BEFORE_V5D_AUTHORING",
        "frozen_at_utc":datetime.now(timezone.utc).isoformat(),
        "contract_sha256":boundary.EXPECTED["contract"],
        "implementation_lock_sha256":boundary.EXPECTED["impl_lock"],
        "results_sha256":sha256(RESULTS),
        "r1_model_id":"BAAI/bge-m3","r1_model_revision":r1_commit,
        "k1_model_id":"BAAI/bge-reranker-v2-m3","k1_model_revision":k1_commit,
        "r1_rank_repeatability_passed":True,"k1_rank_repeatability_passed":True,
        "semantic_quality_assertions_performed":False,
        "catalog_retrieval_performed":False,"bm25_retrieval_performed":False,
        "judge_calls_performed":0,"v5d_authored_or_executed":False,
        "development_selection_calculated":False,"release_metrics_calculated":False,
        "next_stage":"AUTHOR_AND_FREEZE_FRESH_V5D24_QUERY_AND_FACET_SET_BEFORE_ANY_DEVELOPMENT_RETRIEVAL",
    }
    QUAL_LOCK.write_text(json.dumps(lock,indent=2)+"\n",encoding="utf-8")
    print()
    print("V5 R1/K1 PINNED-MODEL QUALIFICATION: PASS")
    print(f"Results SHA-256: {sha256(RESULTS)}")
    print(f"Qualification lock SHA-256: {sha256(QUAL_LOCK)}")
    print("Semantic quality assertions performed: NO")
    print("Catalog/BM25 retrieval performed: NO")
    print("Judge calls performed: 0")
    print("V5D authored/executed: NO")
    print("Next: author and freeze the fresh V5D24 query/facet set before any development retrieval.")

if __name__=="__main__": main()
