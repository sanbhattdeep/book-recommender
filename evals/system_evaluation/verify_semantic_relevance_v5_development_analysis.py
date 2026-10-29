from __future__ import annotations
import hashlib, json
from pathlib import Path
from typing import Any
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
EVALS = ROOT/"evals"
SYS = EVALS/"system_evaluation"
RUN = EVALS/"runs/semantic_relevance_v5_development/20261026T124615Z_candidate_collection"

CONTRACT = SYS/"semantic_relevance_v5_development_analysis_contract.v0.1.0.json"
DESIGN_SPEC = SYS/"semantic_relevance_v5_development_design.v0.1.0.json"
DESIGN_LOCK = SYS/"semantic_relevance_v5_development_design.v0.1.0.lock.json"
RERANK_LOCK = RUN/"v5d_factorial_reranking.lock.json"
LATENCY = RUN/"v5d_reranking_latency.csv"

RECS = {
    "B": RUN/"v5d_B_r0k0_recommendations.csv",
    "R": RUN/"v5d_R_r1k0_recommendations.csv",
    "K": RUN/"v5d_K_r0k1_recommendations.csv",
    "RK": RUN/"v5d_RK_r1k1_recommendations.csv",
}
SCORES = RUN/"v5d_judge_scores.csv"
SCORE_LOCK = RUN/"v5d_judge_scoring.lock.json"

EXPECTED = {
    "contract": "1d920bc842eb0dd540259620110e29706dcdca5427fa21ea4ae92daf03b753b5",
    "design_spec": "ef21af8375d3606cf53038f6986a8cd7e3c389fee8345770a39aff73d4cc4551",
    "design_lock": "89ee03a4328c268a8128d5e373854b92634c41eff204b53f7b93b78c481311c0",
    "rerank_lock": "085a3691ed7e0c8e324482798d5dc0c3a9d15f35c8d26fc37ef44bfd066f22f6",
    "B": "dba45cd656b760d21c1adc58d5e9f97b2407b4498335d9350bb819f2edcb1692",
    "R": "968ebd3dd0b92dd481d3362556daf8d547f02a63280ca8bea09657e56125bc86",
    "K": "c6aa093346b0907b03a6dc6fa730822985fa05ab0e79cb7c9c2a6abe53a61208",
    "RK": "7068a0ed477ad1954512448193f103da84408f335cdf81af6657e150ef49fd9a",
    "scores": "0f5c1453ed6f860ef31eb51de5094460a9b60559ad147202a50b7901ec422822",
    "score_lock": "977cf3db37eb3c0aefe6809567923a74672c6bcbeac8f432ff6368937e20ee9d",
}
EXPECTED_IDS=[f"V5D{i:02d}" for i in range(1,25)]
SYSTEMS=["B","R","K","RK"]

OUTPUT_NAMES = [
    *(f"v5d_{s}_query_metrics.csv" for s in SYSTEMS),
    *(f"v5d_{s}_aggregate_metrics.json" for s in SYSTEMS),
    *(f"v5d_{s}_bootstrap_confidence_intervals.json" for s in SYSTEMS),
    "v5d_factor_contrasts.json",
    "v5d_paired_query_deltas_by_stratum.csv",
    "v5d_interaction_diagnostics.json",
    "v5d_candidate_depth_diagnostics_status.json",
    "v5d_development_selection_result.json",
    "v5d_development_analysis_report.md",
    "v5d_development_analysis_metadata.json",
    "v5d_development_analysis_selection.lock.json",
]

def sha256(p:Path)->str:return hashlib.sha256(p.read_bytes()).hexdigest()
def lfsha(p:Path)->str:
    b=p.read_bytes();return hashlib.sha256(b.replace(b"\r\n",b"\n").replace(b"\r",b"\n")).hexdigest()
def req(c:bool,m:str)->None:
    if not c: raise ValueError(m)
def load(p:Path)->dict[str,Any]:
    req(p.exists(),f"Missing JSON input: {p}")
    return json.loads(p.read_text(encoding="utf-8-sig"))
def verify_raw(p:Path,e:str,l:str)->None:
    req(p.exists(),f"Missing {l}: {p}");a=sha256(p);req(a==e,f"{l} hash mismatch: expected={e} actual={a}");print(f"PASS  {l} raw SHA-256 verified")
def verify_text(p:Path,e:str,l:str)->None:
    req(p.exists(),f"Missing {l}: {p}");a=sha256(p);n=lfsha(p);req(a==e or n==e,f"{l} hash mismatch: expected={e} raw={a} lf={n}");print(f"PASS  {l} {'raw' if a==e else 'canonical-LF'} SHA-256 verified")

def validate():
    verify_raw(CONTRACT,EXPECTED["contract"],"V5D analysis contract")
    verify_text(DESIGN_SPEC,EXPECTED["design_spec"],"V5 development design spec")
    verify_raw(DESIGN_LOCK,EXPECTED["design_lock"],"V5 development design lock")
    verify_raw(RERANK_LOCK,EXPECTED["rerank_lock"],"V5D factorial reranking lock")
    for s in SYSTEMS: verify_raw(RECS[s],EXPECTED[s],f"{s} frozen recommendations")
    verify_raw(SCORES,EXPECTED["scores"],"V5D frozen judge scores")
    verify_raw(SCORE_LOCK,EXPECTED["score_lock"],"V5D judge scoring lock")

    spec=load(DESIGN_SPEC)
    lock=load(SCORE_LOCK)
    rlock=load(RERANK_LOCK)
    contract=load(CONTRACT)

    req(lock.get("status")=="V5D_JUDGE_SCORES_FROZEN_BEFORE_DEVELOPMENT_ANALYSIS","Unexpected V5D score-lock status")
    req(int(lock.get("case_count",-1))==452,"V5D score-lock case count changed")
    req(lock.get("judge_scores_sha256")==EXPECTED["scores"],"Score lock does not bind expected score file")
    req(lock.get("development_metrics_calculated") is False,"Score lock says development metrics already calculated")
    req(lock.get("development_selection_calculated") is False,"Score lock says selection already calculated")
    req(lock.get("release_metrics_calculated") is False,"Score lock says release metrics calculated")

    req(rlock.get("status")=="V5D_B_R_K_RK_RERANKING_FROZEN_BEFORE_JUDGING","Unexpected reranking lock status")
    req(rlock.get("systems")=={"B":"R0K0","R":"R1K0","K":"R0K1","RK":"R1K1"},"Factorial system identities changed")
    outputs=rlock.get("outputs",{})
    req("v5d_reranking_latency.csv" in outputs,"Reranking lock does not bind latency artifact")
    verify_raw(LATENCY,str(outputs["v5d_reranking_latency.csv"]),"frozen system latency artifact")

    sel=spec["development_selection_rule"]
    req(sel["candidate_set"]==SYSTEMS,"Selection candidate set changed")
    req(sel["ordered_criteria"]==[
        "lower catastrophic_query_rate",
        "lower irrelevant_rate_at_10",
        "higher clear_or_strong_rate_at_10",
        "higher macro_mean_relevance_at_10",
        "higher ndcg_at_10",
    ],"Selection criteria/order changed")
    req(sel["exact_tie_policy"]==[
        "if B is tied for best across every ordered criterion, retain B",
        "otherwise prefer fewer changed factors",
        "if R and K remain exactly tied, prefer lower median system inference latency measured on the same frozen V5D run excluding judge time",
        "if still exactly tied, make no V5 selection and require a separately preregistered tie-break evaluation"
    ],"Exact tie policy changed")
    boot=spec["evaluation"]["bootstrap"]
    req(int(boot["replicates"])==5000 and int(boot["seed"])==20261002,"Bootstrap contract changed")

    scores=pd.read_csv(SCORES,dtype=str,keep_default_na=False,encoding="utf-8")
    req(len(scores)==452,"V5D score file is not 452 rows")
    req(not scores.duplicated(["query_id","isbn13"]).any(),"Duplicate pair in V5D score file")
    vals=pd.to_numeric(scores["judge_score"],errors="raise").astype(int)
    req(vals.between(0,4).all(),"Judge score outside 0..4")
    req(vals.value_counts().sort_index().to_dict()=={0:373,1:13,2:56,3:10},"Frozen score-count distribution changed")

    recs={}
    all_score_pairs=set(zip(scores.query_id.astype(str),scores.isbn13.astype(str)))
    for s in SYSTEMS:
        f=pd.read_csv(RECS[s],dtype=str,keep_default_na=False,encoding="utf-8")
        req(len(f)==240,f"{s}: recommendations not 240 rows")
        req(sorted(f.query_id.unique())==EXPECTED_IDS,f"{s}: query IDs changed")
        f["rank_num"]=pd.to_numeric(f["rank"],errors="raise").astype(int)
        req(not f.duplicated(["query_id","rank_num"]).any(),f"{s}: duplicate query/rank")
        req(not f.duplicated(["query_id","isbn13"]).any(),f"{s}: duplicate query/book")
        for qid,g in f.groupby("query_id",sort=True):
            req(sorted(g.rank_num.tolist())==list(range(1,11)),f"{s}/{qid}: ranks are not 1..10")
        pairs=set(zip(f.query_id.astype(str),f.isbn13.astype(str)))
        req(pairs <= all_score_pairs,f"{s}: displayed pair missing frozen judge score")
        recs[s]=f

    lat=pd.read_csv(LATENCY,dtype=str,keep_default_na=False,encoding="utf-8")
    req(len(lat)==96,"Latency artifact is not 4 systems x 24 queries")
    req(set(lat.system)==set(SYSTEMS),"Latency artifact system set changed")
    lat["system_latency_seconds"]=pd.to_numeric(lat["system_latency_seconds"],errors="raise")
    req((lat["system_latency_seconds"]>=0).all(),"Negative system latency")
    req((lat.groupby("system").size()==24).all(),"Latency artifact is not 24 rows/system")

    for name in OUTPUT_NAMES:
        req(not (RUN/name).exists(),f"Refusing because V5D analysis output already exists: {RUN/name}")

    return recs,scores,lat,spec,contract

def main():
    print("Semantic relevance V5 - factorial development-analysis preflight")
    print("--------------------------------------------------------------")
    print("Judge/model/retrieval/reranker execution: DISABLED")
    print("Human relevance labels: NOT USED")
    print("Release gates/decision: NOT CALCULATED")
    print("Selection rule: FROZEN LEXICOGRAPHIC B/R/K/RK")
    print()
    _,scores,lat,_,_=validate()
    med=lat.groupby("system").system_latency_seconds.median().to_dict()
    print()
    print("V5D FACTORIAL DEVELOPMENT-ANALYSIS PREFLIGHT: PASS")
    print("Queries: 24")
    print("Frozen judge scores: 452")
    print("Score counts: {0: 373, 1: 13, 2: 56, 3: 10}")
    print("Systems: B=R0K0, R=R1K0, K=R0K1, RK=R1K1")
    print("Each displayed system: exact 24x10")
    print("Bootstrap: 5000 query-level replicates, seed 20261002")
    print("Selection: frozen five-criterion lexicographic rule")
    print("Median latency evidence available for exact R-vs-K tie only: YES")
    print("Candidate-depth diagnostics calculated in this step: NO")
    print("Judge/model calls performed during preflight: 0")
    print("Development selection calculated during preflight: NO")
    print("Release decision produced: NO")
    print("Next: calculate and freeze display metrics, factorial contrasts, and preregistered development selection.")

if __name__=="__main__": main()
