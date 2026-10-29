from __future__ import annotations
import hashlib,json
from pathlib import Path
from typing import Any
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]; E=ROOT/"evals"; S=E/"system_evaluation"
RUN=E/"runs/semantic_relevance_v5_holdout/20261028T055522Z_retrieval"
V5D=E/"runs/semantic_relevance_v5_development/20261026T124615Z_candidate_collection"
FILES={"analysis_contract":S/"semantic_relevance_v5_holdout_release_analysis_contract.v0.1.0.json",
"scores":RUN/"v5h_judge_scores.csv","scoring_lock":RUN/"v5h_judge_scoring.lock.json",
"B_recs":RUN/"v5h_B_r0k0_recommendations.csv","K_recs":RUN/"v5h_K_r0k1_recommendations.csv",
"rerank_lock":RUN/"v5h_bk_reranking.lock.json","display_union":RUN/"v5h_displayed_pair_union.csv",
"display_lock":RUN/"v5h_displayed_pair_union.lock.json","jinput_lock":RUN/"v5h_judging_input.lock.json",
"selection_lock":V5D/"v5d_development_analysis_selection.lock.json","authoring_lock":S/"semantic_relevance_v5_holdout_authoring.v0.1.0.lock.json"}
EXP={"analysis_contract":"e9e5a3b327e796982cc05b9fee16cbcfb0e2725dbdbc90d7d9726397abe46ac7","scores":"336e4374bbafdd50ebf7149d6edce00895bb072b8f4c258ec070c2ad6b530a83",
"scoring_lock":"35ea4964e039aae36efb92771180531125f5236ae4328e66f632ed10e03a6ae6",
"B_recs":"bc79777f7d096b8a5f327577513eeaacaa47d38a9c6ef9ecf93409a6c0a8369a",
"K_recs":"5f371bcfff0a700b9c1b6bfd831b8f83355f2199d5b78f704e5afe7a7dfbc2aa",
"rerank_lock":"21c5b52bb616fc1454bca40fca916bbe70dfb5277761ff6a41b513b161445797",
"display_union":"13ff542ecfe5316484b62eb0f1b406f9f8f8dd4a7491acc369128a5285d6ff67",
"display_lock":"0a9e7981cb20584b0074af58cd9b7a2e7a27a6689c75a6f685db7bb3617c96b0",
"jinput_lock":"a2ff301aef1788526d51858394ee4249eeb548fc05981e0246a50a2a8643eea7",
"selection_lock":"e5fb7768c0940a8e699c7fc26186eeacfb938b811851adc6a34d1e920e0a519b",
"authoring_lock":"e4836060ba90e49fbf4da05bf12a15246c312bd7926e48ff72d3766f74613751"}
SYSTEM=S/"semantic_relevance_system_eval.v1.0.0.json"
GATES={'macro_mean_relevance_at_10_ge_2_50': ('macro_mean_relevance_at_10', '>=', 2.5), 'macro_mean_relevance_ci_lower_ge_2_00': ('macro_mean_relevance_at_10_ci_lower_95', '>=', 2.0), 'clear_or_strong_rate_at_10_ge_0_60': ('clear_or_strong_rate_at_10', '>=', 0.6), 'irrelevant_rate_at_10_le_0_10': ('irrelevant_rate_at_10', '<=', 0.1), 'hit_rate_at_5_clear_or_strong_ge_0_90': ('hit_rate_at_5_clear_or_strong', '>=', 0.9), 'top1_clear_or_strong_rate_ge_0_75': ('top1_clear_or_strong_rate', '>=', 0.75), 'ndcg_at_10_ge_0_85': ('ndcg_at_10', '>=', 0.85), 'catastrophic_query_rate_eq_0': ('catastrophic_query_rate', '==', 0.0)}
IDS=[f"V5H{i:02d}" for i in range(1,25)]
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def req(c,m):
    if not c: raise ValueError(m)
def js(p): req(p.exists(),f"Missing {p}"); return json.loads(p.read_text(encoding="utf-8-sig"))
def vr(k):
    p=FILES[k]; req(p.exists(),f"Missing {k}: {p}"); a=sha(p); req(a==EXP[k],f"{k} SHA mismatch expected={EXP[k]} actual={a}"); print(f"PASS  {k} raw SHA-256 verified")
def bv(v:Any):
    s=str(v).strip().lower()
    if s in {"true","1","yes"}: return True
    if s in {"false","0","no",""}: return False
    raise ValueError(f"bad bool {v!r}")
def verify_gate_contract():
    c=js(SYSTEM); st=c.get("statistical_uncertainty",{})
    req(int(st.get("bootstrap_replicates",-1))==5000,"bootstrap replicate count changed")
    req(int(st.get("seed",-1))==20261002,"bootstrap seed changed")
    gs=c.get("release_gates",{}); req(list(gs)==list(GATES),"release gate names/order changed")
    for n,(m,o,v) in GATES.items():
        g=gs[n]; req((g.get("metric"),g.get("op"),float(g.get("value")))==(m,o,float(v)),f"gate changed {n}")
    print("PASS  original eight release gates verified")
    print("PASS  bootstrap 5000 / seed 20261002 verified")
    return c
def validate():
    for k in FILES: vr(k)
    l=js(FILES["scoring_lock"])
    req(l.get("status")=="V5H_JUDGE_SCORES_FROZEN_BEFORE_RELEASE_ANALYSIS","bad score-lock status")
    req(int(l.get("case_count",-1))==359,"case count changed"); req(l.get("judge_candidate")=="semantic_relevance_v0.29.0-r5","judge changed")
    req(l.get("judge_scores_sha256")==EXP["scores"],"score hash binding changed")
    req(l.get("historical_score_reuse_performed") is False,"historical reuse occurred")
    req(int(l.get("fresh_judge_cases_completed",-1))==359,"fresh count changed")
    req(l.get("human_relevance_labels_used") is False,"human labels used")
    req(l.get("holdout_metrics_calculated") is False and l.get("release_gates_calculated") is False and l.get("release_decision_produced") is False,"release analysis already ran during judging")
    # The exact V5D development-selection lock is already verified above by
    # its frozen SHA-256. Avoid coupling this preflight to historical field
    # naming inside that immutable lock. The identity actually carried into
    # V5H is independently frozen in the V5H authoring lock below.
    print("PASS  frozen V5D development-selection lock hash bound")
    al=js(FILES["authoring_lock"]); req(al.get("status")=="V5H_QUERY_AND_FACET_SPEC_FROZEN_BEFORE_HOLDOUT_RETRIEVAL","bad V5H authoring lock")
    req(al.get("selected_development_candidate")=="K" and al.get("selected_system_identity")=="R0K1","V5H selected system changed")
    print("PASS  selected holdout identity K=R0K1 verified from frozen V5H authoring lock")
    d=pd.read_csv(FILES["scores"],dtype=str,keep_default_na=False,encoding="utf-8")
    need={"case_id","query_id","query","stratum","isbn13","in_B","B_rank","in_K","K_rank","judge_score"}
    req(not(need-set(d.columns)),f"missing columns {sorted(need-set(d.columns))}"); req(len(d)==359,"expected 359 rows")
    req(not d["case_id"].duplicated().any() and not d.duplicated(["query_id","isbn13"]).any(),"duplicate score rows")
    req(sorted(d["query_id"].unique())==IDS,"query IDs changed"); d["judge_score"]=pd.to_numeric(d["judge_score"],errors="raise").astype(int)
    req(d["judge_score"].value_counts().sort_index().to_dict()=={0:313,1:8,2:32,3:6},"score distribution changed")
    for mem,rank,label in [("in_B","B_rank","B"),("in_K","K_rank","K")]:
        x=d[d[mem].map(bv)].copy(); req(len(x)==240,f"{label} expected 240"); x["rank"]=pd.to_numeric(x[rank],errors="raise").astype(int)
        req(not x.duplicated(["query_id","rank"]).any(),f"{label} duplicate rank")
        for q,g in x.groupby("query_id"): req(sorted(g["rank"])==list(range(1,11)),f"{label}/{q} bad ranks")
    print("PASS  frozen score structure reconstructs B and K as exact 24x10")
    verify_gate_contract()
    req(not (RUN/"v5h_final_release_analysis.lock.json").exists(),"release analysis already frozen")
    return d
def main():
    print("Semantic relevance V5H - FINAL RELEASE-ANALYSIS PREFLIGHT")
    print("Authoritative system: K=R0K1"); print("B=R0K0 role: SECONDARY")
    print("Judge/model/retrieval/reranker calls: DISABLED"); print("Release metrics/decision: NOT CALCULATED")
    d=validate(); print(); print("V5H FINAL RELEASE-ANALYSIS PREFLIGHT: PASS")
    print(f"Frozen unique scored pairs: {len(d)}"); print("B displayed rows: 240"); print("K displayed rows: 240")
    print("Holdout queries: 24"); print("Bootstrap: 5000 query-cluster replicates / seed 20261002")
    print("Authoritative release system: K=R0K1"); print("B-vs-K paired role: SECONDARY / CANNOT OVERRIDE")
    print("Judge/model/retrieval/reranker calls performed: 0"); print("Human relevance labels used: NO")
    print("Release metrics calculated during preflight: NO"); print("Release decision produced during preflight: NO")
    print("Next: calculate standalone K release metrics and secondary B-vs-K evidence.")
if __name__=="__main__": main()
