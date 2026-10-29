from __future__ import annotations
import hashlib,json
from datetime import datetime,timezone
from pathlib import Path
import numpy as np,pandas as pd
import verify_semantic_relevance_v5_holdout_release_analysis as b
RUN=b.RUN; N=24; D=10; REPS=5000; SEED=20261002; EXPECTED_VERIFIER="c6309e7bbe01d368f573c33b36b87f003200f7552a9e3869afc795918beca8ba"
PAIR={"macro_mean_relevance_at_10":("mean_relevance_at_10",True),"clear_or_strong_rate_at_10":("clear_or_strong_rate_at_10",True),"irrelevant_rate_at_10":("irrelevant_rate_at_10",False),"hit_rate_at_5_clear_or_strong":("hit_at_5_clear_or_strong",True),"top1_clear_or_strong_rate":("top1_clear_or_strong",True),"mrr_at_10_clear_or_strong":("mrr_at_10_clear_or_strong",True),"ndcg_at_10":("ndcg_at_10",True),"catastrophic_query_rate":("catastrophic_query",False)}
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def lf(p): return hashlib.sha256(p.read_bytes().replace(b"\r\n",b"\n").replace(b"\r",b"\n")).hexdigest()
def req(c,m):
    if not c: raise ValueError(m)
def ndcg(s):
    g=np.array([2**int(x)-1 for x in s],float); d=np.log2(np.arange(2,len(s)+2,dtype=float)); dc=float(np.sum(g/d)); i=np.sort(g)[::-1]; z=float(np.sum(i/d)); return 0.0 if z==0 else dc/z
def rec(df,mem,rank,label):
    x=df[df[mem].map(b.bv)].copy(); x["rank"]=pd.to_numeric(x[rank]).astype(int); x["judge_score"]=pd.to_numeric(x["judge_score"]).astype(int)
    req(len(x)==240,f"{label} rows"); return x
def qs(g):
    g=g.sort_values("rank"); s=g["judge_score"].astype(int).tolist(); c=np.array(s)>=3; z=np.array(s)==0; rs=[i+1 for i,v in enumerate(c) if v]; first=min(rs) if rs else None
    return {"query_id":str(g["query_id"].iloc[0]),"query":str(g["query"].iloc[0]),"stratum":str(g["stratum"].iloc[0]),"mean_relevance_at_10":float(np.mean(s)),"median_relevance_at_10":float(np.median(s)),"clear_or_strong_rate_at_10":float(np.mean(c)),"irrelevant_rate_at_10":float(np.mean(z)),"hit_at_5_clear_or_strong":int(np.any(c[:5])),"top1_clear_or_strong":int(bool(c[0])),"first_clear_or_strong_rank":first,"mrr_at_10_clear_or_strong":0.0 if first is None else 1.0/first,"ndcg_at_10":ndcg(s),"catastrophic_query":int(float(np.mean(s))<1.5)}
def agg(q):
    return {"macro_mean_relevance_at_10":float(q["mean_relevance_at_10"].mean()),"clear_or_strong_rate_at_10":float(q["clear_or_strong_rate_at_10"].mean()),"irrelevant_rate_at_10":float(q["irrelevant_rate_at_10"].mean()),"hit_rate_at_5_clear_or_strong":float(q["hit_at_5_clear_or_strong"].mean()),"top1_clear_or_strong_rate":float(q["top1_clear_or_strong"].mean()),"mrr_at_10_clear_or_strong":float(q["mrr_at_10_clear_or_strong"].mean()),"ndcg_at_10":float(q["ndcg_at_10"].mean()),"catastrophic_query_rate":float(q["catastrophic_query"].mean())}
def boot(q):
    cols={"macro_mean_relevance_at_10":"mean_relevance_at_10","clear_or_strong_rate_at_10":"clear_or_strong_rate_at_10","irrelevant_rate_at_10":"irrelevant_rate_at_10","hit_rate_at_5_clear_or_strong":"hit_at_5_clear_or_strong","top1_clear_or_strong_rate":"top1_clear_or_strong","mrr_at_10_clear_or_strong":"mrr_at_10_clear_or_strong","ndcg_at_10":"ndcg_at_10"}
    rng=np.random.default_rng(SEED); idx=rng.integers(0,len(q),size=(REPS,len(q))); out={}
    for m,c in cols.items():
        v=q[c].astype(float).to_numpy(); dist=np.mean(v[idx],axis=1); out[m]={"lower_95":float(np.quantile(dist,.025)),"upper_95":float(np.quantile(dist,.975)),"bootstrap_mean":float(np.mean(dist)),"bootstrap_standard_error":float(np.std(dist,ddof=1))}
    return out
def standalone(x):
    q=pd.DataFrame([qs(g) for _,g in x.groupby("query_id",sort=True)]); a=agg(q); ci=boot(q); a["macro_mean_relevance_at_10_ci_lower_95"]=ci["macro_mean_relevance_at_10"]["lower_95"]; a["macro_mean_relevance_at_10_ci_upper_95"]=ci["macro_mean_relevance_at_10"]["upper_95"]; return q,a,ci
def gates(a,c):
    checks={}; failed=[]
    for n,g in c["release_gates"].items():
        actual=float(a[g["metric"]]); th=float(g["value"]); op=g["op"]; p=actual>=th if op==">=" else actual<=th if op=="<=" else actual==th
        checks[n]={"passed":bool(p),"metric":g["metric"],"actual":actual,"operator":op,"threshold":th}
        if not p: failed.append(n)
    cat=float(a["catastrophic_query_rate"]); decision="PASS" if not failed else ("REVIEW" if len(failed)<=2 and cat==0 else "FAIL")
    return {"decision":decision,"passed_gate_count":len(checks)-len(failed),"total_gate_count":8,"failed_gate_count":len(failed),"failed_gates":failed,"checks":checks,"decision_policy":c["decision_policy"]}
def paired(B,K):
    rng=np.random.default_rng(SEED); idx=rng.integers(0,N,size=(REPS,N)); metrics={}; rows=[]
    for m,(col,hib) in PAIR.items():
        x=B[col].astype(float).to_numpy(); y=K[col].astype(float).to_numpy(); d=y-x; imp=d if hib else -d; dist=np.mean(d[idx],axis=1)
        metrics[m]={"B":float(np.mean(x)),"K":float(np.mean(y)),"raw_delta_K_minus_B":float(np.mean(d)),"paired_ci_lower":float(np.quantile(dist,.025)),"paired_ci_upper":float(np.quantile(dist,.975)),"wins_for_K":int(np.sum(imp>1e-15)),"ties":int(np.sum(np.abs(imp)<=1e-15)),"losses_for_K":int(np.sum(imp<-1e-15)),"higher_is_better":hib}
        for i,q in enumerate(B["query_id"].astype(str)): rows.append({"query_id":q,"query":str(B.iloc[i]["query"]),"stratum":str(B.iloc[i]["stratum"]),"metric":m,"B_value":float(x[i]),"K_value":float(y[i]),"raw_delta_K_minus_B":float(d[i]),"outcome_for_K":"WIN" if imp[i]>1e-15 else "LOSS" if imp[i]<-1e-15 else "TIE"})
    return {"status":"COMPLETE","analysis_role":"secondary","cannot_override_standalone_K_release_decision":True,"matched_query_count":24,"bootstrap_replicates":REPS,"bootstrap_seed":SEED,"metrics":metrics},pd.DataFrame(rows)
def main():
    vp=Path(b.__file__).resolve(); req(sha(vp)==EXPECTED_VERIFIER or lf(vp)==EXPECTED_VERIFIER,"verifier changed")
    print("Semantic relevance V5H - FINAL K release analysis"); print("Authoritative system: K=R0K1"); print("B comparison: SECONDARY"); print("Judge/model/retrieval/reranker calls: 0")
    df=b.validate(); sc=b.verify_gate_contract()
    outs=["v5h_B_query_metrics.csv","v5h_K_query_metrics.csv","v5h_B_aggregate_metrics.json","v5h_K_aggregate_metrics.json","v5h_B_bootstrap_confidence_intervals.json","v5h_K_bootstrap_confidence_intervals.json","v5h_B_release_gate_result.json","v5h_K_release_gate_result.json","v5h_B_vs_K_query_deltas.csv","v5h_B_vs_K_paired_comparison.json","v5h_stratum_metrics.csv","v5h_final_release_report.md","v5h_final_release_analysis_metadata.json","v5h_final_release_analysis.lock.json"]
    for n in outs: req(not (RUN/n).exists(),f"refuse overwrite {n}")
    B=rec(df,"in_B","B_rank","B"); K=rec(df,"in_K","K_rank","K"); Bq,Ba,Bci=standalone(B); Kq,Ka,Kci=standalone(K); Bg=gates(Ba,sc); Kg=gates(Ka,sc); pp,dr=paired(Bq,Kq)
    Bq.to_csv(RUN/"v5h_B_query_metrics.csv",index=False); Kq.to_csv(RUN/"v5h_K_query_metrics.csv",index=False)
    for n,x in [("v5h_B_aggregate_metrics.json",Ba),("v5h_K_aggregate_metrics.json",Ka),("v5h_B_bootstrap_confidence_intervals.json",Bci),("v5h_K_bootstrap_confidence_intervals.json",Kci),("v5h_B_release_gate_result.json",Bg),("v5h_K_release_gate_result.json",Kg),("v5h_B_vs_K_paired_comparison.json",pp)]: (RUN/n).write_text(json.dumps(x,indent=2)+"\n",encoding="utf-8")
    dr.to_csv(RUN/"v5h_B_vs_K_query_deltas.csv",index=False)
    sr=[]
    for sys,q in [("B",Bq),("K",Kq)]:
        for st,g in q.groupby("stratum",sort=True): sr.append({"system":sys,"stratum":st,"query_count":len(g),"mean_relevance_at_10":float(g["mean_relevance_at_10"].mean()),"clear_or_strong_rate_at_10":float(g["clear_or_strong_rate_at_10"].mean()),"irrelevant_rate_at_10":float(g["irrelevant_rate_at_10"].mean()),"hit_rate_at_5_clear_or_strong":float(g["hit_at_5_clear_or_strong"].mean()),"top1_clear_or_strong_rate":float(g["top1_clear_or_strong"].mean()),"mrr_at_10_clear_or_strong":float(g["mrr_at_10_clear_or_strong"].mean()),"ndcg_at_10":float(g["ndcg_at_10"].mean()),"catastrophic_query_rate":float(g["catastrophic_query"].mean()),"role":"diagnostic_only_not_release_gating"})
    pd.DataFrame(sr).to_csv(RUN/"v5h_stratum_metrics.csv",index=False)
    report=["# Semantic Relevance V5 — Final Unseen V5H Release Report","",f"**Authoritative standalone K=R0K1 decision: {Kg['decision']} ({Kg['passed_gate_count']}/8 gates)**","","K is authoritative. B and paired B-vs-K are secondary and cannot override K.","","## K release gates",""]
    for n,c in Kg["checks"].items(): report.append(f"- {'PASS' if c['passed'] else 'FAIL'} `{n}`: actual={c['actual']:.6f} {c['operator']} {c['threshold']:.6f}")
    report += ["","## Standalone summary","",f"- B: Mean@10={Ba['macro_mean_relevance_at_10']:.3f}, Clear={Ba['clear_or_strong_rate_at_10']:.1%}, Irrelevant={Ba['irrelevant_rate_at_10']:.1%}, Hit@5={Ba['hit_rate_at_5_clear_or_strong']:.1%}, Top1={Ba['top1_clear_or_strong_rate']:.1%}, MRR={Ba['mrr_at_10_clear_or_strong']:.3f}, nDCG={Ba['ndcg_at_10']:.3f}, Cat={Ba['catastrophic_query_rate']:.1%}",f"- K: Mean@10={Ka['macro_mean_relevance_at_10']:.3f}, Clear={Ka['clear_or_strong_rate_at_10']:.1%}, Irrelevant={Ka['irrelevant_rate_at_10']:.1%}, Hit@5={Ka['hit_rate_at_5_clear_or_strong']:.1%}, Top1={Ka['top1_clear_or_strong_rate']:.1%}, MRR={Ka['mrr_at_10_clear_or_strong']:.3f}, nDCG={Ka['ndcg_at_10']:.3f}, Cat={Ka['catastrophic_query_rate']:.1%}","",f"- B secondary gate result: {Bg['decision']} ({Bg['passed_gate_count']}/8)",f"- K authoritative gate result: {Kg['decision']} ({Kg['passed_gate_count']}/8)","","5,000 query-cluster bootstrap replicates; seed 20261002."]
    (RUN/"v5h_final_release_report.md").write_text("\n".join(report)+"\n",encoding="utf-8")
    meta={"status":"V5H_FINAL_RELEASE_ANALYSIS_COMPLETE","completed_at_utc":datetime.now(timezone.utc).isoformat(),"query_count":24,"judge_scores_sha256":b.EXP["scores"],"judge_scoring_lock_sha256":b.EXP["scoring_lock"],"bootstrap_replicates":REPS,"bootstrap_seed":SEED,"selected_development_candidate":"K","authoritative_system":"K=R0K1","K_final_release_decision":Kg["decision"],"K_passed_gates":Kg["passed_gate_count"],"B_secondary_gate_result":Bg["decision"],"paired_comparison_role":"secondary","judge_calls_performed_during_analysis":0,"model_calls_performed_during_analysis":0,"retrieval_calls_performed_during_analysis":0,"reranker_calls_performed_during_analysis":0,"human_relevance_labels_used":False}
    mp=RUN/"v5h_final_release_analysis_metadata.json"; mp.write_text(json.dumps(meta,indent=2)+"\n",encoding="utf-8")
    names=[x for x in outs if not x.endswith(".lock.json")]
    lock={"status":"V5H_FINAL_RELEASE_ANALYSIS_FROZEN","frozen_at_utc":datetime.now(timezone.utc).isoformat(),"judge_scores_sha256":b.EXP["scores"],"judge_scoring_lock_sha256":b.EXP["scoring_lock"],"release_analysis_contract_sha256":b.EXP["analysis_contract"],"selected_development_candidate":"K","authoritative_system":"K=R0K1","K_final_release_decision":Kg["decision"],"K_passed_gates":Kg["passed_gate_count"],"K_total_gates":8,"B_comparison_role":"secondary","paired_comparison_cannot_override_K_decision":True,"bootstrap_replicates":REPS,"bootstrap_seed":SEED,"judge_calls_performed_during_analysis":0,"model_calls_performed_during_analysis":0,"retrieval_calls_performed_during_analysis":0,"reranker_calls_performed_during_analysis":0,"human_relevance_labels_used":False,"outputs":{n:sha(RUN/n) for n in names}}
    lp=RUN/"v5h_final_release_analysis.lock.json"; lp.write_text(json.dumps(lock,indent=2)+"\n",encoding="utf-8")
    print(); print("B=R0K0 SECONDARY STANDALONE ON V5H"); print(f"Mean@10={Ba['macro_mean_relevance_at_10']:.3f} CI=[{Bci['macro_mean_relevance_at_10']['lower_95']:.3f},{Bci['macro_mean_relevance_at_10']['upper_95']:.3f}] Clear={Ba['clear_or_strong_rate_at_10']:.1%} Irrelevant={Ba['irrelevant_rate_at_10']:.1%} Hit@5={Ba['hit_rate_at_5_clear_or_strong']:.1%} Top1={Ba['top1_clear_or_strong_rate']:.1%} MRR={Ba['mrr_at_10_clear_or_strong']:.3f} nDCG={Ba['ndcg_at_10']:.3f} Cat={Ba['catastrophic_query_rate']:.1%}")
    print(f"Secondary gate result={Bg['decision']} ({Bg['passed_gate_count']}/8)")
    print(); print("K=R0K1 AUTHORITATIVE FINAL UNSEEN V5H"); print(f"Mean@10={Ka['macro_mean_relevance_at_10']:.3f} CI=[{Kci['macro_mean_relevance_at_10']['lower_95']:.3f},{Kci['macro_mean_relevance_at_10']['upper_95']:.3f}] Clear={Ka['clear_or_strong_rate_at_10']:.1%} Irrelevant={Ka['irrelevant_rate_at_10']:.1%} Hit@5={Ka['hit_rate_at_5_clear_or_strong']:.1%} Top1={Ka['top1_clear_or_strong_rate']:.1%} MRR={Ka['mrr_at_10_clear_or_strong']:.3f} nDCG={Ka['ndcg_at_10']:.3f} Cat={Ka['catastrophic_query_rate']:.1%}")
    print(f"FINAL K RELEASE DECISION={Kg['decision']} ({Kg['passed_gate_count']}/8 gates)")
    print(); print("Judge/model/retrieval/reranker calls during analysis: 0"); print("Human relevance labels used: NO"); print(f"Final analysis lock SHA-256: {sha(lp)}"); print("V5H FINAL RELEASE ANALYSIS: PASS")
if __name__=="__main__": main()
