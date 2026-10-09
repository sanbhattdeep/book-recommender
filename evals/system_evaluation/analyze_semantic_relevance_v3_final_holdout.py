from __future__ import annotations
import hashlib, json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
E = ROOT / "evals"
RUN = E / "runs/semantic_relevance_v3_holdout/20261007T124150Z_retrieval"
SCORES = RUN / "holdout_judge_scores.csv"
SLOCK = RUN / "holdout_judge_scoring_lock.json"
UNION = RUN / "holdout_displayed_pair_union.csv"
ULOCK = RUN / "holdout_displayed_pair_union_lock.json"
V2REC = RUN / "v2_holdout_recommendations.csv"
V3REC = RUN / "v3_holdout_recommendations.csv"
CONTRACT = E / "system_evaluation/semantic_relevance_system_eval.v1.0.0.json"

EXPECTED = {
    SCORES: "d1c9ba1848fb4875c1e925b6c5bdafc3483db37a613bad49286a634d96803fa3",
    SLOCK: "4a4be17bdec6a3d1cd95667bde9fbae1fb4a6077e10b5dc4270710787ffe6777",
    UNION: "885850ae548bba5af4fb24fefd74a7dbedbf961241a7090b10e24e1d5598030b",
    ULOCK: "fd5697398a80200386c38d1f4bedf0ebd4dfcfa3c677915c7597829f899a6da9",
    V2REC: "4e8969ee79341b04ada9f2d251a0e838ab790cc238ba51aa4af403ab4a1c2aaa",
    V3REC: "e5e136a6d611a527635d202ec5edb928e9de77a8a0db46fa0fcab295c69a4ba2",
}
NQ, K, REPS, SEED = 24, 10, 5000, 20261002
CI_METRICS = [
    "macro_mean_relevance_at_10","clear_or_strong_rate_at_10",
    "irrelevant_rate_at_10","hit_rate_at_5_clear_or_strong",
    "top1_clear_or_strong_rate","mrr_at_10_clear_or_strong","ndcg_at_10"
]
PAIR = {
    "macro_mean_relevance_at_10":("mean_relevance_at_10",True),
    "clear_or_strong_rate_at_10":("clear_or_strong_rate_at_10",True),
    "irrelevant_rate_at_10":("irrelevant_rate_at_10",False),
    "hit_rate_at_5_clear_or_strong":("hit_at_5_clear_or_strong",True),
    "top1_clear_or_strong_rate":("top1_clear_or_strong",True),
    "mrr_at_10_clear_or_strong":("mrr_at_10_clear_or_strong",True),
    "ndcg_at_10":("ndcg_at_10",True),
    "catastrophic_query_rate":("catastrophic_query",False),
}
GATES = {
    "macro_mean_relevance_at_10_ge_2_50":("macro_mean_relevance_at_10",">=",2.5),
    "macro_mean_relevance_ci_lower_ge_2_00":("macro_mean_relevance_at_10_ci_lower_95",">=",2.0),
    "clear_or_strong_rate_at_10_ge_0_60":("clear_or_strong_rate_at_10",">=",0.6),
    "irrelevant_rate_at_10_le_0_10":("irrelevant_rate_at_10","<=",0.1),
    "hit_rate_at_5_clear_or_strong_ge_0_90":("hit_rate_at_5_clear_or_strong",">=",0.9),
    "top1_clear_or_strong_rate_ge_0_75":("top1_clear_or_strong_rate",">=",0.75),
    "ndcg_at_10_ge_0_85":("ndcg_at_10",">=",0.85),
    "catastrophic_query_rate_eq_0":("catastrophic_query_rate","==",0.0),
}

def sha(p:Path)->str: return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p:Path)->dict[str,Any]: return json.loads(p.read_text(encoding="utf-8-sig"))
def req(x:bool,msg:str):
    if not x: raise ValueError(msg)
def b(v:Any)->bool:
    s=str(v).strip().lower()
    if s in {"true","1","yes"}: return True
    if s in {"false","0","no",""}: return False
    raise ValueError(f"Bad bool {v!r}")
def ndcg(scores:list[int])->float:
    g=np.array([2**int(x)-1 for x in scores],float)
    d=np.log2(np.arange(2,len(scores)+2,dtype=float))
    dcg=float(np.sum(g/d)); ideal=np.sort(g)[::-1]; idcg=float(np.sum(ideal/d))
    return 0.0 if idcg==0 else dcg/idcg

def qsum(g:pd.DataFrame)->dict[str,Any]:
    g=g.sort_values("rank",kind="stable")
    s=g["judge_score"].astype(int).tolist(); req(len(s)==10,"Expected top10")
    c=np.array(s)>=3; z=np.array(s)==0
    cr=[i+1 for i,x in enumerate(c) if x]; first=min(cr) if cr else None
    return {
        "query_id":str(g["query_id"].iloc[0]),"query":str(g["query"].iloc[0]),
        "stratum":str(g["stratum"].iloc[0]),
        "mean_relevance_at_10":float(np.mean(s)),
        "median_relevance_at_10":float(np.median(s)),
        "clear_or_strong_rate_at_10":float(np.mean(c)),
        "irrelevant_rate_at_10":float(np.mean(z)),
        "hit_at_5_clear_or_strong":int(np.any(c[:5])),
        "top1_clear_or_strong":int(bool(c[0])),
        "first_clear_or_strong_rank":first,
        "mrr_at_10_clear_or_strong":0.0 if first is None else 1.0/first,
        "ndcg_at_10":ndcg(s),
        "catastrophic_query":int(float(np.mean(s))<1.5),
    }

def aggregate(qm:pd.DataFrame)->dict[str,float]:
    return {
        "macro_mean_relevance_at_10":float(qm["mean_relevance_at_10"].mean()),
        "clear_or_strong_rate_at_10":float(qm["clear_or_strong_rate_at_10"].mean()),
        "irrelevant_rate_at_10":float(qm["irrelevant_rate_at_10"].mean()),
        "hit_rate_at_5_clear_or_strong":float(qm["hit_at_5_clear_or_strong"].mean()),
        "top1_clear_or_strong_rate":float(qm["top1_clear_or_strong"].mean()),
        "mrr_at_10_clear_or_strong":float(qm["mrr_at_10_clear_or_strong"].mean()),
        "ndcg_at_10":float(qm["ndcg_at_10"].mean()),
        "catastrophic_query_rate":float(qm["catastrophic_query"].mean()),
    }

def bootstrap(qm:pd.DataFrame)->dict[str,Any]:
    cols={
      "macro_mean_relevance_at_10":"mean_relevance_at_10",
      "clear_or_strong_rate_at_10":"clear_or_strong_rate_at_10",
      "irrelevant_rate_at_10":"irrelevant_rate_at_10",
      "hit_rate_at_5_clear_or_strong":"hit_at_5_clear_or_strong",
      "top1_clear_or_strong_rate":"top1_clear_or_strong",
      "mrr_at_10_clear_or_strong":"mrr_at_10_clear_or_strong",
      "ndcg_at_10":"ndcg_at_10"}
    rng=np.random.default_rng(SEED); idx=rng.integers(0,len(qm),size=(REPS,len(qm)))
    out={}
    for m,c in cols.items():
        a=qm[c].astype(float).to_numpy(); dist=np.mean(a[idx],axis=1)
        out[m]={"lower_95":float(np.quantile(dist,.025)),"upper_95":float(np.quantile(dist,.975)),
                "bootstrap_mean":float(np.mean(dist)),"bootstrap_standard_error":float(np.std(dist,ddof=1))}
    return out

def reconstruct(scores:pd.DataFrame, member:str, rankcol:str, name:str)->pd.DataFrame:
    f=scores[scores[member].map(b)].copy()
    f["rank"]=pd.to_numeric(f[rankcol],errors="raise").astype(int)
    f["judge_score"]=pd.to_numeric(f["judge_score"],errors="raise").astype(int)
    req(len(f)==NQ*K,f"{name}: expected 240 rows")
    req(not f.duplicated(["query_id","rank"]).any(),f"{name}: duplicate ranks")
    counts=f.groupby("query_id").size()
    req(len(counts)==NQ and (counts==K).all(),f"{name}: not 24x10")
    for qid,g in f.groupby("query_id"): req(sorted(g["rank"])==list(range(1,11)),f"{name}/{qid}: bad ranks")
    return f

def standalone(f:pd.DataFrame):
    qm=pd.DataFrame([qsum(g) for _,g in f.groupby("query_id",sort=True)])
    agg=aggregate(qm); ci=bootstrap(qm)
    agg["macro_mean_relevance_at_10_ci_lower_95"]=ci["macro_mean_relevance_at_10"]["lower_95"]
    agg["macro_mean_relevance_at_10_ci_upper_95"]=ci["macro_mean_relevance_at_10"]["upper_95"]
    return qm,agg,ci

def passgate(actual:float,op:str,t:float)->bool:
    return actual>=t if op==">=" else actual<=t if op=="<=" else actual==t

def gate_result(agg:dict[str,Any],contract:dict[str,Any])->dict[str,Any]:
    checks={}; failed=[]
    for n,g in contract["release_gates"].items():
        a=float(agg[g["metric"]]); ok=passgate(a,g["op"],float(g["value"]))
        checks[n]={"passed":ok,"metric":g["metric"],"actual":a,"operator":g["op"],"threshold":float(g["value"])}
        if not ok: failed.append(n)
    cat=float(agg["catastrophic_query_rate"])
    decision="PASS" if not failed else ("REVIEW" if len(failed)<=2 and cat==0 else "FAIL")
    return {"decision":decision,"passed_gate_count":len(checks)-len(failed),"total_gate_count":len(checks),
            "failed_gate_count":len(failed),"failed_gates":failed,"checks":checks,"decision_policy":contract["decision_policy"]}

def main():
    print("Semantic relevance v3 - FINAL unseen holdout analysis")
    print("Judge execution: DISABLED | Human labels: NOT USED")
    for p,e in EXPECTED.items():
        req(p.exists(),f"Missing {p}"); a=sha(p); req(a==e,f"Hash mismatch {p.name}: {a}")
        print(f"PASS  frozen {p.name} hash unchanged")
    sl=load(SLOCK)
    req(sl.get("status")=="V3_HOLDOUT_JUDGE_SCORES_FROZEN_BEFORE_ANALYSIS","Scores not frozen")
    req(sl.get("aggregate_metrics_calculated") is False,"Metrics already calculated during scoring")
    req(sl.get("release_decision_produced") is False,"Release decision already produced during scoring")
    contract=load(CONTRACT)
    stat=contract["statistical_uncertainty"]
    req(int(stat["bootstrap_replicates"])==REPS and int(stat["seed"])==SEED,"Bootstrap contract drift")
    req(list(contract["release_gates"])==list(GATES),"Gate names/order drift")
    for n,(m,o,t) in GATES.items():
        g=contract["release_gates"][n]; req((g["metric"],g["op"],float(g["value"]))==(m,o,t),f"Gate drift {n}")
    print("PASS  frozen bootstrap and eight release gates verified")

    scores=pd.read_csv(SCORES,dtype=str,keep_default_na=False,encoding="utf-8")
    req(len(scores)==322 and not scores["case_id"].duplicated().any(),"Bad 322-row score artifact")
    scores["judge_score"]=pd.to_numeric(scores["judge_score"],errors="raise").astype(int)
    req(scores["judge_score"].between(0,4).all(),"Scores outside 0..4")
    v2=reconstruct(scores,"in_v2_top10","v2_rank","v2")
    v3=reconstruct(scores,"in_v3_top10","v3_rank","v3")
    print("PASS  reconstructed v2 and v3 as 24x10")

    v2q,v2a,v2ci=standalone(v2); v3q,v3a,v3ci=standalone(v3)
    req(list(v2q["query_id"])==list(v3q["query_id"]),"Query mismatch")
    v2g=gate_result(v2a,contract); v3g=gate_result(v3a,contract)

    v2q.to_csv(RUN/"v2_holdout_query_metrics.csv",index=False)
    v3q.to_csv(RUN/"v3_holdout_query_metrics.csv",index=False)
    (RUN/"v2_holdout_aggregate_metrics.json").write_text(json.dumps(v2a,indent=2)+"\n")
    (RUN/"v3_holdout_aggregate_metrics.json").write_text(json.dumps(v3a,indent=2)+"\n")
    (RUN/"v2_holdout_bootstrap_confidence_intervals.json").write_text(json.dumps(v2ci,indent=2)+"\n")
    (RUN/"v3_holdout_bootstrap_confidence_intervals.json").write_text(json.dumps(v3ci,indent=2)+"\n")
    (RUN/"v2_holdout_release_gate_result.json").write_text(json.dumps(v2g,indent=2)+"\n")
    (RUN/"v3_holdout_release_gate_result.json").write_text(json.dumps(v3g,indent=2)+"\n")

    rng=np.random.default_rng(SEED); idx=rng.integers(0,NQ,size=(REPS,NQ))
    metrics={}; deltarows=[]; diffs={}
    for m,(col,hib) in PAIR.items():
        a=v2q[col].astype(float).to_numpy(); c=v3q[col].astype(float).to_numpy(); d=c-a
        imp=d if hib else -d; diffs[m]=d
        dist=np.mean(d[idx],axis=1)
        metrics[m]={"v2":float(np.mean(a)),"v3":float(np.mean(c)),"raw_delta_v3_minus_v2":float(np.mean(d)),
                    "paired_ci_lower":float(np.quantile(dist,.025)),"paired_ci_upper":float(np.quantile(dist,.975)),
                    "wins":int(np.sum(imp>1e-15)),"ties":int(np.sum(np.abs(imp)<=1e-15)),"losses":int(np.sum(imp<-1e-15))}
        for i,qid in enumerate(v2q["query_id"]):
            deltarows.append({"query_id":qid,"query":v2q.iloc[i]["query"],"stratum":v2q.iloc[i]["stratum"],
                              "metric":m,"v2_value":float(a[i]),"v3_value":float(c[i]),
                              "raw_delta_v3_minus_v2":float(d[i]),"outcome":"WIN" if imp[i]>1e-15 else "LOSS" if imp[i]<-1e-15 else "TIE"})
    pd.DataFrame(deltarows).to_csv(RUN/"v2_vs_v3_holdout_query_deltas.csv",index=False)

    trans=[]; tc={"fail_to_pass":0,"pass_to_pass":0,"fail_to_fail":0,"pass_to_fail":0}
    for n in v2g["checks"]:
        a=v2g["checks"][n]["passed"]; c=v3g["checks"][n]["passed"]
        key="fail_to_pass" if (not a and c) else "pass_to_pass" if (a and c) else "fail_to_fail" if (not a and not c) else "pass_to_fail"
        tc[key]+=1; trans.append({"gate":n,"v2_passed":a,"v3_passed":c,"transition":key.replace("_","->",1).upper()})
    pd.DataFrame(trans).to_csv(RUN/"v2_vs_v3_holdout_gate_transitions.csv",index=False)
    paired={"status":"COMPLETE","analysis_role":"secondary","cannot_override_standalone_v3_release_decision":True,
            "matched_query_count":NQ,"replicates":REPS,"seed":SEED,"metrics":metrics,
            "gate_transitions":{"v2_decision":v2g["decision"],"v3_decision":v3g["decision"],"counts":tc,"gates":trans}}
    (RUN/"v2_vs_v3_holdout_paired_comparison.json").write_text(json.dumps(paired,indent=2)+"\n")

    report = [
      "# Semantic Relevance v3 — Final Unseen Holdout Report","",
      f"**Final standalone v3 decision: {v3g['decision']} ({v3g['passed_gate_count']}/8 gates)**","",
      "This standalone v3 holdout decision is authoritative; the paired v2→v3 comparison is secondary evidence.","",
      f"- Frozen unique pairs judged: 322",
      f"- Holdout queries: 24",
      f"- Human relevance labels: No",
      f"- Judge calls during analysis: 0",
      f"- Judge score artifact: `{EXPECTED[SCORES]}`","",
      "## Standalone metrics","",
      "| Metric | v2 | v3 |","|---|---:|---:|"
    ]
    for m in CI_METRICS:
        report.append(f"| `{m}` | {v2a[m]:.4f} | {v3a[m]:.4f} |")
    report += ["",f"- v2 catastrophic query rate: {v2a['catastrophic_query_rate']:.1%}",
               f"- v3 catastrophic query rate: {v3a['catastrophic_query_rate']:.1%}",
               f"- v2 decision: {v2g['decision']} ({v2g['passed_gate_count']}/8 gates)",
               f"- v3 decision: {v3g['decision']} ({v3g['passed_gate_count']}/8 gates)","",
               "## Paired v2 → v3 deltas","",
               "| Metric | Delta | 95% paired CI | W/T/L |","|---|---:|---:|---:|"]
    for m,r in metrics.items():
        report.append(f"| `{m}` | {r['raw_delta_v3_minus_v2']:+.4f} | [{r['paired_ci_lower']:+.4f}, {r['paired_ci_upper']:+.4f}] | {r['wins']}/{r['ties']}/{r['losses']} |")
    report += ["","## Gate transitions","",
               f"- FAIL→PASS: {tc['fail_to_pass']}",
               f"- PASS→PASS: {tc['pass_to_pass']}",
               f"- FAIL→FAIL: {tc['fail_to_fail']}",
               f"- PASS→FAIL: {tc['pass_to_fail']}","",
               "Standalone CIs and paired CIs use 5,000 query-level bootstrap replicates with seed `20261002`.",
               "The paired comparison cannot override the standalone v3 release decision."]
    (RUN/"v3_final_unseen_holdout_report.md").write_text("\n".join(report)+"\n",encoding="utf-8")

    meta={"status":"V3_FINAL_UNSEEN_HOLDOUT_ANALYSIS_COMPLETE","completed_at_utc":datetime.now(timezone.utc).isoformat(),
          "judge_scores_sha256":EXPECTED[SCORES],"query_count":NQ,"bootstrap_replicates":REPS,"bootstrap_seed":SEED,
          "judge_calls_performed_during_analysis":0,"human_relevance_labels_used":False,
          "v2_decision":v2g["decision"],"v3_final_decision":v3g["decision"],"paired_comparison_role":"secondary"}
    (RUN/"v3_final_holdout_analysis_metadata.json").write_text(json.dumps(meta,indent=2)+"\n")

    outs=[RUN/x for x in ["v2_holdout_query_metrics.csv","v3_holdout_query_metrics.csv",
      "v2_holdout_aggregate_metrics.json","v3_holdout_aggregate_metrics.json",
      "v2_holdout_bootstrap_confidence_intervals.json","v3_holdout_bootstrap_confidence_intervals.json",
      "v2_holdout_release_gate_result.json","v3_holdout_release_gate_result.json",
      "v2_vs_v3_holdout_query_deltas.csv","v2_vs_v3_holdout_gate_transitions.csv",
      "v2_vs_v3_holdout_paired_comparison.json","v3_final_unseen_holdout_report.md","v3_final_holdout_analysis_metadata.json"]]
    lock={"status":"V3_FINAL_UNSEEN_HOLDOUT_ANALYSIS_FROZEN","frozen_at_utc":datetime.now(timezone.utc).isoformat(),
          "judge_scores_sha256":EXPECTED[SCORES],"v2_decision":v2g["decision"],"v3_final_decision":v3g["decision"],
          "v2_passed_gates":v2g["passed_gate_count"],"v3_passed_gates":v3g["passed_gate_count"],
          "paired_comparison_is_secondary":True,"judge_calls_performed_during_analysis":0,
          "outputs":{p.name:sha(p) for p in outs}}
    lp=RUN/"v3_final_holdout_analysis_lock.json"
    if lp.exists():
        old=load(lp)
        for k,v in lock.items():
            if k!="frozen_at_utc": req(old.get(k)==v,f"Existing analysis lock mismatch: {k}")
    else: lp.write_text(json.dumps(lock,indent=2)+"\n")

    print("\nV2 STANDALONE HOLDOUT")
    print(f"Mean@10={v2a['macro_mean_relevance_at_10']:.3f} CI=[{v2ci['macro_mean_relevance_at_10']['lower_95']:.3f},{v2ci['macro_mean_relevance_at_10']['upper_95']:.3f}]")
    print(f"Clear={v2a['clear_or_strong_rate_at_10']:.1%} Irrelevant={v2a['irrelevant_rate_at_10']:.1%} Hit@5={v2a['hit_rate_at_5_clear_or_strong']:.1%} Top1={v2a['top1_clear_or_strong_rate']:.1%} MRR={v2a['mrr_at_10_clear_or_strong']:.3f} nDCG={v2a['ndcg_at_10']:.3f} Cat={v2a['catastrophic_query_rate']:.1%}")
    print(f"Decision={v2g['decision']} ({v2g['passed_gate_count']}/8 gates)")
    print("\nV3 STANDALONE UNSEEN HOLDOUT")
    print(f"Mean@10={v3a['macro_mean_relevance_at_10']:.3f} CI=[{v3ci['macro_mean_relevance_at_10']['lower_95']:.3f},{v3ci['macro_mean_relevance_at_10']['upper_95']:.3f}]")
    print(f"Clear={v3a['clear_or_strong_rate_at_10']:.1%} Irrelevant={v3a['irrelevant_rate_at_10']:.1%} Hit@5={v3a['hit_rate_at_5_clear_or_strong']:.1%} Top1={v3a['top1_clear_or_strong_rate']:.1%} MRR={v3a['mrr_at_10_clear_or_strong']:.3f} nDCG={v3a['ndcg_at_10']:.3f} Cat={v3a['catastrophic_query_rate']:.1%}")
    print(f"FINAL V3 DECISION={v3g['decision']} ({v3g['passed_gate_count']}/8 gates)")
    print("\nPAIRED V2 -> V3")
    for m,r in metrics.items():
        print(f"{m}: delta={r['raw_delta_v3_minus_v2']:+.4f} CI=[{r['paired_ci_lower']:+.4f},{r['paired_ci_upper']:+.4f}] W/T/L={r['wins']}/{r['ties']}/{r['losses']}")
    print(f"Gate transitions: {tc}")
    print("Judge calls during analysis: 0")
    print(f"Final analysis lock SHA-256: {sha(lp)}")
    print("V3 FINAL UNSEEN HOLDOUT ANALYSIS: PASS")

if __name__=="__main__":
    main()
