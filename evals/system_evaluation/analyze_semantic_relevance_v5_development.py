from __future__ import annotations
import hashlib,json,math
from datetime import datetime,timezone
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
import verify_semantic_relevance_v5_development_analysis as boundary

ROOT=boundary.ROOT;RUN=boundary.RUN
SYSTEMS=boundary.SYSTEMS
NQ=24;K=10;REPS=5000;SEED=20261002
METRICS=[
 "macro_mean_relevance_at_10","clear_or_strong_rate_at_10","irrelevant_rate_at_10",
 "hit_rate_at_5_clear_or_strong","top1_clear_or_strong_rate","mrr_at_10_clear_or_strong",
 "ndcg_at_10","catastrophic_query_rate"
]
CRITERIA=[
 ("catastrophic_query_rate","lower"),
 ("irrelevant_rate_at_10","lower"),
 ("clear_or_strong_rate_at_10","higher"),
 ("macro_mean_relevance_at_10","higher"),
 ("ndcg_at_10","higher"),
]
CHANGED={"B":0,"R":1,"K":1,"RK":2}
EXPECTED_VERIFIER="34295163ee2beea6d17e1b5d6ede7dd42e5ce4fc84ae556579522d7c13dee947"

def sha(p:Path)->str:return hashlib.sha256(p.read_bytes()).hexdigest()
def lfsha(p:Path)->str:
 b=p.read_bytes();return hashlib.sha256(b.replace(b"\r\n",b"\n").replace(b"\r",b"\n")).hexdigest()
def req(c:bool,m:str)->None:
 if not c:raise ValueError(m)

def ndcg(scores:list[int])->float:
 gains=np.asarray([2**int(x)-1 for x in scores],dtype=float)
 discounts=np.log2(np.arange(2,len(scores)+2,dtype=float))
 dcg=float(np.sum(gains/discounts));ideal=np.sort(gains)[::-1];idcg=float(np.sum(ideal/discounts))
 return 0.0 if idcg==0.0 else dcg/idcg

def query_summary(g:pd.DataFrame)->dict[str,Any]:
 g=g.sort_values("rank",kind="stable");scores=g.judge_score.astype(int).tolist();req(len(scores)==10,"Expected top10")
 clear=np.asarray(scores)>=3;cr=[i+1 for i,x in enumerate(clear) if x];first=min(cr) if cr else None
 return {
  "query_id":str(g["query_id"].iloc[0]),"query":str(g["query"].iloc[0]),"stratum":str(g["stratum"].iloc[0]),
  "mean_relevance_at_10":float(np.mean(scores)),"median_relevance_at_10":float(np.median(scores)),
  "clear_or_strong_rate_at_10":float(np.mean(clear)),
  "irrelevant_rate_at_10":float(np.mean(np.asarray(scores)==0)),
  "hit_at_5_clear_or_strong":int(np.any(clear[:5])),
  "top1_clear_or_strong":int(bool(clear[0])),
  "first_clear_or_strong_rank":first,
  "mrr_at_10_clear_or_strong":0.0 if first is None else 1.0/float(first),
  "ndcg_at_10":ndcg(scores),
  "catastrophic_query":int(float(np.mean(scores))<1.5),
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
  "macro_mean_relevance_at_10":"mean_relevance_at_10","clear_or_strong_rate_at_10":"clear_or_strong_rate_at_10",
  "irrelevant_rate_at_10":"irrelevant_rate_at_10","hit_rate_at_5_clear_or_strong":"hit_at_5_clear_or_strong",
  "top1_clear_or_strong_rate":"top1_clear_or_strong","mrr_at_10_clear_or_strong":"mrr_at_10_clear_or_strong",
  "ndcg_at_10":"ndcg_at_10","catastrophic_query_rate":"catastrophic_query",
 }
 rng=np.random.default_rng(SEED);idx=rng.integers(0,len(qm),size=(REPS,len(qm)));out={}
 for metric,col in cols.items():
  v=qm[col].astype(float).to_numpy();dist=np.mean(v[idx],axis=1)
  out[metric]={"lower_95":float(np.quantile(dist,.025)),"upper_95":float(np.quantile(dist,.975)),
               "bootstrap_mean":float(np.mean(dist)),"bootstrap_standard_error":float(np.std(dist,ddof=1))}
 return out

def reconstruct(rec:pd.DataFrame,scores:pd.DataFrame,system:str)->pd.DataFrame:
 s=scores[["query_id","isbn13","judge_score"]].copy()
 out=rec.merge(s,on=["query_id","isbn13"],how="left",validate="one_to_one")
 req(out.judge_score.ne("").all(),f"{system}: missing judge score")
 out["rank"]=pd.to_numeric(out["rank"],errors="raise").astype(int)
 out["judge_score"]=pd.to_numeric(out["judge_score"],errors="raise").astype(int)
 return out

QUERY_COL={
 "macro_mean_relevance_at_10":"mean_relevance_at_10","clear_or_strong_rate_at_10":"clear_or_strong_rate_at_10",
 "irrelevant_rate_at_10":"irrelevant_rate_at_10","hit_rate_at_5_clear_or_strong":"hit_at_5_clear_or_strong",
 "top1_clear_or_strong_rate":"top1_clear_or_strong","mrr_at_10_clear_or_strong":"mrr_at_10_clear_or_strong",
 "ndcg_at_10":"ndcg_at_10","catastrophic_query_rate":"catastrophic_query",
}
HIGHER={
 "macro_mean_relevance_at_10":True,"clear_or_strong_rate_at_10":True,"irrelevant_rate_at_10":False,
 "hit_rate_at_5_clear_or_strong":True,"top1_clear_or_strong_rate":True,
 "mrr_at_10_clear_or_strong":True,"ndcg_at_10":True,"catastrophic_query_rate":False,
}

def contrast(qm:dict[str,pd.DataFrame],after:str,before:str)->dict[str,Any]:
 req(list(qm[after].query_id)==list(qm[before].query_id),f"{after}-{before}: query order differs")
 rng=np.random.default_rng(SEED);idx=rng.integers(0,NQ,size=(REPS,NQ));result={}
 for metric,col in QUERY_COL.items():
  a=qm[after][col].astype(float).to_numpy();b=qm[before][col].astype(float).to_numpy();d=a-b
  imp=d if HIGHER[metric] else -d;dist=np.mean(d[idx],axis=1)
  result[metric]={"raw_delta":float(np.mean(d)),"paired_ci_lower":float(np.quantile(dist,.025)),
   "paired_ci_upper":float(np.quantile(dist,.975)),"wins":int(np.sum(imp>0)),
   "ties":int(np.sum(imp==0)),"losses":int(np.sum(imp<0)),"higher_is_better":HIGHER[metric]}
 return result

def interaction(qm:dict[str,pd.DataFrame])->dict[str,Any]:
 rng=np.random.default_rng(SEED);idx=rng.integers(0,NQ,size=(REPS,NQ));out={}
 for metric,col in QUERY_COL.items():
  b=qm["B"][col].astype(float).to_numpy();r=qm["R"][col].astype(float).to_numpy()
  k=qm["K"][col].astype(float).to_numpy();rk=qm["RK"][col].astype(float).to_numpy()
  x=rk-k-r+b;dist=np.mean(x[idx],axis=1)
  out[metric]={"interaction_mean":float(np.mean(x)),"interaction_ci_lower":float(np.quantile(dist,.025)),
               "interaction_ci_upper":float(np.quantile(dist,.975)),
               "definition":"RK - K - R + B; descriptive factorial interaction, not a selection criterion"}
 return out

def select(agg:dict[str,dict[str,float]],median_latency:dict[str,float])->dict[str,Any]:
 tied=list(SYSTEMS);steps=[]
 for metric,direction in CRITERIA:
  vals={s:float(agg[s][metric]) for s in tied}
  best=(min(vals.values()) if direction=="lower" else max(vals.values()))
  survivors=[s for s in tied if vals[s]==best]
  steps.append({"metric":metric,"direction":direction,"candidates_before":tied.copy(),
                "values":vals,"best_value":best,"survivors":survivors.copy()})
  tied=survivors
  if len(tied)==1:
   return {"status":"V5D_DEVELOPMENT_SELECTION_COMPLETE","selected_candidate":tied[0],
           "decision_stage":"lexicographic_metric","decisive_metric":metric,
           "ordered_metric_steps":steps,"final_tied_candidates":tied,
           "requires_fresh_v5h_holdout":tied[0]!="B","release_decision":False}
 if "B" in tied:
  return {"status":"V5D_DEVELOPMENT_SELECTION_COMPLETE","selected_candidate":"B",
          "decision_stage":"exact_tie_policy_retain_B","decisive_metric":None,
          "ordered_metric_steps":steps,"final_tied_candidates":tied,
          "requires_fresh_v5h_holdout":False,"release_decision":False}
 minchg=min(CHANGED[s] for s in tied);few=[s for s in tied if CHANGED[s]==minchg]
 tie_steps={"candidates_after_metrics":tied.copy(),"changed_factor_counts":{s:CHANGED[s] for s in tied},
            "fewest_changed_factor_survivors":few.copy()}
 if len(few)==1:
  return {"status":"V5D_DEVELOPMENT_SELECTION_COMPLETE","selected_candidate":few[0],
          "decision_stage":"exact_tie_policy_fewer_changed_factors","decisive_metric":None,
          "ordered_metric_steps":steps,"tie_break":tie_steps,"final_tied_candidates":few,
          "requires_fresh_v5h_holdout":True,"release_decision":False}
 req(set(few)=={"R","K"},f"Unexpected unresolved exact-tie set after factor-count rule: {few}")
 lr={s:float(median_latency[s]) for s in few};best=min(lr.values());fast=[s for s in few if lr[s]==best]
 tie_steps["median_system_latency_seconds"]=lr;tie_steps["latency_survivors"]=fast.copy()
 if len(fast)==1:
  return {"status":"V5D_DEVELOPMENT_SELECTION_COMPLETE","selected_candidate":fast[0],
          "decision_stage":"exact_tie_policy_R_vs_K_latency","decisive_metric":None,
          "ordered_metric_steps":steps,"tie_break":tie_steps,"final_tied_candidates":fast,
          "requires_fresh_v5h_holdout":True,"release_decision":False}
 return {"status":"V5D_DEVELOPMENT_SELECTION_NO_SELECTION_EXACT_TIE",
         "selected_candidate":None,"decision_stage":"exact_tie_unresolved_after_latency",
         "decisive_metric":None,"ordered_metric_steps":steps,"tie_break":tie_steps,
         "final_tied_candidates":fast,"requires_fresh_v5h_holdout":False,"release_decision":False,
         "next_stage":"SEPARATELY_PREREGISTER_TIE_BREAK_EVALUATION"}

def main():
 print("Semantic relevance V5 - B/R/K/RK DEVELOPMENT analysis and selection")
 print("------------------------------------------------------------------")
 print("Judge/model/retrieval/reranker execution: DISABLED")
 print("Human relevance labels: NOT USED")
 print("Release gates/decision: NOT CALCULATED")
 print("Selection rule: FROZEN LEXICOGRAPHIC")
 print()
 ver=Path(boundary.__file__).resolve();req(sha(ver)==EXPECTED_VERIFIER or lfsha(ver)==EXPECTED_VERIFIER,
   "Development-analysis preflight verifier changed")
 recs,scores,lat,spec,contract=boundary.validate()

 qm={};agg={};ci={}
 for s in SYSTEMS:
  f=reconstruct(recs[s],scores,s)
  qm[s]=pd.DataFrame([query_summary(g) for _,g in f.groupby("query_id",sort=True)])
  agg[s]=aggregate(qm[s]);ci[s]=bootstrap(qm[s])

 contrasts={
  "R_minus_B":contrast(qm,"R","B"),
  "RK_minus_K":contrast(qm,"RK","K"),
  "K_minus_B":contrast(qm,"K","B"),
  "RK_minus_R":contrast(qm,"RK","R"),
 }
 inter=interaction(qm)

 delta_rows=[]
 pairs=[("R","B","R-B"),("RK","K","RK-K"),("K","B","K-B"),("RK","R","RK-R")]
 for after,before,label in pairs:
  for metric,col in QUERY_COL.items():
   for i,qid in enumerate(qm[after]["query_id"].astype(str)):
    delta_rows.append({"query_id":qid,"query":str(qm[after].iloc[i]["query"]),
      "stratum":str(qm[after].iloc[i]["stratum"]),"contrast":label,"metric":metric,
      "before_value":float(qm[before].iloc[i][col]),"after_value":float(qm[after].iloc[i][col]),
      "raw_delta":float(qm[after].iloc[i][col])-float(qm[before].iloc[i][col])})
 deltas=pd.DataFrame(delta_rows)

 median_latency={s:float(lat.loc[lat["system"]==s, "system_latency_seconds"].median()) for s in SYSTEMS}
 selection=select(agg,median_latency)
 selection["candidate_set"]=SYSTEMS
 selection["median_system_latency_seconds"]=median_latency
 selection["diagnostics_can_override_selection"]=False
 selection["post_result_parameter_search_allowed"]=False
 if selection["selected_candidate"] is not None and selection["selected_candidate"]!="B":
  selection["next_stage"]="AUTHOR_AND_FREEZE_V5H01_V5H24_UNSEEN_HOLDOUT_BEFORE_RETRIEVAL"
 elif selection["selected_candidate"]=="B":
  selection["next_stage"]="NO_NONBASELINE_V5_CANDIDATE_SELECTED; OPTIONAL_NONOVERRIDING_DIAGNOSTICS_OR_NEW_SEPARATE_EXPERIMENT"

 depth_status={
  "status":"NOT_CALCULATED_IN_SELECTION_STEP",
  "candidate_clear_or_strong_coverage_at_50":"PENDING_SEPARATELY_FROZEN_DEEP_CANDIDATE_JUDGING",
  "first_clear_or_strong_candidate_rank_within_50":"PENDING_SEPARATELY_FROZEN_DEEP_CANDIDATE_JUDGING",
  "display_clear_or_strong_conversion_given_candidate_at_50":"PENDING_SEPARATELY_FROZEN_DEEP_CANDIDATE_JUDGING",
  "reason":"The frozen 452-case score set covers the B/R/K/RK displayed-pair union, not every deeper candidate through rank 50.",
  "may_override_selection":False,
 }

 outputs=[]
 for s in SYSTEMS:
  p=RUN/f"v5d_{s}_query_metrics.csv";qm[s].to_csv(p,index=False,encoding="utf-8");outputs.append(p)
  p=RUN/f"v5d_{s}_aggregate_metrics.json";p.write_text(json.dumps(agg[s],indent=2)+"\n",encoding="utf-8");outputs.append(p)
  p=RUN/f"v5d_{s}_bootstrap_confidence_intervals.json";p.write_text(json.dumps(ci[s],indent=2)+"\n",encoding="utf-8");outputs.append(p)
 p=RUN/"v5d_factor_contrasts.json";p.write_text(json.dumps(contrasts,indent=2)+"\n",encoding="utf-8");outputs.append(p)
 p=RUN/"v5d_paired_query_deltas_by_stratum.csv";deltas.to_csv(p,index=False,encoding="utf-8");outputs.append(p)
 p=RUN/"v5d_interaction_diagnostics.json";p.write_text(json.dumps(inter,indent=2)+"\n",encoding="utf-8");outputs.append(p)
 p=RUN/"v5d_candidate_depth_diagnostics_status.json";p.write_text(json.dumps(depth_status,indent=2)+"\n",encoding="utf-8");outputs.append(p)
 p=RUN/"v5d_development_selection_result.json";p.write_text(json.dumps(selection,indent=2)+"\n",encoding="utf-8");outputs.append(p)

 report=RUN/"v5d_development_analysis_report.md"
 lines=["# Semantic Relevance V5 — Factorial Development Analysis","",
  "V5D01-V5D24 are development-only. This report does not make a release claim.","",
  f"**Selected development candidate: {selection['selected_candidate'] if selection['selected_candidate'] else 'NONE — unresolved exact tie'}**",
  f"**Decision stage: {selection['decision_stage']}**","",
  "## Standalone display metrics","",
  "| Metric | B | R | K | RK |","|---|---:|---:|---:|---:|"]
 for m in METRICS:
  lines.append(f"| `{m}` | {agg['B'][m]:.6f} | {agg['R'][m]:.6f} | {agg['K'][m]:.6f} | {agg['RK'][m]:.6f} |")
 lines+=["","## Preregistered lexicographic selection",""]
 for st in selection["ordered_metric_steps"]:
  lines.append(f"- `{st['metric']}` ({st['direction']} is better): "+", ".join(f"{s}={st['values'][s]:.6f}" for s in st["candidates_before"])+f" → survivors {st['survivors']}")
 lines+=["","## Factorial diagnostics","",
  "- R main-effect contrasts: R−B and RK−K.",
  "- K main-effect contrasts: K−B and RK−R.",
  "- Interaction: RK−K−R+B.",
  "- Query-level paired deltas are retained with stratum labels.",
  "- Candidate-depth coverage@50 diagnostics are not calculated here because the frozen judge scores cover displayed pairs only; they cannot override selection.",
  "","No parameter search is permitted after these V5D results are observed."]
 report.write_text("\n".join(lines)+"\n",encoding="utf-8");outputs.append(report)

 meta=RUN/"v5d_development_analysis_metadata.json"
 meta_payload={"status":"V5D_FACTORIAL_DEVELOPMENT_ANALYSIS_AND_SELECTION_COMPLETE",
  "completed_at_utc":datetime.now(timezone.utc).isoformat(),"query_set":"V5D01-V5D24","query_count":24,
  "analysis_contract_sha256":boundary.EXPECTED["contract"],"judge_scores_sha256":boundary.EXPECTED["scores"],
  "judge_scoring_lock_sha256":boundary.EXPECTED["score_lock"],"factorial_reranking_lock_sha256":boundary.EXPECTED["rerank_lock"],
  "bootstrap_replicates":REPS,"bootstrap_seed":SEED,"selected_candidate":selection["selected_candidate"],
  "decision_stage":selection["decision_stage"],"judge_calls_performed_during_analysis":0,
  "model_retrieval_reranker_calls_performed_during_analysis":0,"human_relevance_labels_used":False,
  "candidate_depth_diagnostics_calculated":False,"release_gates_calculated":False,"release_decision_produced":False,
  "requires_fresh_v5h_holdout":selection.get("requires_fresh_v5h_holdout",False),"next_stage":selection["next_stage"]}
 meta.write_text(json.dumps(meta_payload,indent=2)+"\n",encoding="utf-8");outputs.append(meta)

 lock=RUN/"v5d_development_analysis_selection.lock.json"
 lock_payload={"status":"V5D_FACTORIAL_DEVELOPMENT_ANALYSIS_AND_SELECTION_FROZEN",
  "frozen_at_utc":datetime.now(timezone.utc).isoformat(),"analysis_contract_sha256":boundary.EXPECTED["contract"],
  "v5_design_spec_sha256":boundary.EXPECTED["design_spec"],"v5_design_lock_sha256":boundary.EXPECTED["design_lock"],
  "factorial_reranking_lock_sha256":boundary.EXPECTED["rerank_lock"],"judge_scores_sha256":boundary.EXPECTED["scores"],
  "judge_scoring_lock_sha256":boundary.EXPECTED["score_lock"],"query_set":"V5D01-V5D24","query_count":24,
  "bootstrap_replicates":REPS,"bootstrap_seed":SEED,"selection_order":[m for m,_ in CRITERIA],
  "exact_tie_policy":contract["selection"]["exact_tie_policy"],"selected_candidate":selection["selected_candidate"],
  "decision_stage":selection["decision_stage"],"candidate_depth_diagnostics_calculated":False,
  "candidate_depth_diagnostics_can_override_selection":False,"judge_calls_performed_during_analysis":0,
  "model_retrieval_reranker_calls_performed_during_analysis":0,"human_relevance_labels_used":False,
  "release_gates_calculated":False,"release_decision_produced":False,
  "requires_fresh_v5h_holdout":selection.get("requires_fresh_v5h_holdout",False),
  "outputs":{p.name:sha(p) for p in outputs},"next_stage":selection["next_stage"]}
 lock.write_text(json.dumps(lock_payload,indent=2)+"\n",encoding="utf-8")

 print()
 for s in SYSTEMS:
  a=agg[s]
  print(f"{s} ({'R0K0' if s=='B' else 'R1K0' if s=='R' else 'R0K1' if s=='K' else 'R1K1'}): "
    f"Mean@10={a['macro_mean_relevance_at_10']:.3f} Clear={a['clear_or_strong_rate_at_10']:.1%} "
    f"Irrelevant={a['irrelevant_rate_at_10']:.1%} Hit@5={a['hit_rate_at_5_clear_or_strong']:.1%} "
    f"Top1={a['top1_clear_or_strong_rate']:.1%} MRR={a['mrr_at_10_clear_or_strong']:.3f} "
    f"nDCG={a['ndcg_at_10']:.3f} Cat={a['catastrophic_query_rate']:.1%}")
 print()
 print("PRE-REGISTERED V5D DEVELOPMENT SELECTION")
 for st in selection["ordered_metric_steps"]:
  print(f"{st['metric']}: "+ " ".join(f"{s}={st['values'][s]:.6f}" for s in st["candidates_before"]) + f" -> {st['survivors']}")
 print(f"SELECTED DEVELOPMENT CANDIDATE: {selection['selected_candidate'] if selection['selected_candidate'] else 'NONE'}")
 print(f"Decision stage: {selection['decision_stage']}")
 print(f"Fresh V5H holdout required: {'YES' if selection.get('requires_fresh_v5h_holdout',False) else 'NO'}")
 print("Candidate-depth diagnostics calculated: NO")
 print("Judge/model/retrieval/reranker calls during analysis: 0")
 print("Human relevance labels used: NO")
 print("Release gates calculated: NO")
 print("Release decision produced: NO")
 print(f"Development analysis/selection lock SHA-256: {sha(lock)}")
 print("V5D FACTORIAL DEVELOPMENT ANALYSIS AND SELECTION: PASS")

if __name__=="__main__":main()
