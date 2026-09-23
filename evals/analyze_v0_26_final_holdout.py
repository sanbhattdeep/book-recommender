"""Analyze the single frozen v0.26.0 r6 final-holdout run.

The result is final evidence: always report it as observed, even if one or more
pre-registered release checks fail. Do not use this holdout for retuning.
"""
from __future__ import annotations

import argparse, hashlib, json
from pathlib import Path
import pandas as pd
from analyze_v0_26_development import safe_rate, confusion_matrix_no_sklearn, weighted_cohen_kappa

REPO_ROOT=Path(__file__).resolve().parents[1]
EXPECTED_ROLE="unseen_final_holdout"
EXPECTED_SCOPE="one_time_frozen_release_validation"
EXPECTED_CASES=30

def sha256(p:Path)->str: return hashlib.sha256(p.read_bytes()).hexdigest()
def load_json(p:Path)->dict: return json.loads(p.read_text(encoding="utf-8"))
def resolve(raw:str)->Path:
    p=Path(raw)
    if not p.is_absolute(): p=REPO_ROOT/p
    return p.resolve()
def nested(d:dict,path:str):
    cur=d
    for part in path.split("."): cur=cur[part]
    return cur

def main()->None:
    ap=argparse.ArgumentParser()
    ap.add_argument("--run",required=True)
    args=ap.parse_args()
    run=resolve(args.run)
    meta=load_json(run/"run_metadata.json")
    expected={"judge_config_version":"0.26.0","evaluation_dataset_version":"2.0.0","evaluation_dataset_role":EXPECTED_ROLE,"evaluation_scope":EXPECTED_SCOPE,"judge_behavior_frozen":True,"final_holdout_one_time":True,"facet_spec_version":"0.9.3","rubric_version":"0.1.0"}
    bad=[f"{k}: {meta.get(k)!r} != {v!r}" for k,v in expected.items() if str(meta.get(k))!=str(v)]
    if bad: raise ValueError("Not the frozen r6 final-holdout run:\n- "+"\n- ".join(bad))
    dataset=resolve(str(meta["evaluation_dataset_file"]))
    if meta.get("evaluation_dataset_sha256") != sha256(dataset): raise ValueError("Final-holdout dataset hash no longer matches run metadata.")
    rel=resolve(str(meta["release_manifest_file"]))
    if meta.get("release_manifest_sha256") != sha256(rel): raise ValueError("Release manifest hash no longer matches run metadata.")
    release=load_json(rel)

    gold=pd.read_csv(dataset,dtype={"case_id":str,"query_id":str,"isbn13":str},encoding="utf-8")
    judged=pd.read_csv(run/"judge_results.csv",dtype={"case_id":str},encoding="utf-8")
    if len(gold)!=EXPECTED_CASES: raise ValueError(f"Expected {EXPECTED_CASES} gold cases, found {len(gold)}")
    if gold["case_id"].duplicated().any() or judged["case_id"].duplicated().any(): raise ValueError("Duplicate case IDs.")
    cmp=gold[["case_id","query_id","query","title","human_score","human_reason"]].merge(judged,on="case_id",how="inner",validate="one_to_one")
    if len(cmp)!=EXPECTED_CASES:
        missing=sorted(set(gold.case_id)-set(judged.case_id))
        raise ValueError(f"Final holdout incomplete: {len(cmp)}/{EXPECTED_CASES}; missing={missing}")
    cmp["human_score"]=cmp.human_score.astype(int); cmp["judge_score"]=cmp.judge_score.astype(int)
    cmp["signed_difference"]=cmp.judge_score-cmp.human_score
    cmp["absolute_difference"]=cmp.signed_difference.abs()
    cmp["exact_match"]=cmp.human_score.eq(cmp.judge_score)
    cmp["within_one"]=cmp.absolute_difference.le(1)
    h=cmp.human_score; j=cmp.judge_score; labels=[0,1,2,3,4]
    exact=float(cmp.exact_match.mean()); within=float(cmp.within_one.mean()); mae=float(cmp.absolute_difference.mean())
    lk=weighted_cohen_kappa(h,j,labels,"linear"); qk=weighted_cohen_kappa(h,j,labels,"quadratic")
    hp=h>0; jp=j>0
    tp=int((hp&jp).sum()); fp=int((~hp&jp).sum()); fn=int((hp&~jp).sum()); tn=int((~hp&~jp).sum())
    precision=safe_rate(tp,tp+fp); recall=safe_rate(tp,tp+fn); f1=safe_rate(2*precision*recall,precision+recall)
    severe=int((cmp.absolute_difference>=2).sum()); over=int((cmp.signed_difference>=2).sum()); under=int((cmp.signed_difference<=-2).sum())
    false_zero=int(((h>0)&(j==0)).sum()); false_pos=int(((h==0)&(j>0)).sum())
    cue_severe=0
    if "deterministic_direct_cue_count" in cmp.columns:
        cue=pd.to_numeric(cmp.deterministic_direct_cue_count,errors="coerce").fillna(0).astype(int)>0
        cue_severe=int((cue&(h==0)&(j>=2)).sum())
    metrics={
      "within_one_agreement":within,"exact_agreement":exact,"quadratic_weighted_cohens_kappa":qk,"linear_weighted_cohens_kappa":lk,
      "error_counts":{"absolute_difference_ge_2":severe,"false_zeros":false_zero,"false_positive_relevance":false_pos,"over_promotions_by_2_or_more":over,"under_promotions_by_2_or_more":under},
      "binary_relevance_0_vs_positive":{"true_positive":tp,"false_positive":fp,"false_negative":fn,"true_negative":tn,"precision":precision,"recall":recall,"f1":f1},
      "deterministic_cue_severe_false_positives":cue_severe,
    }
    checks={}
    for name,spec in release["final_holdout_protocol"]["pre_registered_release_checks"].items():
        val=nested(metrics,spec["metric"]); target=spec["value"]; op=spec["op"]
        checks[name] = (val>=target if op==">=" else val<=target if op=="<=" else val==target)
    matrix=pd.DataFrame(confusion_matrix_no_sklearn(h,j,labels),index=[f"human_{x}" for x in labels],columns=[f"judge_{x}" for x in labels])
    largest=cmp.sort_values(["absolute_difference","case_id"],ascending=[False,True])[["case_id","query_id","title","human_score","judge_score","signed_difference","absolute_difference"]]
    summary={
      "finality_note":"One-time final holdout result for frozen semantic relevance Judge v0.26.0 r6. Report as observed; do not tune, relabel, or rerun the holdout to improve metrics.",
      "analysis_scope":EXPECTED_SCOPE,"evaluation_dataset_version":"2.0.0","cases_compared":EXPECTED_CASES,
      "exact_agreement":exact,"within_one_agreement":within,"mean_absolute_difference":mae,"linear_weighted_cohens_kappa":lk,"quadratic_weighted_cohens_kappa":qk,
      "binary_relevance_0_vs_positive":metrics["binary_relevance_0_vs_positive"],"error_counts":metrics["error_counts"],
      "deterministic_cue_severe_false_positives":cue_severe,"pre_registered_release_checks":checks,"all_release_checks_pass":all(checks.values()),
      "release_manifest_sha256":sha256(rel),"run_metadata":meta,
    }
    cmp.to_csv(run/"human_vs_judge.final_holdout.csv",index=False,encoding="utf-8")
    matrix.to_csv(run/"confusion_matrix.final_holdout.csv",encoding="utf-8")
    largest.to_csv(run/"largest_disagreements.final_holdout.csv",index=False,encoding="utf-8")
    (run/"final_holdout_summary.json").write_text(json.dumps(summary,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    print("Frozen v0.26.0 r6 final holdout summary")
    print("----------------------------------------")
    print(f"Cases compared:            {EXPECTED_CASES}/{EXPECTED_CASES}")
    print(f"Exact agreement:           {exact:.1%}")
    print(f"Within ±1 agreement:       {within:.1%}")
    print(f"Mean absolute difference:  {mae:.3f}")
    print(f"Linear weighted kappa:     {lk:.3f}")
    print(f"Quadratic weighted kappa:  {qk:.3f}")
    print(); print("Binary relevance (0 vs >0)"); print("--------------------------")
    print(f"TP={tp} FP={fp} FN={fn} TN={tn}")
    print(f"Precision={precision:.1%} Recall={recall:.1%} F1={f1:.1%}")
    print(); print("Error counts"); print("------------")
    print(f"False zeros:               {false_zero}")
    print(f"False-positive relevance:  {false_pos}")
    print(f"|difference| >= 2:         {severe}")
    print(f"Over-promotions >= 2:      {over}")
    print(f"Under-promotions >= 2:     {under}")
    print(); print("Pre-registered release checks"); print("-----------------------------")
    for name,passed in checks.items(): print(f"{'PASS' if passed else 'FAIL':4}  {name}")
    print(); print("Final holdout release checks: "+("PASS" if all(checks.values()) else "REVIEW"))
    print("Result is final evidence regardless of PASS/REVIEW; do not retune on this holdout.")
    print(); print("Largest disagreements"); print("---------------------")
    print(largest.head(15).to_string(index=False))

if __name__=="__main__": main()
