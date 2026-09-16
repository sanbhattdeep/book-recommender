from __future__ import annotations
import argparse, json
from pathlib import Path
import pandas as pd
from fresh_unseen_contract import *

def resolve_run_dir(raw: str) -> Path:
    path = Path(raw)
    if not path.is_absolute():
        path = REPO_ROOT / path
    if not path.exists():
        raise FileNotFoundError(f"Run directory not found: {path}")
    return path


def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_dataset(metadata: dict) -> Path:
    raw = metadata.get("evaluation_dataset_file")
    if not raw:
        raise ValueError(
            "run_metadata.json does not contain evaluation_dataset_file."
        )
    path = Path(str(raw))
    if not path.is_absolute():
        path = REPO_ROOT / path
    if not path.exists():
        raise FileNotFoundError(
            f"Development dataset recorded in metadata was not found: {path}"
        )
    return path


def safe_rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def confusion_matrix_no_sklearn(
    human: pd.Series,
    judge: pd.Series,
    labels: list[int],
) -> list[list[int]]:
    """Rows = human score, columns = judge score."""
    pos = {label: i for i, label in enumerate(labels)}
    matrix = [[0 for _ in labels] for _ in labels]

    for h, j in zip(human.tolist(), judge.tolist()):
        matrix[pos[int(h)]][pos[int(j)]] += 1

    return matrix


def weighted_cohen_kappa(
    human: pd.Series,
    judge: pd.Series,
    labels: list[int],
    mode: str,
) -> float:
    """
    Weighted Cohen's kappa using disagreement weights.

    Equivalent to sklearn's linear/quadratic weighting for the fixed
    ordinal score set 0..4.
    """
    if mode not in {"linear", "quadratic"}:
        raise ValueError("mode must be 'linear' or 'quadratic'")

    n = len(human)
    if n == 0:
        raise ValueError("Cannot compute kappa with zero cases.")

    pos = {label: i for i, label in enumerate(labels)}
    k = len(labels)
    max_distance = k - 1

    observed = [[0.0 for _ in labels] for _ in labels]
    human_counts = [0.0 for _ in labels]
    judge_counts = [0.0 for _ in labels]

    for h, j in zip(human.tolist(), judge.tolist()):
        hi = pos[int(h)]
        ji = pos[int(j)]
        observed[hi][ji] += 1.0
        human_counts[hi] += 1.0
        judge_counts[ji] += 1.0

    observed_weighted = 0.0
    expected_weighted = 0.0

    for i in range(k):
        for j in range(k):
            distance = abs(i - j)

            if mode == "linear":
                weight = distance / max_distance
            else:
                weight = (distance ** 2) / (max_distance ** 2)

            observed_prob = observed[i][j] / n
            expected_prob = (
                (human_counts[i] / n)
                * (judge_counts[j] / n)
            )

            observed_weighted += weight * observed_prob
            expected_weighted += weight * expected_prob

    if expected_weighted == 0:
        return 1.0 if observed_weighted == 0 else 0.0

    return 1.0 - (observed_weighted / expected_weighted)



def analyze_run(run_dir: Path, expected_role: str, expected_scope: str, output_prefix: str, finality_note: str | None = None):
    metadata=load_json(run_dir/'run_metadata.json')
    expected={'judge_config_version':'0.18.0','evaluation_dataset_version':'2.0.0','evaluation_dataset_role':expected_role,'evaluation_scope':expected_scope}
    bad=[f"{k}: run={metadata.get(k)!r}, expected={v!r}" for k,v in expected.items() if str(metadata.get(k))!=str(v)]
    if bad: raise ValueError('Run provenance mismatch:\n- '+'\n- '.join(bad))
    dataset_path=Path(str(metadata['evaluation_dataset_file'])); dataset_path=dataset_path if dataset_path.is_absolute() else REPO_ROOT/dataset_path
    gold=pd.read_csv(dataset_path,dtype={'case_id':str,'query_id':str,'isbn13':str},encoding='utf-8')
    judged=pd.read_csv(run_dir/'judge_results.csv',dtype={'case_id':str},encoding='utf-8')
    if len(gold)!=30: raise ValueError(f'Expected 30 gold cases, found {len(gold)}')
    if gold['case_id'].duplicated().any() or judged['case_id'].duplicated().any(): raise ValueError('Duplicate case_id in gold/results')
    c=gold[['case_id','query_id','query','title','human_score','human_reason']].merge(judged,on='case_id',how='inner',validate='one_to_one')
    if len(c)!=30:
        missing=sorted(set(gold.case_id)-set(judged.case_id)); raise ValueError(f'Run incomplete: {len(c)}/30; missing={missing}')
    c['human_score']=c.human_score.astype(int); c['judge_score']=c.judge_score.astype(int)
    c['signed_difference']=c.judge_score-c.human_score; c['absolute_difference']=c.signed_difference.abs(); c['exact_match']=c.human_score==c.judge_score; c['within_one']=c.absolute_difference<=1
    human=c.human_score; judge=c.judge_score; labels=[0,1,2,3,4]
    exact=float(c.exact_match.mean()); within=float(c.within_one.mean()); mae=float(c.absolute_difference.mean())
    lin=weighted_cohen_kappa(human,judge,labels,'linear'); quad=weighted_cohen_kappa(human,judge,labels,'quadratic')
    hp=human>0; jp=judge>0; tp=int((hp&jp).sum()); fp=int((~hp&jp).sum()); fn=int((hp&~jp).sum()); tn=int((~hp&~jp).sum())
    precision=safe_rate(tp,tp+fp); recall=safe_rate(tp,tp+fn); f1=safe_rate(2*precision*recall,precision+recall)
    severe=int((c.absolute_difference>=2).sum()); over2=int((c.signed_difference>=2).sum()); under2=int((c.signed_difference<=-2).sum())
    false_zero=int(((human>0)&(judge==0)).sum()); false_pos=int(((human==0)&(judge>0)).sum())
    cue_count=pd.to_numeric(c.get('deterministic_direct_cue_count',pd.Series([0]*len(c))),errors='coerce').fillna(0).astype(int)
    cue_severe_fp=int(((cue_count>0)&(human==0)&(judge>=2)).sum())
    checks={'within_one_ge_0_95':within>=0.95,'exact_ge_0_50':exact>=0.50,'quadratic_kappa_ge_0_80':quad>=0.80,'linear_kappa_ge_0_60':lin>=0.60,'absolute_difference_ge_2_le_1_case':severe<=1,'binary_precision_ge_0_90':precision>=0.90,'binary_recall_ge_0_80':recall>=0.80,'deterministic_cue_severe_false_positives_eq_0':cue_severe_fp==0}
    matrix=pd.DataFrame(confusion_matrix_no_sklearn(human,judge,labels),index=[f'human_{x}' for x in labels],columns=[f'judge_{x}' for x in labels])
    agreement=c.groupby('human_score',as_index=False).agg(cases=('case_id','count'),exact_agreement=('exact_match','mean'),within_one_agreement=('within_one','mean'),mean_absolute_difference=('absolute_difference','mean'))
    agreement_q=c.groupby('query_id',as_index=False).agg(cases=('case_id','count'),exact_agreement=('exact_match','mean'),within_one_agreement=('within_one','mean'),mean_absolute_difference=('absolute_difference','mean'))
    largest=c.sort_values(['absolute_difference','case_id'],ascending=[False,True])[[ 'case_id','query_id','title','human_score','judge_score','signed_difference','absolute_difference']]
    summary={'analysis_scope':expected_scope,'evaluation_dataset_version':'2.0.0','cases_compared':30,'exact_agreement':exact,'within_one_agreement':within,'mean_absolute_difference':mae,'linear_weighted_cohens_kappa':lin,'quadratic_weighted_cohens_kappa':quad,'max_absolute_difference':int(c.absolute_difference.max()),'human_score_distribution':{str(k):int(v) for k,v in human.value_counts().sort_index().items()},'judge_score_distribution':{str(k):int(v) for k,v in judge.value_counts().sort_index().items()},'binary_relevance_0_vs_positive':{'true_positive':tp,'false_positive':fp,'false_negative':fn,'true_negative':tn,'precision':precision,'recall':recall,'f1':f1},'error_counts':{'false_zeros':false_zero,'false_positive_relevance':false_pos,'absolute_difference_ge_2':severe,'over_promotions_by_2_or_more':over2,'under_promotions_by_2_or_more':under2},'reference_threshold_checks':checks,'all_reference_thresholds_met':bool(all(checks.values())),'transport_recovery':{'override_count':len(metadata.get('transport_overrides',[])),'case_ids':[x.get('case_id') for x in metadata.get('transport_overrides',[]) if isinstance(x,dict)]},'run_metadata':metadata}
    if finality_note: summary['finality_note']=finality_note
    (run_dir/f'{output_prefix}_summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    c.to_csv(run_dir/f'human_vs_judge.{output_prefix}.csv',index=False,encoding='utf-8'); matrix.to_csv(run_dir/f'confusion_matrix.{output_prefix}.csv',encoding='utf-8'); agreement.to_csv(run_dir/f'agreement_by_human_score.{output_prefix}.csv',index=False,encoding='utf-8'); agreement_q.to_csv(run_dir/f'agreement_by_query.{output_prefix}.csv',index=False,encoding='utf-8'); largest.to_csv(run_dir/f'largest_disagreements.{output_prefix}.csv',index=False,encoding='utf-8')
    print(f'{output_prefix.replace("_"," ").title()} summary'); print('-'*32); print('Cases compared:            30/30'); print(f'Exact agreement:           {exact:.1%}'); print(f'Within ±1 agreement:       {within:.1%}'); print(f'Mean absolute difference:  {mae:.3f}'); print(f'Linear weighted kappa:     {lin:.3f}'); print(f'Quadratic weighted kappa:  {quad:.3f}'); print(); print('Binary relevance (0 vs >0)'); print('--------------------------'); print(f'TP={tp} FP={fp} FN={fn} TN={tn}'); print(f'Precision={precision:.1%} Recall={recall:.1%} F1={f1:.1%}'); print(); print('Error counts'); print('------------'); print(f'False zeros:               {false_zero}'); print(f'False-positive relevance:  {false_pos}'); print(f'|difference| >= 2:         {severe}'); print(f'Over-promotions >= 2:      {over2}'); print(f'Under-promotions >= 2:     {under2}'); print(); print('Pre-registered reference thresholds'); print('-----------------------------------'); [print(f'{"PASS" if v else "FAIL":4}  {k}') for k,v in checks.items()]; print(); print('Reference-threshold comparison: '+('ALL MET' if all(checks.values()) else 'SOME NOT MET')); print(); print('Largest disagreements'); print('---------------------'); print(largest.head(15).to_string(index=False))
    return summary

def main():
    p=argparse.ArgumentParser(); p.add_argument('--run',required=True); args=p.parse_args(); validate_split_contract(require_release=True)
    run_dir=resolve_run_dir(args.run)
    note='This is the one-time final holdout result for frozen Judge v0.18.0 on fresh unseen candidates. Reference thresholds are descriptive only; do not tune or relabel against this holdout.'
    analyze_run(run_dir,'fresh_unseen_final_holdout','fresh_unseen_final_holdout','fresh_holdout',finality_note=note)
    print(); print('Finality note'); print('-------------'); print(note)
if __name__=='__main__': main()
