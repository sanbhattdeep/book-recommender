"""Split the fully human-labelled fresh pool into 30 validation / 30 locked holdout.

Hard constraints:
- 6 cases from each sampling bucket in each split;
- each query contributes 2 or 3 cases to validation (therefore 3 or 2 holdout);
- split occurs before any fresh v0.18 judge output exists.

Among feasible splits, a deterministic seeded search minimizes human-score imbalance.
"""
from __future__ import annotations
import json, random
from pathlib import Path
import pandas as pd
from fresh_unseen_contract import *

SEED=20260918; ITERATIONS=300000
BUCKETS=['T02','T10','T30','T70','NEG']

def objective(val: pd.DataFrame, all_df: pd.DataFrame) -> int:
    total=all_df['human_score'].astype(int).value_counts().to_dict()
    got=val['human_score'].astype(int).value_counts().to_dict()
    # Minimum possible contribution for an odd total is 1.
    return sum(abs(2*got.get(s,0)-total.get(s,0)) for s in range(5))

def main():
    # Prevent accidental post-judge splitting.
    for run_root in [
        EVALS_DIR/'runs'/'semantic_relevance_v0_18_fresh_validation',
        EVALS_DIR/'runs'/'semantic_relevance_v0_18_fresh_holdout',
    ]:
        if run_root.exists() and any(run_root.iterdir()):
            raise RuntimeError(f'Refusing to create split after fresh judge run artifacts exist: {run_root}')
    df=pd.read_csv(LABELLED_POOL_FILE,dtype={'isbn13':str},encoding='utf-8')
    validate_pool_dataframe(df, require_labels=True)
    by_bucket={b:df.index[df['sampling_bucket'].astype(str)==b].tolist() for b in BUCKETS}
    if any(len(v)!=12 for v in by_bucket.values()): raise ValueError('Every sampling bucket must contain 12 cases')
    rng=random.Random(SEED); best=None; best_key=None; feasible=0
    theoretical_min=sum((df['human_score'].astype(int).value_counts().get(s,0)%2) for s in range(5))
    for i in range(ITERATIONS):
        selected=[]
        for b in BUCKETS: selected.extend(rng.sample(by_bucket[b],6))
        val=df.loc[selected]
        qcounts=val['query_id'].astype(str).value_counts().to_dict()
        if len(qcounts)!=12 or any(c not in (2,3) for c in qcounts.values()): continue
        feasible+=1; obj=objective(val,df)
        case_key=tuple(sorted(val['case_id'].astype(str)))
        key=(obj,case_key)
        if best_key is None or key<best_key:
            best_key=key; best=case_key
            if obj==theoretical_min: break
    if best is None: raise RuntimeError('No feasible split found; increase ITERATIONS.')
    val=df[df['case_id'].astype(str).isin(set(best))].copy().sort_values(['query_id','sampling_bucket','case_id'])
    hold=df[~df['case_id'].astype(str).isin(set(best))].copy().sort_values(['query_id','sampling_bucket','case_id'])
    if len(val)!=30 or len(hold)!=30: raise AssertionError('Split size bug')
    if set(val['sampling_bucket'].astype(str).value_counts().values)!={6}: raise AssertionError('Validation bucket balance bug')
    if set(hold['sampling_bucket'].astype(str).value_counts().values)!={6}: raise AssertionError('Holdout bucket balance bug')
    val.to_csv(VALIDATION_FILE,index=False,encoding='utf-8'); hold.to_csv(HOLDOUT_FILE,index=False,encoding='utf-8')
    score_counts=lambda x:{str(k):int(v) for k,v in x['human_score'].astype(int).value_counts().sort_index().items()}
    manifest={
        'version':'2.0.0','split_seed':SEED,'search_iterations_limit':ITERATIONS,'feasible_candidates_examined':feasible,
        'score_balance_objective':int(best_key[0]),'theoretical_min_score_balance_objective':int(theoretical_min),
        'split_created_before_judge_outputs':True,'judge_outputs_used_for_split':False,
        'labelled_pool_sha256':sha256_file(LABELLED_POOL_FILE),'validation_sha256':sha256_file(VALIDATION_FILE),'holdout_sha256':sha256_file(HOLDOUT_FILE),
        'validation_case_ids':val['case_id'].astype(str).tolist(),'holdout_case_ids':hold['case_id'].astype(str).tolist(),
        'validation_human_score_distribution':score_counts(val),'holdout_human_score_distribution':score_counts(hold),
        'validation_sampling_bucket_counts':{str(k):int(v) for k,v in val['sampling_bucket'].astype(str).value_counts().sort_index().items()},
        'holdout_sampling_bucket_counts':{str(k):int(v) for k,v in hold['sampling_bucket'].astype(str).value_counts().sort_index().items()},
        'validation_query_counts':{str(k):int(v) for k,v in val['query_id'].astype(str).value_counts().sort_index().items()},
        'holdout_query_counts':{str(k):int(v) for k,v in hold['query_id'].astype(str).value_counts().sort_index().items()},
    }
    SPLIT_MANIFEST_FILE.write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    validate_split_contract(require_release=False)
    print('Fresh unseen split created')
    print('--------------------------')
    print('Validation: 30 | Holdout: 30')
    print(f'Score-balance objective: {best_key[0]} (theoretical minimum {theoretical_min})')
    print('Each sampling bucket: 6 validation / 6 holdout')
    print('Each query: 2 or 3 validation cases')
    print('Judge outputs used for split: NO')
    print(f'Locked holdout: {HOLDOUT_FILE}')

if __name__=='__main__': main()
