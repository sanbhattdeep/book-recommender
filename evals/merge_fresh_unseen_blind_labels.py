"""Merge completed blind labels back into canonical pool by opaque ID mapping."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import pandas as pd
from fresh_unseen_contract import REPO_ROOT, POOL_FILE, LABELLED_POOL_FILE, validate_pool_dataframe, sha256_file

MAP_FILE=REPO_ROOT/'evals'/'datasets'/'semantic_relevance_fresh_unseen_blind_map.v2.0.0.json'
DEFAULT_CSV=REPO_ROOT/'evals'/'datasets'/'semantic_relevance_fresh_unseen_pool.v2.0.0.BLIND_LABELING.csv'

def main():
    p=argparse.ArgumentParser(); p.add_argument('--labels',type=Path,default=DEFAULT_CSV); args=p.parse_args()
    labels=args.labels if args.labels.is_absolute() else REPO_ROOT/args.labels
    mapping=json.loads(MAP_FILE.read_text(encoding='utf-8'))
    if sha256_file(POOL_FILE)!=mapping['canonical_pool_sha256']: raise ValueError('Canonical pool changed after blind mapping was created.')
    if labels.suffix.lower()=='.xlsx':
        try: blind=pd.read_excel(labels,sheet_name='Labeling',dtype=str,keep_default_na=False)
        except ImportError as e: raise RuntimeError('For XLSX input run with: uv run --with openpyxl python evals/merge_fresh_unseen_blind_labels.py --labels <file.xlsx>') from e
    else:
        blind=pd.read_csv(labels,dtype=str,encoding='utf-8',keep_default_na=False)
    required={'blind_label_id','human_score','human_reason','review_status'}
    missing=required-set(blind.columns)
    if missing: raise ValueError(f'Blind labels missing columns: {sorted(missing)}')
    if blind['blind_label_id'].duplicated().any(): raise ValueError('Duplicate blind_label_id')
    expected=set(mapping['mapping']); actual=set(blind['blind_label_id'].astype(str))
    if actual!=expected: raise ValueError(f'Blind label IDs mismatch. missing={sorted(expected-actual)} extra={sorted(actual-expected)}')
    blind['human_score']=pd.to_numeric(blind['human_score'],errors='raise').astype(int)
    if not blind['human_score'].isin([0,1,2,3,4]).all(): raise ValueError('human_score must be 0..4')
    if not blind['human_reason'].astype(str).str.strip().ne('').all(): raise ValueError('Every row needs human_reason')
    if set(blind['review_status'].astype(str))!={'LABELLED'}: raise ValueError('Every row must be review_status=LABELLED')
    label_by_case={mapping['mapping'][r.blind_label_id]:(int(r.human_score),str(r.human_reason),str(r.review_status)) for r in blind.itertuples(index=False)}
    pool=pd.read_csv(POOL_FILE,dtype={'isbn13':str},encoding='utf-8')
    pool['human_score']=[label_by_case[str(cid)][0] for cid in pool['case_id']]
    pool['human_reason']=[label_by_case[str(cid)][1] for cid in pool['case_id']]
    pool['review_status']=[label_by_case[str(cid)][2] for cid in pool['case_id']]
    validate_pool_dataframe(pool, require_labels=True)
    pool.to_csv(LABELLED_POOL_FILE,index=False,encoding='utf-8')
    print(f'Wrote labelled canonical pool: {LABELLED_POOL_FILE}')

if __name__=='__main__': main()
