"""Create a deterministic, opaque-ID blind human-labeling CSV + private map."""
from __future__ import annotations
import hashlib, json, random
from pathlib import Path
import pandas as pd
from fresh_unseen_contract import REPO_ROOT, POOL_FILE, validate_pool_dataframe, sha256_file

SEED=20260917
OUT_DIR=REPO_ROOT/'evals'/'datasets'
BLIND_FILE=OUT_DIR/'semantic_relevance_fresh_unseen_pool.v2.0.0.BLIND_LABELING.csv'
MAP_FILE=OUT_DIR/'semantic_relevance_fresh_unseen_blind_map.v2.0.0.json'

def main():
    df=pd.read_csv(POOL_FILE,dtype={'isbn13':str},encoding='utf-8')
    validate_pool_dataframe(df, require_labels=False)
    indices=list(range(len(df))); random.Random(SEED).shuffle(indices)
    shuffled=df.iloc[indices].reset_index(drop=True)
    blind=pd.DataFrame({
        'blind_label_id':[f'B{i:03d}' for i in range(1,len(shuffled)+1)],
        'query':shuffled['query'], 'title':shuffled['title'], 'authors':shuffled['authors'],
        'description':shuffled['description'], 'human_score':['']*len(shuffled),
        'human_reason':['']*len(shuffled), 'review_status':['UNLABELLED']*len(shuffled),
    })
    blind.to_csv(BLIND_FILE,index=False,encoding='utf-8')
    mapping={bid:case for bid,case in zip(blind['blind_label_id'],shuffled['case_id'].astype(str))}
    payload={
        'version':'2.0.0','blind_order_seed':SEED,'canonical_pool_sha256':sha256_file(POOL_FILE),
        'blind_csv_sha256':sha256_file(BLIND_FILE),'mapping':mapping,
        'note':'Keep this mapping separate while labeling. blind_label_id intentionally hides query rank/random-negative identity.',
    }
    MAP_FILE.write_text(json.dumps(payload,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print(f'Blind labeling CSV: {BLIND_FILE}')
    print(f'Private mapping:     {MAP_FILE}')
    print('Do not inspect/use the mapping while assigning human labels.')

if __name__=='__main__': main()
