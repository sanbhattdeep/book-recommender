"""Build the 150-case v0.27 r6 consumed-development dataset.

Combines the already-consumed 120-case v0.27 development set with the first
30-case v0.27 validation set after that validation was evaluated and reviewed.
The independent 30-case final holdout is NOT read into the output; its frozen
hash and LOCKED_DO_NOT_RUN / judge_run_count=0 state are verified only as a
safety precondition.
"""
from __future__ import annotations
from pathlib import Path
import hashlib, json, re, unicodedata
import pandas as pd

EVALS=Path(__file__).resolve().parent
D=EVALS/'datasets'
DEV120=D/'semantic_relevance_v0.27_development.v3.0.0.csv'
VAL30=D/'semantic_relevance_validation.v3.0.0.csv'
FINAL30=D/'semantic_relevance_final_holdout.v3.0.0.DO_NOT_RUN_YET.csv'
LOCK=D/'semantic_relevance_final_holdout_lock.v3.0.0.json'
OUT=D/'semantic_relevance_v0.27_development.v3.1.0.csv'
MANIFEST=D/'semantic_relevance_v0.27_development_manifest.v1.1.0.json'
EXPECTED_VAL_SHA='55f4099240c9108d91a3342d809ea73210568a1dd0df98d7b056742ea35cb24f'
EXPECTED_FINAL_SHA='cce944bd48b102452c024134560b7b87866b85f5fa1bc1194775138b8976eb2e'

def sha(p:Path)->str: return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p:Path)->pd.DataFrame: return pd.read_csv(p,dtype={'case_id':str,'isbn13':str},keep_default_na=False,encoding='utf-8-sig')
def norm(v:str)->str:
    s=unicodedata.normalize('NFKC',str(v)).casefold().strip()
    return re.sub(r'\s+',' ',s)
def title_author(df:pd.DataFrame)->set[str]:
    return {norm(t)+'||'+norm(a) for t,a in zip(df['title'],df['authors']) if norm(t) or norm(a)}

def main()->None:
    for p in (DEV120,VAL30,FINAL30,LOCK):
        if not p.exists(): raise FileNotFoundError(f'Required source not found: {p}')
    if sha(VAL30)!=EXPECTED_VAL_SHA: raise ValueError('Validation CSV hash differs from the frozen consumed validation split.')
    if sha(FINAL30)!=EXPECTED_FINAL_SHA: raise ValueError('Final holdout hash changed; refusing to build r6 development data.')
    lock=json.loads(LOCK.read_text(encoding='utf-8'))
    if lock.get('status')!='LOCKED_DO_NOT_RUN': raise ValueError('Final holdout is not LOCKED_DO_NOT_RUN.')
    if int(lock.get('judge_run_count',-1))!=0: raise ValueError('Final holdout judge_run_count is not 0.')
    dev=load(DEV120); val=load(VAL30)
    if len(dev)!=120 or dev['case_id'].nunique()!=120: raise ValueError('Expected 120 unique source development cases.')
    if len(val)!=30 or val['case_id'].nunique()!=30: raise ValueError('Expected 30 unique consumed validation cases.')
    if not val['case_id'].str.startswith('U3_').all(): raise ValueError('Consumed validation must contain only U3_ cases.')
    if set(dev['case_id']) & set(val['case_id']): raise ValueError('case_id overlap between 120-case development and validation.')
    dev_isbn={x for x in dev['isbn13'].astype(str) if x}; val_isbn={x for x in val['isbn13'].astype(str) if x}
    if dev_isbn & val_isbn: raise ValueError('ISBN overlap between source development and consumed validation.')
    if title_author(dev)&title_author(val): raise ValueError('title+author overlap between source development and consumed validation.')
    for name,frame in [('development',dev),('validation',val)]:
        if not frame['human_score'].astype(int).isin([0,1,2,3,4]).all(): raise ValueError(f'{name} contains invalid human_score.')
        if set(frame['rubric_version'].astype(str))!={'0.1.0'}: raise ValueError(f'{name} rubric version mismatch.')
        if not frame['human_reason'].astype(str).str.strip().ne('').all(): raise ValueError(f'{name} contains blank human_reason.')
                # Historical consumed-development rows may carry provenance-preserving
        # statuses such as adjudicated/revised labels.  Do not rewrite those
        # statuses merely to construct the r6 consumed-development dataset.
        #
        # What matters here is that no source row is still awaiting human
        # review and that every row has a valid human gold score/reason.

        review_status = (
            frame["review_status"]
            .fillna("")
            .astype(str)
            .str.strip()
        )

        unfinished_statuses = {
            "",
            "UNREVIEWED",
            "UNLABELLED",
        }

        unfinished_mask = (
            review_status
            .str.upper()
            .isin(unfinished_statuses)
        )

        if unfinished_mask.any():
            bad = frame.loc[
                unfinished_mask,
                ["case_id", "review_status"],
            ]

            raise ValueError(
                f"{name} contains unfinished review statuses:\n"
                + bad.to_string(index=False)
            )

        scores = pd.to_numeric(
            frame["human_score"],
            errors="coerce",
        )

        if (
            scores.isna().any()
            or not scores.isin([0, 1, 2, 3, 4]).all()
        ):
            raise ValueError(
                f"{name} contains missing/invalid human_score values."
            )

        reasons = (
            frame["human_reason"]
            .fillna("")
            .astype(str)
            .str.strip()
        )

        if reasons.eq("").any():
            bad_ids = frame.loc[
                reasons.eq(""),
                "case_id",
            ].tolist()

            raise ValueError(
                f"{name} contains blank human_reason values: "
                f"{bad_ids}"
            )
    dev=dev.copy(); val=val.copy()
    if 'development_source' not in dev.columns: dev['development_source']='consumed_pre_r6_development_120'
    val['development_source']='consumed_v0_27_validation_30'
    columns=list(dict.fromkeys(list(dev.columns)+list(val.columns)))
    combined=pd.concat([dev.reindex(columns=columns,fill_value=''),val.reindex(columns=columns,fill_value='')],ignore_index=True)
    combined['dataset_version']='3.1.0'; combined['rubric_version']='0.1.0'
    if len(combined)!=150 or combined['case_id'].nunique()!=150: raise ValueError('r6 development output must contain 150 unique cases.')
    counts={p:int(combined['case_id'].str.startswith(p).sum()) for p in ('U_','U2_','U3_')}
    if counts!={'U_':60,'U2_':60,'U3_':30}: raise ValueError(f'Unexpected case-prefix counts: {counts}')
    combined.to_csv(OUT,index=False,encoding='utf-8')
    manifest={
      'version':'1.1.0','dataset_version':'3.1.0','rubric_version':'0.1.0','case_count':150,
      'methodological_status':'post_validation_consumed_development_only',
      'sources':[{'path':str(DEV120.relative_to(EVALS.parent)),'cases':120,'sha256':sha(DEV120)}, {'path':str(VAL30.relative_to(EVALS.parent)),'cases':30,'sha256':sha(VAL30),'status':'consumed_after_r5_unseen_validation'}],
      'output':str(OUT.relative_to(EVALS.parent)),'output_sha256':sha(OUT),
      'independent_final_holdout':{'path':str(FINAL30.relative_to(EVALS.parent)),'cases':30,'sha256':sha(FINAL30),'status':lock.get('status'),'judge_run_count':int(lock.get('judge_run_count',-1)),'included_in_development':False},
      'methodology_note':'The 30-case validation set became development evidence only after its first r5 evaluation and diagnostic review. The 30-case final holdout remains untouched and is the only independent evidence for r6.'
    }
    MANIFEST.write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print('v0.27 r6 consumed-development dataset built')
    print('------------------------------------------')
    print('PASS  source development: 120 cases')
    print('PASS  consumed validation: 30 cases')
    print('PASS  output development: 150 cases')
    print(f'PASS  prefix counts: {counts}')
    print('PASS  validation hash pinned')
    print('PASS  final holdout hash pinned and still LOCKED_DO_NOT_RUN')
    print('PASS  final holdout judge_run_count = 0')
    print(f'Output SHA-256: {sha(OUT)}')
    print(f'Output: {OUT}')
    print(f'Manifest: {MANIFEST}')
if __name__=='__main__': main()
