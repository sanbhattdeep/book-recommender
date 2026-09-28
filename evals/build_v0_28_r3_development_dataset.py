"""Build v0.28.0 r3's 180-case consumed-development dataset.

The v0.27 final holdout has been consumed exactly once and is now diagnostic /
regression evidence only.  This builder therefore combines the frozen 150-case
v0.27 development view with those 30 consumed final-holdout cases.  It refuses
to run unless the one-time holdout lock records FINAL_HOLDOUT_COMPLETED with
judge_run_count=1 and the original holdout bytes still match the frozen hash.
No new unseen v0.28 evidence is created here.
"""
from __future__ import annotations
from pathlib import Path
import hashlib,json,re,unicodedata
import pandas as pd
E=Path(__file__).resolve().parent; D=E/'datasets'; RUNS=E/'runs'
DEV150=D/'semantic_relevance_v0.27_development.v3.1.0.csv'
FINAL30=D/'semantic_relevance_final_holdout.v3.0.0.DO_NOT_RUN_YET.csv'
LOCK=D/'semantic_relevance_final_holdout_lock.v3.0.0.json'
OUT=D/'semantic_relevance_v0.28_development.v4.0.0.csv'
MANIFEST=D/'semantic_relevance_v0.28_development_manifest.v1.0.0.json'
EXPECTED_DEV150_SHA='72810fa61cc71e28f05beaec9664f2ccdee55d5834272c44e2f1cc97cafeee7d'
EXPECTED_FINAL_SHA='cce944bd48b102452c024134560b7b87866b85f5fa1bc1194775138b8976eb2e'
def sha(p:Path)->str:return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p:Path)->pd.DataFrame:return pd.read_csv(p,dtype={'case_id':str,'isbn13':str},keep_default_na=False,encoding='utf-8-sig')
def norm(v:str)->str:return re.sub(r'\s+',' ',unicodedata.normalize('NFKC',str(v)).casefold().strip())
def ta(df):return {norm(t)+'||'+norm(a) for t,a in zip(df['title'],df['authors']) if norm(t) or norm(a)}
def main():
    for p in (DEV150,FINAL30,LOCK):
        if not p.exists():raise FileNotFoundError(f'Required source not found: {p}')
    if sha(DEV150)!=EXPECTED_DEV150_SHA:raise ValueError('150-case v0.27 consumed-development hash changed.')
    if sha(FINAL30)!=EXPECTED_FINAL_SHA:raise ValueError('Consumed v0.27 final-holdout bytes changed.')
    lock=json.loads(LOCK.read_text(encoding='utf-8'))
    if lock.get('status')!='FINAL_HOLDOUT_COMPLETED':raise ValueError('v0.27 final holdout is not FINAL_HOLDOUT_COMPLETED.')
    if int(lock.get('judge_run_count',-1))!=1:raise ValueError('v0.27 final holdout judge_run_count must equal 1.')
    dev=load(DEV150); final=load(FINAL30)
    if len(dev)!=150 or dev.case_id.nunique()!=150:raise ValueError('Expected 150 unique consumed v0.27 development cases.')
    if len(final)!=30 or final.case_id.nunique()!=30:raise ValueError('Expected 30 unique consumed v0.27 final-holdout cases.')
    if set(dev.case_id)&set(final.case_id):raise ValueError('case_id overlap between v0.27 development and consumed final holdout.')
    if {x for x in dev.isbn13 if x}&{x for x in final.isbn13 if x}:raise ValueError('ISBN overlap between sources.')
    if ta(dev)&ta(final):raise ValueError('title+author overlap between sources.')
    for name,frame in [('development',dev),('consumed_final_holdout',final)]:
        scores=pd.to_numeric(frame.human_score,errors='raise').astype(int)
        if not scores.isin([0,1,2,3,4]).all():raise ValueError(f'{name} invalid human_score.')
        if not frame.human_reason.astype(str).str.strip().ne('').all():raise ValueError(f'{name} blank human_reason.')
        if set(frame.rubric_version.astype(str))!={'0.1.0'}:raise ValueError(f'{name} rubric mismatch.')
    dev=dev.copy(); final=final.copy()
    if 'development_source' not in dev.columns:dev['development_source']='consumed_v0_27_development_150'
    final['development_source']='consumed_v0_27_final_holdout_30'
    cols=list(dict.fromkeys(list(dev.columns)+list(final.columns)))
    combined=pd.concat([dev.reindex(columns=cols,fill_value=''),final.reindex(columns=cols,fill_value='')],ignore_index=True)
    combined['dataset_version']='4.0.0'; combined['rubric_version']='0.1.0'
    if len(combined)!=180 or combined.case_id.nunique()!=180:raise ValueError('v0.28 development output must contain 180 unique cases.')
    counts={p:int(combined.case_id.str.startswith(p).sum()) for p in ('U_','U2_','U3_')}
    if counts!={'U_':60,'U2_':60,'U3_':60}:raise ValueError(f'Unexpected prefix counts: {counts}')
    combined.to_csv(OUT,index=False,encoding='utf-8')
    manifest={'version':'1.0.0','dataset_version':'4.0.0','rubric_version':'0.1.0','case_count':180,
      'methodological_status':'post_v0_27_final_holdout_consumed_development_only',
      'sources':[{'path':str(DEV150.relative_to(E.parent)),'cases':150,'sha256':sha(DEV150),'status':'consumed_v0_27_development'},
                 {'path':str(FINAL30.relative_to(E.parent)),'cases':30,'sha256':sha(FINAL30),'status':'consumed_v0_27_final_holdout','prior_lock_status':lock.get('status'),'prior_judge_run_count':int(lock.get('judge_run_count'))}],
      'output':str(OUT.relative_to(E.parent)),'output_sha256':sha(OUT),
      'independent_v0_28_evidence':None,
      'methodology_note':'All 180 cases are consumed development/regression evidence. The v0.27 final holdout must never be reused as unseen validation or holdout. Any v0.28 generalization claim requires a newly sampled/blind-labelled unseen pool and a new untouched final holdout.'}
    MANIFEST.write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print('v0.28 r3 consumed-development dataset built')
    print('-----------------------------------------')
    print('PASS  consumed v0.27 development: 150 cases')
    print('PASS  consumed v0.27 final holdout: 30 cases')
    print('PASS  output development: 180 cases')
    print(f'PASS  prefix counts: {counts}')
    print('PASS  v0.27 final holdout hash pinned')
    print('PASS  prior final holdout status = FINAL_HOLDOUT_COMPLETED')
    print('PASS  prior final holdout judge_run_count = 1')
    print('NOTE  no independent v0.28 holdout exists yet')
    print(f'Output SHA-256: {sha(OUT)}')
    print(f'Output: {OUT}')
    print(f'Manifest: {MANIFEST}')
if __name__=='__main__':main()
