from __future__ import annotations
import hashlib,json,math
from pathlib import Path
from typing import Any
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
SYS=ROOT/'evals/system_evaluation'
RUN=ROOT/'evals/runs/semantic_relevance_v5_development/20261026T124615Z_candidate_collection'
CONTRACT=SYS/'semantic_relevance_v5_development_factorial_reranking_contract.v0.1.0.json'
CAND_CONTRACT=SYS/'semantic_relevance_v5_development_candidate_collection_contract.v0.1.0.json'
CAND_LOCK=RUN/'v5d_candidate_collection.lock.json'
R0_DENSE=RUN/'v5d_r0_dense_top50.csv'; R1_DENSE=RUN/'v5d_r1_dense_top50.csv'; BM25=RUN/'v5d_bm25_top50.csv'
R0_UNION=RUN/'v5d_r0_hybrid_union.csv'; R1_UNION=RUN/'v5d_r1_hybrid_union.csv'; CAND_LAT=RUN/'v5d_candidate_collection_latency.csv'
DESIGN_LOCK=SYS/'semantic_relevance_v5_development_design.v0.1.0.lock.json'
ARCH_LOCK=SYS/'semantic_relevance_v5_r1_k1_architecture.v0.1.0.lock.json'
IMPL_LOCK=SYS/'semantic_relevance_v5_r1_k1_implementation.v0.1.0.lock.json'
QUAL_LOCK=SYS/'semantic_relevance_v5_r1_k1_model_qualification.v0.1.0.lock.json'
AUTHORING_LOCK=SYS/'semantic_relevance_v5_development_authoring.v0.1.0.lock.json'
K0_SOURCE=SYS/'semantic_relevance_v3_reranker.py'; V4B_RANK=SYS/'semantic_relevance_v4b_ranking.py'; K1_SOURCE=SYS/'semantic_relevance_v5_k1_reranker.py'
EXPECTED={
'contract':'622f2fbb599d455d1641535884a0adf08c7e4b6b52f7db227a2b3845022c1683','candidate_contract':'0b9dc202a43a64099ee03257f18f644e06d398dedb5d07391e9555d87cba6ad5','candidate_lock':'d6c863d5dcd69ac6230ac00e60ee05aa9867701b2d15e42b91de0d05c8cd2e03',
'r0_dense':'c118140b77c7813c12f865bab591229141e295e8573940e9a6146fc5c895bd22','r1_dense':'94f8b03640bc542b1caf5379cfc11529d881bd4613bd4fcc0057e13c308bc4c8','bm25':'22f3f4a76e509965b3fa1ad2baecdc78a960e87faec2a29f743ed7ae6e43bb1c','r0_union':'221f2f6036a8cb3564a47fcfbc8f197ff12bab6776c1cafcafa83b459a1f916f','r1_union':'e54989797c1ebc7c12eacecf296172c3f38342ee3be679bab56682b6bd63d09d',
'design':'89ee03a4328c268a8128d5e373854b92634c41eff204b53f7b93b78c481311c0','arch':'2b298d3263c7a88090a983700da34c30767a71877a4b32ea2e1ae8f3524ade87','impl':'4d166df6b1acff66b23829126ee18c080bd3756de7ed98953e4520db8de9f72b','qual':'67b1dd15403b5825086cce4c9e5c5af14cc883945f5a9626abaa4e5894b77211','authoring':'ffca0f7a10c4d05ee79fc08d584df5793f93de2242a2ed2bf09436e2df80ba3e',
'k0_source':'5e7a7d6b35c0674f32644e758218830d8a340a61588e2ff0b3f20582fdbb0f60','v4b_rank':'5729b078cee1a2706a8c3d9bcaa9e8449292bc1f9f1a2816053b84758db7d677','k1_source':'4a6ba7a13d03176108c65512a4185845b532ff5d85f3af512af013b8c8d60207'}
IDS=[f'V5D{i:02d}' for i in range(1,25)]
OUTPUTS=['v5d_B_r0k0_ranked_full_union.csv','v5d_B_r0k0_recommendations.csv','v5d_R_r1k0_ranked_full_union.csv','v5d_R_r1k0_recommendations.csv','v5d_K_r0k1_ranked_full_union.csv','v5d_K_r0k1_recommendations.csv','v5d_RK_r1k1_ranked_full_union.csv','v5d_RK_r1k1_recommendations.csv','v5d_reranking_latency.csv','v5d_factorial_reranking_metadata.json','v5d_factorial_reranking.lock.json']
def sha(p:Path)->str:return hashlib.sha256(p.read_bytes()).hexdigest()
def lfsha(p:Path)->str:
 b=p.read_bytes();return hashlib.sha256(b.replace(b'\r\n',b'\n').replace(b'\r',b'\n')).hexdigest()
def req(c:bool,m:str)->None:
 if not c:raise ValueError(m)
def load(p:Path)->dict[str,Any]:return json.loads(p.read_text(encoding='utf-8-sig'))
def vr(p,e,l):
 req(p.exists(),f'Missing {l}: {p}');a=sha(p);req(a==e,f'{l} hash mismatch: expected={e} actual={a}');print(f'PASS  {l} raw SHA-256 verified')
def vt(p,e,l):
 req(p.exists(),f'Missing {l}: {p}');a=sha(p);n=lfsha(p);req(a==e or n==e,f'{l} hash mismatch: expected={e} raw={a} lf={n}');print(f"PASS  {l} {'raw' if a==e else 'canonical-LF'} SHA-256 verified")
def _validate_union(path:Path,expected_hash:str,expected_rows:int,label:str)->pd.DataFrame:
 vr(path,expected_hash,label);f=pd.read_csv(path,dtype=str,keep_default_na=False,encoding='utf-8');need={'query_id','query','stratum','hybrid_retrieval_rank','isbn13','title','description'};req(not(need-set(f.columns)),f'{label} missing columns: {sorted(need-set(f.columns))}');f['hybrid_retrieval_rank']=pd.to_numeric(f['hybrid_retrieval_rank'],errors='raise').astype(int);req(len(f)==expected_rows,f'{label} rows changed: expected={expected_rows} actual={len(f)}');req(sorted(f.query_id.unique())==IDS,f'{label} query IDs changed');req(not f.duplicated(['query_id','isbn13']).any(),f'Duplicate pair in {label}');
 for qid,g in f.groupby('query_id',sort=True):req(50<=len(g)<=100,f'{qid}: {label} size outside [50,100]');req(sorted(g.hybrid_retrieval_rank.tolist())==list(range(1,len(g)+1)),f'{qid}: {label} ranks not contiguous')
 return f

def validate():
 vr(CONTRACT,EXPECTED['contract'],'V5D factorial reranking contract');vr(CAND_CONTRACT,EXPECTED['candidate_contract'],'V5D candidate-collection contract');vr(CAND_LOCK,EXPECTED['candidate_lock'],'V5D candidate lock');vr(DESIGN_LOCK,EXPECTED['design'],'V5 design lock');vr(ARCH_LOCK,EXPECTED['arch'],'V5 architecture lock');vr(IMPL_LOCK,EXPECTED['impl'],'V5 implementation lock');vr(QUAL_LOCK,EXPECTED['qual'],'V5 model qualification lock');vr(AUTHORING_LOCK,EXPECTED['authoring'],'V5D authoring lock');vt(K0_SOURCE,EXPECTED['k0_source'],'K0 reranker source');vt(V4B_RANK,EXPECTED['v4b_rank'],'V4B score-only ranking source');vt(K1_SOURCE,EXPECTED['k1_source'],'K1 reranker source');vr(R0_DENSE,EXPECTED['r0_dense'],'R0 dense artifact');vr(R1_DENSE,EXPECTED['r1_dense'],'R1 dense artifact');vr(BM25,EXPECTED['bm25'],'shared BM25 artifact')
 lock=load(CAND_LOCK);req(lock.get('status')=='V5D_R0_R1_CANDIDATES_FROZEN_BEFORE_RERANKING_OR_JUDGING','Unexpected candidate-lock status');req(lock.get('r0_repeat_order_identical_all_queries') is True and lock.get('r1_repeat_order_identical_all_queries') is True,'Candidate repeatability not frozen PASS');req(lock.get('reranker_inference_performed') is False and lock.get('judge_execution_performed') is False,'Candidate lock says later execution already occurred');outs=lock.get('outputs',{});req(outs.get('v5d_r0_hybrid_union.csv')==EXPECTED['r0_union'] and outs.get('v5d_r1_hybrid_union.csv')==EXPECTED['r1_union'],'Candidate lock union hashes mismatch')
 r0=_validate_union(R0_UNION,EXPECTED['r0_union'],2231,'R0 frozen hybrid union');r1=_validate_union(R1_UNION,EXPECTED['r1_union'],2242,'R1 frozen hybrid union')
 req(CAND_LAT.exists(),'Missing frozen candidate latency artifact');lat=pd.read_csv(CAND_LAT,dtype=str,keep_default_na=False,encoding='utf-8');req(outs.get(CAND_LAT.name)==sha(CAND_LAT),'Candidate latency hash differs from frozen candidate lock');need={'query_id','r0_dense_first_pass_seconds','r1_dense_first_pass_seconds','shared_bm25_seconds'};req(not(need-set(lat.columns)),'Candidate latency columns changed');req(sorted(lat.query_id.unique())==IDS and len(lat)==24,'Candidate latency must be exact V5D24');
 for c in ['r0_dense_first_pass_seconds','r1_dense_first_pass_seconds','shared_bm25_seconds']:
  x=pd.to_numeric(lat[c],errors='raise');req(x.map(math.isfinite).all() and (x>=0).all(),f'Invalid latency column {c}')
 c=load(CONTRACT);req(c.get('status')=='FROZEN_BY_PACKAGE_BEFORE_V5D_K0_K1_RERANKING','Unexpected reranking contract status');req(c.get('K0',{}).get('model_revision')=='2cfc18c9415c912f9d8155881c133215df768a70','K0 revision changed');req(c.get('K1',{}).get('model_revision')=='953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e','K1 revision changed')
 for name in OUTPUTS:req(not (RUN/name).exists(),f'Refusing because reranking output already exists: {RUN/name}')
 return r0,r1,lat

def main():
 print('Semantic relevance V5 - factorial reranking preflight');print('---------------------------------------------------');print('New retrieval: DISABLED');print('K0/K1 inference: DISABLED');print('Judge execution: DISABLED');print('Development selection: NOT CALCULATED');print();r0,r1,_=validate();over=0
 for qid in IDS:over+=len(set(r0[r0.query_id==qid].isbn13)&set(r1[r1.query_id==qid].isbn13))
 print();print('V5D FACTORIAL RERANKING PREFLIGHT: PASS');print(f'R0 union rows: {len(r0)}');print(f'R1 union rows: {len(r1)}');print(f'R0/R1 same-query candidate overlap rows: {over}');print('Systems to materialize: B=R0K0, R=R1K0, K=R0K1, RK=R1K1');print('K0 passes per union: 2');print('K1 passes per union: 2');print('New retrieval performed: NO');print('Reranker inference performed during preflight: NO');print('Judge calls performed: 0');print('Development selection calculated: NO');print('Next: execute K0/K1 reranking only on the frozen R0/R1 unions.')
if __name__=='__main__':main()
