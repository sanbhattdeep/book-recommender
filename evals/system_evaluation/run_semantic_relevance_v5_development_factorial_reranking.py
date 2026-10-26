from __future__ import annotations
import argparse,gc,hashlib,json,math,time
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import pandas as pd
import verify_semantic_relevance_v5_development_factorial_reranking as boundary
from semantic_relevance_v3_reranker import RerankerRuntimeConfig,candidate_text as k0_text,load_cross_encoder,score_query_candidate_pairs as score_k0
from semantic_relevance_v4b_ranking import rank_v4b
from semantic_relevance_v5_k1_reranker import K1RuntimeConfig,candidate_text as k1_text,load_reranker,score_query_candidate_pairs as score_k1,rank_k1
ROOT=boundary.ROOT;RUN=boundary.RUN
EXPECTED_VERIFIER='95a5f9f396211c5c8ee75a366a4d911d44c97e10fa3bc5006bbd4a72727c1f0b'
def sha(p:Path)->str:return hashlib.sha256(p.read_bytes()).hexdigest()
def lfsha(p:Path)->str:
 b=p.read_bytes();return hashlib.sha256(b.replace(b'\r\n',b'\n').replace(b'\r',b'\n')).hexdigest()
def req(c:bool,m:str)->None:
 if not c:raise ValueError(m)
def score_union(frame,*,system,model_kind,tokenizer,model,device):
 ranked_parts=[];lat=[];max_delta=0.0
 for qid,g in frame.groupby('query_id',sort=True):
  g=g.sort_values('hybrid_retrieval_rank',kind='stable').copy();query=str(g['query'].iloc[0]);queries=[query]*len(g)
  if model_kind=='K0':texts=[k0_text(t,d) for t,d in zip(g.title,g.description)]; t=time.perf_counter();s1=score_k0(queries=queries,candidates=texts,tokenizer=tokenizer,model=model,device=device,max_length_tokens=512,batch_size=8);first=time.perf_counter()-t;s2=score_k0(queries=queries,candidates=texts,tokenizer=tokenizer,model=model,device=device,max_length_tokens=512,batch_size=8);p1=g.copy();p2=g.copy();p1['reranker_score']=s1;p2['reranker_score']=s2;r1=rank_v4b(p1);r2=rank_v4b(p2);rankcol='v4b_rank'
  else:texts=[k1_text(t,d) for t,d in zip(g.title,g.description)];t=time.perf_counter();s1=score_k1(queries=queries,candidates=texts,tokenizer=tokenizer,model=model,device=device);first=time.perf_counter()-t;s2=score_k1(queries=queries,candidates=texts,tokenizer=tokenizer,model=model,device=device);p1=g.copy();p2=g.copy();p1['reranker_score']=s1;p2['reranker_score']=s2;r1=rank_k1(p1);r2=rank_k1(p2);rankcol='k1_rank'
  req(np.isfinite(s1).all() and np.isfinite(s2).all(),f'{system} {qid}: non-finite scores');delta=float(np.max(np.abs(s1-s2)));req(math.isfinite(delta),f'{system} {qid}: non-finite score delta');max_delta=max(max_delta,delta);sig1=tuple(r1.sort_values(rankcol).isbn13.astype(str));sig2=tuple(r2.sort_values(rankcol).isbn13.astype(str));req(sig1==sig2,f'{system} {qid}: repeated rank order differs');r1['system']=system;r1['system_rank']=r1[rankcol].astype(int);ranked_parts.append(r1);lat.append({'query_id':qid,'system':system,'reranker':model_kind,'candidate_rows':len(g),'reranker_first_pass_seconds':first,'repeat_max_score_abs_delta':delta})
 out=pd.concat(ranked_parts,ignore_index=True).sort_values(['query_id','system_rank'],kind='stable').reset_index(drop=True);return out,pd.DataFrame(lat),max_delta

def save_system(system,ranked):
 full=RUN/f'v5d_{system}_{"r0k0" if system=="B" else "r1k0" if system=="R" else "r0k1" if system=="K" else "r1k1"}_ranked_full_union.csv';rec=RUN/f'v5d_{system}_{"r0k0" if system=="B" else "r1k0" if system=="R" else "r0k1" if system=="K" else "r1k1"}_recommendations.csv';ranked.to_csv(full,index=False,encoding='utf-8');top=ranked[ranked.system_rank<=10].copy();top['rank']=top.system_rank.astype(int);top=top.sort_values(['query_id','rank'],kind='stable');counts=top.groupby('query_id').size();req(len(top)==240 and len(counts)==24 and (counts==10).all(),f'{system} top10 is not 24x10');req(not top.duplicated(['query_id','isbn13']).any(),f'{system} duplicate top10 pair');top.to_csv(rec,index=False,encoding='utf-8');return full,rec

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--device',choices=['auto','cuda','cpu'],default='auto');args=ap.parse_args();print('Semantic relevance V5 - B/R/K/RK factorial reranking');print('--------------------------------------------------');print('New retrieval: DISABLED');print('K0/K1 inference: ENABLED');print('Judge execution: DISABLED');print('Development selection: DISABLED');print('Two passes per model/union: REQUIRED');print();r0,r1,candlat=boundary.validate();ver=Path(__file__).with_name('verify_semantic_relevance_v5_development_factorial_reranking.py');req(sha(ver)==EXPECTED_VERIFIER or lfsha(ver)==EXPECTED_VERIFIER,'Factorial reranking preflight verifier changed')
 print('Loading frozen K0 BAAI/bge-reranker-base...');cfg=RerankerRuntimeConfig(model_id='BAAI/bge-reranker-base',model_revision='2cfc18c9415c912f9d8155881c133215df768a70',max_length_tokens=512,truncation='only_second',batch_size=8,device=args.device);tok,m,dev0=load_cross_encoder(cfg);print(f'INFO  K0 device: {dev0}');B,lb,dB=score_union(r0,system='B',model_kind='K0',tokenizer=tok,model=m,device=dev0);print('PASS  B=R0K0 repeated full-union rank order identical');R,lr,dR=score_union(r1,system='R',model_kind='K0',tokenizer=tok,model=m,device=dev0);print('PASS  R=R1K0 repeated full-union rank order identical');del m,tok;gc.collect()
 try:
  import torch
  if torch.cuda.is_available():torch.cuda.empty_cache()
 except Exception:pass
 print('Loading frozen K1 BAAI/bge-reranker-v2-m3...');tok,m,dev1=load_reranker(K1RuntimeConfig(device=args.device));print(f'INFO  K1 device: {dev1}');K,lk,dK=score_union(r0,system='K',model_kind='K1',tokenizer=tok,model=m,device=dev1);print('PASS  K=R0K1 repeated full-union rank order identical');RK,lrk,dRK=score_union(r1,system='RK',model_kind='K1',tokenizer=tok,model=m,device=dev1);print('PASS  RK=R1K1 repeated full-union rank order identical')
 files={};
 for system,df in [('B',B),('R',R),('K',K),('RK',RK)]:
  f,r=save_system(system,df);files[f.name]=sha(f);files[r.name]=sha(r)
 lat=pd.concat([lb,lr,lk,lrk],ignore_index=True);base=candlat.copy();num=['r0_dense_first_pass_seconds','r1_dense_first_pass_seconds','shared_bm25_seconds'];
 for c in num:base[c]=pd.to_numeric(base[c],errors='raise')
 lat['reranker_first_pass_seconds']=pd.to_numeric(lat['reranker_first_pass_seconds'],errors='raise');lat=lat.merge(base[['query_id']+num],on='query_id',how='left',validate='many_to_one');lat['candidate_generation_first_pass_seconds']=np.where(lat.system.isin(['B','K']),lat.r0_dense_first_pass_seconds,lat.r1_dense_first_pass_seconds)+lat.shared_bm25_seconds;lat['system_latency_seconds']=lat.candidate_generation_first_pass_seconds+lat.reranker_first_pass_seconds;latp=RUN/'v5d_reranking_latency.csv';lat.to_csv(latp,index=False,encoding='utf-8');files[latp.name]=sha(latp)
 med={s:float(lat[lat.system==s].system_latency_seconds.median()) for s in ['B','R','K','RK']}
 meta={'status':'V5D_FACTORIAL_RERANKING_COMPLETE','completed_at_utc':datetime.now(timezone.utc).isoformat(),'query_set':'V5D01-V5D24','query_count':24,'source_candidate_lock_sha256':boundary.EXPECTED['candidate_lock'],'reranking_contract_sha256':boundary.EXPECTED['contract'],'K0':{'model_id':'BAAI/bge-reranker-base','revision':'2cfc18c9415c912f9d8155881c133215df768a70','device':dev0,'batch_size':8,'passes_per_union':2,'B_max_abs_score_delta':dB,'R_max_abs_score_delta':dR},'K1':{'model_id':'BAAI/bge-reranker-v2-m3','revision':'953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e','device':dev1,'batch_size':4,'passes_per_union':2,'K_max_abs_score_delta':dK,'RK_max_abs_score_delta':dRK},'latency_definition':boundary.load(boundary.CONTRACT)['latency_measurement'],'median_system_latency_seconds':med,'outputs':files,'new_retrieval_performed':False,'candidate_mutation_performed':False,'judge_execution_performed':False,'development_selection_calculated':False}
 mp=RUN/'v5d_factorial_reranking_metadata.json';mp.write_text(json.dumps(meta,indent=2)+'\n',encoding='utf-8');files[mp.name]=sha(mp)
 lock={'status':'V5D_B_R_K_RK_RERANKING_FROZEN_BEFORE_JUDGING','frozen_at_utc':datetime.now(timezone.utc).isoformat(),'query_set':'V5D01-V5D24','query_count':24,'source_candidate_lock_sha256':boundary.EXPECTED['candidate_lock'],'reranking_contract_sha256':boundary.EXPECTED['contract'],'systems':{'B':'R0K0','R':'R1K0','K':'R0K1','RK':'R1K1'},'repeatability_passes':2,'B_repeat_order_identical':True,'R_repeat_order_identical':True,'K_repeat_order_identical':True,'RK_repeat_order_identical':True,'outputs':files,'new_retrieval_performed':False,'judge_execution_performed':False,'human_relevance_labels_used':False,'development_selection_calculated':False,'release_metrics_calculated':False,'next_stage':'FREEZE_UNIQUE_B_R_K_RK_DISPLAYED_PAIR_UNION_BEFORE_JUDGING'}
 lp=RUN/'v5d_factorial_reranking.lock.json';lp.write_text(json.dumps(lock,indent=2)+'\n',encoding='utf-8');print();print('V5D B/R/K/RK FACTORIAL RERANKING: PASS');print(f'B recommendations SHA-256: {files["v5d_B_r0k0_recommendations.csv"]}');print(f'R recommendations SHA-256: {files["v5d_R_r1k0_recommendations.csv"]}');print(f'K recommendations SHA-256: {files["v5d_K_r0k1_recommendations.csv"]}');print(f'RK recommendations SHA-256: {files["v5d_RK_r1k1_recommendations.csv"]}');print(f'Reranking lock SHA-256: {sha(lp)}');print('New retrieval performed: NO');print('Judge calls performed: 0');print('Development selection calculated: NO');print('Next: freeze unique B/R/K/RK displayed-pair union before judging.')
if __name__=='__main__':main()
