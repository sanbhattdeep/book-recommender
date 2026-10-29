from __future__ import annotations
import ast,hashlib,json,os,re,subprocess,time
from datetime import datetime,timezone
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
import verify_semantic_relevance_v5_development_candidate_collection as boundary
from semantic_relevance_v4a_hybrid import BM25Config,BM25Index,build_hybrid_union
from semantic_relevance_v5_r1_retrieval import R1DenseRuntimeConfig,build_dense_index,load_dense_encoder,rank_query

ROOT=boundary.ROOT; SYS=boundary.SYS; RUN_ROOT=boundary.RUN_ROOT; CATALOG=boundary.CATALOG; TAGGED=boundary.TAGGED; R0_SOURCE=boundary.R0_SOURCE
EXPECTED_DEPTH=50

def sha(p:Path)->str:return hashlib.sha256(p.read_bytes()).hexdigest()
def lfsha(p:Path)->str:
    b=p.read_bytes(); return hashlib.sha256(b.replace(b'\r\n',b'\n').replace(b'\r',b'\n')).hexdigest()
def req(c:bool,m:str)->None:
    if not c: raise ValueError(m)
def cisbn(v:Any)->str:
    t=str(v).strip().strip('"').strip("'")
    if re.fullmatch(r'\d+\.0',t):t=t[:-2]
    req(bool(t),'Empty ISBN'); return t

def load_r0_backend()->dict[str,Any]:
    source=R0_SOURCE.read_text(encoding='utf-8'); tree=ast.parse(source,filename=str(R0_SOURCE)); idx=None
    for i,node in enumerate(tree.body):
        if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)) and node.name=='retrieve_semantic_recommedations': idx=i; break
    req(idx is not None,'Could not find frozen R0 recommender function'); bt=ast.Module(body=tree.body[:idx+1],type_ignores=[]); ast.fix_missing_locations(bt); ns={'__name__':'_v5_r0_backend','__file__':str(R0_SOURCE)}; old=Path.cwd()
    try:
        os.chdir(ROOT); exec(compile(bt,filename=str(R0_SOURCE),mode='exec'),ns,ns)
    finally: os.chdir(old)
    req('db_books' in ns and 'books' in ns,'R0 backend missing db_books/books'); return ns

def parse_doc_isbn(text:str)->str:
    toks=str(text).strip('"').split(); req(bool(toks),'Empty R0 page_content'); return cisbn(toks[0])

def r0_dense(qid,query,stratum,db,books,lookup):
    t0=time.perf_counter(); a=db.similarity_search(query,k=50); first=time.perf_counter()-t0
    b=db.similarity_search(query,k=50); ids1=[parse_doc_isbn(x.page_content) for x in a]; ids2=[parse_doc_isbn(x.page_content) for x in b]
    req(len(a)==50 and len(b)==50,f'{qid}: R0 did not return 50 twice'); req(ids1==ids2,f'{qid}: R0 repeat order differs'); req(len(set(ids1))==50,f'{qid}: duplicate R0 ISBN')
    rows=[]
    for rank,doc in enumerate(a,1):
        isbn=ids1[rank-1]; inds=lookup.get(isbn,[]); req(len(inds)==1,f'{qid}: R0 ISBN metadata multiplicity {isbn}: {len(inds)}'); m=books.iloc[inds[0]]
        rows.append({'query_id':qid,'query':query,'stratum':stratum,'dense_rank':rank,'isbn13':isbn,'title':str(m.get('title','')),'description':str(m.get('description','')),'authors':str(m.get('authors','')),'retrieval_document_text':str(doc.page_content)})
    return pd.DataFrame(rows),first

def main():
    print('Semantic relevance V5 - V5D R0/R1 candidate collection'); print('------------------------------------------------------'); print('R0 dense retrieval: ENABLED'); print('R1 dense retrieval: ENABLED'); print('Shared BM25 retrieval: ENABLED'); print('K0/K1 reranker inference: DISABLED'); print('Judge execution: DISABLED'); print('Development selection: DISABLED'); print()
    qs=boundary.validate(); verifier=Path(__file__).with_name('verify_semantic_relevance_v5_development_candidate_collection.py'); expected_verifier='4fc1a165a240a37b277cbb871267ff79567dbbb994c60c9dcc16fbf11aeb798a'; req(sha(verifier)==expected_verifier or lfsha(verifier)==expected_verifier,'Candidate preflight verifier changed')
    existing=sorted(RUN_ROOT.glob('*_candidate_collection')) if RUN_ROOT.exists() else []; req(not existing,'Existing V5D candidate-collection run found')
    catalog=pd.read_csv(CATALOG,dtype=str,keep_default_na=False,encoding='utf-8');
    if 'authors' not in catalog.columns: catalog['authors']=''
    catalog=catalog.copy().reset_index(drop=True); catalog['isbn13']=catalog['isbn13'].map(cisbn); catlook=catalog.set_index('isbn13',drop=False)
    lines=[x for x in TAGGED.read_text(encoding='utf-8').splitlines() if x.strip()]; tagged_ids=[cisbn(x.strip('"').split()[0]) for x in lines]
    print('Loading frozen R0 backend / nomic-embed-text index...'); ns=load_r0_backend(); db=ns['db_books']; books=ns['books'].copy().reset_index(drop=True); books['_isbn_key']=books['isbn13'].map(cisbn); lookup={}
    for i,k in enumerate(books['_isbn_key']): lookup.setdefault(k,[]).append(i)
    req(set(books['_isbn_key'])==set(catalog['isbn13']),'R0 backend ISBN universe differs')
    print('Loading frozen R1 BGE-M3 and building exact tagged-description dense index...'); r1_model,r1_device=load_dense_encoder(R1DenseRuntimeConfig(device='auto')); ti0=time.perf_counter(); r1_index=build_dense_index(isbn13=tagged_ids,document_texts=lines,model=r1_model); r1_index_seconds=time.perf_counter()-ti0; print(f'PASS  R1 dense index built: docs={len(tagged_ids)} device={r1_device} seconds={r1_index_seconds:.3f}')
    bm25=BM25Index(isbn13=catalog['isbn13'].tolist(),title=catalog['title'].tolist(),description=catalog['description'].tolist(),config=BM25Config(k1=1.5,b=0.75,candidate_depth=50)); print(f'PASS  shared BM25 index built: docs={bm25.n_docs}')
    r0ds=[];r1ds=[];bms=[];r0us=[];r1us=[];lat=[]
    for i,q in enumerate(qs,1):
        qid,query,stratum=q['query_id'],q['query'],q['stratum']
        r0,r0sec=r0_dense(qid,query,stratum,db,books,lookup); r0ds.append(r0)
        t=time.perf_counter(); r1a=rank_query(query=query,index=r1_index,model=r1_model); r1sec=time.perf_counter()-t; r1b=rank_query(query=query,index=r1_index,model=r1_model)
        req(tuple(r1a['isbn13'])==tuple(r1b['isbn13']),f'{qid}: R1 repeat order differs'); delta=float(np.max(np.abs(r1a['dense_score'].to_numpy()-r1b['dense_score'].to_numpy()))); req(delta<=1e-5,f'{qid}: R1 repeat score delta {delta} > 1e-5')
        r1a.insert(0,'query_id',qid);r1a.insert(1,'query',query);r1a.insert(2,'stratum',stratum); meta=catlook.loc[r1a['isbn13'].tolist()][['title','description','authors']].reset_index(drop=True); r1out=pd.concat([r1a.reset_index(drop=True),meta],axis=1); r1ds.append(r1out)
        tb=time.perf_counter(); bm=bm25.rank_query(query,top_k=50); bmsec=time.perf_counter()-tb; bmout=bm.copy();bmout.insert(0,'query_id',qid);bmout.insert(1,'query',query);bmout.insert(2,'stratum',stratum)
        if len(bmout): bmout=pd.concat([bmout.reset_index(drop=True),catlook.loc[bmout['isbn13'].tolist()][['title','description','authors']].reset_index(drop=True)],axis=1)
        else: bmout[['title','description','authors']]=''
        bms.append(bmout)
        def union(dense):
            u=build_hybrid_union(dense[['isbn13','dense_rank']],bm[['isbn13','bm25_rank','bm25_score']],rrf_k=60.0); req(50<=len(u)<=100,f'{qid}: union size {len(u)}'); req(not u['isbn13'].duplicated().any(),f'{qid}: duplicate union ISBN'); req(u['hybrid_retrieval_rank'].tolist()==list(range(1,len(u)+1)),f'{qid}: union ranks not contiguous'); u['dense_rank']=pd.to_numeric(u['dense_rank'],errors='coerce').astype('Int64');u['bm25_rank']=pd.to_numeric(u['bm25_rank'],errors='coerce').astype('Int64'); u=pd.concat([u.reset_index(drop=True),catlook.loc[u['isbn13'].tolist()][['title','description','authors']].reset_index(drop=True)],axis=1);u.insert(0,'query_id',qid);u.insert(1,'query',query);u.insert(2,'stratum',stratum);return u
        u0=union(r0);u1=union(r1a);r0us.append(u0);r1us.append(u1)
        lat.append({'query_id':qid,'r0_dense_first_pass_seconds':r0sec,'r1_dense_first_pass_seconds':r1sec,'shared_bm25_seconds':bmsec,'r1_repeat_max_score_abs_delta':delta})
        print(f'{i:2d}/24 {qid}: R0=50 R1=50 BM25={len(bm):2d} R0union={len(u0):3d} R1union={len(u1):3d} R0repeat=PASS R1repeat=PASS')
    frames={'v5d_r0_dense_top50.csv':pd.concat(r0ds,ignore_index=True),'v5d_r1_dense_top50.csv':pd.concat(r1ds,ignore_index=True),'v5d_bm25_top50.csv':pd.concat(bms,ignore_index=True),'v5d_r0_hybrid_union.csv':pd.concat(r0us,ignore_index=True),'v5d_r1_hybrid_union.csv':pd.concat(r1us,ignore_index=True),'v5d_candidate_collection_latency.csv':pd.DataFrame(lat)}
    req(len(frames['v5d_r0_dense_top50.csv'])==1200 and len(frames['v5d_r1_dense_top50.csv'])==1200,'Dense output not 24x50')
    ts=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'); run=RUN_ROOT/f'{ts}_candidate_collection';run.mkdir(parents=True,exist_ok=False)
    hashes={}
    for name,df in frames.items(): p=run/name;df.to_csv(p,index=False,encoding='utf-8');hashes[name]=sha(p)
    metadata={'status':'V5D_R0_R1_CANDIDATE_COLLECTION_COMPLETE','completed_at_utc':datetime.now(timezone.utc).isoformat(),'git_sha':subprocess.run(['git','rev-parse','HEAD'],cwd=ROOT,capture_output=True,text=True).stdout.strip(),'query_set':'V5D01-V5D24','query_count':24,'contract_sha256':boundary.EXPECTED['contract'],'v5d_authoring_lock_sha256':boundary.EXPECTED['authoring'],'r1_device':r1_device,'r1_index_build_seconds':r1_index_seconds,'latency_definition':boundary.load(boundary.CONTRACT)['latency_measurement'],'outputs':hashes,'reranker_inference_performed':False,'judge_execution_performed':False,'development_selection_calculated':False}
    mp=run/'v5d_candidate_collection_metadata.json';mp.write_text(json.dumps(metadata,indent=2)+'\n',encoding='utf-8');hashes[mp.name]=sha(mp)
    lock={'status':'V5D_R0_R1_CANDIDATES_FROZEN_BEFORE_RERANKING_OR_JUDGING','frozen_at_utc':datetime.now(timezone.utc).isoformat(),'query_set':'V5D01-V5D24','query_count':24,'candidate_collection_contract_sha256':boundary.EXPECTED['contract'],'v5d_authoring_lock_sha256':boundary.EXPECTED['authoring'],'r0_dense_depth':50,'r1_dense_depth':50,'bm25_depth':50,'hybrid_rrf_k':60.0,'maximum_unique_candidates_per_query':100,'r0_repeat_order_identical_all_queries':True,'r1_repeat_order_identical_all_queries':True,'r1_repeat_score_tolerance_abs':1e-5,'outputs':hashes,'reranker_inference_performed':False,'judge_execution_performed':False,'human_relevance_labels_used':False,'development_selection_calculated':False,'release_metrics_calculated':False,'next_stage':'PREPARE_K0_AND_K1_RERANKING_ON_FROZEN_R0_AND_R1_V5D_CANDIDATE_UNIONS'}
    lp=run/'v5d_candidate_collection.lock.json';lp.write_text(json.dumps(lock,indent=2)+'\n',encoding='utf-8')
    print(); print('V5D R0/R1 CANDIDATE COLLECTION: PASS');print(f'Run directory: {run}');print(f'R0 dense SHA-256: {hashes["v5d_r0_dense_top50.csv"]}');print(f'R1 dense SHA-256: {hashes["v5d_r1_dense_top50.csv"]}');print(f'BM25 SHA-256: {hashes["v5d_bm25_top50.csv"]}');print(f'R0 union SHA-256: {hashes["v5d_r0_hybrid_union.csv"]}');print(f'R1 union SHA-256: {hashes["v5d_r1_hybrid_union.csv"]}');print(f'Candidate lock SHA-256: {sha(lp)}');print('Reranker inference performed: NO');print('Judge calls performed: 0');print('Development selection calculated: NO');print('Next: prepare K0/K1 reranking on these frozen R0/R1 unions.')
if __name__=='__main__':main()
