from __future__ import annotations
import hashlib,json,re,subprocess,importlib.metadata
from pathlib import Path
from typing import Any
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]; SYS=ROOT/'evals/system_evaluation'; DATA=ROOT/'evals/datasets'; FACETS=ROOT/'evals/facets/semantic_relevance'
QUERY=DATA/'semantic_relevance_v5_development_queries.v0.1.0.json'
FACET=FACETS/'semantic_relevance_v5_development_query_facets.v0.1.0.json'
AUTHORING=SYS/'semantic_relevance_v5_development_authoring.v0.1.0.lock.json'
DESIGN=SYS/'semantic_relevance_v5_development_design.v0.1.0.lock.json'
ARCH=SYS/'semantic_relevance_v5_r1_k1_architecture.v0.1.0.lock.json'
IMPL=SYS/'semantic_relevance_v5_r1_k1_implementation.v0.1.0.lock.json'
QUAL=SYS/'semantic_relevance_v5_r1_k1_model_qualification.v0.1.0.lock.json'
R1_SOURCE=SYS/'semantic_relevance_v5_r1_retrieval.py'
R0_SOURCE=ROOT/'gradio-dashboard.py'; HYBRID=SYS/'semantic_relevance_v4a_hybrid.py'
CATALOG=ROOT/'books_with_emotions.csv'; TAGGED=ROOT/'tagged_description.txt'
CONTRACT=SYS/'semantic_relevance_v5_development_candidate_collection_contract.v0.1.0.json'
RUN_ROOT=ROOT/'evals/runs/semantic_relevance_v5_development'
EXPECTED={
'design':'89ee03a4328c268a8128d5e373854b92634c41eff204b53f7b93b78c481311c0','arch':'2b298d3263c7a88090a983700da34c30767a71877a4b32ea2e1ae8f3524ade87','impl':'4d166df6b1acff66b23829126ee18c080bd3756de7ed98953e4520db8de9f72b','qual':'67b1dd15403b5825086cce4c9e5c5af14cc883945f5a9626abaa4e5894b77211',
'query':'1d93979d21d081afaecac63ee46c56330e80a2e7dbeeeea5d3bfa55a4b56387a','facet':'355ac59604fc73039fe394ba36b9ef37c04cd0be447ff470bce669d8adeb4746','authoring':'ffca0f7a10c4d05ee79fc08d584df5793f93de2242a2ed2bf09436e2df80ba3e',
'r0':'cdd0458ff23bd669185a8c882ceb132438d037af1a29f4dd60bd7fef9c405b60','hybrid':'6f31504e29078fffbab86ddb86ef26b02aaa22082d3bfdb6944792bf689f6db7','r1':'eabadeccc01a98fb2f962e21ef94c4696e066b78d118f597b83120be4eeaf511',
'catalog':'22244d277a643e56bf23a744d0d45eb2247db4a05792ec69ffd942c6d5fdbf6e','tagged':'cb3f34f397be0d65af4f931bc24265b69b1ed151b0a76ade03885fbaf3c5e1a5',
'contract':'0b9dc202a43a64099ee03257f18f644e06d398dedb5d07391e9555d87cba6ad5'}
R0_BLOB='89a769f7a752a517ebc7079e08c531e8df415bce'; HYBRID_BLOB='3c1a612dc4d33a7680dd1f93d40558cb094efa1b'
IDS=[f'V5D{i:02d}' for i in range(1,25)]

def sha(p:Path)->str:return hashlib.sha256(p.read_bytes()).hexdigest()
def lfsha(p:Path)->str:return hashlib.sha256(p.read_bytes().replace(b'\r\n',b'\n').replace(b'\r',b'\n')).hexdigest()
def req(c:bool,m:str)->None:
    if not c: raise ValueError(m)
def load(p:Path)->dict[str,Any]:return json.loads(p.read_text(encoding='utf-8-sig'))
def verify_raw(p:Path,e:str,l:str):
    req(p.exists(),f'Missing {l}: {p}'); a=sha(p); req(a==e,f'{l} hash mismatch: expected={e} actual={a}'); print(f'PASS  {l} raw SHA-256 verified')
def verify_text(p:Path,e:str,l:str):
    req(p.exists(),f'Missing {l}: {p}'); a=sha(p)
    if a==e: mode='raw'
    else:
        n=lfsha(p); req(n==e,f'{l} hash mismatch: expected={e} raw={a} lf={n}'); mode='canonical-LF'
    print(f'PASS  {l} {mode} SHA-256 verified')
def git_blob(path:str)->str:
    return subprocess.run(['git','rev-parse',f'HEAD:{path}'],cwd=ROOT,capture_output=True,text=True,check=True).stdout.strip()
def tracked(path:str,blob:str,label:str):
    req(git_blob(path)==blob,f'{label} Git blob changed'); d=subprocess.run(['git','diff','--quiet','--',path],cwd=ROOT); req(d.returncode==0,f'{label} has working-tree modifications'); print(f'PASS  {label} exact tracked Git blob verified')
def cisbn(v:Any)->str:
    t=str(v).strip().strip('"').strip("'");
    if re.fullmatch(r'\d+\.0',t):t=t[:-2]
    req(bool(t),'Empty ISBN'); return t

def validate()->list[dict[str,str]]:
    verify_raw(CONTRACT,EXPECTED['contract'],'V5D candidate-collection contract')
    for p,k,l in [(DESIGN,'design','V5 design lock'),(ARCH,'arch','V5 R1/K1 architecture lock'),(IMPL,'impl','V5 R1/K1 implementation lock'),(QUAL,'qual','V5 R1/K1 model qualification lock'),(QUERY,'query','V5D query file'),(FACET,'facet','V5D facet file'),(AUTHORING,'authoring','V5D authoring lock')]: verify_raw(p,EXPECTED[k],l)
    verify_text(R0_SOURCE,EXPECTED['r0'],'R0 dense candidate-generator source'); verify_text(HYBRID,EXPECTED['hybrid'],'hybrid retrieval source'); verify_text(R1_SOURCE,EXPECTED['r1'],'R1 retrieval source')
    tracked('gradio-dashboard.py',R0_BLOB,'R0 dense candidate-generator source'); tracked('evals/system_evaluation/semantic_relevance_v4a_hybrid.py',HYBRID_BLOB,'hybrid retrieval source')
    verify_raw(CATALOG,EXPECTED['catalog'],'runtime catalog'); verify_raw(TAGGED,EXPECTED['tagged'],'tagged-description corpus')
    req(importlib.metadata.version('sentence-transformers')=='6.2.0','sentence-transformers version changed'); print('PASS  sentence-transformers version pinned at 6.2.0')
    ad=load(AUTHORING); req(ad.get('status')=='V5D_QUERY_AND_FACET_SPEC_FROZEN_BEFORE_DEVELOPMENT_RETRIEVAL','Unexpected V5D authoring lock status'); req(ad.get('v5d_retrieval_performed') is False,'Authoring lock says retrieval already occurred')
    qd,fd=load(QUERY),load(FACET); qs=qd['queries']; fs=fd['queries']; req([q['query_id'] for q in qs]==IDS,'V5D IDs/order changed'); req([(q['query_id'],q['query'],q['stratum']) for q in qs]==[(q['query_id'],q['query'],q['stratum']) for q in fs],'Query/facet mismatch')
    cat=pd.read_csv(CATALOG,dtype=str,keep_default_na=False,encoding='utf-8'); req(len(cat)==5197,'Catalog row count changed'); req({'isbn13','title','description'}<=set(cat.columns),'Catalog columns missing'); ci=[cisbn(x) for x in cat['isbn13']]; req(len(set(ci))==5197,'Duplicate catalog ISBN')
    lines=[x for x in TAGGED.read_text(encoding='utf-8').splitlines() if x.strip()]; req(len(lines)==5197,'Tagged-description row count changed'); ti=[cisbn(x.strip('"').split()[0]) for x in lines]; req(len(set(ti))==5197,'Duplicate tagged ISBN'); req(set(ci)==set(ti),'Catalog/tagged ISBN sets differ'); print('PASS  runtime catalog/tagged-description structure and ISBN-set alignment verified')
    c=load(CONTRACT); req(c.get('status')=='FROZEN_BY_PACKAGE_BEFORE_V5D_CANDIDATE_COLLECTION','Contract status changed'); req(c['R0']['candidate_depth']==50 if 'candidate_depth' in c['R0'] else True,'R0 depth changed'); req(c['execution_boundary']['K0_reranker_inference'] is False and c['execution_boundary']['K1_reranker_inference'] is False,'Reranker unexpectedly enabled'); req(c['execution_boundary']['judge_execution'] is False,'Judge unexpectedly enabled')
    existing=sorted(RUN_ROOT.glob('*_candidate_collection')) if RUN_ROOT.exists() else []; req(not existing,'Existing V5D candidate-collection run found: '+', '.join(str(x) for x in existing))
    return [{'query_id':str(x['query_id']),'query':str(x['query']),'stratum':str(x['stratum'])} for x in qs]

def main():
    print('Semantic relevance V5 - V5D R0/R1 candidate-collection preflight'); print('--------------------------------------------------------------'); print('R0/R1 retrieval: DISABLED'); print('K0/K1 reranker inference: DISABLED'); print('Judge execution: DISABLED'); print('Development selection: NOT CALCULATED'); print()
    qs=validate(); print(); print('V5D R0/R1 CANDIDATE COLLECTION PREFLIGHT: PASS'); print(f'Queries: {len(qs)} (V5D01-V5D24)'); print('R0 dense depth: 50'); print('R1 dense depth: 50'); print('Shared BM25 depth: 50'); print('Maximum unique candidates/query/system: 100'); print('Retrieval/model calls performed during preflight: 0'); print('Reranker inference performed: NO'); print('Judge calls performed: 0'); print('Development selection calculated: NO'); print('Next: execute and freeze R0/R1 candidate collection only.')
if __name__=='__main__':main()
