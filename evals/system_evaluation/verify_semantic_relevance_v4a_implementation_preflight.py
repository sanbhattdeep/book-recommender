from __future__ import annotations
import hashlib,json,subprocess,sys
from pathlib import Path
REPO_ROOT=Path(__file__).resolve().parents[2]
BASE=REPO_ROOT/'evals'/'system_evaluation'
FILES=[('frozen JSON spec',BASE/'semantic_relevance_v4a_hybrid_retrieval_experiment.v0.1.0.json','6aba9c8354f201ad41044dd82d4476612f81ebd27a5ce4f8bf4f327a8b3281b9'),('frozen Markdown spec',BASE/'semantic_relevance_v4a_hybrid_retrieval_experiment.v0.1.0.md','5af31f1b986187b668b0d486d4209da198e88eb6ea03e3b12818ea8c04e33b90'),('frozen spec lock',BASE/'semantic_relevance_v4a_hybrid_retrieval_experiment.v0.1.0.lock.json','c9fee869cda6ccfd26c1ad0b2624cbcf693b9c896366b6baaa5c0b7b98371db2'),('v4a implementation',BASE/'semantic_relevance_v4a_hybrid.py','6f31504e29078fffbab86ddb86ef26b02aaa22082d3bfdb6944792bf689f6db7'),('v4a synthetic tests',BASE/'test_semantic_relevance_v4a_hybrid.py','0376710871da371e5bbeb5089efccfdad79beb4f8599176b9a48a19587ab06bd')]
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def req(c,m):
    if not c: raise ValueError(m)
def main():
    print('Semantic relevance v4a - implementation candidate preflight')
    print('-----------------------------------------------------------')
    print('Experiment retrieval: DISABLED'); print('Reranker inference:   DISABLED'); print('Judge execution:      DISABLED'); print('Catalog indexing:     DISABLED'); print()
    for label,p,e in FILES:
        req(p.exists(),f'Missing {label}: {p}'); a=sha(p); req(a==e,f'{label} hash mismatch: expected={e} actual={a}'); print(f'PASS  {label} SHA-256 verified')
    spec=json.loads((BASE/'semantic_relevance_v4a_hybrid_retrieval_experiment.v0.1.0.json').read_text(encoding='utf-8-sig'))
    lock=json.loads((BASE/'semantic_relevance_v4a_hybrid_retrieval_experiment.v0.1.0.lock.json').read_text(encoding='utf-8-sig'))
    req(spec['status']=='FROZEN_BEFORE_V4A_IMPLEMENTATION','Spec is not frozen.'); req(lock['status']=='V4A_EXPERIMENT_SPEC_FROZEN_BEFORE_IMPLEMENTATION','Lock status changed.')
    req(spec['bm25_corpus']['search_fields']==['title','description'],'BM25 fields changed.'); req(spec['bm25_analyzer']['stop_words']=='none','Stop-word policy changed.'); req(spec['bm25_analyzer']['stemming']=='none','Stemming changed.')
    req(float(spec['bm25_scoring']['k1'])==1.5,'k1 changed.'); req(float(spec['bm25_scoring']['b'])==0.75,'b changed.'); req(int(spec['bm25_scoring']['candidate_depth'])==50,'BM25 depth changed.'); req(int(spec['hybrid_retrieval_rank']['k'])==60,'Hybrid RRF changed.'); req(int(spec['final_v4a_ranking']['k'])==60,'Final RRF changed.')
    print('PASS  frozen v4a configuration contract verified'); print(); print('Running synthetic-only unit tests...')
    r=subprocess.run([sys.executable,'-m','unittest','test_semantic_relevance_v4a_hybrid','-v'],cwd=BASE,text=True)
    req(r.returncode==0,'Synthetic v4a unit tests failed.')
    print(); print('V4A IMPLEMENTATION CANDIDATE PREFLIGHT: PASS'); print('Synthetic unit tests: PASS'); print('Experiment retrieval executed: NO'); print('Catalog indexed:              NO'); print('Reranker inference executed:  NO'); print('Judge calls performed:        0'); print('Human relevance labels used:  NO'); print('Next: freeze implementation source hashes before v4a development retrieval.')
if __name__=='__main__': main()
