from __future__ import annotations
import hashlib, json
from pathlib import Path

REPO_ROOT=Path(__file__).resolve().parents[2]
BASE=REPO_ROOT/'evals'/'system_evaluation'
EXPECTED={
  'semantic_relevance_v4a_hybrid_retrieval_experiment.v0.1.0.json':'6aba9c8354f201ad41044dd82d4476612f81ebd27a5ce4f8bf4f327a8b3281b9',
  'semantic_relevance_v4a_hybrid_retrieval_experiment.v0.1.0.md':'5af31f1b986187b668b0d486d4209da198e88eb6ea03e3b12818ea8c04e33b90',
  'semantic_relevance_v4a_hybrid_retrieval_experiment.v0.1.0.lock.json':'c9fee869cda6ccfd26c1ad0b2624cbcf693b9c896366b6baaa5c0b7b98371db2',
  'semantic_relevance_v4a_hybrid.py':'6f31504e29078fffbab86ddb86ef26b02aaa22082d3bfdb6944792bf689f6db7',
  'test_semantic_relevance_v4a_hybrid.py':'0376710871da371e5bbeb5089efccfdad79beb4f8599176b9a48a19587ab06bd',
  'verify_semantic_relevance_v4a_implementation_preflight.py':'3f149941e25f60a17069833cbd6ab1de79ed67380fe76350066779f196a84a37',
  'semantic_relevance_v4a_implementation.lock.json':'b67d305ca5c5da7df82997f43129ad596c7c79eb085571bb4a83b8638f016775',
}

def sha256(p:Path)->str: return hashlib.sha256(p.read_bytes()).hexdigest()
def req(c:bool,m:str)->None:
    if not c: raise ValueError(m)

def main()->None:
    print('Semantic relevance v4a - implementation freeze verification')
    print('----------------------------------------------------------')
    print('Runtime catalog read:     DISABLED')
    print('BM25 catalog indexing:    DISABLED')
    print('Development retrieval:    DISABLED')
    print('Reranker inference:       DISABLED')
    print('Judge execution:          DISABLED')
    print()
    for name,expected in EXPECTED.items():
        p=BASE/name
        req(p.exists(),f'Missing frozen file: {p}')
        actual=sha256(p)
        req(actual==expected,f'Hash mismatch for {name}: expected={expected} actual={actual}')
        print(f'PASS  {name} SHA-256 verified')
    lock=json.loads((BASE/'semantic_relevance_v4a_implementation.lock.json').read_text(encoding='utf-8-sig'))
    req(lock.get('status')=='V4A_IMPLEMENTATION_FROZEN_BEFORE_DEVELOPMENT_RETRIEVAL','Implementation lock status changed.')
    b=lock['execution_boundary']
    req(b['runtime_catalog_read_during_freeze'] is False,'Runtime catalog unexpectedly read during freeze.')
    req(b['bm25_catalog_index_built_during_freeze'] is False,'BM25 catalog unexpectedly indexed during freeze.')
    req(b['development_retrieval_executed_during_freeze'] is False,'Development retrieval unexpectedly executed during freeze.')
    req(b['reranker_inference_executed_during_freeze'] is False,'Reranker unexpectedly executed during freeze.')
    req(b['judge_calls_performed_during_freeze']==0,'Judge calls unexpectedly performed during freeze.')
    c=lock['critical_contract']
    req(float(c['bm25_k1'])==1.5 and float(c['bm25_b'])==0.75,'BM25 parameters changed.')
    req(int(c['bm25_candidate_depth'])==50 and int(c['dense_candidate_depth'])==50,'Candidate depth changed.')
    req(int(c['hybrid_rrf_k'])==60 and int(c['final_rrf_k'])==60,'RRF k changed.')
    req(c['union_truncated_before_reranker'] is False,'Union truncation contract changed.')
    req(c['reranker_model_revision']=='2cfc18c9415c912f9d8155881c133215df768a70','Reranker revision changed.')
    print('PASS  frozen implementation critical contract verified')
    print('PASS  zero-experiment freeze boundary verified')
    print()
    print('V4A IMPLEMENTATION FREEZE VERIFICATION: PASS')
    print('Runtime catalog read:      NO')
    print('BM25 catalog indexed:      NO')
    print('Development retrieval run: NO')
    print('Reranker inference run:    NO')
    print('Judge calls performed:     0')
    print('Next: prepare v4a development retrieval execution package.')

if __name__=='__main__': main()
