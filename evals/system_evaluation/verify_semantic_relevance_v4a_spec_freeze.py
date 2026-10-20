from __future__ import annotations
import hashlib,json
from pathlib import Path
REPO_ROOT=Path(__file__).resolve().parents[2]
BASE=REPO_ROOT/'evals'/'system_evaluation'
SPEC_JSON=BASE/'semantic_relevance_v4a_hybrid_retrieval_experiment.v0.1.0.json'
SPEC_MD=BASE/'semantic_relevance_v4a_hybrid_retrieval_experiment.v0.1.0.md'
LOCK=BASE/'semantic_relevance_v4a_hybrid_retrieval_experiment.v0.1.0.lock.json'
EXPECTED_JSON_SHA256='6aba9c8354f201ad41044dd82d4476612f81ebd27a5ce4f8bf4f327a8b3281b9'
EXPECTED_MD_SHA256='5af31f1b986187b668b0d486d4209da198e88eb6ea03e3b12818ea8c04e33b90'
EXPECTED_LOCK_SHA256='c9fee869cda6ccfd26c1ad0b2624cbcf693b9c896366b6baaa5c0b7b98371db2'
def sha256(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def require(c,m):
    if not c: raise ValueError(m)
def main():
    print('Semantic relevance v4a - frozen spec verification')
    print('--------------------------------------------------')
    for label,path,expected in [('JSON spec',SPEC_JSON,EXPECTED_JSON_SHA256),('Markdown spec',SPEC_MD,EXPECTED_MD_SHA256),('spec lock',LOCK,EXPECTED_LOCK_SHA256)]:
        require(path.exists(),f'Missing {label}: {path}')
        actual=sha256(path); require(actual==expected,f'{label} hash mismatch: expected={expected} actual={actual}')
        print(f'PASS  {label} SHA-256 verified')
    spec=json.loads(SPEC_JSON.read_text(encoding='utf-8')); lock=json.loads(LOCK.read_text(encoding='utf-8'))
    require(spec['status']=='FROZEN_BEFORE_V4A_IMPLEMENTATION','Spec is not frozen.')
    require(lock['status']=='V4A_EXPERIMENT_SPEC_FROZEN_BEFORE_IMPLEMENTATION','Lock status changed.')
    require(lock['execution_boundary']['implementation_code_included'] is False,'Implementation code unexpectedly included.')
    require(lock['execution_boundary']['retrieval_executed'] is False,'Retrieval unexpectedly executed.')
    require(lock['execution_boundary']['reranker_inference_executed'] is False,'Reranker unexpectedly executed.')
    require(lock['execution_boundary']['judge_calls_performed']==0,'Judge calls unexpectedly performed.')
    require(spec['bm25_corpus']['search_fields']==['title','description'],'BM25 fields changed.')
    require(float(spec['bm25_scoring']['k1'])==1.5,'BM25 k1 changed.')
    require(float(spec['bm25_scoring']['b'])==0.75,'BM25 b changed.')
    require(int(spec['bm25_scoring']['candidate_depth'])==50,'BM25 candidate depth changed.')
    require(int(spec['candidate_generation']['dense_branch']['candidate_depth'])==50,'Dense candidate depth changed.')
    require(int(spec['hybrid_retrieval_rank']['k'])==60,'Hybrid RRF k changed.')
    require(int(spec['final_v4a_ranking']['k'])==60,'Final RRF k changed.')
    require(spec['reranker']['model_id']=='BAAI/bge-reranker-base','Reranker model changed.')
    require(spec['reranker']['model_revision']=='2cfc18c9415c912f9d8155881c133215df768a70','Reranker revision changed.')
    print('PASS  frozen v4a critical contract verified')
    print('PASS  implementation/retrieval/reranker/judge execution boundary verified')
    print()
    print('V4A SPEC FREEZE VERIFICATION: PASS')
    print('Implementation code included: NO')
    print('Retrieval executed:           NO')
    print('Reranker inference executed:  NO')
    print('Judge calls performed:        0')
    print('Next: prepare v4a implementation package with zero experiment execution.')
if __name__=='__main__': main()
