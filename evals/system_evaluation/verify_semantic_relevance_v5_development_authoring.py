from __future__ import annotations
import hashlib, json, re
from collections import Counter
from pathlib import Path
from typing import Any

ROOT=Path(__file__).resolve().parents[2]
SYS=ROOT/'evals/system_evaluation'; DATA=ROOT/'evals/datasets'; FACETS=ROOT/'evals/facets/semantic_relevance'
QUERY_FILE=DATA/'semantic_relevance_v5_development_queries.v0.1.0.json'
FACET_FILE=FACETS/'semantic_relevance_v5_development_query_facets.v0.1.0.json'
PRIOR=SYS/'semantic_relevance_prior_query_texts_for_v5d.v0.1.0.json'
DESIGN_LOCK=SYS/'semantic_relevance_v5_development_design.v0.1.0.lock.json'
ARCH_LOCK=SYS/'semantic_relevance_v5_r1_k1_architecture.v0.1.0.lock.json'
IMPL_LOCK=SYS/'semantic_relevance_v5_r1_k1_implementation.v0.1.0.lock.json'
QUAL_LOCK=SYS/'semantic_relevance_v5_r1_k1_model_qualification.v0.1.0.lock.json'
AUTHORING_LOCK=SYS/'semantic_relevance_v5_development_authoring.v0.1.0.lock.json'
EXPECTED={
 'query':'1d93979d21d081afaecac63ee46c56330e80a2e7dbeeeea5d3bfa55a4b56387a','facet':'355ac59604fc73039fe394ba36b9ef37c04cd0be447ff470bce669d8adeb4746','prior':'32015f2426c1d72b2eb65db0e11c4be7ffdf390e14e00e00f29d0307e9f89056',
 'design_lock':'89ee03a4328c268a8128d5e373854b92634c41eff204b53f7b93b78c481311c0',
 'arch_lock':'2b298d3263c7a88090a983700da34c30767a71877a4b32ea2e1ae8f3524ade87',
 'impl_lock':'4d166df6b1acff66b23829126ee18c080bd3756de7ed98953e4520db8de9f72b',
 'qual_lock':'67b1dd15403b5825086cce4c9e5c5af14cc883945f5a9626abaa4e5894b77211',
}
STRATA={'two_core_theme':4,'three_core_conjunctive':4,'core_plus_qualifier':4,'relationship_or_role_constraint':4,'hard_exclusion_or_contrast':4,'abstract_nonfiction_multi_facet':4}
STOP=set('a an the about and or of to for in on with as whose how not into from between after'.split())
MAX_JACCARD=0.35

def sha256(p:Path)->str: return hashlib.sha256(p.read_bytes()).hexdigest()
def req(c:bool,m:str)->None:
    if not c: raise ValueError(m)
def load(p:Path)->dict[str,Any]: return json.loads(p.read_text(encoding='utf-8-sig'))
def verify_raw(p:Path,e:str,l:str)->None:
    req(p.exists(),f'Missing {l}: {p}'); a=sha256(p); req(a==e,f'{l} hash mismatch: expected={e} actual={a}'); print(f'PASS  {l} raw SHA-256 verified')
def tokens(t:str)->set[str]: return {x for x in re.findall(r'[a-z0-9]+',t.casefold()) if x not in STOP}
def jaccard(a:str,b:str)->float:
    x,y=tokens(a),tokens(b); return 0.0 if not (x or y) else len(x&y)/len(x|y)

def verify_boundary(*,require_lock_absent:bool=True)->tuple[dict,dict,dict]:
    for p,k,l in [(QUERY_FILE,'query','V5D query file'),(FACET_FILE,'facet','V5D facet file'),(PRIOR,'prior','V5D prior-query anti-leakage manifest'),(DESIGN_LOCK,'design_lock','V5 development design lock'),(ARCH_LOCK,'arch_lock','V5 R1/K1 architecture lock'),(IMPL_LOCK,'impl_lock','V5 R1/K1 implementation lock'),(QUAL_LOCK,'qual_lock','V5 R1/K1 model qualification lock')]: verify_raw(p,EXPECTED[k],l)
    if require_lock_absent: req(not AUTHORING_LOCK.exists(),f'Refusing because V5D authoring lock already exists: {AUTHORING_LOCK}')
    d,a,i,q=[load(x) for x in [DESIGN_LOCK,ARCH_LOCK,IMPL_LOCK,QUAL_LOCK]]
    req(d.get('status')=='V5_DEVELOPMENT_DESIGN_FROZEN_BEFORE_COMPONENT_IMPLEMENTATION','Unexpected V5 design-lock status')
    req(a.get('status')=='V5_R1_K1_ARCHITECTURE_FROZEN_BEFORE_COMPONENT_IMPLEMENTATION','Unexpected architecture-lock status')
    req(i.get('status')=='V5_R1_K1_IMPLEMENTATION_SOURCE_FROZEN_BEFORE_MODEL_QUALIFICATION','Unexpected implementation-lock status')
    req(q.get('status')=='V5_R1_K1_PINNED_MODELS_QUALIFIED_BEFORE_V5D_AUTHORING','Unexpected qualification-lock status')
    req(q.get('v5d_authored_or_executed') is False,'Qualification lock says V5D was already authored/executed')
    qdoc,fdoc,pdoc=load(QUERY_FILE),load(FACET_FILE),load(PRIOR)
    req(qdoc.get('status')=='frozen_before_v5d_retrieval','Unexpected V5D query status')
    req(fdoc.get('status')=='frozen_before_v5d_retrieval','Unexpected V5D facet status')
    qs,fs=qdoc['queries'],fdoc['queries']; expected=[f'V5D{n:02d}' for n in range(1,25)]
    req(len(qs)==24 and len(fs)==24,'Expected exactly 24 V5D queries/facet entries')
    req([x['query_id'] for x in qs]==expected,'V5D query IDs/order changed'); req([x['query_id'] for x in fs]==expected,'V5D facet IDs/order changed')
    req(Counter(x['stratum'] for x in qs)==Counter(STRATA),'V5D stratum balance changed'); req(Counter(x['stratum'] for x in fs)==Counter(STRATA),'V5D facet stratum balance changed')
    req([(x['query_id'],x['query'],x['stratum']) for x in qs]==[(x['query_id'],x['query'],x['stratum']) for x in fs],'Query/facet files disagree')
    for qrow,frow in zip(qs,fs):
        fl=frow.get('facets',[]); req(bool(fl),f"{qrow['query_id']}: no facets")
        for n,fac in enumerate(fl,1):
            req(fac.get('facet_id')==f'F{n}',f"{qrow['query_id']}: facet IDs must be sequential")
            req(fac.get('facet_type') in {'core','qualifier'},f"{qrow['query_id']}: invalid facet type")
            req(bool(fac.get('semantic_definition')),f"{qrow['query_id']}: missing semantic definition")
            req(bool(fac.get('required_components')),f"{qrow['query_id']}: missing required component")
            for c in fac['required_components']:
                req(bool(c.get('definition')) and bool(c.get('negative_boundaries')),f"{qrow['query_id']}: incomplete component boundary")
        s=qrow['stratum']
        if s=='two_core_theme': req(len(fl)==2 and all(x['facet_type']=='core' for x in fl),f"{qrow['query_id']}: expected two core facets")
        elif s=='three_core_conjunctive': req(len(fl)==3 and all(x['facet_type']=='core' for x in fl),f"{qrow['query_id']}: expected three core facets")
        elif s=='core_plus_qualifier': req(len(fl)==2 and [x['facet_type'] for x in fl]==['core','qualifier'],f"{qrow['query_id']}: expected core + qualifier")
        elif s=='relationship_or_role_constraint': req(len(fl)==1 and fl[0]['facet_type']=='core',f"{qrow['query_id']}: expected one relational core facet")
        elif s=='hard_exclusion_or_contrast': req(len(fl)==1 and bool(fl[0].get('hard_exclusions')),f"{qrow['query_id']}: expected one hard-exclusion core facet")
        elif s=='abstract_nonfiction_multi_facet': req(len(fl)==3 and all(x['facet_type']=='core' for x in fl),f"{qrow['query_id']}: expected three nonfiction core facets")
    req(pdoc.get('query_count')==63 and len(pdoc.get('queries',[]))==63,'Prior-query anti-leakage manifest must contain 63 prior/probe queries')
    expected_prior=[f'Q{n:02d}' for n in range(1,13)]+[f'H{n:02d}' for n in range(1,25)]+[f'V4H{n:02d}' for n in range(1,25)]+[f'P{n:02d}' for n in range(1,4)]
    req([x['query_id'] for x in pdoc['queries']]==expected_prior,'Prior-query manifest IDs/order changed')
    prior=[(x['query_id'],x['query']) for x in pdoc['queries']]; prior_norm={re.sub(r'\s+',' ',t.strip().casefold()) for _,t in prior}
    max_prior=(0.0,None,None); max_internal=(0.0,None,None)
    for qrow in qs:
        norm=re.sub(r'\s+',' ',qrow['query'].strip().casefold()); req(norm not in prior_norm,f"{qrow['query_id']}: exact normalized reuse of prior/probe query")
        for pid,ptext in prior:
            sim=jaccard(qrow['query'],ptext)
            if sim>max_prior[0]: max_prior=(sim,qrow['query_id'],pid)
            req(sim<MAX_JACCARD,f"{qrow['query_id']}: excessive token overlap with {pid}: jaccard={sim:.3f}")
    for idx,x in enumerate(qs):
        for y in qs[idx+1:]:
            sim=jaccard(x['query'],y['query'])
            if sim>max_internal[0]: max_internal=(sim,x['query_id'],y['query_id'])
            req(sim<MAX_JACCARD,f"Internal V5D query overlap too high: {x['query_id']} vs {y['query_id']} jaccard={sim:.3f}")
    anti=qdoc.get('anti_leakage',{})
    for k in ['catalog_inspected_during_authoring','v5d_retrieval_outputs_inspected_during_authoring','v5d_reranker_outputs_inspected_during_authoring','judge_outputs_inspected_during_authoring','human_relevance_labels_used','prior_query_texts_used_for_parameter_or_model_selection','paraphrased_reuse_detected']:
        req(anti.get(k) is False,f'Authoring anti-leakage field must be false: {k}')
    req(anti.get('manual_semantic_overlap_review_completed') is True,'Manual semantic-overlap review not recorded')
    print('PASS  exact V5D01-V5D24 IDs and 6x4 stratum balance verified')
    print('PASS  query file and facet file agree exactly')
    print('PASS  facet-shape and component-boundary contracts verified')
    print(f'PASS  no exact/near-duplicate prior/probe query detected; max Jaccard={max_prior[0]:.3f} ({max_prior[1]} vs {max_prior[2]})')
    print(f'PASS  no near-duplicate within V5D24; max internal Jaccard={max_internal[0]:.3f} ({max_internal[1]} vs {max_internal[2]})')
    print('PASS  candidate-blind V5D authoring declarations verified')
    return qdoc,fdoc,pdoc

def main()->None:
    print('Semantic relevance V5 - fresh V5D24 authoring preflight')
    print('------------------------------------------------------')
    print('V5D retrieval/reranking: DISABLED'); print('Judge execution: DISABLED'); print('Development selection: NOT CALCULATED'); print('Catalog inspection: DISABLED'); print()
    verify_boundary(require_lock_absent=True)
    print(); print('V5D24 AUTHORING PREFLIGHT: PASS')
    print(f'Query file SHA-256: {sha256(QUERY_FILE)}'); print(f'Facet file SHA-256: {sha256(FACET_FILE)}'); print(f'Prior-query manifest SHA-256: {sha256(PRIOR)}')
    print('V5D retrieval performed: NO'); print('Reranker inference performed: NO'); print('Judge calls performed: 0'); print('Development selection calculated: NO')
    print('Next: freeze V5D24 query/facet artifacts before any V5D retrieval.')
if __name__=='__main__': main()
