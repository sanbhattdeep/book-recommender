from pathlib import Path
import hashlib,json,subprocess,sys
R=Path(__file__).resolve().parents[1]
EXPECTED={
"evals/semantic_relevance_facet_judge.py":"890498d37d827e9bb2a2dce1c9fc02358908097468f28dcd39ab9a71695abf92",
"evals/semantic_relevance_facet_scoring.py":"8d40b3f773e3c763d9b6436ff036e63966ddc2b3877ecf4bf5f8989664d2587b",
"evals/facets/semantic_relevance/semantic_relevance_query_facets.v0.10.1.json":"083c88fe4de381e8c2e51e30af75e999c9d09cff6cfb03b134472acf881629a0",
"evals/judge_configs/semantic_relevance_judge.v0.29.0-r4.json":"5293d9473d06c4ad5ab4496191f72fbc93e5f175bfcc65a912ad1541ea5427f4",
"evals/datasets/semantic_relevance_v0.29_regression_manifest.v2.3.0.json":"9d17da01aac280d52bc68515430fcc3fbd3aaf1e5f4e2262f32ec8868a56c611"}
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def req(c,m):
    assert c,m; print("PASS ",m)
def main():
    print("v0.29.0 r4 package verification\n--------------------------------\nJUDGE EXECUTION: DISABLED")
    for rel,h in EXPECTED.items():
        p=R/rel; req(p.exists(),f"artifact exists: {rel}"); req(sha(p)==h,f"hash pinned: {rel}")
    ds=R/'evals/datasets/semantic_relevance_v0.29_development.v6.1.0.csv'; req(ds.exists() and sha(ds)=='cfcc465497079944fec6da17f14bbe5f3350d9385ea9ddc646610b7751d72da7','adjudicated 240-case v0.29 development dataset hash pinned')
    cp=R/'evals/releases/semantic_relevance_v0.28.0_r8_final_evaluation_closeout.json'; req(cp.exists() and sha(cp)=='24f90865503f4572d54e168d7f2cca33647a7c9a3c294bfb1ab727fd544cb2de','v0.28 r8 closeout hash pinned')
    c=json.loads(cp.read_text(encoding='utf-8-sig')); req(c.get('closeout_status')=='EVALUATED_NOT_FINAL_QUALIFIED','r8 closeout status preserved'); req(c.get('final_holdout_decision')=='FINAL_HOLDOUT_REVIEW','r8 final decision preserved')
    b=c.get('evidence_boundary_for_next_lineage',{}); req(b.get('next_candidate_lineage')=='semantic_relevance_v0.29.0','r8 closeout authorizes v0.29 lineage'); req(b.get('r8_final_holdout_may_be_reused_as_independent_evidence') is False,'r8 final holdout remains unavailable as independent evidence')
    lock=json.loads((R/'evals/datasets/semantic_relevance_final_holdout_lock.v4.0.0.json').read_text(encoding='utf-8-sig')); req(lock.get('status')=='FINAL_HOLDOUT_COMPLETED','v0.28 final holdout remains completed'); req(int(lock.get('judge_run_count',-1))==1,'v0.28 final holdout remains consumed exactly once')
    eb=json.loads((R/'evals/datasets/semantic_relevance_v0.29_evidence_boundary.v1.0.0.json').read_text(encoding='utf-8-sig')); req(eb.get('new_independent_evidence_required') is True,'fresh independent v0.29 evidence still required after freeze')
    for s in ['evals/test_v0_28_r5_localized_guards.py','evals/test_v0_28_r8_localized_guards.py','evals/test_v0_29_r1_localized_guards.py','evals/test_v0_29_r2_localized_guards.py','evals/test_v0_29_r3_localized_guards.py','evals/test_v0_29_r4_localized_guards.py','evals/test_v0_29_r4_contract.py']:
        subprocess.run([sys.executable,str(R/s)],check=True,cwd=R)
    print('v0.29.0 r4 package verification passed.\nNo judge calls performed.')
if __name__=='__main__': main()
