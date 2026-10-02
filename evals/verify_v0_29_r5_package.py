from __future__ import annotations
from pathlib import Path
import hashlib,json,subprocess,sys
R=Path(__file__).resolve().parents[1]
EXPECTED={
"evals/semantic_relevance_facet_judge.py":"41b3b6e9650e7e4a1f1ad9693c2b0a145c844c3f813852d6c53ed8896ea9384a",
"evals/semantic_relevance_facet_scoring.py":"8d40b3f773e3c763d9b6436ff036e63966ddc2b3877ecf4bf5f8989664d2587b",
"evals/facets/semantic_relevance/semantic_relevance_query_facets.v0.10.1.json":"083c88fe4de381e8c2e51e30af75e999c9d09cff6cfb03b134472acf881629a0",
"evals/judge_configs/semantic_relevance_judge.v0.29.0-r5.json":"47c2bc8013d34939a699b590f33b521ef3aa17074b288f3bfcf09e2892572275",
"evals/datasets/semantic_relevance_v0.29_regression_manifest.v2.4.0.json":"095c222d05cffd18e7e5f2b41d5409cee143760182ccc11d77f0a84c9b920af6",}
DATASET_SHA="cfcc465497079944fec6da17f14bbe5f3350d9385ea9ddc646610b7751d72da7"
CLOSEOUT_SHA="24f90865503f4572d54e168d7f2cca33647a7c9a3c294bfb1ab727fd544cb2de"
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def req(c,m):
    if not c: raise AssertionError(m)
    print("PASS ",m)
def main():
    print("v0.29.0 r5 package verification")
    print("--------------------------------")
    print("JUDGE EXECUTION: DISABLED")
    for rel,exp in EXPECTED.items():
        p=R/rel; req(p.exists(),f"artifact exists: {rel}"); req(sha(p)==exp,f"hash pinned: {rel}")
    ds=R/"evals/datasets/semantic_relevance_v0.29_development.v6.1.0.csv"
    close=R/"evals/releases/semantic_relevance_v0.28.0_r8_final_evaluation_closeout.json"
    lock=R/"evals/datasets/semantic_relevance_final_holdout_lock.v4.0.0.json"
    eb=R/"evals/datasets/semantic_relevance_v0.29_evidence_boundary.v1.0.0.json"
    prereqs=[ds,close,lock,eb]
    if all(p.exists() for p in prereqs):
        req(sha(ds)==DATASET_SHA,"adjudicated 240-case v0.29 development dataset hash pinned")
        req(sha(close)==CLOSEOUT_SHA,"v0.28 r8 closeout hash pinned")
        c=json.loads(close.read_text(encoding="utf-8-sig")); req(c.get("closeout_status")=="EVALUATED_NOT_FINAL_QUALIFIED","r8 closeout status preserved"); req(c.get("final_holdout_decision")=="FINAL_HOLDOUT_REVIEW","r8 final decision preserved")
        b=c.get("evidence_boundary_for_next_lineage",{}); req(b.get("next_candidate_lineage")=="semantic_relevance_v0.29.0","r8 closeout authorizes v0.29 lineage"); req(b.get("r8_final_holdout_may_be_reused_as_independent_evidence") is False,"r8 final holdout remains unavailable as independent evidence")
        l=json.loads(lock.read_text(encoding="utf-8-sig")); req(l.get("status")=="FINAL_HOLDOUT_COMPLETED","v0.28 final holdout remains completed"); req(int(l.get("judge_run_count",-1))==1,"v0.28 final holdout remains consumed exactly once")
        e=json.loads(eb.read_text(encoding="utf-8-sig")); req(e.get("new_independent_evidence_required") is True,"fresh independent v0.29 evidence still required after freeze")
    else:
        print("INFO  repository-only dataset/lifecycle prerequisites are not bundled in this overlay; strict checks are deferred until overlay into the repository")
    for s in ["evals/test_v0_28_r5_localized_guards.py","evals/test_v0_28_r8_localized_guards.py","evals/test_v0_29_r1_localized_guards.py","evals/test_v0_29_r2_localized_guards.py","evals/test_v0_29_r3_localized_guards.py","evals/test_v0_29_r4_localized_guards.py","evals/test_v0_29_r5_localized_guards.py","evals/test_v0_29_r5_contract.py"]:
        subprocess.run([sys.executable,str(R/s)],check=True,cwd=R)
    print("v0.29.0 r5 package verification passed.")
    print("No judge calls performed.")
if __name__=="__main__": main()
