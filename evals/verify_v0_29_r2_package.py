from pathlib import Path
import hashlib,json,subprocess,sys
R=Path(__file__).resolve().parents[1]
EXPECTED={
"evals/semantic_relevance_facet_judge.py":"ff8ef4fefb69b37b490b548d5bd7daf3acdb6a4f4d57be237c2e0820d5f5cf5f",
"evals/semantic_relevance_facet_scoring.py":"8d40b3f773e3c763d9b6436ff036e63966ddc2b3877ecf4bf5f8989664d2587b",
"evals/facets/semantic_relevance/semantic_relevance_query_facets.v0.10.1.json":"083c88fe4de381e8c2e51e30af75e999c9d09cff6cfb03b134472acf881629a0",
"evals/judge_configs/semantic_relevance_judge.v0.29.0-r2.json":"5da2304cdecbf46f1c1c076badab1774d3d2237afcce77000b430a2679445708",
"evals/datasets/semantic_relevance_v0.29_regression_manifest.v2.1.0.json":"c2c7cb1fe6d0500e8db9d5cd3223629088b645961eb100ba69380b7d40bf9b55"}
DATASET_SHA="cfcc465497079944fec6da17f14bbe5f3350d9385ea9ddc646610b7751d72da7"
CLOSEOUT_SHA="24f90865503f4572d54e168d7f2cca33647a7c9a3c294bfb1ab727fd544cb2de"
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def req(c,m):
    if not c: raise AssertionError(m)
    print("PASS ",m)
def main():
    print("v0.29.0 r2 package verification"); print("--------------------------------"); print("JUDGE EXECUTION: DISABLED")
    for rel,h in EXPECTED.items():
        p=R/rel; req(p.exists(),f"artifact exists: {rel}"); req(sha(p)==h,f"hash pinned: {rel}")
    ds=R/"evals/datasets/semantic_relevance_v0.29_development.v6.1.0.csv"; req(ds.exists() and sha(ds)==DATASET_SHA,"adjudicated 240-case v0.29 development dataset hash pinned")
    close=R/"evals/releases/semantic_relevance_v0.28.0_r8_final_evaluation_closeout.json"; req(close.exists() and sha(close)==CLOSEOUT_SHA,"v0.28 r8 closeout hash pinned")
    adj=json.loads((R/"evals/datasets/semantic_relevance_v0.29_adjudications.v1.0.0.json").read_text(encoding="utf-8-sig")); req(adj.get("output_dataset_sha256")==DATASET_SHA,"adjudication output dataset SHA pinned"); req(adj.get("judge_calls_made") is False and adj.get("semantic_changes_made") is False,"adjudication provenance preserved")
    lock=json.loads((R/"evals/datasets/semantic_relevance_final_holdout_lock.v4.0.0.json").read_text(encoding="utf-8-sig")); req(lock.get("status")=="FINAL_HOLDOUT_COMPLETED","v0.28 final holdout remains completed"); req(int(lock.get("judge_run_count",-1))==1,"v0.28 final holdout remains consumed exactly once")
    bp=json.loads((R/"evals/datasets/semantic_relevance_v0.29_evidence_boundary.v1.0.0.json").read_text(encoding="utf-8-sig")); req(bp.get("new_independent_evidence_required") is True,"new independent v0.29 evidence still required after candidate freeze")
    for script in ["evals/test_v0_28_r5_localized_guards.py","evals/test_v0_28_r8_localized_guards.py","evals/test_v0_29_r1_localized_guards.py","evals/test_v0_29_r2_localized_guards.py","evals/test_v0_29_r2_contract.py"]: subprocess.run([sys.executable,str(R/script)],check=True,cwd=R)
    print("v0.29.0 r2 package verification passed."); print("No judge calls performed.")
if __name__=="__main__": main()
