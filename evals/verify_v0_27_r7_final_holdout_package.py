from pathlib import Path
import hashlib, py_compile, subprocess, sys
ROOT=Path(__file__).resolve().parents[1]; E=ROOT/'evals'
subprocess.run([sys.executable,str(E/'verify_v0_27_release_candidate_package.py')],cwd=ROOT,check=True)
expected={
 'semantic_relevance_facet_judge.py':'d7504a62721f7120dda50cdc6cea759fced8ea85c98d1766baf62bfd2fb716d1',
 'semantic_relevance_facet_scoring.py':'8447a75aee29d1ebf34147e625a7fe8c1f8ccf4b409029e917ec4c47285e8808',
 'facets/semantic_relevance/semantic_relevance_query_facets.v0.9.7.json':'2f54203336c8125e9bec6751afba6fc8aa45e046527e992550468f6a8aaa25de',
 'judge_configs/semantic_relevance_judge.v0.27.0.json':'d96347852d982d1f9242520519eac55ff7e0dd8c598b0eb02f82eeeb6c17bff2',
 'rubrics/semantic_relevance/semantic_relevance_rubric.v0.1.0.json':'658fe8ef49f1b73e4d6d6bdb143ab4d43a040ed7ea347fc2d4a41a3051ee7d2e',
}
for rel,h in expected.items():
 p=E/rel; assert hashlib.sha256(p.read_bytes()).hexdigest()==h,(rel,'hash mismatch')
new=['run_judge_v0_27_r7_final_holdout.py','run_judge_v0_27_r7_final_holdout_direct_transport.py','analyze_v0_27_r7_final_holdout.py','preflight_v0_27_r7_final_holdout.py']
for n in new:
 p=E/n; assert p.exists(),p; py_compile.compile(str(p),doraise=True)
runner=(E/'run_judge_v0_27_r7_final_holdout.py').read_text(encoding='utf-8')
for token in [
 'EXPECTED_HOLDOUT_CASES = 30','EXPECTED_HOLDOUT_DATASET_VERSION = "3.0.0"',
 'FINAL_HOLDOUT_STARTED.json','FINAL_V0_27_R7','Targeted fresh holdout runs are forbidden',
 'EXPECTED_RELEASE_MANIFEST_SHA256 = "5822551371fe21bd48cb5eb423f4c516b12770a75d7d11d5bb754ae5ca0dfbfc"',
 'EXPECTED_HOLDOUT_SHA256 = "cce944bd48b102452c024134560b7b87866b85f5fa1bc1194775138b8976eb2e"',
 'FINAL_HOLDOUT_STARTED','judge_run_count": 1',
]: assert token in runner,token
an=(E/'analyze_v0_27_r7_final_holdout.py').read_text(encoding='utf-8')
assert 'Result is final evidence regardless of PASS/REVIEW' in an
assert 'FINAL_HOLDOUT_COMPLETED' in an
assert 'pre_registered_release_checks' in an
pre=(E/'preflight_v0_27_r7_final_holdout.py').read_text(encoding='utf-8')
assert 'JUDGE EXECUTION: DISABLED' in pre and 'NO HOLDOUT JUDGE CALLS WERE MADE.' in pre
print('v0.27.0 r7 final-holdout package verification passed.')
