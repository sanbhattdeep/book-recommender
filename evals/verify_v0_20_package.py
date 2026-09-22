from pathlib import Path
import json, hashlib, py_compile, subprocess, sys
ROOT=Path(__file__).resolve().parents[1]
E=ROOT/'evals'
required=[E/'semantic_relevance_facet_judge.py',E/'semantic_relevance_facet_scoring.py',E/'judge_configs/semantic_relevance_judge.v0.20.0.json',E/'facets/semantic_relevance/semantic_relevance_query_facets.v0.8.0.json',E/'rubrics/semantic_relevance/semantic_relevance_rubric.v0.1.0.json']
for p in required:
    assert p.exists(), p
r=json.loads(required[-1].read_text(encoding="utf-8"))
assert r.get('version')=='0.1.0' and set(r.get('scores',{}))=={'0','1','2','3','4'}
for p in E.glob('*.py'): py_compile.compile(str(p),doraise=True)
tests=['test_rubric_integrity.py','test_v0_20_boundary_spec.py','test_v0_20_hard_exclusion_contract.py','test_v0_20_context_role_contract.py','test_v0_20_audit_serialization.py','test_v0_20_no_case_specific_behavior.py','test_v0_20_dataset_builder_contract.py','test_facet_scoring.py','test_deterministic_direct_cues.py','test_polarity_aware_direct_cues.py','test_evidence_candidate_selection.py','test_composite_relation_consistency.py','test_composite_verifier_recovery.py']
for name in tests:
    subprocess.run([sys.executable,str(E/name)],cwd=E,check=True)
print('v0.20 package verification passed.')
print('Rubric bytes:', required[-1].stat().st_size)
print('Rubric SHA-256:', hashlib.sha256(required[-1].read_bytes()).hexdigest())
