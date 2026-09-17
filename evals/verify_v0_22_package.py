from pathlib import Path
import json, hashlib, py_compile, subprocess, sys
ROOT=Path(__file__).resolve().parents[1]
E=ROOT/'evals'
required=[
 E/'semantic_relevance_facet_judge.py', E/'semantic_relevance_facet_scoring.py',
 E/'judge_configs/semantic_relevance_judge.v0.22.0.json',
 E/'facets/semantic_relevance/semantic_relevance_query_facets.v0.8.0.json',
 E/'rubrics/semantic_relevance/semantic_relevance_rubric.v0.1.0.json',
 E/'datasets/semantic_relevance_v0.22_regression_manifest.v1.0.0.json',
 E/'analyze_v0_22_targeted.py', E/'run_judge_development.py', E/'build_v0_22_development_dataset.py'
]
for p in required: assert p.exists(), p
cfg=json.loads(required[2].read_text())
assert cfg['version']=='0.22.0'
assert cfg['facet_spec_version']=='0.8.0'
assert cfg['rubric_version']=='0.1.0'
assert 'book_subject_stage' in cfg
assert cfg['pipeline'].startswith('facet_independent_book_subject')
for p in E.glob('*.py'): py_compile.compile(str(p), doraise=True)
for name in [
 'test_v0_22_frozen_artifacts.py',
 'test_v0_22_subject_contract.py',
 'test_v0_22_subject_reuse.py',
 'test_v0_22_role_contract.py',
 'test_v0_22_precheck_contract.py',
 'test_v0_22_audit_contract.py',
 'test_facet_scoring.py',
 'test_deterministic_direct_cues.py',
 'test_polarity_aware_direct_cues.py',
 'test_evidence_candidate_selection.py',
 'test_composite_relation_consistency.py',
 'test_composite_verifier_recovery.py',
]:
 subprocess.run([sys.executable,str(E/name)],cwd=E,check=True)
print('v0.22.0 package verification passed.')
print('Rubric SHA-256:', hashlib.sha256(required[4].read_bytes()).hexdigest())
