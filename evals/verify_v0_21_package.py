from pathlib import Path
import json, hashlib, py_compile, subprocess, sys
ROOT=Path(__file__).resolve().parents[1]
E=ROOT/'evals'
required=[
 E/'semantic_relevance_facet_judge.py', E/'semantic_relevance_facet_scoring.py',
 E/'judge_configs/semantic_relevance_judge.v0.21.2.json',
 E/'facets/semantic_relevance/semantic_relevance_query_facets.v0.8.0.json',
 E/'rubrics/semantic_relevance/semantic_relevance_rubric.v0.1.0.json',
 E/'datasets/semantic_relevance_v0.21_regression_manifest.v1.1.0.json',
 E/'analyze_v0_21_targeted.py', E/'run_judge_development.py'
]
for p in required: assert p.exists(), p
cfg=json.loads(required[2].read_text(encoding="utf-8"))
assert cfg['version']=='0.21.2'
assert cfg['facet_spec_version']=='0.8.0'
assert cfg['rubric_version']=='0.1.0'
for p in E.glob('*.py'): py_compile.compile(str(p), doraise=True)
for name in ['test_v0_21_frozen_artifacts.py','test_v0_21_role_contract.py','test_v0_21_precheck_contract.py','test_v0_21_causal_background_contract.py','test_facet_scoring.py','test_deterministic_direct_cues.py','test_polarity_aware_direct_cues.py','test_evidence_candidate_selection.py','test_composite_relation_consistency.py','test_composite_verifier_recovery.py']:
 subprocess.run([sys.executable,str(E/name)],cwd=E,check=True)
print('v0.21.2 package verification passed.')
print('Rubric SHA-256:', hashlib.sha256(required[4].read_bytes()).hexdigest())
