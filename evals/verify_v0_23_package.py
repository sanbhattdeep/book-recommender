from pathlib import Path
import hashlib, json, py_compile, subprocess, sys
ROOT=Path(__file__).resolve().parents[1]
E=ROOT/'evals'
required=[
 E/'semantic_relevance_facet_judge.py', E/'semantic_relevance_facet_scoring.py',
 E/'judge_configs/semantic_relevance_judge.v0.23.0.json',
 E/'facets/semantic_relevance/semantic_relevance_query_facets.v0.8.0.json',
 E/'rubrics/semantic_relevance/semantic_relevance_rubric.v0.1.0.json',
 E/'datasets/semantic_relevance_v0.23_regression_manifest.v2.0.0.json',
 E/'datasets/semantic_relevance_label_revisions.v1.0.0.json',
 E/'analyze_v0_23_targeted.py', E/'analyze_v0_23_development.py',
 E/'run_judge_development.py', E/'build_v0_23_development_dataset.py'
]
for p in required: assert p.exists(), p
cfg=json.loads(required[2].read_text(encoding='utf-8'))
assert cfg['version']=='0.23.0'
assert cfg['dataset_version']=='0.5.0'
assert cfg['facet_spec_version']=='0.8.0'
assert cfg['rubric_version']=='0.1.0'
assert 'component_complete_verification' in cfg['pipeline']
assert 'book_subject_stage' in cfg
runner_text=(E/'run_judge_development.py').read_text(encoding='utf-8')
assert 'EVALUATION_DATASET_VERSION = "2.1.0"' in runner_text
assert 'JUDGE_CALIBRATION_DATASET_VERSION = "0.5.0"' in runner_text
# Frozen/scoring artifacts must remain byte-identical to the v0.22.2 package.
expected_hashes={
 E/'semantic_relevance_facet_scoring.py':'69cb7cea0ba288724bd623679d4d0b270f3a9ad1a1810146a02c4383e8f9b933',
 E/'facets/semantic_relevance/semantic_relevance_query_facets.v0.8.0.json':'970274175c115b442893be9a057352ea43c444937f21bb38f95c1990e58d9ac6',
 E/'rubrics/semantic_relevance/semantic_relevance_rubric.v0.1.0.json':'658fe8ef49f1b73e4d6d6bdb143ab4d43a040ed7ea347fc2d4a41a3051ee7d2e',
}
for path, expected in expected_hashes.items():
    actual=hashlib.sha256(path.read_bytes()).hexdigest()
    assert actual==expected, (path, actual, expected)
for p in E.glob('*.py'): py_compile.compile(str(p), doraise=True)
for name in [
 'test_v0_22_frozen_artifacts.py',
 'test_v0_22_subject_contract.py',
 'test_v0_22_subject_retry_recovery.py',
 'test_v0_22_subject_reuse.py',
 'test_v0_22_role_contract.py',
 'test_v0_22_subject_relation_invariants.py',
 'test_v0_22_precheck_contract.py',
 'test_v0_22_audit_contract.py',
 'test_v0_23_component_completeness.py',
 'test_v0_23_verifier_recovery.py',
 'test_v0_23_no_hard_exclusion_normalization.py',
 'test_v0_23_label_revision_manifest.py',
 'test_facet_scoring.py',
 'test_deterministic_direct_cues.py',
 'test_polarity_aware_direct_cues.py',
 'test_evidence_candidate_selection.py',
 'test_composite_relation_consistency.py',
 'test_composite_verifier_recovery.py',
]:
 subprocess.run([sys.executable,str(E/name)],cwd=E,check=True)
print('v0.23.0 package verification passed.')
print('Scoring SHA-256:', hashlib.sha256((E/'semantic_relevance_facet_scoring.py').read_bytes()).hexdigest())
print('Facet-spec SHA-256:', hashlib.sha256((E/'facets/semantic_relevance/semantic_relevance_query_facets.v0.8.0.json').read_bytes()).hexdigest())
print('Rubric SHA-256:', hashlib.sha256((E/'rubrics/semantic_relevance/semantic_relevance_rubric.v0.1.0.json').read_bytes()).hexdigest())
