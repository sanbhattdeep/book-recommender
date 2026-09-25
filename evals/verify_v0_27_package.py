"""Static/unit verifier for the v0.27.0 r6 overlay package. No Ollama calls."""
from pathlib import Path
import hashlib,json,subprocess,sys,re
EVALS=Path(__file__).resolve().parent; ROOT=EVALS.parent
def sha(p:Path)->str:return hashlib.sha256(p.read_bytes()).hexdigest()
required=[
 EVALS/'semantic_relevance_facet_judge.py',EVALS/'semantic_relevance_facet_scoring.py',EVALS/'run_judge_development.py',EVALS/'run_judge_development_direct_transport.py',
 EVALS/'analyze_v0_27_targeted.py',EVALS/'analyze_v0_27_development.py',EVALS/'build_v0_27_r6_development_dataset.py',
 EVALS/'datasets/semantic_relevance_v0.27_regression_manifest.v1.2.0.json',EVALS/'facets/semantic_relevance/semantic_relevance_query_facets.v0.9.6.json',EVALS/'judge_configs/semantic_relevance_judge.v0.27.0.json']
for p in required: assert p.exists(),p
cfg=json.loads((EVALS/'judge_configs/semantic_relevance_judge.v0.27.0.json').read_text(encoding='utf-8'))
assert cfg['version']=='0.27.0' and cfg['facet_spec_version']=='0.9.6'
assert cfg['text_licensed_inference_contract']['audit_field']=='external_knowledge_required'
reg=json.loads((EVALS/'datasets/semantic_relevance_v0.27_regression_manifest.v1.2.0.json').read_text(encoding='utf-8'))
assert len(reg['targeted_expectations'])==34
assert reg['targeted_expectations']['U3_Q05_T01']=={'judge_min':2,'judge_max':3}
assert reg['targeted_expectations']['U3_Q01_T03']=={'judge_max':0}
assert reg['targeted_expectations']['U3_Q02_T05']=={'judge_max':0}
assert reg['targeted_expectations']['U3_Q07_T01']=={'judge_min':1,'judge_max':2}
target=(EVALS/'analyze_v0_27_targeted.py').read_text(encoding='utf-8')
assert 'semantic_relevance_v0.27_regression_manifest.v1.2.0.json' in target and 'semantic_relevance_v0.27_development.v3.1.0.csv' in target
dev=(EVALS/'analyze_v0_27_development.py').read_text(encoding='utf-8')
for tok in ['EXPECTED_EVALUATION_DATASET_VERSION = "3.1.0"','EXPECTED_EVALUATION_DATASET_ROLE = "post_validation_development"','len(gold) != 150','len(comparison) != 150'] : assert tok in dev
a=(EVALS/'run_judge_development.py').read_text(encoding='utf-8')
for tok in ['JUDGE_CONFIG_VERSION = "0.27.0"','EVALUATION_DATASET_VERSION = "3.1.0"','FACET_SPEC_VERSION = "0.9.6"','len(dataset) != 150','startswith("U3_").sum()) != 30'] : assert tok in a,tok
# no bare read_text() on Windows-sensitive Python source
for p in EVALS.glob('*.py'):
    t=p.read_text(encoding='utf-8-sig'); assert re.search(r'\.read_text\(\s*\)',t) is None,f'bare read_text(): {p.name}'; compile(t,str(p),'exec')
for name in ['test_v0_27_r6_validation_boundaries.py','test_v0_27_external_knowledge_repair.py','test_v0_27_text_licensed_inference.py','test_v0_27_facet_clarifications.py','test_v0_27_post_holdout_label_revisions.py','test_v0_26_local_contract_isolation.py','test_v0_26_suspense_boundary.py','test_v0_26_grief_non_regression.py','test_v0_26_parent_child_referent_binding.py','test_v0_26_personal_growth_scope.py','test_v0_26_r4_label_revisions.py']:
    subprocess.run([sys.executable,str(EVALS/name)],cwd=ROOT,check=True)
print('v0.27.0 r6 package verification passed.')
print('Judge SHA-256:',sha(EVALS/'semantic_relevance_facet_judge.py'))
print('Scoring SHA-256:',sha(EVALS/'semantic_relevance_facet_scoring.py'))
print('Facet-spec SHA-256:',sha(EVALS/'facets/semantic_relevance/semantic_relevance_query_facets.v0.9.6.json'))
print('Judge-config SHA-256:',sha(EVALS/'judge_configs/semantic_relevance_judge.v0.27.0.json'))
print('Rubric SHA-256:',sha(EVALS/'rubrics/semantic_relevance/semantic_relevance_rubric.v0.1.0.json'))
