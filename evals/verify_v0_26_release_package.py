from pathlib import Path
import hashlib, json, py_compile, subprocess, sys
ROOT=Path(__file__).resolve().parents[1]; E=ROOT/"evals"
# First prove the underlying r6 package still passes all semantic/contracts.
subprocess.run([sys.executable,str(E/"verify_v0_26_package.py")],cwd=ROOT,check=True)
expected={
 "semantic_relevance_facet_judge.py":"3f1f54bd7dba9fd08ddaafa55684c2f8bba8c7c888571996258cce57baeb3614",
 "semantic_relevance_facet_scoring.py":"8447a75aee29d1ebf34147e625a7fe8c1f8ccf4b409029e917ec4c47285e8808",
 "facets/semantic_relevance/semantic_relevance_query_facets.v0.9.3.json":"1821180bdbd9284db67d9bea661e69049713c29051b1ba5ea297d41690735c01",
 "rubrics/semantic_relevance/semantic_relevance_rubric.v0.1.0.json":"658fe8ef49f1b73e4d6d6bdb143ab4d43a040ed7ea347fc2d4a41a3051ee7d2e",
}
for rel,h in expected.items():
 p=E/rel; assert hashlib.sha256(p.read_bytes()).hexdigest()==h,(rel,"hash mismatch")
new=["freeze_v0_26_r6_release_candidate.py","run_judge_final_holdout.py","run_judge_final_holdout_direct_transport.py","analyze_v0_26_final_holdout.py"]
for n in new:
 p=E/n; assert p.exists(),p; py_compile.compile(str(p),doraise=True)
runner=(E/"run_judge_final_holdout.py").read_text(encoding="utf-8")
assert 'EXPECTED_HOLDOUT_CASES = 30' in runner
assert 'EXPECTED_HOLDOUT_DATASET_VERSION = "2.0.0"' in runner
assert 'FINAL_HOLDOUT_STARTED.json' in runner
assert 'Targeted fresh holdout runs are forbidden' in runner
assert 'judge_behavior_frozen": True' in runner
assert 'Final holdout overlaps development ISBNs' in runner
freeze=(E/"freeze_v0_26_r6_release_candidate.py").read_text(encoding="utf-8")
assert 'Exactly three --stability-run directories are required' in freeze
assert 'all 21 targeted regression checks' in freeze.lower()
an=(E/"analyze_v0_26_final_holdout.py").read_text(encoding="utf-8")
assert 'Result is final evidence regardless of PASS/REVIEW' in an
assert 'pre_registered_release_checks' in an
# This verifier is designed to run AFTER the release package has been overlaid
# into an existing repository. Root-level package metadata/docs may legitimately
# be omitted by an overlay that copies only evals/, so do not require them here.
# Instead verify the frozen runtime hashes and the release tooling that must exist
# under evals/. Historical/consumed holdout CSVs already present in the repository
# are also allowed; the final-holdout runner separately enforces dataset_version=2.0.0
# and overlap/finality guards before any case is consumed.
required_release_tools = [
    "freeze_v0_26_r6_release_candidate.py",
    "run_judge_final_holdout.py",
    "run_judge_final_holdout_direct_transport.py",
    "analyze_v0_26_final_holdout.py",
]
for name in required_release_tools:
    path = E / name
    assert path.exists(), f"Missing release tool after overlay: {path}"
    py_compile.compile(str(path), doraise=True)

# Do not scan repository CSVs here. An existing repository may contain historical
# holdouts or a separately staged untouched holdout. The one-time runner performs
# the authoritative dataset-version, overlap, and finality checks immediately
# before consumption.

print("v0.26.0 r6 final-holdout package r3 verification passed.")
