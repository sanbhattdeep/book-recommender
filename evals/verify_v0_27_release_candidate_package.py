"""Static verification for the v0.27.0 r7 freeze overlay. Performs no judge calls."""
from pathlib import Path
import hashlib
import py_compile
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
E = ROOT / "evals"

subprocess.run([sys.executable, str(E / "verify_v0_27_package.py")], cwd=ROOT, check=True)

expected = {
    "semantic_relevance_facet_judge.py": "d7504a62721f7120dda50cdc6cea759fced8ea85c98d1766baf62bfd2fb716d1",
    "semantic_relevance_facet_scoring.py": "8447a75aee29d1ebf34147e625a7fe8c1f8ccf4b409029e917ec4c47285e8808",
    "facets/semantic_relevance/semantic_relevance_query_facets.v0.9.7.json": "2f54203336c8125e9bec6751afba6fc8aa45e046527e992550468f6a8aaa25de",
    "judge_configs/semantic_relevance_judge.v0.27.0.json": "d96347852d982d1f9242520519eac55ff7e0dd8c598b0eb02f82eeeb6c17bff2",
    "rubrics/semantic_relevance/semantic_relevance_rubric.v0.1.0.json": "658fe8ef49f1b73e4d6d6bdb143ab4d43a040ed7ea347fc2d4a41a3051ee7d2e",
}
for rel, expected_hash in expected.items():
    path = E / rel
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    assert actual == expected_hash, (rel, actual, expected_hash)

for name in ["freeze_v0_27_r7_release_candidate.py", "verify_v0_27_release_candidate_package.py"]:
    path = E / name
    assert path.exists(), path
    py_compile.compile(str(path), doraise=True)

freeze = (E / "freeze_v0_27_r7_release_candidate.py").read_text(encoding="utf-8")
assert "Exactly three --stability-run directories are required" in freeze
assert "complete 150-case development analysis" in freeze
assert "all 34 targeted regression checks" in freeze
assert "semantic_relevance_final_holdout.v3.0.0.DO_NOT_RUN_YET.csv" in freeze
assert "EXPECTED_FINAL_HOLDOUT_SHA" in freeze
assert "judge_run_count" in freeze
assert "NO JUDGE CALLS WERE MADE BY THIS FREEZE COMMAND" in freeze
assert "identical_across_three_runs" in freeze

wrapper = (ROOT / "freeze_v027_r7.ps1").read_text(encoding="utf-8")
assert "20260925T123917Z" in wrapper
assert "20260927T023402Z" in wrapper
assert "20260927T031200Z" in wrapper
assert "20260927T040746Z" in wrapper
assert "verify_v0_27_release_candidate_package.py" in wrapper
assert "freeze_v0_27_r7_release_candidate.py" in wrapper

print("v0.27.0 r7 freeze-overlay verification passed.")
print("No judge calls performed.")
