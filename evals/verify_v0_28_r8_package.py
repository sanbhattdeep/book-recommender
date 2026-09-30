"""Static verifier for semantic relevance v0.28.0 r8. No judge calls."""
from __future__ import annotations
import hashlib, json, py_compile, subprocess, sys
from pathlib import Path

R = Path(__file__).resolve().parents[1]
EXPECTED = {
    "evals/semantic_relevance_facet_judge.py": "e4efec54d03601e3016b376741c7f14fc1259ab3d63d1fdff75925cc2bfb43c1",
    "evals/semantic_relevance_facet_scoring.py": "8447a75aee29d1ebf34147e625a7fe8c1f8ccf4b409029e917ec4c47285e8808",
    "evals/facets/semantic_relevance/semantic_relevance_query_facets.v0.9.13.json": "0952d80ebed8fbdebfb3b625f5648be9ccc08148d87d6753adc83d010953e289",
    "evals/judge_configs/semantic_relevance_judge.v0.28.0-r8.json": "bdc4157b457e3453ff94b860cb45022a7c7c83f311432ef9841a1c1fe5a9bb8d",
    "evals/datasets/semantic_relevance_v0.28_regression_manifest.v1.7.0.json": "da8d7ffe0f9131dd5ab976a0af72d7c478ee592b4cf1b495e105e483d908c585",
    "evals/datasets/semantic_relevance_v0.28_post_validation_label_adjudications.v1.0.0.json": "68f5d85fd5905034d813234b9f9df8202252ef1598e9676b5247902e55a7b893",
}
FINAL_HOLDOUT_SHA = "0c7b55065629b6c1fbdb3f0d530cf8bdc8e15011b0d213ab4be6259b95a43e85"

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def require(c,m):
    if not c: raise AssertionError(m)
    print("PASS ", m)

def main():
    print("v0.28.0 r8 package verification")
    print("--------------------------------")
    print("JUDGE EXECUTION: DISABLED")
    for rel,expected in EXPECTED.items():
        p=R/rel
        require(p.exists(), f"artifact exists: {rel}")
        require(sha(p)==expected, f"hash pinned: {rel}")

    hold=R/"evals/datasets/semantic_relevance_final_holdout.v4.0.0.DO_NOT_RUN_YET.csv"
    lockp=R/"evals/datasets/semantic_relevance_final_holdout_lock.v4.0.0.json"
    require(hold.exists() and sha(hold)==FINAL_HOLDOUT_SHA, "independent final-holdout hash remains pinned")
    lock=json.loads(lockp.read_text(encoding="utf-8-sig"))
    require(lock.get("status")=="LOCKED_DO_NOT_RUN", "independent final holdout remains LOCKED_DO_NOT_RUN")
    require(int(lock.get("judge_run_count",-1))==0, "independent final holdout judge_run_count = 0")

    vlock=json.loads((R/"evals/datasets/semantic_relevance_validation_lock.v4.0.0.json").read_text(encoding="utf-8-sig"))
    require(vlock.get("status")=="INDEPENDENT_VALIDATION_COMPLETED", "r4 validation remains consumed once")
    require(int(vlock.get("judge_run_count",-1))==1, "r4 validation judge_run_count = 1")
    require(vlock.get("validation_decision")=="REVIEW_STOP_FINAL_HOLDOUT", "r4 validation failure decision preserved")

    for rel in [
        "evals/semantic_relevance_facet_judge.py",
        "evals/run_judge_v0_28_r8_development.py",
        "evals/build_v0_28_r8_development_dataset.py",
        "evals/analyze_v0_28_r8_targeted.py",
        "evals/test_v0_28_r8_contract.py",
        "evals/test_v0_28_r8_localized_guards.py",
    ]:
        py_compile.compile(str(R/rel), doraise=True)

    for script in [
        "evals/test_v0_28_r5_localized_guards.py",
        "evals/test_v0_28_r8_contract.py",
        "evals/test_v0_28_r8_localized_guards.py",
    ]:
        subprocess.run([sys.executable, str(R/script)], check=True, cwd=R)

    print("v0.28.0 r8 package verification passed.")
    print("No judge calls performed.")

if __name__=="__main__":
    main()
