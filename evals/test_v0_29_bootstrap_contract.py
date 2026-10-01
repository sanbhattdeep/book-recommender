from pathlib import Path
R = Path(__file__).resolve().parents[1]
B = (R/"evals/build_v0_29_development_dataset.py").read_text(encoding="utf-8")
A = (R/"evals/reconstruct_v0_29_r8_baseline.py").read_text(encoding="utf-8")
P = (R/"prepare_v029_bootstrap.ps1").read_text(encoding="utf-8-sig")
for token in [
    "semantic_relevance_v0.29_development.v6.0.0.csv",
    "240",
    "consumed_v0_28_final_holdout_30",
    "labels_changed_during_bootstrap",
    "new_independent_evidence_required",
]:
    assert token in B, token
for token in [
    "r8_baseline_240",
    "semantic_relevance_v0.29_severe_review.v1.0.0.csv",
    "U4_Q03_T01","U4_Q03_T02","U4_Q04_T02","U4_Q06_T01","U4_Q09_T04",
    "no judge calls made",
]:
    assert token in A.lower() if token == "no judge calls made" else token in A, token
for token in [
    "closeout_v028_r8.ps1",
    "build_v0_29_development_dataset.py",
    "reconstruct_v0_29_r8_baseline.py",
    "JUDGE EXECUTION: DISABLED",
    "LABEL CHANGES: NONE",
]:
    assert token in P, token
compile(B, str(R/"evals/build_v0_29_development_dataset.py"), "exec")
compile(A, str(R/"evals/reconstruct_v0_29_r8_baseline.py"), "exec")
print("v0.29 bootstrap contract passed")
