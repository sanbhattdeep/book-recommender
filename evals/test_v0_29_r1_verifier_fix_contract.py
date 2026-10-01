from pathlib import Path
R = Path(__file__).resolve().parents[1]
V = (R / "evals/verify_v0_29_r1_package.py").read_text(encoding="utf-8")
P = (R / "prepare_v029_r1.ps1").read_text(encoding="utf-8-sig")

for token in [
    "adjudication manifest source dataset SHA pinned",
    "adjudication manifest output dataset SHA pinned",
    "adjudicated-review actual SHA matches adjudication manifest",
    "exact three approved adjudication score transitions pinned",
    "exact two reviewed-but-unchanged labels pinned",
    "semantic repair targets pinned",
    'ADJUDICATION_SHA',
    'ADJUDICATED_REVIEW_SHA',
]:
    if token in {"ADJUDICATION_SHA", "ADJUDICATED_REVIEW_SHA"}:
        assert token not in V, token
    else:
        assert token in V, token

assert "verify_v0_29_r1_package.py" in P
assert "JUDGE EXECUTION: DISABLED" in P
compile(V, str(R / "evals/verify_v0_29_r1_package.py"), "exec")
print("v0.29 r1 verifier-fix contract passed")
