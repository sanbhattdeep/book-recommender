"""Verify the v0.27 post-holdout preparation overlay."""
from __future__ import annotations

from pathlib import Path
import json
import py_compile
import subprocess
import sys

EVALS = Path(__file__).resolve().parent
required = [
    EVALS / "build_v0_27_development_dataset.py",
    EVALS / "analyze_v0_27_post_adjudication_baseline.py",
    EVALS / "test_v0_27_post_holdout_label_revisions.py",
    EVALS / "datasets/semantic_relevance_post_holdout_label_revisions.v1.0.0.json",
    EVALS / "datasets/semantic_relevance_v0.27_regression_manifest.v1.0.0.json",
]
for p in required:
    assert p.exists(), p
for p in EVALS.glob("*.py"):
    if p.name.startswith(("build_v0_27", "analyze_v0_27", "test_v0_27", "verify_v0_27")):
        py_compile.compile(str(p), doraise=True)

rev = json.loads(required[3].read_text(encoding="utf-8"))
assert len(rev["revisions"]) == 4
reg = json.loads(required[4].read_text(encoding="utf-8"))
assert len(reg["targeted_expectations"]) == 24

subprocess.run([sys.executable, str(EVALS / "test_v0_27_post_holdout_label_revisions.py")], check=True)
print("v0.27 post-holdout prep package verification passed.")
