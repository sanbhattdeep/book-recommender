"""Static regression test for the v0.18 fresh split runner contract."""
from __future__ import annotations

from pathlib import Path

HERE = Path(__file__).resolve().parent

for filename in [
    "run_judge_fresh_validation.py",
    "run_judge_fresh_holdout.py",
]:
    text = (HERE / filename).read_text(encoding="utf-8")

    assert "if len(dataset) != 30:" in text, filename
    assert 'startswith("U2_")' in text, filename

    assert "if len(dataset) != 60:" not in text, filename
    assert "The v0.17 development dataset must contain exactly 60 cases" not in text, filename
    assert 'startswith("U_")' not in text, filename

print("Fresh validation/holdout runner 30-case contract regression test passed.")
