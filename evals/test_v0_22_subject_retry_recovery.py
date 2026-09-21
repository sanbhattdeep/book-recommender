"""Regression contract for v0.22.2 Stage-A malformed-span retry recovery."""
from __future__ import annotations

import json
from pathlib import Path

from semantic_relevance_facet_judge import (
    BookSubjectAnalysis,
    assess_book_subject,
    build_book_subject_prompt,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "evals/judge_configs/semantic_relevance_judge.v0.26.0.json"
config = json.loads(CONFIG.read_text(encoding="utf-8"))
spans = {
    "S6": "The central situation begins.",
    "S7": "The conflict develops.",
    "S8": "The consequences deepen.",
    "S9": "The principal subject remains in focus.",
}

base_prompt = build_book_subject_prompt(spans=spans, config=config)
assert "Never combine IDs into a range" in base_prompt
assert "S6-S9" in base_prompt


class RepairingModel:
    def __init__(self) -> None:
        self.calls = 0

    def generate(self, prompt, schema, **kwargs):
        assert schema is BookSubjectAnalysis
        self.calls += 1
        if self.calls == 1:
            return BookSubjectAnalysis(
                primary_subject_summary="The central situation and conflict.",
                primary_subject_span_ids=["S6-S9"],
                reason="The subject is supported across the range.",
            )

        assert "PREVIOUS VALIDATION FAILURE" in prompt
        assert "unknown span IDs" in prompt
        assert "S6-S9" in prompt
        assert '["S6", "S7", "S8", "S9"]' in prompt
        return BookSubjectAnalysis(
            primary_subject_summary="The central situation and conflict.",
            primary_subject_span_ids=["S6", "S7", "S8", "S9"],
            reason="The individual spans support the subject.",
        )


model = RepairingModel()
result, retries = assess_book_subject(model, spans, config)
assert model.calls == 2
assert retries == 1
assert result.primary_subject_span_ids == ["S6", "S7", "S8", "S9"]
print("v0.22.2 malformed book-subject span retry recovery passed")
