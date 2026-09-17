"""Static and mock-model contracts for v0.22.0 frozen book-subject extraction."""
from __future__ import annotations
import json
from pathlib import Path
from semantic_relevance_facet_judge import (
    BookSubjectAnalysis,
    build_book_subject_prompt,
    build_prominence_prompt,
)
from semantic_relevance_facet_scoring import QueryFacet

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "evals/judge_configs/semantic_relevance_judge.v0.22.0.json"
config = json.loads(CONFIG.read_text(encoding="utf-8"))
spans = {"S1": "A principal subject is described.", "S2": "A secondary cause is mentioned."}
subject = BookSubjectAnalysis(
    primary_subject_summary="A principal subject is described.",
    primary_subject_span_ids=["S1"],
    reason="S1 states the principal subject.",
)
facet = QueryFacet(
    facet_id="F1", text="synthetic factor", facet_type="core",
    semantic_definition="A synthetic factor used only for contract testing.",
)
subject_prompt = build_book_subject_prompt(spans, config)
assert "FACET" not in subject_prompt
assert facet.text not in subject_prompt
assert "No user query or query facet is available" in subject_prompt
assert "S1: A principal subject is described." in subject_prompt
role_prompt = build_prominence_prompt(facet, "S2: A secondary cause is mentioned.", subject, spans, config)
assert "FROZEN FACET-INDEPENDENT BOOK SUBJECT" in role_prompt
assert subject.primary_subject_summary in role_prompt
assert "Do NOT return or redefine the book subject" in role_prompt
print("v0.22.0 facet-independent subject contract passed")
