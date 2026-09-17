"""Deterministic polarity-aware DIRECT cue tests for v0.21.0. No LLM is called."""

from __future__ import annotations

import json
from pathlib import Path

from semantic_relevance_facet_judge import (
    _scan_deterministic_direct_cue,
    build_description_spans,
    find_deterministic_direct_cue,
)
from semantic_relevance_facet_scoring import QueryFacet

EVALS_DIR = Path(__file__).resolve().parent
CONFIG_FILE = EVALS_DIR / "judge_configs" / "semantic_relevance_judge.v0.21.0.json"
config = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))

teamwork = QueryFacet(
    facet_id="F2",
    text="teamwork",
    facet_type="core",
    semantic_definition="People collaborating, coordinating, cooperating, or functioning collectively toward a shared goal.",
)

# Positive lexical assertion remains a deterministic DIRECT cue.
spans = build_description_spans("The teams rely on cooperation and coordination to succeed.")
match, blocked = _scan_deterministic_direct_cue("Q10", teamwork, spans, config)
assert match is not None
assert match.cue_id == "Q10_F2_teamwork_lexical"
assert blocked == 0

# The v0.15 failure class: concept explicitly contrasted away.
spans = build_description_spans(
    "The system produces growing conflict instead of cooperation and eventually dissolves."
)
match, blocked = _scan_deterministic_direct_cue("Q10", teamwork, spans, config)
assert match is None
assert blocked >= 1

# Explicit absence/negation also suppresses the cue.
for text in [
    "The organization succeeds without teamwork.",
    "There is no cooperation between the departments.",
    "Teamwork is absent from the culture.",
    "The factions lack cooperation during the crisis.",
]:
    spans = build_description_spans(text)
    match, blocked = _scan_deterministic_direct_cue("Q10", teamwork, spans, config)
    assert match is None, text
    assert blocked >= 1, text

# "not only" is additive rather than negative and must not be suppressed.
spans = build_description_spans("The project requires not only cooperation but also careful coordination.")
match = find_deterministic_direct_cue("Q10", teamwork, spans, config)
assert match is not None

# A negated early occurrence does not prevent a later genuinely positive occurrence.
spans = build_description_spans(
    "At first there was no cooperation. Later, cooperation became the team's defining strength."
)
match, blocked = _scan_deterministic_direct_cue("Q10", teamwork, spans, config)
assert match is not None
assert blocked >= 1
assert match.evidence_span_id == "S2"

print("All v0.21.0 polarity-aware deterministic-cue tests passed.")
