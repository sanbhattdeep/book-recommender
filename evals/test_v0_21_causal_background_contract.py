"""Static contract tests for the v0.21.2 causal-background role refinement.

No LLM is called. These assertions protect the general classifier instruction
and the v0.21.1 deterministic resolver precedence that v0.21.2 intentionally
keeps unchanged.
"""

from __future__ import annotations

import json
from pathlib import Path

from semantic_relevance_facet_judge import (
    ProminenceAssessment,
    _context_role_from_decomposed_signals,
)
from semantic_relevance_facet_scoring import FacetContextRole

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "evals/judge_configs/semantic_relevance_judge.v0.21.2.json"

config = json.loads(CONFIG.read_text(encoding="utf-8"))
instructions = "\n".join(config["prominence_stage"]["instructions"])

# The classifier must require development of the facet itself, not simply a
# causal mention that explains some other primary subject.
assert "facet itself receives meaningful development" in instructions
assert "one item in a list of causes" in instructions
assert "counterfactual check" in instructions
assert "background-only rather than substantively examined" in instructions
assert "Causal importance alone is not substantive treatment" in instructions
assert "develops the facet beyond its role" in instructions

# Preserve the v0.21.1 fix: when the LLM genuinely supports both signals,
# substantive treatment outranks background cause deterministically.
result = ProminenceAssessment(
    primary_subject_summary="generic subject",
    is_primary_subject=False,
    is_background_cause_or_factor=True,
    is_example_or_illustration=False,
    is_meta_discussion=False,
    is_substantively_examined=True,
    supporting_span_ids=["S1"],
    reason="generic contract fixture",
)
assert (
    _context_role_from_decomposed_signals(result)
    == FacetContextRole.SUBSTANTIVE_SUBJECT
)

print("v0.21.2 causal-background role contract passed")
