"""
Deterministic prompt-contract tests for Judge Config v0.15.0.

No LLM is called. These tests ensure the approved frozen semantic definition is
actually present in every semantic LLM stage while the full query is not needed
by those prompt builders.
"""

from __future__ import annotations

import json
from pathlib import Path

from semantic_relevance_facet_judge import (
    build_evidence_selection_prompt,
    build_prominence_prompt,
    build_verification_prompt,
)
from semantic_relevance_facet_scoring import QueryFacet


EVALS_DIR = Path(__file__).resolve().parent
CONFIG_FILE = (
    EVALS_DIR
    / "judge_configs"
    / "semantic_relevance_judge.v0.15.0.json"
)

with CONFIG_FILE.open("r", encoding="utf-8") as f:
    config = json.load(f)

definition = (
    "Social or economic inequality. Generic legal injustice alone does not "
    "establish this facet."
)

facet = QueryFacet(
    facet_id="F1",
    text="inequality",
    facet_type="core",
    semantic_definition=definition,
)

spans = {
    "S1": "The book examines an individual case of legal injustice.",
    "S2": "It also discusses another topic.",
}

selection_prompt = build_evidence_selection_prompt(
    facet=facet,
    spans=spans,
    config=config,
)
verification_prompt = build_verification_prompt(
    facet=facet,
    evidence_text=spans["S1"],
    config=config,
)
prominence_prompt = build_prominence_prompt(
    facet=facet,
    evidence_text=spans["S1"],
    spans=spans,
    config=config,
)

for name, prompt in [
    ("selection", selection_prompt),
    ("verification", verification_prompt),
    ("prominence", prominence_prompt),
]:
    assert facet.text in prompt, f"{name}: missing facet text"
    assert definition in prompt, f"{name}: missing semantic definition"
    assert "FROZEN SEMANTIC DEFINITION" in prompt

assert spans["S1"] in selection_prompt
assert spans["S1"] in verification_prompt
assert spans["S1"] in prominence_prompt

# Contract checks for the v0.12 semantic-boundary and precedence instructions.
verification_instructions = " ".join(
    config["verification_stage"]["instructions"]
).lower()
prominence_instructions = " ".join(
    config["prominence_stage"]["instructions"]
).lower()

assert "morphological" in verification_instructions
assert "do not skip directly to adjacent" in verification_instructions
assert "unsupported" in verification_instructions
assert "adjacent" in verification_instructions
assert "decision precedence" in verification_prompt.lower()
assert "candidate ranking priority" in selection_prompt.lower()
assert "excluded_or_wrong_sense_gate" in verification_prompt
assert "direct_gate" in verification_prompt
assert "entailed_gate" in verification_prompt
assert "adjacent_gate" in verification_prompt
assert "literal_or_lexical_match" in selection_prompt
assert "removal test" in prominence_instructions
assert "incidental" in prominence_instructions

print("All v0.15.0 semantic-definition/prompt-precedence tests passed.")
