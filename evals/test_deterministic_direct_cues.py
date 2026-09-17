"""Deterministic v0.16 direct-cue tests. No LLM is called."""

from __future__ import annotations

import json
from pathlib import Path

from semantic_relevance_facet_judge import (
    build_description_spans,
    find_deterministic_direct_cue,
)
from semantic_relevance_facet_scoring import QueryFacet

EVALS_DIR = Path(__file__).resolve().parent
CONFIG_FILE = EVALS_DIR / "judge_configs" / "semantic_relevance_judge.v0.21.2.json"
with CONFIG_FILE.open("r", encoding="utf-8") as f:
    config = json.load(f)


def facet(fid: str, text: str, definition: str, facet_type: str = "core") -> QueryFacet:
    return QueryFacet(
        facet_id=fid,
        text=text,
        facet_type=facet_type,
        semantic_definition=definition,
    )

# Q06: ordinary lexical entailment must not depend on Stage A remembering that
# an orphan has experienced loss.
spans = build_description_spans(
    "A haunting novel about an orphaned boy and his aunt surviving a war."
)
match = find_deterministic_direct_cue(
    "Q06",
    facet("F2", "loss", "Significant deprivation or ending."),
    spans,
    config,
)
assert match is not None
assert match.cue_id == "Q06_F2_loss_unmistakable"
assert "orphaned" in match.evidence_text.lower()

# Q08 core: explicit subject wording is enough for DIRECT relation.
spans = build_description_spans(
    "This introduction to astronomy features an exceptionally clear writing style for students."
)
match = find_deterministic_direct_cue(
    "Q08",
    facet("F1", "astronomy and the universe", "Astronomy/cosmology subject matter."),
    spans,
    config,
)
assert match is not None
assert match.cue_id == "Q08_F1_astronomy_subject"

# Q08 qualifier: strict composite introduction + clear/student language.
match = find_deterministic_direct_cue(
    "Q08",
    facet("F2", "beginner-friendly explanation", "Accessible introductory explanation.", "qualifier"),
    spans,
    config,
)
assert match is not None
assert match.cue_id == "Q08_F2_beginner_explicit"

# Excluded near-concepts remain fall-through: these MUST NOT become deterministic
# positives merely because they were previous development failures.
spans = build_description_spans("The characters learn how to adapt to change successfully.")
match = find_deterministic_direct_cue(
    "Q01",
    facet("F2", "redemption", "Restoration after wrongdoing; adaptation alone is excluded."),
    spans,
    config,
)
assert match is None

spans = build_description_spans("A lawyer confronts an individual case of injustice.")
match = find_deterministic_direct_cue(
    "Q12",
    facet("F1", "inequality", "Social/economic inequality; individual injustice alone is excluded."),
    spans,
    config,
)
assert match is None

print("All v0.21.2 deterministic-direct-cue tests passed.")
