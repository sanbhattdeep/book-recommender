from __future__ import annotations
from types import SimpleNamespace

from semantic_relevance_facet_judge import (
    IsolatedComponentVerification,
    _q11_r7_explicit_classical_myth_content_guard,
)

def facet(text): return SimpleNamespace(text=text)
def comp(cid): return SimpleNamespace(component_id=cid)
def missing(cid): return IsolatedComponentVerification(
    component_id=cid, grounding_relation="missing", reason="test"
)

def main():
    positive = (
        "Odysseus sails home. The enchantress warns that the Greeks will face "
        "Scylla, a six-headed monster, and Charybdis. This book is based on "
        "episodes from Homer's Odyssey."
    )
    r = _q11_r7_explicit_classical_myth_content_guard(
        facet("mythology"), comp("mythic_or_mythological_basis"),
        positive, missing("mythic_or_mythological_basis")
    )
    assert r.grounding_relation == "entailed"

    # Provenance scholarship must remain excluded.
    analytical = (
        "The study presents historical evidence for the ritual origin and "
        "provenance of the legend of Beowulf."
    )
    r = _q11_r7_explicit_classical_myth_content_guard(
        facet("mythology"), comp("mythic_or_mythological_basis"),
        analytical, missing("mythic_or_mythological_basis")
    )
    assert r.grounding_relation == "missing"

    # A classical source name alone, without mythic narrative content, is insufficient.
    weak = "The book discusses a translation of Homer's Odyssey and its publication history."
    r = _q11_r7_explicit_classical_myth_content_guard(
        facet("mythology"), comp("mythic_or_mythological_basis"),
        weak, missing("mythic_or_mythological_basis")
    )
    assert r.grounding_relation == "missing"

    print("v0.28 r7 localized Q11 myth-content guard tests passed")

if __name__ == "__main__":
    main()
