from __future__ import annotations
from types import SimpleNamespace

from semantic_relevance_facet_judge import (
    _q11_r8_cross_span_classical_myth_component_check,
)

def facet(text):
    return SimpleNamespace(text=text)

def comp(cid):
    return SimpleNamespace(component_id=cid)

def main():
    # Positive: source/retelling and mythic-content signals are intentionally
    # separated across exact spans.
    spans = {
        "S1": "Odysseus returned alive from the terrifying Land of the Dead.",
        "S2": "The enchantress Circe warns of Scylla, a six-headed monster, and Charybdis.",
        "S3": "This book is based on episodes from Homer's Odyssey.",
    }
    r = _q11_r8_cross_span_classical_myth_component_check(
        facet("mythology"), comp("mythic_or_mythological_basis"), spans
    )
    assert r is not None
    assert r.established is True
    assert r.grounding_relation == "entailed"
    assert "S3" in r.supporting_span_ids
    assert "S1" in r.supporting_span_ids or "S2" in r.supporting_span_ids

    # Negative: classical source name but no supernatural mythic narrative content.
    weak = {
        "S1": "The book discusses a translation of Homer's Odyssey.",
        "S2": "It focuses on publication history and textual scholarship.",
    }
    assert _q11_r8_cross_span_classical_myth_component_check(
        facet("mythology"), comp("mythic_or_mythological_basis"), weak
    ) is None

    # Negative: analytical legend-origin/provenance prose remains blocked.
    provenance = {
        "S1": "The study presents historical evidence for the ritual origin of the legend.",
        "S2": "It discusses a monster motif in later manuscripts.",
        "S3": "The analysis compares the work with Homer's Odyssey.",
    }
    assert _q11_r8_cross_span_classical_myth_component_check(
        facet("mythology"), comp("mythic_or_mythological_basis"), provenance
    ) is None

    print("v0.28 r8 cross-span Q11 component tests passed")

if __name__ == "__main__":
    main()
