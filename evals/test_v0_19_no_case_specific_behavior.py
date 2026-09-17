from pathlib import Path
ROOT=Path(__file__).resolve().parent
behavior_files=[
    ROOT/"semantic_relevance_facet_judge.py",
    ROOT/"semantic_relevance_facet_scoring.py",
    ROOT/"judge_configs"/"semantic_relevance_judge.v0.19.0.json",
    ROOT/"facets"/"semantic_relevance"/"semantic_relevance_query_facets.v0.7.0.json",
]
forbidden=[
    "Outsiders Within", "The Dead Yard", "How to Read Literature Like a Professor", "Drum Calls",
    "U2_Q12_T30", "U2_Q05_NEG", "U2_Q11_T30", "U2_Q04_T10",
]
for path in behavior_files:
    text=path.read_text(encoding="utf-8")
    hits=[x for x in forbidden if x in text]
    assert not hits, f"case-specific leakage in {path}: {hits}"
print("v0.19 behavior leakage scan passed.")
