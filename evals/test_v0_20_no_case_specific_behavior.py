from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
behavior=[ROOT/'evals/semantic_relevance_facet_judge.py',ROOT/'evals/semantic_relevance_facet_scoring.py',ROOT/'evals/judge_configs/semantic_relevance_judge.v0.20.0.json',ROOT/'evals/facets/semantic_relevance/semantic_relevance_query_facets.v0.8.0.json']
forbidden=['Outsiders Within','The Dead Yard','How to Read Literature Like a Professor','The Girls of Lighthouse Lane','U2_Q12_T30','U2_Q05_NEG','U2_Q11_T30','U_Q02_NEG']
for p in behavior:
    s=p.read_text(encoding='utf-8')
    for token in forbidden:
        assert token not in s, f'{token!r} leaked into {p}'
print('v0.20 no-case-specific-behavior test passed.')
