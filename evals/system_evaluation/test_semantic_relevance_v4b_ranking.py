from __future__ import annotations

import pandas as pd

from semantic_relevance_v4b_ranking import rank_v4b, top_k_v4b


def _frame(rows):
    return pd.DataFrame(rows)


def test_primary_order_is_reranker_score_descending():
    frame = _frame([
        {"query_id": "H01", "isbn13": "3", "hybrid_retrieval_rank": 1, "reranker_score": 0.1},
        {"query_id": "H01", "isbn13": "2", "hybrid_retrieval_rank": 99, "reranker_score": 0.9},
        {"query_id": "H01", "isbn13": "1", "hybrid_retrieval_rank": 2, "reranker_score": 0.5},
    ])
    ranked = rank_v4b(frame)
    assert ranked["isbn13"].tolist() == ["2", "1", "3"]


def test_hybrid_rank_only_breaks_exact_score_tie():
    frame = _frame([
        {"query_id": "H01", "isbn13": "2", "hybrid_retrieval_rank": 20, "reranker_score": 1.0},
        {"query_id": "H01", "isbn13": "1", "hybrid_retrieval_rank": 5, "reranker_score": 1.0},
        {"query_id": "H01", "isbn13": "3", "hybrid_retrieval_rank": 1, "reranker_score": 0.999999999999},
    ])
    ranked = rank_v4b(frame)
    assert ranked["isbn13"].tolist() == ["1", "2", "3"]


def test_isbn_breaks_remaining_exact_tie():
    frame = _frame([
        {"query_id": "H01", "isbn13": "200", "hybrid_retrieval_rank": 7, "reranker_score": 2.0},
        {"query_id": "H01", "isbn13": "100", "hybrid_retrieval_rank": 7, "reranker_score": 2.0},
    ])
    ranked = rank_v4b(frame)
    assert ranked["isbn13"].tolist() == ["100", "200"]


def test_queries_are_ranked_independently():
    frame = _frame([
        {"query_id": "H01", "isbn13": "1", "hybrid_retrieval_rank": 2, "reranker_score": 0.2},
        {"query_id": "H02", "isbn13": "2", "hybrid_retrieval_rank": 2, "reranker_score": 0.1},
        {"query_id": "H01", "isbn13": "3", "hybrid_retrieval_rank": 1, "reranker_score": 0.3},
        {"query_id": "H02", "isbn13": "4", "hybrid_retrieval_rank": 1, "reranker_score": 0.4},
    ])
    ranked = rank_v4b(frame)
    assert ranked.loc[ranked["query_id"] == "H01", "v4b_rank"].tolist() == [1, 2]
    assert ranked.loc[ranked["query_id"] == "H02", "v4b_rank"].tolist() == [1, 2]


def test_top_k_requires_ten():
    frame = _frame([
        {"query_id": "H01", "isbn13": str(i), "hybrid_retrieval_rank": i, "reranker_score": float(20-i)}
        for i in range(1, 12)
    ])
    out = top_k_v4b(frame)
    assert len(out) == 10
    assert out["rank"].tolist() == list(range(1, 11))

    try:
        top_k_v4b(frame, top_k=5)
    except ValueError as exc:
        assert "top_k=10" in str(exc)
    else:
        raise AssertionError("Expected frozen top_k guard to reject top_k=5")


def test_duplicate_pair_is_rejected():
    frame = _frame([
        {"query_id": "H01", "isbn13": "1", "hybrid_retrieval_rank": 1, "reranker_score": 1.0},
        {"query_id": "H01", "isbn13": "1", "hybrid_retrieval_rank": 2, "reranker_score": 0.5},
    ])
    try:
        rank_v4b(frame)
    except ValueError as exc:
        assert "Duplicate query-book pair" in str(exc)
    else:
        raise AssertionError("Expected duplicate guard to fail")
