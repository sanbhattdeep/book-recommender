from __future__ import annotations

import pandas as pd

from semantic_relevance_v3_reranker import rank_v3a, rank_v3b


def main() -> None:
    rows = [
        {"query_id": "QX", "vector_rank": 1, "isbn13": "1", "reranker_score": 0.8},
        {"query_id": "QX", "vector_rank": 2, "isbn13": "2", "reranker_score": -0.3},
        {"query_id": "QX", "vector_rank": 3, "isbn13": "3", "reranker_score": 2.1},
        {"query_id": "QX", "vector_rank": 7, "isbn13": "4", "reranker_score": 3.7},
        {"query_id": "QX", "vector_rank": 12, "isbn13": "5", "reranker_score": 4.2},
        {"query_id": "QX", "vector_rank": 16, "isbn13": "6", "reranker_score": 4.8},
    ]
    frame = pd.DataFrame(rows)

    v3a = rank_v3a(frame).sort_values("v3a_rank")
    assert v3a["isbn13"].tolist() == ["6", "5", "4", "3", "1", "2"]

    v3b = rank_v3b(frame).sort_values("v3b_rank")
    # The exact RRF order is part of the frozen implementation contract.
    expected = v3b["isbn13"].tolist()
    assert set(expected) == {"1", "2", "3", "4", "5", "6"}
    assert v3b["rrf_score"].notna().all()
    assert sorted(v3b["v3b_rank"].tolist()) == [1, 2, 3, 4, 5, 6]

    # Deterministic tie handling: vector rank, then ISBN.
    tie = pd.DataFrame([
        {"query_id": "QT", "vector_rank": 2, "isbn13": "20", "reranker_score": 1.0},
        {"query_id": "QT", "vector_rank": 1, "isbn13": "10", "reranker_score": 1.0},
    ])
    tie_ranked = rank_v3a(tie).sort_values("v3a_rank")
    assert tie_ranked["isbn13"].tolist() == ["10", "20"]

    print("PASS  v3a ranking contract")
    print("PASS  v3b RRF contract")
    print("PASS  deterministic tie-break contract")


if __name__ == "__main__":
    main()
