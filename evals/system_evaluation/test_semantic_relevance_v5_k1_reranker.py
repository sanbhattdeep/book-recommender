from __future__ import annotations

import pandas as pd
import pytest

from semantic_relevance_v5_k1_reranker import (
    K1RuntimeConfig,
    candidate_text,
    rank_k1,
    reranker_order_signature,
    top_k_k1,
)


def test_candidate_text_contract():
    assert candidate_text("Book", "Desc") == "Title: Book\nDescription: Desc"
    assert candidate_text("Book", "") == "Title: Book"


def test_frozen_config_rejects_parameter_change():
    K1RuntimeConfig()
    with pytest.raises(ValueError):
        K1RuntimeConfig(batch_size=8)
    with pytest.raises(ValueError):
        K1RuntimeConfig(max_length_tokens=1024)
    with pytest.raises(ValueError):
        K1RuntimeConfig(model_id="other")


def test_rank_k1_is_raw_score_only_with_exact_tie_breaks():
    frame = pd.DataFrame([
        {"query_id":"Q1","isbn13":"300","hybrid_retrieval_rank":1,"reranker_score":0.5},
        {"query_id":"Q1","isbn13":"200","hybrid_retrieval_rank":3,"reranker_score":0.7},
        {"query_id":"Q1","isbn13":"100","hybrid_retrieval_rank":2,"reranker_score":0.7},
        {"query_id":"Q1","isbn13":"050","hybrid_retrieval_rank":2,"reranker_score":0.7},
    ])
    ranked = rank_k1(frame)
    assert ranked["isbn13"].tolist() == ["050", "100", "200", "300"]
    assert ranked["k1_rank"].tolist() == [1,2,3,4]


def test_k1_does_not_bucket_near_equal_scores():
    frame = pd.DataFrame([
        {"query_id":"Q1","isbn13":"1","hybrid_retrieval_rank":1,"reranker_score":0.5000000000},
        {"query_id":"Q1","isbn13":"2","hybrid_retrieval_rank":2,"reranker_score":0.5000000001},
    ])
    ranked = rank_k1(frame)
    assert ranked["isbn13"].tolist() == ["2", "1"]


def test_top10_and_signature():
    rows=[]
    for i in range(12):
        rows.append({"query_id":"Q1","isbn13":str(100+i),"hybrid_retrieval_rank":i+1,"reranker_score":float(i)})
    frame=pd.DataFrame(rows)
    top=top_k_k1(frame)
    assert len(top)==10
    sig1=reranker_order_signature(frame)
    sig2=reranker_order_signature(frame.copy())
    assert sig1==sig2
