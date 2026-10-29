from __future__ import annotations

import numpy as np
import pytest

from semantic_relevance_v5_r1_retrieval import (
    BATCH_SIZE,
    CANDIDATE_DEPTH,
    EMBEDDING_DIMENSION,
    R1DenseIndex,
    R1DenseRuntimeConfig,
    dense_order_signature,
)


def unit(*indices: int) -> np.ndarray:
    x = np.zeros(EMBEDDING_DIMENSION, dtype=np.float32)
    for i in indices:
        x[i] = 1.0
    x /= np.linalg.norm(x)
    return x


def test_frozen_config_rejects_parameter_change():
    R1DenseRuntimeConfig()
    with pytest.raises(ValueError):
        R1DenseRuntimeConfig(candidate_depth=51)
    with pytest.raises(ValueError):
        R1DenseRuntimeConfig(batch_size=BATCH_SIZE + 1)
    with pytest.raises(ValueError):
        R1DenseRuntimeConfig(model_id="other")


def test_dense_rank_uses_normalized_dot_product_and_isbn_tie_break():
    docs = np.stack([unit(0), unit(1), unit(0)], axis=0)
    index = R1DenseIndex(("300", "200", "100"), docs)
    ranked = index.rank_query_embedding(unit(0), top_k=CANDIDATE_DEPTH)
    assert ranked["isbn13"].tolist()[:3] == ["100", "300", "200"]
    assert ranked["dense_rank"].tolist() == [1, 2, 3]
    assert ranked["dense_score"].tolist()[:2] == pytest.approx([1.0, 1.0])


def test_dense_index_rejects_duplicate_isbn_and_non_normalized_vectors():
    docs = np.stack([unit(0), unit(1)], axis=0)
    with pytest.raises(ValueError):
        R1DenseIndex(("1", "1"), docs)
    bad = docs.copy(); bad[0] *= 2.0
    with pytest.raises(ValueError):
        R1DenseIndex(("1", "2"), bad)


def test_dense_order_signature_is_deterministic():
    docs = np.stack([unit(0), unit(1), unit(2)], axis=0)
    index = R1DenseIndex(("3", "2", "1"), docs)
    a = index.rank_query_embedding(unit(1), top_k=CANDIDATE_DEPTH)
    b = index.rank_query_embedding(unit(1), top_k=CANDIDATE_DEPTH)
    assert dense_order_signature(a) == dense_order_signature(b)
