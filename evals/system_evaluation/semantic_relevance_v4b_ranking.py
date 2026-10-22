from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _safe_text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    return str(value).strip()


def canonical_isbn(value: Any) -> str:
    text = _safe_text(value).strip('"').strip("'")
    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]
    if not text:
        raise ValueError("ISBN is empty.")
    return text


def rank_v4b(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply the frozen V4B final-ranking contract.

    V4B changes only the final ordering over the already-scored V4A hybrid
    candidate union.

    Ordering within each query:
      1. reranker_score descending;
      2. hybrid_retrieval_rank ascending only when reranker_score is exactly equal;
      3. isbn13 ascending.

    No score rounding, epsilon tie bucket, normalization, calibration, or
    retrieval-rank weighting is allowed.
    """
    required = {
        "query_id",
        "isbn13",
        "hybrid_retrieval_rank",
        "reranker_score",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing V4B ranking columns: {sorted(missing)}")

    if frame.empty:
        raise ValueError("V4B ranking input is empty.")

    working = frame.copy()
    working["query_id"] = working["query_id"].astype(str)
    working["isbn13"] = working["isbn13"].map(canonical_isbn)
    working["hybrid_retrieval_rank"] = pd.to_numeric(
        working["hybrid_retrieval_rank"], errors="raise"
    ).astype(int)
    working["reranker_score"] = pd.to_numeric(
        working["reranker_score"], errors="raise"
    ).astype(float)

    if not np.isfinite(working["reranker_score"].to_numpy(dtype=float)).all():
        raise ValueError("V4B reranker_score contains non-finite values.")
    if (working["hybrid_retrieval_rank"] < 1).any():
        raise ValueError("V4B hybrid_retrieval_rank must be >= 1.")
    if working.duplicated(["query_id", "isbn13"]).any():
        raise ValueError("Duplicate query-book pair in V4B ranking input.")

    ranked_groups: list[pd.DataFrame] = []
    for _, group in working.groupby("query_id", sort=False):
        g = group.copy()
        g = g.sort_values(
            ["reranker_score", "hybrid_retrieval_rank", "isbn13"],
            ascending=[False, True, True],
            kind="stable",
        )
        g["v4b_rank"] = np.arange(1, len(g) + 1, dtype=int)
        ranked_groups.append(g)

    return pd.concat(ranked_groups, ignore_index=True)


def top_k_v4b(frame: pd.DataFrame, *, top_k: int = 10) -> pd.DataFrame:
    if int(top_k) != 10:
        raise ValueError("Frozen V4B contract requires top_k=10.")

    ranked = rank_v4b(frame)
    out = ranked[ranked["v4b_rank"] <= 10].copy()
    out["rank"] = out["v4b_rank"].astype(int)
    return out.sort_values(["query_id", "rank"], kind="stable").reset_index(drop=True)
