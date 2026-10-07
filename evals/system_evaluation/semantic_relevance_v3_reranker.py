from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RerankerRuntimeConfig:
    model_id: str
    model_revision: str
    max_length_tokens: int = 512
    truncation: str = "only_second"
    batch_size: int = 8
    device: str = "auto"


def safe_text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    return str(value).strip()


def candidate_text(title: Any, description: Any) -> str:
    title_text = safe_text(title)
    description_text = safe_text(description)
    if description_text:
        return f"Title: {title_text}\nDescription: {description_text}"
    return f"Title: {title_text}"


def canonical_isbn(value: Any) -> str:
    text = safe_text(value).strip('"').strip("'")
    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]
    if not text:
        raise ValueError("ISBN is empty.")
    return text


def assign_reranker_rank(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"query_id", "vector_rank", "isbn13", "reranker_score"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing reranker columns: {sorted(missing)}")

    chunks: list[pd.DataFrame] = []
    for _, group in frame.groupby("query_id", sort=False):
        g = group.copy()
        g["_isbn_sort"] = g["isbn13"].map(canonical_isbn)
        g = g.sort_values(
            ["reranker_score", "vector_rank", "_isbn_sort"],
            ascending=[False, True, True],
            kind="stable",
        )
        g["reranker_rank"] = np.arange(1, len(g) + 1, dtype=int)
        g = g.drop(columns=["_isbn_sort"])
        chunks.append(g)
    return pd.concat(chunks, ignore_index=True)


def rank_v3a(frame: pd.DataFrame) -> pd.DataFrame:
    ranked = assign_reranker_rank(frame)
    ranked["v3a_rank"] = ranked["reranker_rank"].astype(int)
    return ranked


def rank_v3b(frame: pd.DataFrame, *, rrf_k: float = 60.0) -> pd.DataFrame:
    if float(rrf_k) != 60.0:
        raise ValueError("Frozen v3b contract requires RRF k=60.")

    ranked = assign_reranker_rank(frame)
    ranked["rrf_score"] = (
        1.0 / (rrf_k + ranked["vector_rank"].astype(float))
        + 1.0 / (rrf_k + ranked["reranker_rank"].astype(float))
    )

    chunks: list[pd.DataFrame] = []
    for _, group in ranked.groupby("query_id", sort=False):
        g = group.copy()
        g["_isbn_sort"] = g["isbn13"].map(canonical_isbn)
        g = g.sort_values(
            ["rrf_score", "reranker_rank", "vector_rank", "_isbn_sort"],
            ascending=[False, True, True, True],
            kind="stable",
        )
        g["v3b_rank"] = np.arange(1, len(g) + 1, dtype=int)
        g = g.drop(columns=["_isbn_sort"])
        chunks.append(g)

    return pd.concat(chunks, ignore_index=True)


def choose_device(requested: str) -> str:
    import torch

    requested = str(requested).strip().lower()
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False.")
    if requested not in {"cuda", "cpu"}:
        raise ValueError("device must be one of: auto, cuda, cpu")
    return requested


def load_cross_encoder(config: RerankerRuntimeConfig):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    device = choose_device(config.device)

    torch.manual_seed(20261002)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(20261002)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    tokenizer = AutoTokenizer.from_pretrained(
        config.model_id,
        revision=config.model_revision,
        trust_remote_code=False,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        config.model_id,
        revision=config.model_revision,
        trust_remote_code=False,
        use_safetensors=True,
    )
    model.eval()
    model.to(device)
    return tokenizer, model, device


def score_query_candidate_pairs(
    *,
    queries: list[str],
    candidates: list[str],
    tokenizer: Any,
    model: Any,
    device: str,
    max_length_tokens: int,
    batch_size: int,
) -> np.ndarray:
    import torch

    if len(queries) != len(candidates):
        raise ValueError("queries and candidates must have equal length.")
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1.")

    scores: list[float] = []
    with torch.inference_mode():
        for start in range(0, len(queries), batch_size):
            q_batch = queries[start:start + batch_size]
            c_batch = candidates[start:start + batch_size]

            inputs = tokenizer(
                q_batch,
                c_batch,
                padding=True,
                truncation="only_second",
                max_length=max_length_tokens,
                return_tensors="pt",
            )
            inputs = {key: value.to(device) for key, value in inputs.items()}
            logits = model(**inputs, return_dict=True).logits.view(-1).float()
            scores.extend(float(x) for x in logits.detach().cpu().tolist())

    return np.asarray(scores, dtype=float)


def reranker_order_signature(frame: pd.DataFrame) -> dict[str, tuple[str, ...]]:
    ranked = assign_reranker_rank(frame)
    signatures: dict[str, tuple[str, ...]] = {}
    for qid, group in ranked.groupby("query_id", sort=False):
        g = group.sort_values("reranker_rank", kind="stable")
        signatures[str(qid)] = tuple(g["isbn13"].map(canonical_isbn).tolist())
    return signatures
