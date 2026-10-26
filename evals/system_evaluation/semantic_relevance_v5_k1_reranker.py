from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

MODEL_ID = "BAAI/bge-reranker-v2-m3"
MODEL_REVISION = "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"
MAX_LENGTH_TOKENS = 512
BATCH_SIZE = 4
SEED = 20261002


def safe_text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    return str(value).strip()


def canonical_isbn(value: Any) -> str:
    text = safe_text(value).strip('"').strip("'")
    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]
    if not text:
        raise ValueError("ISBN is empty.")
    return text


def candidate_text(title: Any, description: Any) -> str:
    title_text = safe_text(title)
    description_text = safe_text(description)
    if description_text:
        return f"Title: {title_text}\nDescription: {description_text}"
    return f"Title: {title_text}"


@dataclass(frozen=True)
class K1RuntimeConfig:
    model_id: str = MODEL_ID
    model_revision: str = MODEL_REVISION
    max_length_tokens: int = MAX_LENGTH_TOKENS
    batch_size: int = BATCH_SIZE
    device: str = "auto"

    def __post_init__(self) -> None:
        if self.model_id != MODEL_ID:
            raise ValueError("Frozen V5 K1 requires BAAI/bge-reranker-v2-m3.")
        if self.model_revision != MODEL_REVISION:
            raise ValueError("Frozen V5 K1 model revision changed.")
        if int(self.max_length_tokens) != MAX_LENGTH_TOKENS:
            raise ValueError("Frozen V5 K1 requires max_length_tokens=512.")
        if int(self.batch_size) != BATCH_SIZE:
            raise ValueError("Frozen V5 K1 requires batch_size=4.")
        if self.device not in {"auto", "cuda", "cpu"}:
            raise ValueError("device must be one of: auto, cuda, cpu")


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


def load_reranker(config: K1RuntimeConfig | None = None):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    cfg = config or K1RuntimeConfig()
    device = choose_device(cfg.device)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model_id,
        revision=cfg.model_revision,
        trust_remote_code=False,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model_id,
        revision=cfg.model_revision,
        trust_remote_code=False,
        use_safetensors=True,
        torch_dtype=torch.float32,
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
    max_length_tokens: int = MAX_LENGTH_TOKENS,
    batch_size: int = BATCH_SIZE,
) -> np.ndarray:
    import torch

    if len(queries) != len(candidates):
        raise ValueError("queries and candidates must have equal length.")
    if not queries:
        raise ValueError("K1 scoring input is empty.")
    if int(max_length_tokens) != MAX_LENGTH_TOKENS:
        raise ValueError("Frozen V5 K1 requires max_length_tokens=512.")
    if int(batch_size) != BATCH_SIZE:
        raise ValueError("Frozen V5 K1 requires batch_size=4.")

    scores: list[float] = []
    with torch.inference_mode():
        for start in range(0, len(queries), BATCH_SIZE):
            q_batch = queries[start : start + BATCH_SIZE]
            c_batch = candidates[start : start + BATCH_SIZE]
            inputs = tokenizer(
                q_batch,
                c_batch,
                padding=True,
                truncation="only_second",
                max_length=MAX_LENGTH_TOKENS,
                return_tensors="pt",
            )
            inputs = {key: value.to(device) for key, value in inputs.items()}
            logits = model(**inputs, return_dict=True).logits.view(-1).float()
            scores.extend(float(x) for x in logits.detach().cpu().tolist())

    result = np.asarray(scores, dtype=float)
    if result.shape != (len(queries),):
        raise ValueError(f"Unexpected K1 score shape: {result.shape}")
    if not np.isfinite(result).all():
        raise ValueError("K1 scores contain non-finite values.")
    return result


def rank_k1(frame: pd.DataFrame) -> pd.DataFrame:
    """Frozen K1 ordering: raw logit desc; exact tie -> hybrid retrieval rank -> ISBN."""
    required = {"query_id", "isbn13", "hybrid_retrieval_rank", "reranker_score"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing K1 ranking columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("K1 ranking input is empty.")

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
        raise ValueError("K1 reranker_score contains non-finite values.")
    if (working["hybrid_retrieval_rank"] < 1).any():
        raise ValueError("K1 hybrid_retrieval_rank must be >=1.")
    if working.duplicated(["query_id", "isbn13"]).any():
        raise ValueError("Duplicate query-book pair in K1 ranking input.")

    chunks: list[pd.DataFrame] = []
    for _, group in working.groupby("query_id", sort=False):
        g = group.copy()
        g = g.sort_values(
            ["reranker_score", "hybrid_retrieval_rank", "isbn13"],
            ascending=[False, True, True],
            kind="stable",
        )
        g["k1_rank"] = np.arange(1, len(g) + 1, dtype=int)
        chunks.append(g)
    return pd.concat(chunks, ignore_index=True)


def top_k_k1(frame: pd.DataFrame, *, top_k: int = 10) -> pd.DataFrame:
    if int(top_k) != 10:
        raise ValueError("Frozen V5 K1 requires top_k=10.")
    ranked = rank_k1(frame)
    out = ranked[ranked["k1_rank"] <= 10].copy()
    out["rank"] = out["k1_rank"].astype(int)
    return out.sort_values(["query_id", "rank"], kind="stable").reset_index(drop=True)


def reranker_order_signature(frame: pd.DataFrame) -> dict[str, tuple[str, ...]]:
    ranked = rank_k1(frame)
    signatures: dict[str, tuple[str, ...]] = {}
    for qid, group in ranked.groupby("query_id", sort=False):
        g = group.sort_values("k1_rank", kind="stable")
        signatures[str(qid)] = tuple(g["isbn13"].map(canonical_isbn).tolist())
    return signatures
