from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd

MODEL_ID = "BAAI/bge-m3"
MODEL_REVISION = "5617a9f61b028005a4858fdac845db406aefb181"
EMBEDDING_DIMENSION = 1024
MAX_SEQUENCE_LENGTH = 512
CANDIDATE_DEPTH = 50
BATCH_SIZE = 16
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


@dataclass(frozen=True)
class R1DenseRuntimeConfig:
    model_id: str = MODEL_ID
    model_revision: str = MODEL_REVISION
    max_sequence_length: int = MAX_SEQUENCE_LENGTH
    embedding_dimension: int = EMBEDDING_DIMENSION
    candidate_depth: int = CANDIDATE_DEPTH
    batch_size: int = BATCH_SIZE
    device: str = "auto"

    def __post_init__(self) -> None:
        if self.model_id != MODEL_ID:
            raise ValueError("Frozen V5 R1 requires BAAI/bge-m3.")
        if self.model_revision != MODEL_REVISION:
            raise ValueError("Frozen V5 R1 model revision changed.")
        if int(self.max_sequence_length) != MAX_SEQUENCE_LENGTH:
            raise ValueError("Frozen V5 R1 requires max_sequence_length=512.")
        if int(self.embedding_dimension) != EMBEDDING_DIMENSION:
            raise ValueError("Frozen V5 R1 requires 1024-dimensional embeddings.")
        if int(self.candidate_depth) != CANDIDATE_DEPTH:
            raise ValueError("Frozen V5 R1 requires candidate_depth=50.")
        if int(self.batch_size) != BATCH_SIZE:
            raise ValueError("Frozen V5 R1 requires batch_size=16.")
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


def load_dense_encoder(config: R1DenseRuntimeConfig | None = None):
    """Load the exact frozen R1 dense encoder. No prompt/instruction is applied."""
    import torch
    from sentence_transformers import SentenceTransformer

    cfg = config or R1DenseRuntimeConfig()
    device = choose_device(cfg.device)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    model = SentenceTransformer(
        cfg.model_id,
        revision=cfg.model_revision,
        device=device,
        trust_remote_code=False,
    )
    model.max_seq_length = cfg.max_sequence_length
    return model, device


def encode_texts(model: Any, texts: Sequence[Any], *, batch_size: int = BATCH_SIZE) -> np.ndarray:
    cleaned = [safe_text(x) for x in texts]
    if not cleaned or any(not x for x in cleaned):
        raise ValueError("R1 encode_texts requires non-empty text values.")
    if int(batch_size) != BATCH_SIZE:
        raise ValueError("Frozen V5 R1 requires batch_size=16.")

    embeddings = model.encode(
        cleaned,
        batch_size=BATCH_SIZE,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    arr = np.asarray(embeddings, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[0] != len(cleaned):
        raise ValueError(f"Unexpected R1 embedding shape: {arr.shape}")
    if arr.shape[1] != EMBEDDING_DIMENSION:
        raise ValueError(
            f"Unexpected R1 embedding dimension: expected={EMBEDDING_DIMENSION} actual={arr.shape[1]}"
        )
    if not np.isfinite(arr).all():
        raise ValueError("R1 embeddings contain non-finite values.")
    norms = np.linalg.norm(arr, axis=1)
    if not np.allclose(norms, 1.0, rtol=1e-4, atol=1e-5):
        raise ValueError("R1 embeddings are not L2-normalized.")
    return arr


@dataclass(frozen=True)
class R1DenseIndex:
    isbn13: tuple[str, ...]
    document_embeddings: np.ndarray

    def __post_init__(self) -> None:
        isbns = tuple(canonical_isbn(x) for x in self.isbn13)
        object.__setattr__(self, "isbn13", isbns)
        if not isbns:
            raise ValueError("R1 dense index is empty.")
        if len(set(isbns)) != len(isbns):
            raise ValueError("Duplicate ISBN13 values are not allowed in R1 dense index.")
        arr = np.asarray(self.document_embeddings, dtype=np.float32)
        if arr.ndim != 2 or arr.shape != (len(isbns), EMBEDDING_DIMENSION):
            raise ValueError(
                f"R1 document embedding matrix must be ({len(isbns)}, {EMBEDDING_DIMENSION}); got {arr.shape}."
            )
        if not np.isfinite(arr).all():
            raise ValueError("R1 document embeddings contain non-finite values.")
        norms = np.linalg.norm(arr, axis=1)
        if not np.allclose(norms, 1.0, rtol=1e-4, atol=1e-5):
            raise ValueError("R1 document embeddings are not L2-normalized.")
        object.__setattr__(self, "document_embeddings", arr)

    def rank_query_embedding(self, query_embedding: np.ndarray, *, top_k: int = CANDIDATE_DEPTH) -> pd.DataFrame:
        if int(top_k) != CANDIDATE_DEPTH:
            raise ValueError("Frozen V5 R1 requires top_k=50.")
        q = np.asarray(query_embedding, dtype=np.float32).reshape(-1)
        if q.shape != (EMBEDDING_DIMENSION,):
            raise ValueError(f"R1 query embedding must have shape ({EMBEDDING_DIMENSION},); got {q.shape}.")
        if not np.isfinite(q).all():
            raise ValueError("R1 query embedding contains non-finite values.")
        qnorm = float(np.linalg.norm(q))
        if not np.isclose(qnorm, 1.0, rtol=1e-4, atol=1e-5):
            raise ValueError("R1 query embedding is not L2-normalized.")

        scores = self.document_embeddings @ q
        rows = [
            {"isbn13": isbn, "dense_score": float(score)}
            for isbn, score in zip(self.isbn13, scores)
        ]
        rows.sort(key=lambda row: (-row["dense_score"], row["isbn13"]))
        rows = rows[: min(CANDIDATE_DEPTH, len(rows))]
        for rank, row in enumerate(rows, start=1):
            row["dense_rank"] = rank
        return pd.DataFrame(rows, columns=["isbn13", "dense_score", "dense_rank"])


def build_dense_index(*, isbn13: Sequence[Any], document_texts: Sequence[Any], model: Any) -> R1DenseIndex:
    if len(isbn13) != len(document_texts):
        raise ValueError("isbn13 and document_texts must have equal length.")
    embeddings = encode_texts(model, document_texts)
    return R1DenseIndex(tuple(canonical_isbn(x) for x in isbn13), embeddings)


def rank_query(*, query: Any, index: R1DenseIndex, model: Any) -> pd.DataFrame:
    query_text = safe_text(query)
    if not query_text:
        raise ValueError("R1 query is empty.")
    query_embedding = encode_texts(model, [query_text])[0]
    return index.rank_query_embedding(query_embedding, top_k=CANDIDATE_DEPTH)


def dense_order_signature(frame: pd.DataFrame) -> tuple[str, ...]:
    required = {"isbn13", "dense_rank"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"R1 signature frame missing columns: {sorted(missing)}")
    ranked = frame.copy()
    ranked["dense_rank"] = pd.to_numeric(ranked["dense_rank"], errors="raise").astype(int)
    ranked["isbn13"] = ranked["isbn13"].map(canonical_isbn)
    ranked = ranked.sort_values(["dense_rank", "isbn13"], kind="stable")
    return tuple(ranked["isbn13"].tolist())
