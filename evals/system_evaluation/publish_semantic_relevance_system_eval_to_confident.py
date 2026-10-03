from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from deepeval import evaluate
from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase

REPO_ROOT = Path(__file__).resolve().parents[2]
EVALS = REPO_ROOT / "evals"

RUN_DIR = (
    EVALS / "runs" / "semantic_relevance_system_eval_v1"
    / "20261002T152856Z_recommendations"
)
JUDGE_SCORES_FILE = RUN_DIR / "judge_scores.csv"
JUDGE_INPUT_FILE = RUN_DIR / "judge_input.csv"
SCORING_LOCK_FILE = RUN_DIR / "judge_scoring_lock.json"
COLLECTION_LOCK_FILE = RUN_DIR / "recommendation_collection_lock.json"
PUBLICATION_METADATA_FILE = RUN_DIR / "confident_ai_publication.json"

IDENTIFIER = "book-recommender-semantic-system-eval-v1-20261002T152856Z"

def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8-sig"))

class SemanticRelevanceMetric(BaseMetric):
    """Expose the already-computed r5 score to DeepEval/Confident AI.

    This metric deliberately performs NO LLM inference. The semantic-relevance
    score has already been computed and frozen by v0.29.0-r5.

    DeepEval metrics use a normalized 0..1 score, so the frozen 0..4 score is
    divided by four for the platform. The raw 0..4 score remains in test-case
    metadata and in the metric reason.
    """

    def __init__(self, threshold: float = 0.75):
        self.threshold = threshold
        self.include_reason = True
        self.strict_mode = False
        self.async_mode = False
        self.evaluation_model = "precomputed-semantic_relevance_v0.29.0-r5"

    def measure(self, test_case: LLMTestCase) -> float:
        metadata = test_case.metadata or {}
        raw = int(metadata["raw_semantic_relevance_score"])
        if raw < 0 or raw > 4:
            raise ValueError(f"Invalid frozen semantic relevance score: {raw}")

        self.score = raw / 4.0
        self.success = self.score >= self.threshold
        self.reason = (
            f"Frozen semantic relevance score: {raw}/4 "
            f"({metadata.get('match_level', 'unknown')}). "
            f"{metadata.get('judge_reason', '')}"
        ).strip()
        return self.score

    async def a_measure(self, test_case: LLMTestCase) -> float:
        return self.measure(test_case)

    @property
    def __name__(self) -> str:
        return "Semantic Relevance (r5 precomputed)"

def verify_inputs() -> tuple[pd.DataFrame, pd.DataFrame, dict, dict]:
    if not os.getenv("CONFIDENT_API_KEY"):
        raise EnvironmentError(
            "CONFIDENT_API_KEY is not available. Put the project API key in "
            "the repo .env as CONFIDENT_API_KEY=... or export it in the shell. "
            "Do not place the key in source control."
        )

    scoring_lock = load_json(SCORING_LOCK_FILE)
    collection_lock = load_json(COLLECTION_LOCK_FILE)

    if scoring_lock.get("status") != "JUDGE_SCORES_FROZEN_BEFORE_METRIC_ANALYSIS":
        raise ValueError(
            "Local judge scoring is not complete/frozen. Publish only after "
            "all 120 pairs have been scored."
        )
    if int(scoring_lock.get("completed_pairs", -1)) != 120:
        raise ValueError("Expected 120 completed frozen judge scores.")
    if scoring_lock.get("human_relevance_labels_used") is not False:
        raise ValueError("System evaluation unexpectedly used human labels.")

    expected_scores_sha = scoring_lock["judge_scores_sha256"]
    actual_scores_sha = sha256(JUDGE_SCORES_FILE)
    if actual_scores_sha != expected_scores_sha:
        raise ValueError(
            "judge_scores.csv changed after freeze: "
            f"expected={expected_scores_sha} actual={actual_scores_sha}"
        )

    expected_input_sha = collection_lock["judge_input_sha256"]
    actual_input_sha = sha256(JUDGE_INPUT_FILE)
    if actual_input_sha != expected_input_sha:
        raise ValueError(
            "judge_input.csv changed after collection freeze: "
            f"expected={expected_input_sha} actual={actual_input_sha}"
        )

    scores = pd.read_csv(
        JUDGE_SCORES_FILE,
        dtype={"case_id": str, "query_id": str, "isbn13": str},
        keep_default_na=False,
        encoding="utf-8",
    )
    inputs = pd.read_csv(
        JUDGE_INPUT_FILE,
        dtype={"query_id": str, "isbn13": str},
        keep_default_na=False,
        encoding="utf-8",
    )

    if len(scores) != 120 or scores["case_id"].duplicated().any():
        raise ValueError("Expected exactly 120 unique frozen judge-score rows.")
    if len(inputs) != 120:
        raise ValueError("Expected exactly 120 frozen judge-input rows.")

    inputs = inputs.copy()
    inputs["rank"] = inputs["rank"].astype(int)
    inputs["case_id"] = inputs.apply(
        lambda r: f"SYS_{r['query_id']}_R{int(r['rank']):02d}",
        axis=1,
    )

    return scores, inputs, scoring_lock, collection_lock

def build_test_cases(scores: pd.DataFrame, inputs: pd.DataFrame) -> list[LLMTestCase]:
    merged = scores.merge(
        inputs[["case_id", "description"]],
        on="case_id",
        how="inner",
        validate="one_to_one",
    )
    if len(merged) != 120:
        raise ValueError("Score/input merge did not produce 120 cases.")

    cases: list[LLMTestCase] = []

    for row in merged.to_dict(orient="records"):
        raw_score = int(row["judge_score"])
        rank = int(row["rank"])

        metadata = {
            "evaluation_id": "semantic-relevance-system-eval",
            "evaluation_version": "1.0.0",
            "judge_candidate": "semantic_relevance_v0.29.0-r5",
            "query_id": str(row["query_id"]),
            "rank": rank,
            "isbn13": str(row["isbn13"]),
            "title": str(row["title"]),
            "authors": str(row.get("authors", "")),
            "raw_semantic_relevance_score": raw_score,
            "normalized_semantic_relevance_score": raw_score / 4.0,
            "match_level": str(row["match_level"]),
            "judge_reason": str(row["judge_reason"]),
            "scoring_explanation": str(row["scoring_explanation"]),
            "semantic_generation_mode": str(row["semantic_generation_mode"]),
            "human_relevance_labels_used": False,
        }

        actual_output = (
            f"Recommended book at rank {rank}: {row['title']}\n\n"
            f"Description:\n{row['description']}"
        )

        cases.append(
            LLMTestCase(
                name=str(row["case_id"]),
                input=str(row["query"]),
                actual_output=actual_output,
                metadata=metadata,
                tags=[
                    "book-recommender",
                    "semantic-relevance",
                    "system-eval-v1",
                    "v0.29.0-r5",
                    f"query:{row['query_id']}",
                    f"rank:{rank}",
                ],
            )
        )

    return cases

def main() -> None:
    print("Confident AI publication - semantic relevance system evaluation")
    print("--------------------------------------------------------------")
    print("Local scores are reused; no judge LLM calls will be made.")

    scores, inputs, scoring_lock, collection_lock = verify_inputs()
    cases = build_test_cases(scores, inputs)

    metric = SemanticRelevanceMetric(threshold=0.75)

    # CONFIDENT_API_KEY causes evaluate() results to be uploaded to Confident
    # AI. CONFIDENT_OPEN_BROWSER=0 can be set by the wrapper to avoid automatic
    # browser launch on Windows.
    result = evaluate(
        test_cases=cases,
        metrics=[metric],
        identifier=IDENTIFIER,
        hyperparameters={
            "system": "book-recommender",
            "evaluation": "semantic-relevance-system-eval-v1",
            "judge": "semantic_relevance_v0.29.0-r5",
            "top_k": 10,
            "query_count": 12,
            "pair_count": 120,
        },
    )

    publication = {
        "status": "PUBLISHED_VIA_DEEPEVAL_EVALUATE",
        "published_at_utc": datetime.now(timezone.utc).isoformat(),
        "identifier": IDENTIFIER,
        "test_case_count": 120,
        "metric_name": "Semantic Relevance (r5 precomputed)",
        "metric_threshold_normalized": 0.75,
        "raw_threshold_equivalent": "3/4",
        "judge_scores_sha256": sha256(JUDGE_SCORES_FILE),
        "judge_input_sha256": sha256(JUDGE_INPUT_FILE),
        "local_scores_recomputed": False,
        "judge_llm_calls_performed_during_publication": 0,
        "human_relevance_labels_used": False,
        "note": (
            "DeepEval metric score is normalized to 0..1 for platform "
            "compatibility; raw 0..4 score is preserved in test-case metadata."
        ),
    }
    PUBLICATION_METADATA_FILE.write_text(
        json.dumps(publication, indent=2) + "\n",
        encoding="utf-8",
    )

    print()
    print("CONFIDENT AI PUBLICATION: COMPLETE")
    print(f"Identifier: {IDENTIFIER}")
    print("Test cases published: 120")
    print("Judge LLM calls during publication: 0")
    print("Human relevance labels used: NO")
    print(f"Publication metadata: {PUBLICATION_METADATA_FILE}")
    print()
    print("Run `uv run deepeval view` to open the latest test run.")

if __name__ == "__main__":
    main()
