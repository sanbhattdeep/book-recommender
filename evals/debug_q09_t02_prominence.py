"""
Diagnose the U_Q09_T02 v0.13 prominence hang WITHOUT using DeepEval.

Run from repository root after copying this file to evals/:

    uv run python evals/debug_q09_t02_prominence.py

That only prints/writes the exact v0.13 prominence prompt and deterministic cue.

To send that exact prompt + the same structured-output schema directly to Ollama:

    uv run python evals/debug_q09_t02_prominence.py --call-ollama --timeout 120

Interpretation:
- Direct Ollama call also hangs/times out -> model/schema/prompt pathology in Ollama.
- Direct Ollama call succeeds quickly -> DeepEval sync timeout/wrapper path is the likely problem.

This script does not read human_score/human_reason and does not write judge results.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
EVALS_DIR = REPO_ROOT / "evals"

# Allow imports from evals/ when invoked from repo root.
sys.path.insert(0, str(EVALS_DIR))

from semantic_relevance_facet_judge import (  # noqa: E402
    ProminenceAssessment,
    build_description_spans,
    build_prominence_prompt,
    find_deterministic_direct_cue,
)
from semantic_relevance_facet_scoring import (  # noqa: E402
    QueryFacet,
)

CASE_ID = "U_Q09_T02"
QUERY_ID = "Q09"

DATASET_FILE = (
    EVALS_DIR
    / "datasets"
    / "semantic_relevance_validation.v1.0.0.csv"
)
FACET_FILE = (
    EVALS_DIR
    / "facets"
    / "semantic_relevance"
    / "semantic_relevance_query_facets.v0.3.0.json"
)
CONFIG_FILE = (
    EVALS_DIR
    / "judge_configs"
    / "semantic_relevance_judge.v0.13.0.json"
)
OUTPUT_FILE = (
    EVALS_DIR
    / "debug_U_Q09_T02_prominence_prompt.txt"
)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--call-ollama",
        action="store_true",
        help="Send the exact prominence prompt/schema directly to Ollama.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="Direct HTTP timeout in seconds (default: 120).",
    )
    args = parser.parse_args()

    dataset = pd.read_csv(
        DATASET_FILE,
        dtype={
            "case_id": str,
            "query_id": str,
            "isbn13": str,
        },
        encoding="utf-8",
    )

    matches = dataset[
        dataset["case_id"] == CASE_ID
    ]

    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one {CASE_ID} row; found {len(matches)}."
        )

    row = matches.iloc[0]

    # Deliberately use ONLY the blind judge inputs.
    description = str(
        row["description"]
    )

    facet_payload = load_json(
        FACET_FILE
    )
    config = load_json(
        CONFIG_FILE
    )

    q09 = next(
        q
        for q in facet_payload["queries"]
        if str(q["query_id"]) == QUERY_ID
    )

    facets = [
        QueryFacet(
            **raw
        )
        for raw in q09["facets"]
    ]

    spans = build_description_spans(
        description
    )

    print(f"CASE: {CASE_ID}")
    print(f"TITLE: {row['title']}")
    print()
    print("DESCRIPTION SPANS")
    print("-----------------")
    for span_id, span_text in spans.items():
        print(f"{span_id}: {span_text}")

    print()
    print("DETERMINISTIC DIRECT CUES")
    print("-------------------------")

    cue_facets = []

    for facet in facets:
        cue = find_deterministic_direct_cue(
            spec_query_id=QUERY_ID,
            facet=facet,
            spans=spans,
            config=config,
        )

        if cue is None:
            print(
                f"{facet.facet_id} {facet.text}: NONE"
            )
            continue

        print(
            f"{facet.facet_id} {facet.text}: "
            f"{cue.cue_id} -> {cue.evidence_span_id}"
        )
        print(
            f"  matched_expression={cue.matched_expression}"
        )
        print(
            f"  evidence={cue.evidence_text}"
        )

        if facet.facet_type == "core":
            cue_facets.append(
                (
                    facet,
                    cue,
                )
            )

    if not cue_facets:
        raise RuntimeError(
            "No deterministic DIRECT core cue fired; "
            "this does not match the suspected failing path."
        )

    # The runtime evaluates facets in frozen order. The first deterministic
    # core cue is therefore the most useful exact prominence repro.
    facet, cue = cue_facets[0]

    prompt = build_prominence_prompt(
        facet=facet,
        evidence_text=cue.evidence_text,
        spans=spans,
        config=config,
    )

    OUTPUT_FILE.write_text(
        prompt + "\n",
        encoding="utf-8",
    )

    print()
    print("PROMINENCE REPRO")
    print("----------------")
    print(f"facet={facet.facet_id} {facet.text}")
    print(f"prompt_file={OUTPUT_FILE}")
    print(f"prompt_chars={len(prompt)}")
    print()
    print(prompt)

    if not args.call_ollama:
        print()
        print(
            "No model call made. Re-run with --call-ollama "
            "to test Ollama directly."
        )
        return

    base_url = str(
        config["base_url"]
    ).rstrip("/")

    endpoint = (
        base_url
        + "/api/chat"
    )

    payload = {
        "model": config["model"],
        "messages": [
            {
                "role": "user",
                "content": prompt,
            }
        ],
        "stream": False,
        "format": ProminenceAssessment.model_json_schema(),
        "options": {
            "temperature": config["temperature"],
        },
    }

    request = urllib.request.Request(
        endpoint,
        data=json.dumps(
            payload
        ).encode(
            "utf-8"
        ),
        headers={
            "Content-Type": "application/json",
        },
        method="POST",
    )

    print()
    print("DIRECT OLLAMA CALL")
    print("------------------")
    print(
        f"POST {endpoint}, timeout={args.timeout}s"
    )

    start = time.monotonic()

    try:
        with urllib.request.urlopen(
            request,
            timeout=args.timeout,
        ) as response:
            raw = response.read().decode(
                "utf-8"
            )

        elapsed = (
            time.monotonic()
            - start
        )

        print(
            f"SUCCESS after {elapsed:.2f}s"
        )
        print(raw)

    except Exception as error:
        elapsed = (
            time.monotonic()
            - start
        )

        print(
            f"FAILED after {elapsed:.2f}s"
        )
        print(
            f"{type(error).__name__}: {error}"
        )
        raise


if __name__ == "__main__":
    main()
