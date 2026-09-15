"""
Recover the single stuck validation case U_Q09_T02 using direct Ollama transport.

Why this exists
---------------
The frozen v0.13 judge pipeline repeatedly hangs inside DeepEval's synchronous
OllamaModel.generate(...) wrapper for this case, while the exact same
prominence prompt + Pydantic schema succeeds when sent directly to Ollama.

This wrapper does NOT change:
- Judge v0.13 prompts
- frozen facet definitions
- deterministic cues
- Pydantic schemas
- model name
- temperature
- deterministic Python scoring
- case data

It only replaces the DeepEval transport wrapper with a bounded direct call to
Ollama's /api/chat endpoint for this one case.

Usage from repository root
--------------------------
Copy this file to evals/, then run:

    uv run python evals/run_q09_t02_direct_transport.py ^
      --resume evals/runs/semantic_relevance_validation/20260914T103318Z

PowerShell multiline equivalent:

    uv run python evals/run_q09_t02_direct_transport.py `
      --resume evals/runs/semantic_relevance_validation/20260914T103318Z

Optional timeout:

    --timeout 180

The script invokes the existing run_judge_validation.py with:
    --resume <same run>
    --case-id U_Q09_T02

After success it appends transport provenance to run_metadata.json and writes a
separate transport-override JSON file in the run directory.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import BaseModel


CASE_ID = "U_Q09_T02"

EVALS_DIR = Path(__file__).resolve().parent
REPO_ROOT = EVALS_DIR.parent

# Import the already-frozen validation runner.
sys.path.insert(0, str(EVALS_DIR))
import run_judge_validation as runner  # noqa: E402


class DirectOllamaTransport:
    """
    Drop-in constructor-compatible replacement for DeepEval's OllamaModel.

    The v0.13 pipeline only needs .generate(prompt=..., schema=...).
    """

    def __init__(
        self,
        model: str,
        base_url: str,
        temperature: float = 0.0,
        timeout_seconds: float = 180.0,
        **_: Any,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.temperature = float(temperature)
        self.timeout_seconds = float(timeout_seconds)

    def generate(
        self,
        prompt: str,
        schema: type[BaseModel] | None = None,
        **_: Any,
    ):
        if schema is None:
            raise TypeError(
                "Frozen v0.13 validation expects structured-output schemas."
            )

        endpoint = self.base_url + "/api/chat"

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            "stream": False,
            "format": schema.model_json_schema(),
            "options": {
                "temperature": self.temperature,
            },
        }

        request = urllib.request.Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
            },
            method="POST",
        )

        started = time.monotonic()

        with urllib.request.urlopen(
            request,
            timeout=self.timeout_seconds,
        ) as response:
            raw = response.read().decode("utf-8")

        elapsed = time.monotonic() - started
        envelope = json.loads(raw)

        content = (
            envelope
            .get("message", {})
            .get("content")
        )

        if not isinstance(content, str) or not content.strip():
            raise RuntimeError(
                "Ollama returned no message.content for structured generation."
            )

        result = schema.model_validate_json(content)

        print(
            f"[direct-ollama] schema={schema.__name__}, "
            f"elapsed={elapsed:.2f}s"
        )

        # semantic_relevance_facet_judge.unpack_generated_model accepts a
        # schema instance directly.
        return result


def resolve_run_dir(raw: str) -> Path:
    path = Path(raw)

    if not path.is_absolute():
        path = REPO_ROOT / path

    if not path.exists():
        raise FileNotFoundError(
            f"Run directory not found: {path}"
        )

    return path


def ensure_case_is_pending(
    run_dir: Path,
) -> None:
    results_file = (
        run_dir
        / "judge_results.csv"
    )

    if not results_file.exists():
        return

    results = pd.read_csv(
        results_file,
        dtype={
            "case_id": str,
        },
        encoding="utf-8",
    )

    if CASE_ID in set(
        results["case_id"].astype(str)
    ):
        raise RuntimeError(
            f"{CASE_ID} is already present in judge_results.csv; "
            "refusing to overwrite/re-run it."
        )


def record_transport_provenance(
    run_dir: Path,
    timeout_seconds: float,
) -> None:
    note = {
        "case_id": CASE_ID,
        "timestamp_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "reason": (
            "DeepEval OllamaModel.generate repeatedly hung/timed out for this "
            "case. The exact prominence prompt and structured schema were "
            "verified to succeed via direct Ollama /api/chat."
        ),
        "transport": "direct_ollama_api_chat",
        "deep_eval_transport_bypassed": True,
        "judge_behavior_changed": False,
        "judge_version": "0.13.0",
        "model": "jeffnyman/ts-evaluator",
        "temperature": 0.0,
        "structured_output": True,
        "http_timeout_seconds": timeout_seconds,
        "scope": "U_Q09_T02 only",
        "prompts_facets_scoring_unchanged": True,
    }

    note_file = (
        run_dir
        / "transport_override_U_Q09_T02.json"
    )

    note_file.write_text(
        json.dumps(
            note,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    metadata_file = (
        run_dir
        / "run_metadata.json"
    )

    if metadata_file.exists():
        metadata = json.loads(
            metadata_file.read_text(
                encoding="utf-8"
            )
        )

        overrides = metadata.get(
            "transport_overrides",
            [],
        )

        if not isinstance(
            overrides,
            list,
        ):
            overrides = [
                overrides
            ]

        overrides = [
            item
            for item in overrides
            if not (
                isinstance(item, dict)
                and item.get("case_id") == CASE_ID
            )
        ]

        overrides.append(
            note
        )

        metadata[
            "transport_overrides"
        ] = overrides

        metadata_file.write_text(
            json.dumps(
                metadata,
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run only U_Q09_T02 through the frozen v0.13 validation pipeline "
            "using direct Ollama transport."
        )
    )

    parser.add_argument(
        "--resume",
        required=True,
        help=(
            "Existing semantic_relevance_validation run directory."
        ),
    )

    parser.add_argument(
        "--timeout",
        type=float,
        default=180.0,
        help=(
            "Per direct Ollama HTTP call timeout in seconds. Default: 180."
        ),
    )

    args = parser.parse_args()

    run_dir = resolve_run_dir(
        args.resume
    )

    ensure_case_is_pending(
        run_dir
    )

    timeout_seconds = float(
        args.timeout
    )

    # Replace only the runner module's constructor reference. The existing
    # v0.13 semantic pipeline remains untouched.
    class BoundDirectOllamaTransport(
        DirectOllamaTransport
    ):
        def __init__(
            self,
            model: str,
            base_url: str,
            temperature: float = 0.0,
            **kwargs: Any,
        ) -> None:
            super().__init__(
                model=model,
                base_url=base_url,
                temperature=temperature,
                timeout_seconds=timeout_seconds,
                **kwargs,
            )

    runner.OllamaModel = (
        BoundDirectOllamaTransport
    )

    original_argv = sys.argv[:]

    try:
        sys.argv = [
            "run_judge_validation.py",
            "--resume",
            str(run_dir),
            "--case-id",
            CASE_ID,
        ]

        runner.main()

    finally:
        sys.argv = original_argv

    # Confirm persistence before recording the transport note.
    results_file = (
        run_dir
        / "judge_results.csv"
    )

    results = pd.read_csv(
        results_file,
        dtype={
            "case_id": str,
        },
        encoding="utf-8",
    )

    if CASE_ID not in set(
        results["case_id"].astype(str)
    ):
        raise RuntimeError(
            f"{CASE_ID} was not persisted successfully."
        )

    record_transport_provenance(
        run_dir=run_dir,
        timeout_seconds=timeout_seconds,
    )

    print()
    print(
        f"{CASE_ID} completed and persisted."
    )
    print(
        "Transport provenance recorded in run_metadata.json and "
        "transport_override_U_Q09_T02.json."
    )


if __name__ == "__main__":
    main()
