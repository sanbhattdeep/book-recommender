"""
Run one pending development case through the v0.23 development pipeline using direct
Ollama structured-output transport instead of DeepEval's transport wrapper.

Use only as an operational recovery path for a reproducible DeepEval timeout or
hang. Prompt/config/facet/scoring behavior is unchanged and provenance is
recorded in the run directory.

Example:
    uv run python evals/run_judge_development_direct_transport.py `
      --resume evals/runs/semantic_relevance_v0_23_development/<RUN_ID> `
      --case-id U_Q09_T02
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

EVALS_DIR = Path(__file__).resolve().parent
REPO_ROOT = EVALS_DIR.parent
sys.path.insert(0, str(EVALS_DIR))

import run_judge_development as runner  # noqa: E402


class DirectOllamaTransport:
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
            raise TypeError("v0.23 development expects structured-output schemas.")

        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "format": schema.model_json_schema(),
            "options": {"temperature": self.temperature},
        }

        request = urllib.request.Request(
            self.base_url + "/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        start = time.monotonic()
        with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
            envelope = json.loads(response.read().decode("utf-8"))
        elapsed = time.monotonic() - start

        content = envelope.get("message", {}).get("content")
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError("Ollama returned no message.content.")

        result = schema.model_validate_json(content)
        print(f"[direct-ollama] schema={schema.__name__}, elapsed={elapsed:.2f}s")
        return result


def resolve_run_dir(raw: str) -> Path:
    path = Path(raw)
    if not path.is_absolute():
        path = REPO_ROOT / path
    if not path.exists():
        raise FileNotFoundError(f"Run directory not found: {path}")
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", required=True)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--timeout", type=float, default=180.0)
    args = parser.parse_args()

    run_dir = resolve_run_dir(args.resume)
    case_id = str(args.case_id)
    results_file = run_dir / "judge_results.csv"

    if results_file.exists():
        current = pd.read_csv(results_file, dtype={"case_id": str}, encoding="utf-8")
        if case_id in set(current["case_id"].astype(str)):
            raise RuntimeError(f"{case_id} is already complete; refusing to overwrite it.")

    timeout_seconds = float(args.timeout)

    class BoundDirectOllamaTransport(DirectOllamaTransport):
        def __init__(self, model: str, base_url: str, temperature: float = 0.0, **kwargs: Any):
            super().__init__(
                model=model,
                base_url=base_url,
                temperature=temperature,
                timeout_seconds=timeout_seconds,
                **kwargs,
            )

    runner.OllamaModel = BoundDirectOllamaTransport

    original_argv = sys.argv[:]
    try:
        sys.argv = [
            "run_judge_development.py",
            "--resume", str(run_dir),
            "--case-id", case_id,
        ]
        runner.main()
    finally:
        sys.argv = original_argv

    result = pd.read_csv(results_file, dtype={"case_id": str}, encoding="utf-8")
    if case_id not in set(result["case_id"].astype(str)):
        raise RuntimeError(f"{case_id} was not persisted successfully.")

    note = {
        "case_id": case_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "transport": "direct_ollama_api_chat",
        "deep_eval_transport_bypassed": True,
        "judge_behavior_changed": False,
        "judge_version": runner.JUDGE_CONFIG_VERSION,
        "structured_output": True,
        "http_timeout_seconds": timeout_seconds,
        "prompts_facets_scoring_unchanged": True,
    }

    note_path = run_dir / f"transport_override_{case_id}.json"
    note_path.write_text(json.dumps(note, indent=2) + "\n", encoding="utf-8")

    metadata_path = run_dir / "run_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    overrides = metadata.get("transport_overrides", [])
    if not isinstance(overrides, list):
        overrides = [overrides]
    overrides = [x for x in overrides if not (isinstance(x, dict) and x.get("case_id") == case_id)]
    overrides.append(note)
    metadata["transport_overrides"] = overrides
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"{case_id} completed; direct-transport provenance recorded.")


if __name__ == "__main__":
    main()
