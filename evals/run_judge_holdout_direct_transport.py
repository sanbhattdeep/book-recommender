"""
Recover exactly one actually failed pending final-holdout case using direct
Ollama structured-output transport.

This is an operational transport fallback only. It cannot be used to select an
arbitrary holdout case: --case-id must match last_failure.json from the same
run, and the case must still be pending.
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

import run_judge_holdout as runner  # noqa: E402


class DirectOllamaTransport:
    def __init__(self, model: str, base_url: str, temperature: float = 0.0,
                 timeout_seconds: float = 180.0, **_: Any) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.temperature = float(temperature)
        self.timeout_seconds = float(timeout_seconds)

    def generate(self, prompt: str, schema: type[BaseModel] | None = None, **_: Any):
        if schema is None:
            raise TypeError("Frozen v0.15 holdout expects structured-output schemas.")
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
        started = time.monotonic()
        with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
            envelope = json.loads(response.read().decode("utf-8"))
        elapsed = time.monotonic() - started
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
    failure_file = run_dir / "last_failure.json"

    if not failure_file.exists():
        raise RuntimeError(
            "No last_failure.json exists. Direct holdout transport may only recover "
            "an actually failed pending case."
        )
    failure = json.loads(failure_file.read_text(encoding="utf-8"))
    failed_case_id = str(failure.get("case_id"))
    if case_id != failed_case_id:
        raise RuntimeError(
            f"Refusing arbitrary holdout selection: requested={case_id!r}, "
            f"recorded_failed_case={failed_case_id!r}."
        )

    if results_file.exists():
        current = pd.read_csv(results_file, dtype={"case_id": str}, encoding="utf-8")
        if case_id in set(current["case_id"].astype(str)):
            raise RuntimeError(f"{case_id} is already complete; refusing to overwrite it.")

    timeout_seconds = float(args.timeout)

    class BoundDirectOllamaTransport(DirectOllamaTransport):
        def __init__(self, model: str, base_url: str, temperature: float = 0.0, **kwargs: Any) -> None:
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
            "run_judge_holdout.py",
            "--resume", str(run_dir),
            "--recover-failed-case", case_id,
        ]
        runner.main()
    finally:
        sys.argv = original_argv

    results = pd.read_csv(results_file, dtype={"case_id": str}, encoding="utf-8")
    if case_id not in set(results["case_id"].astype(str)):
        raise RuntimeError(f"{case_id} was not persisted successfully.")

    metadata_file = run_dir / "run_metadata.json"
    metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
    note = {
        "case_id": case_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "reason": "Recovery of the actually failed pending case after DeepEval transport/execution failure.",
        "transport": "direct_ollama_api_chat",
        "deep_eval_transport_bypassed": True,
        "judge_behavior_changed": False,
        "judge_version": "0.15.0",
        "facet_spec_version": "0.5.0",
        "model": metadata.get("judge_model"),
        "temperature": metadata.get("temperature"),
        "structured_output": True,
        "http_timeout_seconds": timeout_seconds,
        "prompts_facets_scoring_unchanged": True,
        "recovery_was_failure_gated": True,
    }
    overrides = metadata.get("transport_overrides", [])
    if not isinstance(overrides, list):
        overrides = [overrides]
    overrides = [
        item for item in overrides
        if not (isinstance(item, dict) and item.get("case_id") == case_id)
    ]
    overrides.append(note)
    metadata["transport_overrides"] = overrides
    metadata_file.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    if failure_file.exists():
        failure_file.unlink()

    note_file = run_dir / f"transport_override_{case_id}.json"
    note_file.write_text(json.dumps(note, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print()
    print(f"{case_id} recovered and persisted through direct Ollama transport.")
    print("Resume the normal holdout runner to continue the remaining pending cases.")


if __name__ == "__main__":
    main()
