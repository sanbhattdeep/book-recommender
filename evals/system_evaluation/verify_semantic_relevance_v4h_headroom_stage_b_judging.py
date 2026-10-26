from __future__ import annotations
import pandas as pd
import score_semantic_relevance_v4h_headroom_stage_b as execution


def req(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def main() -> None:
    print("Semantic relevance V4H - headroom Stage-B judge execution preflight")
    print("-----------------------------------------------------------------")
    print("Role: consumed-holdout diagnostic only")
    print("Fresh cases: 251")
    print("Judge execution: DISABLED")
    print("Release decision modification: DISABLED")
    print()

    frame, _ = execution.verify_input_boundary()
    execution.verify_measurement_instrument()
    req(not execution.FINAL_LOCK.exists(), "Final Stage-B fresh-score lock already exists")

    if execution.FRESH_SCORES.exists():
        existing = pd.read_csv(execution.FRESH_SCORES, dtype=str, keep_default_na=False, encoding="utf-8")
        execution.validate_existing_scores(existing, frame)
        print(f"INFO  resume-state score artifact exists: {len(existing)}/251 persisted")
    else:
        print("INFO  no Stage-B fresh-score artifact exists yet")

    print()
    print("V4H HEADROOM STAGE-B JUDGE PREFLIGHT: PASS")
    print("Fresh frozen pairs: 251")
    print("Exact reused pairs external: 1")
    print("Measurement instrument: semantic_relevance_v0.29.0-r5")
    print("Resume-safe persistence: ENABLED")
    print("Judge calls performed during preflight: 0")
    print("Release decision modified: NO")
    print("Next: run a 3-case smoke test.")


if __name__ == "__main__":
    main()
