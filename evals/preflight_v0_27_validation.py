"""Non-judging preflight for the frozen 30-case v0.27 validation split."""
from __future__ import annotations

from pathlib import Path
import hashlib
import json
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
EVALS = REPO_ROOT / "evals"
D = EVALS / "datasets"
POOL = D / "semantic_relevance_unseen_pool.v3.0.0.csv"
VAL = D / "semantic_relevance_validation.v3.0.0.csv"
HOLD = D / "semantic_relevance_final_holdout.v3.0.0.DO_NOT_RUN_YET.csv"
MAN = D / "semantic_relevance_unseen_pool_manifest.v3.0.0.json"
LOCK = D / "semantic_relevance_final_holdout_lock.v3.0.0.json"
FREEZE = EVALS / "releases" / "semantic_relevance_v0.27.0_r5_development_candidate.json"
EXPECTED_FREEZE_SHA = "3de6ee3a6274a5d066f17ff24bc965b05dfb8d60b785c0210d45d743538d84e5"
EXPECTED_VAL_SCORES = {0: 18, 1: 2, 2: 2, 3: 8, 4: 0}
EXPECTED_HOLD_SCORES = {0: 18, 1: 2, 2: 2, 3: 7, 4: 1}
EXPECTED_SLOTS = {f"T{i:02d}": 6 for i in range(1, 6)}


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def score_counts(df: pd.DataFrame) -> dict[int, int]:
    s = pd.to_numeric(df["human_score"], errors="raise").astype(int)
    return {i: int((s == i).sum()) for i in range(5)}


def slot_counts(df: pd.DataFrame) -> dict[str, int]:
    slots = df["case_id"].astype(str).str.rsplit("_", n=1).str[-1]
    return {f"T{i:02d}": int((slots == f"T{i:02d}").sum()) for i in range(1, 6)}


def main() -> None:
    for p in (POOL, VAL, HOLD, MAN, LOCK, FREEZE):
        if not p.exists():
            raise FileNotFoundError(f"Required file not found: {p}")

    m = json.loads(MAN.read_text(encoding="utf-8-sig"))
    l = json.loads(LOCK.read_text(encoding="utf-8-sig"))
    pool = pd.read_csv(POOL, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    val = pd.read_csv(VAL, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    hold = pd.read_csv(HOLD, dtype=str, keep_default_na=False, encoding="utf-8-sig")

    def ck(name: str, cond: bool, msg: str) -> None:
        print(f"{'PASS' if cond else 'FAIL':4}  {name}")
        if not cond:
            raise ValueError(msg)

    print("v0.27 validation preflight")
    print("--------------------------")
    ck("frozen r5 candidate hash", sha(FREEZE) == EXPECTED_FREEZE_SHA, "Frozen candidate hash mismatch.")
    ck("pool cases: 60", len(pool) == 60, "Pool count mismatch.")
    ck("validation cases: 30", len(val) == 30, "Validation count mismatch.")
    ck("final holdout cases: 30", len(hold) == 30, "Holdout count mismatch.")
    ck("validation/holdout disjoint", not (set(val.case_id) & set(hold.case_id)), "Split overlap.")
    ck("validation + holdout = pool", set(val.case_id) | set(hold.case_id) == set(pool.case_id), "Split does not reconstruct pool.")
    ck("pool hash matches manifest", sha(POOL) == m["pool"]["sha256"], "Pool hash changed after split.")
    ck("validation hash matches manifest", sha(VAL) == m["validation"]["sha256"], "Validation file changed after split.")
    ck("final holdout hash matches manifest", sha(HOLD) == m["final_holdout"]["sha256"], "Final holdout changed after split.")
    ck("judge outputs were not used for split", m.get("judge_outputs_used_for_split") is False, "Split provenance says judge output was used.")
    ck("validation score distribution balanced", score_counts(val) == EXPECTED_VAL_SCORES, "Validation human-score distribution mismatch.")
    ck("holdout score distribution balanced", score_counts(hold) == EXPECTED_HOLD_SCORES, "Holdout human-score distribution mismatch.")
    ck("validation sampling slots balanced", slot_counts(val) == EXPECTED_SLOTS, "Validation sampling-slot distribution mismatch.")
    ck("holdout sampling slots balanced", slot_counts(hold) == EXPECTED_SLOTS, "Holdout sampling-slot distribution mismatch.")
    ck("final holdout lock status", l.get("status") == "LOCKED_DO_NOT_RUN", "Final holdout is not locked.")
    ck("final holdout judge run count = 0", int(l.get("judge_run_count", -1)) == 0, "Final holdout already marked as run.")
    ck("all validation labels present", pd.to_numeric(val.human_score, errors="coerce").isin([0, 1, 2, 3, 4]).all(), "Validation labels incomplete.")

    print()
    print(f"Validation score distribution: {score_counts(val)}")
    print(f"Final holdout score distribution: {score_counts(hold)}")
    print(f"Validation SHA-256:    {sha(VAL)}")
    print(f"Final holdout SHA-256: {sha(HOLD)}")
    print("PRE-FLIGHT: PASS")


if __name__ == "__main__":
    main()
