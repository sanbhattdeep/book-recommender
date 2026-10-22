from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
EVALS = REPO_ROOT / "evals"
SYS = EVALS / "system_evaluation"
RUN = EVALS / "runs/semantic_relevance_v4a_development/20261019T104415Z_retrieval"
HOLDOUT_RUN = EVALS / "runs/semantic_relevance_v3_holdout/20261007T124150Z_retrieval"

BASE_RUNNER = SYS / "score_semantic_relevance_v4a_dev_novel_pairs.py"
EXPECTED_BASE_BLOB = "da066ad865955f37ed7d418b6a26fa72cab6f665"

NOVEL_INPUT = RUN / "v4b_dev_novel_display_pairs_to_judge.csv"
SCORE_INVENTORY = RUN / "v4b_dev_display_score_inventory.csv"
JUDGING_INPUT_LOCK = RUN / "v4b_dev_judging_input.lock.json"
HOLDOUT_SCORING_LOCK = HOLDOUT_RUN / "holdout_judge_scoring_lock.json"
JUDGE_FILE = EVALS / "semantic_relevance_facet_judge.py"
SCORING_FILE = EVALS / "semantic_relevance_facet_scoring.py"
JUDGE_CONFIG = EVALS / "judge_configs/semantic_relevance_judge.v0.29.0-r5.json"
RUBRIC_FILE = EVALS / "rubrics/semantic_relevance/semantic_relevance_rubric.v0.1.0.json"
FACETS = EVALS / "facets/semantic_relevance/semantic_relevance_v3_holdout_query_facets.v0.1.0.json"
PATCH = SYS / "semantic_relevance_v3_posthoc_execution_patch.span-echo-normalization1.json"

NEW_SCORES = RUN / "v4b_dev_novel_judge_scores.csv"
METADATA = RUN / "v4b_dev_novel_judging_metadata.json"
SCORING_LOCK = RUN / "v4b_dev_novel_judge_scoring.lock.json"

EXPECTED_NOVEL = "5eeb07e860ba70248a3c8f6df238f347535d7fe0a6d3c76314815b493ef753d0"
EXPECTED_INV = "f2c3d8905e582cc9df6b3a9417f3647d71496eed45ea5240d95c1cf1ece85cc1"
EXPECTED_INPUT_LOCK = "9099bae135a7a97459633243a0659543b631fcc9bb3987aee0a86b6ee5fedf20"
EXPECTED_HOLDOUT_LOCK = "4a4be17bdec6a3d1cd95667bde9fbae1fb4a6077e10b5dc4270710787ffe6777"
EXPECTED_PATCH = "85488eb50ae0d9d4b8f0f2c213f9874a3f3d24881131af57e1bf61e5776e4266"

def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()

def lf_sha256(p: Path) -> str:
    b = p.read_bytes()
    return hashlib.sha256(b.replace(b"\r\n", b"\n").replace(b"\r", b"\n")).hexdigest()

def req(c: bool, m: str) -> None:
    if not c:
        raise ValueError(m)

def load(p: Path) -> dict[str, Any]:
    return json.loads(p.read_text(encoding="utf-8-sig"))

def bool_value(v: Any) -> bool:
    t = str(v).strip().lower()
    if t in {"true","1","yes"}: return True
    if t in {"false","0","no",""}: return False
    raise ValueError(f"Invalid boolean: {v!r}")

def verify_raw(p: Path, expected: str, label: str) -> None:
    req(p.exists(), f"Missing {label}: {p}")
    actual = sha256(p)
    req(actual == expected, f"{label} hash mismatch: expected={expected} actual={actual}")
    print(f"PASS  {label} raw SHA-256 verified")

def verify_text(p: Path, expected: str, label: str) -> None:
    req(p.exists(), f"Missing {label}: {p}")
    raw = sha256(p)
    if raw == expected:
        mode = "raw"
    else:
        norm = lf_sha256(p)
        req(norm == expected, f"{label} hash mismatch: expected={expected} raw={raw} lf={norm}")
        mode = "canonical-LF"
    print(f"PASS  {label} {mode} SHA-256 verified")

def main() -> None:
    print("Semantic relevance V4B - novel judge preflight")
    print("---------------------------------------------")
    print("Judge/model execution: DISABLED")
    print("Human relevance labels: NOT USED")
    print("Development comparison: NOT CALCULATED")
    print("Release analysis: DISABLED")
    print()

    verify_raw(NOVEL_INPUT, EXPECTED_NOVEL, "V4B novel input")
    verify_raw(SCORE_INVENTORY, EXPECTED_INV, "V4B score inventory")
    verify_raw(JUDGING_INPUT_LOCK, EXPECTED_INPUT_LOCK, "V4B judging-input lock")
    verify_raw(HOLDOUT_SCORING_LOCK, EXPECTED_HOLDOUT_LOCK, "V3 holdout scoring lock")
    verify_text(PATCH, EXPECTED_PATCH, "span-echo patch record")

    req(BASE_RUNNER.exists(), f"Missing base V4A runner: {BASE_RUNNER}")
    repo_path = "evals/system_evaluation/score_semantic_relevance_v4a_dev_novel_pairs.py"
    blob = subprocess.run(
        ["git","rev-parse",f"HEAD:{repo_path}"],
        cwd=str(REPO_ROOT), capture_output=True, text=True, check=True
    ).stdout.strip()
    req(blob == EXPECTED_BASE_BLOB,
        f"Base V4A runner Git blob changed: expected={EXPECTED_BASE_BLOB} actual={blob}")
    diff = subprocess.run(
        ["git","diff","--quiet","--",repo_path],
        cwd=str(REPO_ROOT)
    )
    req(diff.returncode == 0, "Base V4A judge runner has working-tree modifications")
    print("PASS  exact reviewed V4A judge runner tracked Git blob verified")
    print("PASS  base V4A judge runner working tree is unmodified")

    for p in [NEW_SCORES, METADATA, SCORING_LOCK]:
        req(not p.exists(), f"Refusing preflight because V4B judge output already exists: {p}")

    ilock = load(JUDGING_INPUT_LOCK)
    req(ilock.get("status") == "V4B_DEV_JUDGING_INPUT_FROZEN_BEFORE_JUDGE_EXECUTION",
        "Unexpected V4B judging-input status")
    req(int(ilock.get("novel_pairs_to_judge",-1)) == 22, "Novel count changed")
    req(ilock.get("judge_execution_performed") is False, "Input lock says judge already executed")

    hlock = load(HOLDOUT_SCORING_LOCK)
    req(hlock.get("judge_candidate") == "semantic_relevance_v0.29.0-r5", "Judge candidate changed")
    for p,key,label in [
        (JUDGE_FILE,"judge_source_sha256","judge source"),
        (SCORING_FILE,"scoring_source_sha256","deterministic scoring"),
        (JUDGE_CONFIG,"judge_config_sha256","judge config"),
        (RUBRIC_FILE,"rubric_sha256","rubric"),
        (FACETS,"holdout_facet_spec_sha256","H01-H24 facet spec"),
    ]:
        expected = str(hlock.get(key,""))
        req(bool(expected), f"Holdout lock missing {key}")
        verify_text(p, expected, label)

    patch = load(PATCH)
    expected_patch_contract = {
        "status": "APPLIED_NON_SEMANTIC_EXECUTION_PATCH",
        "patch_id": "v3-posthoc-prominence-span-echo-normalization1",
        "judge_source_modified": False,
        "judge_prompt_modified": False,
        "semantic_rules_changed": False,
        "deterministic_scoring_rules_changed": False,
        "facet_spec_changed": False,
        "judge_config_changed": False,
        "rubric_changed": False,
        "release_metrics_or_gates_changed": False,
        "frozen_v3_release_decision_modified": False,
    }
    for key, value in expected_patch_contract.items():
        req(patch.get(key) == value, f"Span-echo patch contract changed: {key}")
    print("PASS  accepted span-echo patch semantic contract verified")

    config = load(JUDGE_CONFIG)
    req(config.get("version") == "0.29.0-r5", "Judge config version changed")
    req(float(config.get("temperature",-1)) == 0.0, "Temperature changed")

    novel = pd.read_csv(NOVEL_INPUT, dtype=str, keep_default_na=False, encoding="utf-8")
    req(len(novel) == 22, f"Expected 22 novel rows, got {len(novel)}")
    req(not novel["case_id"].duplicated().any(), "Duplicate case IDs")
    req(not novel.duplicated(["query_id","isbn13"]).any(), "Duplicate query-book pair")
    req(novel["pair_source"].eq("v4b_only").all(), "Novel input includes non-V4B-only pair")
    req((~novel["in_v3b"].map(bool_value)).all(), "Novel input includes V3B pair")
    req(novel["in_v4b"].map(bool_value).all(), "Novel input includes non-V4B pair")
    ranks = pd.to_numeric(novel["v4b_rank"], errors="raise").astype(int)
    req(ranks.between(1,10).all(), "V4B rank outside 1..10")
    req(not ((novel["query_id"]=="H14") & (novel["isbn13"]=="9781582972947")).any(),
        "Unresolved H14 quarantine pair entered judge input")

    print("PASS  exact frozen 22-case V4B judging population verified")
    print("PASS  unresolved H14 quarantine pair absent")
    print()
    print("V4B DEVELOPMENT NOVEL-JUDGE PREFLIGHT: PASS")
    print("Novel pairs: 22")
    print("Judge candidate: semantic_relevance_v0.29.0-r5")
    print("Temperature: 0.0")
    print("Judge calls performed: 0")
    print("Model calls performed: 0")
    print("Human relevance labels used: NO")
    print("Development comparison calculated: NO")
    print("Release analysis: NO")
    print("Next: execute the frozen judge only for these 22 novel displayed pairs.")

if __name__ == "__main__":
    main()
