from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
SYS = REPO_ROOT / "evals/system_evaluation"
RUN = REPO_ROOT / "evals/runs/semantic_relevance_v4a_development/20261019T104415Z_retrieval"

V4B_SPEC = SYS / "semantic_relevance_v4b_reranker_only_hybrid_experiment.v0.1.0.json"
V4B_DESIGN_LOCK = SYS / "semantic_relevance_v4b_reranker_only_hybrid_experiment.v0.1.0.lock.json"
COMPLETE_MAP = RUN / "v4b_dev_complete_display_score_map.csv"
FINALIZATION_LOCK = RUN / "v4b_dev_complete_score_map.lock.json"
V4A_COMPARISON_LOCK = RUN / "v4a_dev_comparison_selection.lock.json"
BASE_ANALYZER = SYS / "analyze_semantic_relevance_v4a_dev_comparison.py"

EXPECTED_V4B_SPEC = "c72b5edbc330feae0620314fa5d5c6f90a55b356174d70a075ae77b47a8d0fca"
EXPECTED_V4B_DESIGN_LOCK = "e60ca1032f7fa733f3e7a51ec87a229d7454645aa6109b5e557588af50383c6d"
EXPECTED_COMPLETE_MAP = "6dbf399369292acfdea4adbb30e993aecb3f39f2be64d91c2cc4fd0455187eff"
EXPECTED_FINALIZATION_LOCK = "229c5e1329a8dcbbd74615fb63d8ed75f74d57a186ea923e5834e568a8664a75"
EXPECTED_V4A_COMPARISON_LOCK = "c9a33feabf061e9bec373cbf5a5bdfb8b83ac03fd700f9fa55ffdfaa77e5056f"
EXPECTED_BASE_ANALYZER_BLOB = "ac29a122e58c8627fd011f0db8a0e78c60d0bfe9"

OUTPUTS = [
    RUN / "v3b_dev_query_metrics_for_v4b.csv",
    RUN / "v4b_dev_query_metrics.csv",
    RUN / "v3b_dev_aggregate_metrics_for_v4b.json",
    RUN / "v4b_dev_aggregate_metrics.json",
    RUN / "v3b_dev_bootstrap_confidence_intervals_for_v4b.json",
    RUN / "v4b_dev_bootstrap_confidence_intervals.json",
    RUN / "v3b_vs_v4b_dev_query_deltas.csv",
    RUN / "v3b_vs_v4b_dev_paired_comparison.json",
    RUN / "v4b_dev_selection_result.json",
    RUN / "v3b_vs_v4b_dev_comparison_report.md",
    RUN / "v4b_dev_comparison_metadata.json",
    RUN / "v4b_dev_comparison_selection.lock.json",
]

def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def lf_sha256(path: Path) -> str:
    b = path.read_bytes()
    return hashlib.sha256(b.replace(b"\r\n", b"\n").replace(b"\r", b"\n")).hexdigest()

def req(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)

def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))

def b(value: Any) -> bool:
    t = str(value).strip().lower()
    if t in {"true","1","yes"}: return True
    if t in {"false","0","no",""}: return False
    raise ValueError(f"Invalid boolean value: {value!r}")

def verify_text(path: Path, expected: str, label: str) -> None:
    req(path.exists(), f"Missing frozen input: {path}")
    raw = sha256(path)
    if raw == expected:
        mode = "raw"
    else:
        norm = lf_sha256(path)
        req(norm == expected, f"{label} hash mismatch: expected={expected} raw={raw} lf={norm}")
        mode = "canonical-LF"
    print(f"PASS  {label} {mode} SHA-256 verified")

def verify_raw(path: Path, expected: str, label: str) -> None:
    req(path.exists(), f"Missing frozen input: {path}")
    actual = sha256(path)
    req(actual == expected, f"{label} hash mismatch: expected={expected} actual={actual}")
    print(f"PASS  {label} raw SHA-256 verified")

def main() -> None:
    print("Semantic relevance V4B - development comparison preflight")
    print("---------------------------------------------------------")
    print("Development metrics: DISABLED")
    print("Development selection: DISABLED")
    print("Judge/model execution: DISABLED")
    print("Human relevance labels: NOT USED")
    print("Release gates/decision: DISABLED")
    print()

    verify_text(V4B_SPEC, EXPECTED_V4B_SPEC, "V4B frozen design spec")
    verify_text(V4B_DESIGN_LOCK, EXPECTED_V4B_DESIGN_LOCK, "V4B design lock")
    verify_raw(COMPLETE_MAP, EXPECTED_COMPLETE_MAP, "V4B complete score map")
    verify_raw(FINALIZATION_LOCK, EXPECTED_FINALIZATION_LOCK, "V4B finalization lock")
    verify_raw(V4A_COMPARISON_LOCK, EXPECTED_V4A_COMPARISON_LOCK, "prior V4A comparison/selection lock")

    repo_path = "evals/system_evaluation/analyze_semantic_relevance_v4a_dev_comparison.py"
    blob = subprocess.run(
        ["git","rev-parse",f"HEAD:{repo_path}"],
        cwd=str(REPO_ROOT), capture_output=True, text=True, check=True
    ).stdout.strip()
    req(blob == EXPECTED_BASE_ANALYZER_BLOB,
        f"Base V4A analyzer Git blob changed: expected={EXPECTED_BASE_ANALYZER_BLOB} actual={blob}")
    diff = subprocess.run(["git","diff","--quiet","--",repo_path], cwd=str(REPO_ROOT))
    req(diff.returncode == 0, "Base V4A comparison analyzer has working-tree modifications")
    print("PASS  exact frozen V4A metric implementation tracked Git blob verified")
    print("PASS  base V4A comparison analyzer working tree is unmodified")

    for output in OUTPUTS:
        req(not output.exists(), f"Refusing preflight because V4B comparison output already exists: {output}")

    spec = load(V4B_SPEC)
    req(spec.get("status") == "PROPOSED_FOR_FREEZE_BEFORE_V4B_IMPLEMENTATION",
        "Unexpected V4B design-spec status")
    proto = spec.get("development_protocol", {})
    req(proto.get("query_set") == "H01-H24", "Development query set changed")
    req(int(proto.get("query_count",-1)) == 24, "Development query count changed")
    bs = proto.get("bootstrap", {})
    req(int(bs.get("replicates",-1)) == 5000, "Bootstrap replicate count changed")
    req(int(bs.get("seed",-1)) == 20261002, "Bootstrap seed changed")
    selection = spec.get("development_selection_rule", {})
    req(selection.get("ordered_criteria") == [
        "lower catastrophic_query_rate",
        "lower irrelevant_rate_at_10",
        "higher clear_or_strong_rate_at_10",
        "higher macro_mean_relevance_at_10",
        "higher ndcg_at_10",
    ], "V4B selection order changed")
    req(selection.get("final_tie_break") == "retain v3b", "V4B final tie-break changed")
    req(selection.get("retrieval_diagnostics_can_explain_but_not_override_selection") is True,
        "Retrieval-diagnostic override rule changed")
    req(selection.get("v4a_results_cannot_override_v3b_vs_v4b_selection") is True,
        "V4A override rule changed")
    req(selection.get("no_parameter_search_after_results_seen") is True,
        "Post-result parameter-search prohibition changed")

    dlock = load(V4B_DESIGN_LOCK)
    req(dlock.get("status") == "V4B_DESIGN_FROZEN_BEFORE_IMPLEMENTATION",
        "Unexpected V4B design-lock status")
    req(dlock.get("development_selection_calculated") is False,
        "V4B design lock says selection was already calculated")

    flock = load(FINALIZATION_LOCK)
    req(flock.get("status") == "V4B_DEV_COMPLETE_DISPLAY_SCORE_MAP_FROZEN_BEFORE_COMPARISON",
        "Unexpected V4B finalization-lock status")
    req(flock.get("complete_score_map_sha256") == EXPECTED_COMPLETE_MAP,
        "Complete-map hash differs inside finalization lock")
    req(int(flock.get("displayed_pair_count",-1)) == 367,
        "V4B complete-map row count changed")
    req(flock.get("known_h14_quarantine_pair_present") is False,
        "Finalization lock says unresolved H14 pair is present")
    req(flock.get("development_comparison_calculated") is False,
        "Comparison was already calculated during finalization")
    req(flock.get("development_selection_decision_produced") is False,
        "Selection was already produced during finalization")

    prior = load(V4A_COMPARISON_LOCK)
    req(prior.get("status") == "V4A_DEVELOPMENT_COMPARISON_AND_SELECTION_FROZEN",
        "Unexpected prior V4A comparison lock status")
    req(prior.get("selected_candidate") == "v3b",
        "Prior frozen V4A selection no longer records V3B")
    req(prior.get("release_decision_produced") is False,
        "Prior V4A comparison unexpectedly produced a release decision")

    frame = pd.read_csv(COMPLETE_MAP, dtype=str, keep_default_na=False, encoding="utf-8")
    required = {
        "query_id","query","stratum","isbn13","in_v3b","v3b_rank",
        "in_v4b","v4b_rank","final_judge_score",
    }
    req(not (required-set(frame.columns)), f"Complete map missing columns: {sorted(required-set(frame.columns))}")
    req(len(frame) == 367, "Complete score map is not 367 rows")
    req(frame["query_id"].nunique() == 24, "Complete map is not H01-H24")
    req(not frame.duplicated(["query_id","isbn13"]).any(), "Duplicate pair in complete score map")
    req(int(frame["in_v3b"].map(b).sum()) == 240, "V3B membership is not 240")
    req(int(frame["in_v4b"].map(b).sum()) == 240, "V4B membership is not 240")
    vals = pd.to_numeric(frame["final_judge_score"], errors="raise").astype(int)
    req(vals.between(0,4).all(), "Complete score map contains score outside 0..4")
    req(not ((frame["query_id"]=="H14") & (frame["isbn13"]=="9781582972947")).any(),
        "Unresolved H14 quarantine pair entered comparison map")

    print("PASS  frozen V3B/V4B comparison structure verified: 367 pairs / 24 queries / 240 rows per system")
    print("PASS  pre-registered lexicographic selection contract verified")
    print("PASS  unresolved H14 quarantine pair absent")
    print()
    print("V4B DEVELOPMENT COMPARISON PREFLIGHT: PASS")
    print("Development query set: H01-H24")
    print("Bootstrap: 5000 query-resamples / seed 20261002")
    print("Selection: catastrophic -> irrelevant -> clear/strong -> mean -> nDCG -> retain V3B")
    print("Development metrics calculated: NO")
    print("Development selection produced: NO")
    print("Judge calls: 0")
    print("Model calls: 0")
    print("Human labels used: NO")
    print("Release gates/decision: NO")
    print("Next: calculate frozen V3B-vs-V4B development metrics and apply the pre-registered selection rule.")

if __name__ == "__main__":
    main()
