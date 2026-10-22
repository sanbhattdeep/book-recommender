from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
SYS = REPO_ROOT / "evals/system_evaluation"
RUN = REPO_ROOT / "evals/runs/semantic_relevance_v4a_development/20261019T104415Z_retrieval"

V4B_SPEC = SYS / "semantic_relevance_v4b_reranker_only_hybrid_experiment.v0.1.0.json"
V4B_DESIGN_LOCK = SYS / "semantic_relevance_v4b_reranker_only_hybrid_experiment.v0.1.0.lock.json"
COMPLETE_MAP = RUN / "v4b_dev_complete_display_score_map.csv"
FINALIZATION_LOCK = RUN / "v4b_dev_complete_score_map.lock.json"
V4A_COMPARISON_LOCK = RUN / "v4a_dev_comparison_selection.lock.json"
FROZEN_V3_AGG = RUN / "v3b_dev_aggregate_metrics.json"
BASE_ANALYZER = SYS / "analyze_semantic_relevance_v4a_dev_comparison.py"

EXPECTED_V4B_SPEC = "c72b5edbc330feae0620314fa5d5c6f90a55b356174d70a075ae77b47a8d0fca"
EXPECTED_V4B_DESIGN_LOCK = "e60ca1032f7fa733f3e7a51ec87a229d7454645aa6109b5e557588af50383c6d"
EXPECTED_COMPLETE_MAP = "6dbf399369292acfdea4adbb30e993aecb3f39f2be64d91c2cc4fd0455187eff"
EXPECTED_FINALIZATION_LOCK = "229c5e1329a8dcbbd74615fb63d8ed75f74d57a186ea923e5834e568a8664a75"
EXPECTED_V4A_COMPARISON_LOCK = "c9a33feabf061e9bec373cbf5a5bdfb8b83ac03fd700f9fa55ffdfaa77e5056f"
EXPECTED_BASE_ANALYZER_BLOB = "ac29a122e58c8627fd011f0db8a0e78c60d0bfe9"

NQ = 24
K = 10
REPS = 5000
SEED = 20261002

METRICS = [
    "macro_mean_relevance_at_10",
    "clear_or_strong_rate_at_10",
    "irrelevant_rate_at_10",
    "hit_rate_at_5_clear_or_strong",
    "top1_clear_or_strong_rate",
    "mrr_at_10_clear_or_strong",
    "ndcg_at_10",
    "catastrophic_query_rate",
]
SELECTION_CRITERIA = [
    ("catastrophic_query_rate","lower"),
    ("irrelevant_rate_at_10","lower"),
    ("clear_or_strong_rate_at_10","higher"),
    ("macro_mean_relevance_at_10","higher"),
    ("ndcg_at_10","higher"),
]

V3_QUERY_METRICS = RUN / "v3b_dev_query_metrics_for_v4b.csv"
V4_QUERY_METRICS = RUN / "v4b_dev_query_metrics.csv"
V3_AGG = RUN / "v3b_dev_aggregate_metrics_for_v4b.json"
V4_AGG = RUN / "v4b_dev_aggregate_metrics.json"
V3_CI = RUN / "v3b_dev_bootstrap_confidence_intervals_for_v4b.json"
V4_CI = RUN / "v4b_dev_bootstrap_confidence_intervals.json"
QUERY_DELTAS = RUN / "v3b_vs_v4b_dev_query_deltas.csv"
PAIRED = RUN / "v3b_vs_v4b_dev_paired_comparison.json"
SELECTION = RUN / "v4b_dev_selection_result.json"
REPORT = RUN / "v3b_vs_v4b_dev_comparison_report.md"
METADATA = RUN / "v4b_dev_comparison_metadata.json"
LOCK = RUN / "v4b_dev_comparison_selection.lock.json"

def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))

def req(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)

def b(value: Any) -> bool:
    t = str(value).strip().lower()
    if t in {"true","1","yes"}: return True
    if t in {"false","0","no",""}: return False
    raise ValueError(f"Invalid boolean value: {value!r}")

def load_base():
    repo_path = "evals/system_evaluation/analyze_semantic_relevance_v4a_dev_comparison.py"
    blob = subprocess.run(
        ["git","rev-parse",f"HEAD:{repo_path}"],
        cwd=str(REPO_ROOT), capture_output=True, text=True, check=True
    ).stdout.strip()
    req(blob == EXPECTED_BASE_ANALYZER_BLOB,
        f"Base analyzer Git blob changed: expected={EXPECTED_BASE_ANALYZER_BLOB} actual={blob}")
    diff = subprocess.run(["git","diff","--quiet","--",repo_path], cwd=str(REPO_ROOT))
    req(diff.returncode == 0, "Base V4A analyzer has working-tree modifications")
    spec = importlib.util.spec_from_file_location("v4a_comparison_base_for_v4b", BASE_ANALYZER)
    req(spec is not None and spec.loader is not None, "Could not import base V4A comparison analyzer")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    req(module.REPS == REPS and module.SEED == SEED, "Base bootstrap contract differs")
    return module

def reconstruct(frame: pd.DataFrame, membership: str, rank_col: str, label: str) -> pd.DataFrame:
    mask = frame[membership].map(b)
    out = frame.loc[mask].copy()
    req(len(out) == NQ*K, f"{label}: expected {NQ*K} rows")
    out["rank"] = pd.to_numeric(out[rank_col], errors="raise").astype(int)
    out["judge_score"] = pd.to_numeric(out["final_judge_score"], errors="raise").astype(int)
    req(out["judge_score"].between(0,4).all(), f"{label}: score outside 0..4")
    req(not out.duplicated(["query_id","rank"]).any(), f"{label}: duplicate query/rank")
    counts = out.groupby("query_id").size()
    req(len(counts) == NQ and (counts == K).all(), f"{label}: not exact 24x10")
    for qid, group in out.groupby("query_id", sort=False):
        req(sorted(group["rank"].tolist()) == list(range(1,K+1)), f"{label}/{qid}: ranks are not 1..10")
    return out

def paired_comparison(v3q: pd.DataFrame, v4q: pd.DataFrame) -> tuple[dict[str,Any], pd.DataFrame]:
    req(list(v3q["query_id"]) == list(v4q["query_id"]), "Paired query IDs differ")
    mapping = {
        "macro_mean_relevance_at_10": ("mean_relevance_at_10", True),
        "clear_or_strong_rate_at_10": ("clear_or_strong_rate_at_10", True),
        "irrelevant_rate_at_10": ("irrelevant_rate_at_10", False),
        "hit_rate_at_5_clear_or_strong": ("hit_at_5_clear_or_strong", True),
        "top1_clear_or_strong_rate": ("top1_clear_or_strong", True),
        "mrr_at_10_clear_or_strong": ("mrr_at_10_clear_or_strong", True),
        "ndcg_at_10": ("ndcg_at_10", True),
        "catastrophic_query_rate": ("catastrophic_query", False),
    }
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0,NQ,size=(REPS,NQ))
    metrics, rows = {}, []
    for metric,(column,higher_is_better) in mapping.items():
        before = v3q[column].astype(float).to_numpy()
        after = v4q[column].astype(float).to_numpy()
        delta = after-before
        improvement = delta if higher_is_better else -delta
        distribution = np.mean(delta[idx], axis=1)
        metrics[metric] = {
            "v3b": float(np.mean(before)),
            "v4b": float(np.mean(after)),
            "raw_delta_v4b_minus_v3b": float(np.mean(delta)),
            "paired_ci_lower": float(np.quantile(distribution,0.025)),
            "paired_ci_upper": float(np.quantile(distribution,0.975)),
            "wins": int(np.sum(improvement>0.0)),
            "ties": int(np.sum(improvement==0.0)),
            "losses": int(np.sum(improvement<0.0)),
            "higher_is_better": higher_is_better,
        }
        for i,qid in enumerate(v3q["query_id"].astype(str)):
            rows.append({
                "query_id":qid,
                "query":str(v3q.iloc[i]["query"]),
                "stratum":str(v3q.iloc[i]["stratum"]),
                "metric":metric,
                "v3b_value":float(before[i]),
                "v4b_value":float(after[i]),
                "raw_delta_v4b_minus_v3b":float(delta[i]),
                "outcome_for_v4b":"WIN" if improvement[i]>0 else "LOSS" if improvement[i]<0 else "TIE",
            })
    return {
        "status":"COMPLETE",
        "analysis_role":"development_secondary_evidence",
        "query_count":NQ,
        "bootstrap_replicates":REPS,
        "bootstrap_seed":SEED,
        "metrics":metrics,
    }, pd.DataFrame(rows)

def select_candidate(v3: dict[str,float], v4: dict[str,float]) -> dict[str,Any]:
    evaluations = []
    selected = "v3b"
    decisive = None
    decisive_direction = None
    for metric,direction in SELECTION_CRITERIA:
        baseline = float(v3[metric]); candidate = float(v4[metric])
        if candidate == baseline:
            outcome = "TIE_CONTINUE"
        elif direction == "lower":
            outcome = "V4B_BETTER" if candidate < baseline else "V3B_BETTER"
        else:
            outcome = "V4B_BETTER" if candidate > baseline else "V3B_BETTER"
        evaluations.append({
            "metric":metric,"preferred_direction":direction,
            "v3b":baseline,"v4b":candidate,"outcome":outcome,
        })
        if outcome == "V4B_BETTER":
            selected,decisive,decisive_direction = "v4b",metric,direction
            break
        if outcome == "V3B_BETTER":
            selected,decisive,decisive_direction = "v3b",metric,direction
            break
    exact_tie = decisive is None
    if exact_tie:
        selected = "v3b"
    return {
        "status":"V4B_DEVELOPMENT_SELECTION_COMPLETE",
        "selection_type":"pre_registered_lexicographic_v3b_vs_v4b",
        "ordered_criteria":[m for m,_ in SELECTION_CRITERIA],
        "evaluations_until_decision":evaluations,
        "selected_candidate":selected,
        "decisive_metric":decisive,
        "decisive_preferred_direction":decisive_direction,
        "exact_tie_retains_v3b":exact_tie,
        "retrieval_diagnostics_can_override_selection":False,
        "v4a_results_can_override_selection":False,
        "parameter_search_after_results_seen":False,
        "development_only":True,
        "release_decision":False,
        "requires_new_unseen_v4_holdout":selected=="v4b",
        "next_stage": (
            "AUTHOR_AND_FREEZE_V4H01_V4H24_UNSEEN_HOLDOUT_BEFORE_RETRIEVAL"
            if selected=="v4b"
            else "V4B_NOT_SELECTED_STOP_OR_DESIGN_SEPARATE_NEXT_EXPERIMENT"
        ),
    }

def main() -> None:
    print("Semantic relevance V4B - V3B vs V4B DEVELOPMENT comparison")
    print("-----------------------------------------------------------")
    print("Development query set: H01-H24")
    print("Judge/model execution: DISABLED")
    print("Human relevance labels: NOT USED")
    print("Release gates/decision: NOT CALCULATED")
    print("Selection rule: FROZEN LEXICOGRAPHIC")
    print()

    outputs = [V3_QUERY_METRICS,V4_QUERY_METRICS,V3_AGG,V4_AGG,V3_CI,V4_CI,
               QUERY_DELTAS,PAIRED,SELECTION,REPORT,METADATA,LOCK]
    for path in outputs:
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite existing V4B comparison artifact: {path}")

    req(sha256(COMPLETE_MAP)==EXPECTED_COMPLETE_MAP, "Complete-map hash changed")
    req(sha256(FINALIZATION_LOCK)==EXPECTED_FINALIZATION_LOCK, "Finalization-lock hash changed")
    req(sha256(V4A_COMPARISON_LOCK)==EXPECTED_V4A_COMPARISON_LOCK, "Prior V4A comparison-lock hash changed")

    spec = load_json(V4B_SPEC)
    selection_contract = spec["development_selection_rule"]
    req(selection_contract["ordered_criteria"] == [
        "lower catastrophic_query_rate",
        "lower irrelevant_rate_at_10",
        "higher clear_or_strong_rate_at_10",
        "higher macro_mean_relevance_at_10",
        "higher ndcg_at_10",
    ], "Selection order changed")
    req(selection_contract["final_tie_break"]=="retain v3b", "Final tie-break changed")

    base = load_base()
    frame = pd.read_csv(COMPLETE_MAP, dtype=str, keep_default_na=False, encoding="utf-8")
    req(len(frame)==367 and frame["query_id"].nunique()==24, "Complete-map structure changed")
    req(not ((frame["query_id"]=="H14") & (frame["isbn13"]=="9781582972947")).any(),
        "Unresolved H14 quarantine pair entered comparison")

    v3 = reconstruct(frame,"in_v3b","v3b_rank","v3b")
    v4 = reconstruct(frame,"in_v4b","v4b_rank","v4b")
    print("PASS  reconstructed V3B and V4B as exact 24x10 systems")

    v3q,v3a,v3ci = base.standalone(v3)
    v4q,v4a,v4ci = base.standalone(v4)

    prior_lock = load_json(V4A_COMPARISON_LOCK)
    frozen_v3_hash = prior_lock.get("outputs",{}).get(FROZEN_V3_AGG.name)
    req(bool(frozen_v3_hash), "Prior comparison lock does not freeze V3B aggregate metrics")
    req(FROZEN_V3_AGG.exists(), "Frozen prior V3B aggregate metrics artifact is missing")
    req(sha256(FROZEN_V3_AGG)==frozen_v3_hash, "Prior V3B aggregate metrics hash changed")
    prior_v3 = load_json(FROZEN_V3_AGG)
    for metric in METRICS:
        req(abs(float(v3a[metric])-float(prior_v3[metric])) < 1e-15,
            f"V3B baseline drift detected for {metric}")
    print("PASS  reconstructed V3B aggregate exactly matches frozen prior V3B development baseline")

    paired,delta_rows = paired_comparison(v3q,v4q)
    selection = select_candidate(v3a,v4a)

    v3q.to_csv(V3_QUERY_METRICS,index=False,encoding="utf-8")
    v4q.to_csv(V4_QUERY_METRICS,index=False,encoding="utf-8")
    V3_AGG.write_text(json.dumps(v3a,indent=2)+"\n",encoding="utf-8")
    V4_AGG.write_text(json.dumps(v4a,indent=2)+"\n",encoding="utf-8")
    V3_CI.write_text(json.dumps(v3ci,indent=2)+"\n",encoding="utf-8")
    V4_CI.write_text(json.dumps(v4ci,indent=2)+"\n",encoding="utf-8")
    delta_rows.to_csv(QUERY_DELTAS,index=False,encoding="utf-8")
    PAIRED.write_text(json.dumps(paired,indent=2)+"\n",encoding="utf-8")
    SELECTION.write_text(json.dumps(selection,indent=2)+"\n",encoding="utf-8")

    lines = [
        "# Semantic Relevance V4B — Development Comparison","",
        "H01-H24 are development-only. This report does not make a release claim.","",
        f"**Selected development candidate: {selection['selected_candidate']}**",
        f"**Decisive criterion: {selection['decisive_metric'] or 'exact tie; retain v3b'}**","",
        "## Standalone development metrics","",
        "| Metric | V3B | V4B |","|---|---:|---:|",
    ]
    for metric in METRICS:
        lines.append(f"| `{metric}` | {v3a[metric]:.6f} | {v4a[metric]:.6f} |")
    lines += ["","## Pre-registered lexicographic selection",""]
    for item in selection["evaluations_until_decision"]:
        lines.append(
            f"- `{item['metric']}` ({item['preferred_direction']} is better): "
            f"V3B={item['v3b']:.6f}, V4B={item['v4b']:.6f} → {item['outcome']}"
        )
    lines += [
        "",
        "Retrieval diagnostics and V4A results may explain the result but cannot override this selection.",
        "No parameter search is permitted after these results are observed.",
        "A new unseen V4H01-V4H24 holdout is required only if V4B is selected.",
    ]
    REPORT.write_text("\n".join(lines)+"\n",encoding="utf-8")

    output_files = [V3_QUERY_METRICS,V4_QUERY_METRICS,V3_AGG,V4_AGG,V3_CI,V4_CI,
                    QUERY_DELTAS,PAIRED,SELECTION,REPORT]
    metadata = {
        "status":"V4B_DEVELOPMENT_COMPARISON_AND_SELECTION_COMPLETE",
        "completed_at_utc":datetime.now(timezone.utc).isoformat(),
        "development_query_set":"H01-H24",
        "development_query_count":NQ,
        "complete_score_map_sha256":EXPECTED_COMPLETE_MAP,
        "finalization_lock_sha256":EXPECTED_FINALIZATION_LOCK,
        "v4b_design_spec_sha256":EXPECTED_V4B_SPEC,
        "v4b_design_lock_sha256":EXPECTED_V4B_DESIGN_LOCK,
        "prior_v4a_comparison_lock_sha256":EXPECTED_V4A_COMPARISON_LOCK,
        "bootstrap_replicates":REPS,
        "bootstrap_seed":SEED,
        "v3b_baseline_reproduced_exactly":True,
        "selected_candidate":selection["selected_candidate"],
        "decisive_metric":selection["decisive_metric"],
        "judge_calls_performed_during_comparison":0,
        "model_calls_performed_during_comparison":0,
        "human_relevance_labels_used":False,
        "release_gates_calculated":False,
        "release_decision_produced":False,
        "requires_new_unseen_v4_holdout":selection["requires_new_unseen_v4_holdout"],
        "next_stage":selection["next_stage"],
    }
    METADATA.write_text(json.dumps(metadata,indent=2)+"\n",encoding="utf-8")

    lock = {
        "status":"V4B_DEVELOPMENT_COMPARISON_AND_SELECTION_FROZEN",
        "frozen_at_utc":datetime.now(timezone.utc).isoformat(),
        "v4b_design_spec_sha256":EXPECTED_V4B_SPEC,
        "v4b_design_lock_sha256":EXPECTED_V4B_DESIGN_LOCK,
        "complete_score_map_sha256":EXPECTED_COMPLETE_MAP,
        "finalization_lock_sha256":EXPECTED_FINALIZATION_LOCK,
        "prior_v4a_comparison_lock_sha256":EXPECTED_V4A_COMPARISON_LOCK,
        "base_metric_analyzer_git_blob":EXPECTED_BASE_ANALYZER_BLOB,
        "development_query_set":"H01-H24",
        "development_query_count":NQ,
        "bootstrap_replicates":REPS,
        "bootstrap_seed":SEED,
        "selection_type":"pre_registered_lexicographic_v3b_vs_v4b",
        "selection_order":[m for m,_ in SELECTION_CRITERIA],
        "final_tie_break":"retain v3b",
        "v3b_baseline_reproduced_exactly":True,
        "selected_candidate":selection["selected_candidate"],
        "decisive_metric":selection["decisive_metric"],
        "requires_new_unseen_v4_holdout":selection["requires_new_unseen_v4_holdout"],
        "judge_calls_performed_during_comparison":0,
        "model_calls_performed_during_comparison":0,
        "human_relevance_labels_used":False,
        "release_gates_calculated":False,
        "release_decision_produced":False,
        "outputs":{p.name:sha256(p) for p in output_files},
        "metadata_sha256":sha256(METADATA),
        "next_stage":selection["next_stage"],
    }
    LOCK.write_text(json.dumps(lock,indent=2)+"\n",encoding="utf-8")

    print()
    print("V3B DEVELOPMENT")
    print(
        f"Mean@10={v3a['macro_mean_relevance_at_10']:.3f} "
        f"Clear={v3a['clear_or_strong_rate_at_10']:.1%} "
        f"Irrelevant={v3a['irrelevant_rate_at_10']:.1%} "
        f"Hit@5={v3a['hit_rate_at_5_clear_or_strong']:.1%} "
        f"Top1={v3a['top1_clear_or_strong_rate']:.1%} "
        f"MRR={v3a['mrr_at_10_clear_or_strong']:.3f} "
        f"nDCG={v3a['ndcg_at_10']:.3f} "
        f"Cat={v3a['catastrophic_query_rate']:.1%}"
    )
    print("V4B DEVELOPMENT")
    print(
        f"Mean@10={v4a['macro_mean_relevance_at_10']:.3f} "
        f"Clear={v4a['clear_or_strong_rate_at_10']:.1%} "
        f"Irrelevant={v4a['irrelevant_rate_at_10']:.1%} "
        f"Hit@5={v4a['hit_rate_at_5_clear_or_strong']:.1%} "
        f"Top1={v4a['top1_clear_or_strong_rate']:.1%} "
        f"MRR={v4a['mrr_at_10_clear_or_strong']:.3f} "
        f"nDCG={v4a['ndcg_at_10']:.3f} "
        f"Cat={v4a['catastrophic_query_rate']:.1%}"
    )
    print()
    print("PRE-REGISTERED DEVELOPMENT SELECTION")
    for item in selection["evaluations_until_decision"]:
        print(
            f"{item['metric']}: V3B={item['v3b']:.6f} V4B={item['v4b']:.6f} "
            f"preferred={item['preferred_direction']} -> {item['outcome']}"
        )
    print(f"SELECTED DEVELOPMENT CANDIDATE: {selection['selected_candidate']}")
    print(f"Decisive metric: {selection['decisive_metric'] or 'exact tie -> retain V3B'}")
    print(f"New unseen V4 holdout required: {'YES' if selection['requires_new_unseen_v4_holdout'] else 'NO'}")
    print("Judge calls during comparison: 0")
    print("Model calls during comparison: 0")
    print("Human relevance labels used: NO")
    print("Release gates calculated: NO")
    print("Release decision produced: NO")
    print(f"Comparison/selection lock SHA-256: {sha256(LOCK)}")
    print("V4B DEVELOPMENT COMPARISON AND SELECTION: PASS")

if __name__ == "__main__":
    main()
