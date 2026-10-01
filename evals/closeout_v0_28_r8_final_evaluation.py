from __future__ import annotations
from datetime import datetime, timezone
import csv, hashlib, json
from pathlib import Path

R = Path(__file__).resolve().parents[1]
E = R / "evals"
D = E / "datasets"
REL = E / "releases"
RUN_ROOT = E / "runs" / "semantic_relevance_v0_28_final_holdout"

RELEASE = REL / "semantic_relevance_v0.28.0_r8_release_candidate.json"
PROTOCOL = REL / "semantic_relevance_v0.28.0_r8_final_holdout_protocol.v1.0.0.json"
FINAL_LOCK = D / "semantic_relevance_final_holdout_lock.v4.0.0.json"
VALIDATION_LOCK = D / "semantic_relevance_validation_lock.v4.0.0.json"
FINAL_HOLDOUT = D / "semantic_relevance_final_holdout.v4.0.0.DO_NOT_RUN_YET.csv"
DEVELOPMENT = D / "semantic_relevance_v0.28_development.v5.0.0.csv"
MARKER = RUN_ROOT / "FINAL_HOLDOUT_STARTED.json"
CLOSEOUT = REL / "semantic_relevance_v0.28.0_r8_final_evaluation_closeout.json"

EXPECTED = {
    RELEASE: "c25759ccaba3a3258552a5c138170821b3a9afd48acb968859a5943b4fc91cab",
    PROTOCOL: "7aa066815dd7c084be20773daec70c09981feb6c646061b658ed4fea1fca87f8",
    FINAL_HOLDOUT: "0c7b55065629b6c1fbdb3f0d530cf8bdc8e15011b0d213ab4be6259b95a43e85",
    DEVELOPMENT: "21e53ee575fab0f99e69ffabee705f8afcc843878a33d640d6ada795be1ea520",
}
SEMANTIC = {
    "evals/semantic_relevance_facet_judge.py": "e4efec54d03601e3016b376741c7f14fc1259ab3d63d1fdff75925cc2bfb43c1",
    "evals/semantic_relevance_facet_scoring.py": "8447a75aee29d1ebf34147e625a7fe8c1f8ccf4b409029e917ec4c47285e8808",
    "evals/facets/semantic_relevance/semantic_relevance_query_facets.v0.9.13.json": "0952d80ebed8fbdebfb3b625f5648be9ccc08148d87d6753adc83d010953e289",
    "evals/judge_configs/semantic_relevance_judge.v0.28.0-r8.json": "bdc4157b457e3453ff94b860cb45022a7c7c83f311432ef9841a1c1fe5a9bb8d",
    "evals/datasets/semantic_relevance_v0.28_regression_manifest.v1.7.0.json": "da8d7ffe0f9131dd5ab976a0af72d7c478ee592b4cf1b495e105e483d908c585",
    "evals/rubrics/semantic_relevance/semantic_relevance_rubric.v0.1.0.json": "658fe8ef49f1b73e4d6d6bdb143ab4d43a040ed7ea347fc2d4a41a3051ee7d2e",
}
FAILED = {
    "within_one_ge_0_95",
    "quadratic_kappa_ge_0_80",
    "linear_kappa_ge_0_60",
    "absolute_difference_ge_2_le_1_case",
    "binary_precision_ge_0_90",
}
PASSED = {
    "exact_ge_0_50",
    "binary_recall_ge_0_80",
    "deterministic_cue_severe_false_positives_eq_0",
}
SEVERE = {"U4_Q03_T01","U4_Q03_T02","U4_Q04_T02","U4_Q06_T01","U4_Q09_T04"}

def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()

def load(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8-sig"))

def require(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)
    print("PASS ", msg)

def rel(p: Path) -> str:
    try:
        return str(p.resolve().relative_to(R.resolve()))
    except ValueError:
        return str(p.resolve())

def main() -> None:
    print("v0.28.0 r8 final evaluation closeout")
    print("-----------------------------------")
    print("JUDGE EXECUTION: DISABLED")
    print("SEMANTIC CHANGES: NONE")

    for p, expected in EXPECTED.items():
        require(p.exists(), f"frozen evidence exists: {p.relative_to(R)}")
        require(sha(p) == expected, f"frozen hash pinned: {p.relative_to(R)}")
    for rp, expected in SEMANTIC.items():
        p = R / rp
        require(p.exists(), f"semantic artifact exists: {rp}")
        require(sha(p) == expected, f"semantic hash pinned: {rp}")

    release = load(RELEASE)
    require(release.get("release_candidate") == "semantic_relevance_v0.28.0-r8",
            "release candidate identity = semantic_relevance_v0.28.0-r8")
    require(release.get("release_status") == "FROZEN_BEFORE_FINAL_HOLDOUT",
            "release candidate was frozen before final holdout")

    vlock = load(VALIDATION_LOCK)
    require(vlock.get("status") == "INDEPENDENT_VALIDATION_COMPLETED",
            "historical r4 validation remains completed")
    require(int(vlock.get("judge_run_count", -1)) == 1,
            "historical r4 validation judge_run_count remains 1")
    require(vlock.get("validation_decision") == "REVIEW_STOP_FINAL_HOLDOUT",
            "historical r4 validation failure decision preserved")

    lock = load(FINAL_LOCK)
    require(lock.get("evidence_role") == "independent_final_holdout",
            "final-holdout lock role correct")
    require(lock.get("status") == "FINAL_HOLDOUT_COMPLETED",
            "final holdout status = FINAL_HOLDOUT_COMPLETED")
    require(int(lock.get("judge_run_count", -1)) == 1,
            "final holdout judge_run_count = 1")
    require(lock.get("finality_marker_present") is True,
            "final holdout lock records finality marker")
    require(lock.get("final_holdout_decision") == "FINAL_HOLDOUT_REVIEW",
            "final holdout decision = FINAL_HOLDOUT_REVIEW")
    require(lock.get("all_final_holdout_checks_pass") is False,
            "final holdout records not all checks passed")

    require(MARKER.exists(), "finality marker exists")
    marker = load(MARKER)
    require(marker.get("completed") is True, "finality marker records completed=true")
    require(marker.get("release_candidate_id") == "semantic_relevance_v0.28.0-r8",
            "finality marker release candidate identity correct")
    require(marker.get("final_holdout_decision") == "FINAL_HOLDOUT_REVIEW",
            "finality marker records FINAL_HOLDOUT_REVIEW")

    run_dir = Path(str(lock["run_directory"])).resolve()
    require(run_dir.exists(), f"canonical final-holdout run exists: {run_dir}")
    require(Path(str(marker["run_directory"])).resolve() == run_dir,
            "lock and marker point to same canonical final run")

    summary_path = run_dir / "final_holdout_summary.json"
    results_path = run_dir / "judge_results.csv"
    metadata_path = run_dir / "run_metadata.json"
    hvj_path = run_dir / "human_vs_judge.final_holdout.csv"
    confusion_path = run_dir / "confusion_matrix.final_holdout.csv"
    largest_path = run_dir / "largest_disagreements.final_holdout.csv"
    for p in [summary_path,results_path,metadata_path,hvj_path,confusion_path,largest_path]:
        require(p.exists(), f"final evidence artifact exists: {p.name}")

    with results_path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    require(len(rows) == 30, "canonical final run contains exactly 30 judge rows")
    require(len({r["case_id"] for r in rows}) == 30,
            "canonical final run case IDs are unique")

    summary = load(summary_path)
    require(summary.get("evidence_role") == "independent_final_holdout",
            "summary evidence role = independent_final_holdout")
    require(summary.get("candidate_release") == "semantic_relevance_v0.28.0-r8",
            "summary candidate release identity correct")
    require(int(summary.get("cases_compared", -1)) == 30,
            "summary compares all 30 final-holdout cases")
    require(summary.get("final_holdout_decision") == "FINAL_HOLDOUT_REVIEW",
            "summary decision = FINAL_HOLDOUT_REVIEW")
    require(summary.get("all_final_holdout_checks_pass") is False,
            "summary records final qualification failure")

    require(abs(float(summary["exact_agreement"]) - 20/30) < 1e-12,
            "exact agreement = 20/30 = 66.7%")
    require(abs(float(summary["within_one_agreement"]) - 25/30) < 1e-12,
            "within ±1 agreement = 25/30 = 83.3%")
    require(abs(float(summary["mean_absolute_difference"]) - 0.5) < 1e-12,
            "mean absolute difference = 0.500")
    require(abs(float(summary["linear_weighted_kappa"]) - 0.575) < 0.001,
            "linear weighted kappa ~= 0.575")
    require(abs(float(summary["quadratic_weighted_kappa"]) - 0.702) < 0.001,
            "quadratic weighted kappa ~= 0.702")

    b = summary["binary_relevance_0_vs_positive"]
    require((b["true_positive"],b["false_positive"],b["false_negative"],b["true_negative"]) == (9,3,2,16),
            "binary confusion = TP9 FP3 FN2 TN16")
    require(abs(float(b["precision"]) - 0.75) < 1e-12, "binary precision = 75.0%")
    require(abs(float(b["recall"]) - 9/11) < 1e-12, "binary recall = 81.8%")

    e = summary["error_counts"]
    require(int(e["absolute_difference_ge_2"]) == 5,
            "final holdout has exactly five severe disagreements")
    require(int(e["over_promotions_by_2_or_more"]) == 3,
            "final holdout severe over-promotions = 3")
    require(int(e["under_promotions_by_2_or_more"]) == 2,
            "final holdout severe under-promotions = 2")

    checks = summary["preregistered_checks"]
    failed = {k for k,v in checks.items() if not v["passed"]}
    passed = {k for k,v in checks.items() if v["passed"]}
    require(failed == FAILED, "exact five preregistered final-holdout checks failed")
    require(passed == PASSED, "exact three preregistered final-holdout checks passed")

    import pandas as pd
    hvj = pd.read_csv(hvj_path, dtype={"case_id": str}, encoding="utf-8-sig")
    severe = set(hvj.loc[hvj["absolute_difference"].astype(int) >= 2, "case_id"].astype(str))
    require(severe == SEVERE, "exact five severe final-holdout case IDs pinned")

    require(lock.get("final_holdout_summary_sha256") == sha(summary_path),
            "lock pins final_holdout_summary SHA")
    require(marker.get("final_holdout_summary_sha256") == sha(summary_path),
            "marker pins final_holdout_summary SHA")

    hashes = {
        "final_holdout_dataset": sha(FINAL_HOLDOUT),
        "final_holdout_lock": sha(FINAL_LOCK),
        "finality_marker": sha(MARKER),
        "judge_results": sha(results_path),
        "run_metadata": sha(metadata_path),
        "summary": sha(summary_path),
        "human_vs_judge": sha(hvj_path),
        "confusion_matrix": sha(confusion_path),
        "largest_disagreements": sha(largest_path),
        "release_candidate_manifest": sha(RELEASE),
        "final_holdout_protocol": sha(PROTOCOL),
    }

    out = {
        "schema_version": "1.0.0",
        "lineage": "semantic_relevance_v0.28.0",
        "candidate": "semantic_relevance_v0.28.0-r8",
        "closeout_status": "EVALUATED_NOT_FINAL_QUALIFIED",
        "final_holdout_decision": "FINAL_HOLDOUT_REVIEW",
        "closed_at_utc": datetime.now(timezone.utc).isoformat(),
        "semantic_identity": {
            "judge_config_version": "0.28.0-r8",
            "facet_spec_version": "0.9.13",
            "rubric_version": "0.1.0",
            "regression_manifest_version": "1.7.0",
            "development_dataset_version": "5.0.0",
        },
        "qualification_history": {
            "targeted_regression": "PASS",
            "full_210_consumed_development": "PASS",
            "stability_20_cases_x3": "PASS",
            "release_candidate_freeze": "PASS",
            "final_holdout_preflight": "PASS",
            "one_time_independent_final_holdout": "REVIEW",
        },
        "final_holdout_result": {
            "run_directory": rel(run_dir),
            "cases": 30,
            "exact_agreement": summary["exact_agreement"],
            "within_one_agreement": summary["within_one_agreement"],
            "mean_absolute_difference": summary["mean_absolute_difference"],
            "linear_weighted_kappa": summary["linear_weighted_kappa"],
            "quadratic_weighted_kappa": summary["quadratic_weighted_kappa"],
            "binary_relevance_0_vs_positive": b,
            "error_counts": e,
            "passed_checks": sorted(passed),
            "failed_checks": sorted(failed),
            "severe_case_ids": sorted(severe),
        },
        "final_evidence_hashes": hashes,
        "evidence_boundary_for_next_lineage": {
            "next_candidate_lineage": "semantic_relevance_v0.29.0",
            "v0_28_development_210": "consumed_development_evidence",
            "r4_independent_validation_30": "consumed_development_evidence",
            "r8_final_holdout_30": "consumed_final_evidence_may_be_diagnostic_or_development_only",
            "r8_final_holdout_may_be_reused_as_independent_evidence": False,
            "r8_final_holdout_may_be_rerun_for_r8": False,
            "r8_final_holdout_may_be_relabelled_for_r8": False,
            "r8_may_be_retuned_after_final_result": False,
            "v0_29_may_use_r8_final_cases_for_diagnostics": True,
            "v0_29_may_use_r8_final_cases_for_development_regression": True,
            "v0_29_independent_qualification_requirement": (
                "Create a new unseen human-labelled independent evidence set after "
                "v0.29 development is frozen. No U4 case consumed by v0.28 may count "
                "as independent validation or final-holdout evidence for v0.29."
            ),
        },
        "methodology_note": (
            "v0.28.0-r8 passed all consumed-development qualification gates but "
            "failed five of eight preregistered gates on the one-time 30-case "
            "independent final holdout. The holdout is permanently consumed. "
            "Any repair using these cases must occur in a new candidate lineage "
            "and be qualified on newly created unseen evidence."
        ),
    }

    REL.mkdir(parents=True, exist_ok=True)
    if CLOSEOUT.exists():
        existing = load(CLOSEOUT)
        a,b2 = dict(existing),dict(out)
        a.pop("closed_at_utc",None); b2.pop("closed_at_utc",None)
        require(a == b2, "existing closeout manifest matches immutable evidence")
        print("Closeout already exists; manifest left unchanged.")
    else:
        CLOSEOUT.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print()
    print("v0.28.0 r8 FINAL EVALUATION CLOSEOUT: PASS")
    print("Closeout:", CLOSEOUT)
    print("Closeout SHA-256:", sha(CLOSEOUT))
    print("Status: EVALUATED_NOT_FINAL_QUALIFIED")
    print("Final decision: FINAL_HOLDOUT_REVIEW")
    print("Final holdout remains consumed exactly once; judge_run_count=1.")
    print("Next lineage boundary: v0.29.0 requires new independent unseen evidence.")
    print("NO JUDGE CALLS WERE MADE BY THIS CLOSEOUT TOOL.")

if __name__ == "__main__":
    main()
