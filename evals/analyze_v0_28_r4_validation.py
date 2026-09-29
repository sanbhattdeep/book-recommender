"""Analyze the one-time independent validation run for frozen v0.28.0 r4.

The validation result is independent evidence. It may be diagnosed if the
candidate fails, but the validation set must not be rerun to improve metrics.
The independent final holdout remains locked regardless of this analyzer's
result and requires a separate reviewed authorization step.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import pandas as pd

from analyze_v0_27_development import safe_rate, confusion_matrix_no_sklearn, weighted_cohen_kappa
import run_judge_v0_28_r4_validation as validation_runner

REPO_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_CASES = validation_runner.EXPECTED_CASES


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def resolve(raw: str) -> Path:
    p = Path(raw)
    if not p.is_absolute():
        p = REPO_ROOT / p
    return p.resolve()


def nested(d: dict, path: str):
    cur = d
    for part in path.split("."):
        cur = cur[part]
    return cur


def evaluate_check(value, spec: dict) -> bool:
    op = spec["op"]
    target = spec["value"]
    if op == ">=":
        return value >= target
    if op == "<=":
        return value <= target
    if op == "==":
        return value == target
    raise ValueError(f"Unsupported validation check operator: {op}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    args = ap.parse_args()

    run = resolve(args.run)
    metadata = load_json(run / "run_metadata.json")
    expected = {
        "judge_config_version": "0.28.0",
        "evaluation_dataset_version": "4.0.0",
        "evaluation_dataset_role": validation_runner.ROLE,
        "evaluation_scope": validation_runner.SCOPE,
        "judge_behavior_frozen": True,
        "independent_validation_one_time": True,
        "facet_spec_version": "0.9.9",
        "rubric_version": "0.1.0",
    }
    bad = [
        f"{k}: {metadata.get(k)!r} != {v!r}"
        for k, v in expected.items()
        if str(metadata.get(k)) != str(v)
    ]
    if bad:
        raise ValueError("Not the frozen r4 independent-validation run:\n- " + "\n- ".join(bad))

    dataset = resolve(str(metadata["evaluation_dataset_file"]))
    if sha256(dataset) != validation_runner.EXPECTED_VALIDATION_SHA256:
        raise ValueError("Validation dataset hash no longer matches the frozen SHA.")
    if metadata.get("evaluation_dataset_sha256") != sha256(dataset):
        raise ValueError("Validation dataset hash no longer matches run metadata.")

    release = resolve(str(metadata["release_manifest_file"]))
    validation_runner.validate_release_manifest(release)
    if metadata.get("release_manifest_sha256") != sha256(release):
        raise ValueError("Release-manifest hash no longer matches run metadata.")

    protocol_path = resolve(str(metadata["validation_protocol_file"]))
    protocol = validation_runner.validate_protocol(protocol_path)
    if metadata.get("validation_protocol_sha256") != sha256(protocol_path):
        raise ValueError("Validation-protocol hash no longer matches run metadata.")

    validation_runner.validate_resume_finality(run, dataset, release, protocol_path)
    validation_runner.validate_final_holdout_untouched()

    gold = pd.read_csv(dataset, dtype={"case_id": str, "query_id": str, "isbn13": str}, encoding="utf-8-sig")
    judged = pd.read_csv(run / "judge_results.csv", dtype={"case_id": str}, encoding="utf-8")

    if len(gold) != EXPECTED_CASES:
        raise ValueError(f"Expected {EXPECTED_CASES} validation cases, found {len(gold)}.")
    if gold["case_id"].duplicated().any() or judged["case_id"].duplicated().any():
        raise ValueError("Duplicate case IDs in validation evidence.")

    cmp = gold[["case_id", "query_id", "query", "title", "human_score", "human_reason"]].merge(
        judged, on="case_id", how="inner", validate="one_to_one"
    )
    if len(cmp) != EXPECTED_CASES:
        missing = sorted(set(gold.case_id) - set(judged.case_id))
        raise ValueError(f"Validation incomplete: {len(cmp)}/{EXPECTED_CASES}; missing={missing}")

    cmp["human_score"] = cmp["human_score"].astype(int)
    cmp["judge_score"] = cmp["judge_score"].astype(int)
    cmp["signed_difference"] = cmp["judge_score"] - cmp["human_score"]
    cmp["absolute_difference"] = cmp["signed_difference"].abs()
    cmp["exact_match"] = cmp["human_score"].eq(cmp["judge_score"])
    cmp["within_one"] = cmp["absolute_difference"].le(1)

    h = cmp["human_score"]
    j = cmp["judge_score"]
    labels = [0, 1, 2, 3, 4]

    exact = float(cmp["exact_match"].mean())
    within = float(cmp["within_one"].mean())
    mae = float(cmp["absolute_difference"].mean())
    linear_kappa = weighted_cohen_kappa(h, j, labels, "linear")
    quadratic_kappa = weighted_cohen_kappa(h, j, labels, "quadratic")

    hp = h > 0
    jp = j > 0
    tp = int((hp & jp).sum())
    fp = int((~hp & jp).sum())
    fn = int((hp & ~jp).sum())
    tn = int((~hp & ~jp).sum())
    precision = safe_rate(tp, tp + fp)
    recall = safe_rate(tp, tp + fn)
    f1 = safe_rate(2 * precision * recall, precision + recall)

    severe = int((cmp["absolute_difference"] >= 2).sum())
    over = int((cmp["signed_difference"] >= 2).sum())
    under = int((cmp["signed_difference"] <= -2).sum())
    false_zero = int(((h > 0) & (j == 0)).sum())
    false_positive = int(((h == 0) & (j > 0)).sum())

    cue_severe = 0
    if "deterministic_direct_cue_count" in cmp.columns:
        cue = pd.to_numeric(cmp["deterministic_direct_cue_count"], errors="coerce").fillna(0).astype(int) > 0
        cue_severe = int((cue & (h == 0) & (j >= 2)).sum())

    metrics = {
        "within_one_agreement": within,
        "exact_agreement": exact,
        "mean_absolute_difference": mae,
        "quadratic_weighted_cohens_kappa": quadratic_kappa,
        "linear_weighted_cohens_kappa": linear_kappa,
        "error_counts": {
            "absolute_difference_ge_2": severe,
            "false_zeros": false_zero,
            "false_positive_relevance": false_positive,
            "over_promotions_by_2_or_more": over,
            "under_promotions_by_2_or_more": under,
        },
        "binary_relevance_0_vs_positive": {
            "true_positive": tp,
            "false_positive": fp,
            "false_negative": fn,
            "true_negative": tn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        },
        "deterministic_cue_severe_false_positives": cue_severe,
    }

    checks = {}
    check_observations = {}
    for name, spec in protocol["pre_registered_checks"].items():
        value = nested(metrics, spec["metric"])
        passed = evaluate_check(value, spec)
        checks[name] = bool(passed)
        check_observations[name] = {
            "metric": spec["metric"],
            "observed": value,
            "op": spec["op"],
            "threshold": spec["value"],
            "passed": bool(passed),
        }

    all_pass = all(checks.values())
    decision = "PASS_TO_FINAL_HOLDOUT_REVIEW" if all_pass else "REVIEW_STOP_FINAL_HOLDOUT"

    matrix = pd.DataFrame(
        confusion_matrix_no_sklearn(h, j, labels),
        index=[f"human_{x}" for x in labels],
        columns=[f"judge_{x}" for x in labels],
    )
    largest = cmp.sort_values(
        ["absolute_difference", "case_id"], ascending=[False, True]
    )[["case_id", "query_id", "title", "human_score", "judge_score", "signed_difference", "absolute_difference"]]

    summary = {
        "evidence_role": "independent_validation",
        "candidate_release": "v0.28.0_r4",
        "cases_compared": EXPECTED_CASES,
        "exact_agreement": exact,
        "within_one_agreement": within,
        "mean_absolute_difference": mae,
        "linear_weighted_cohens_kappa": linear_kappa,
        "quadratic_weighted_cohens_kappa": quadratic_kappa,
        "binary_relevance_0_vs_positive": metrics["binary_relevance_0_vs_positive"],
        "error_counts": metrics["error_counts"],
        "deterministic_cue_severe_false_positives": cue_severe,
        "pre_registered_checks": check_observations,
        "all_validation_checks_pass": bool(all_pass),
        "validation_decision": decision,
        "final_holdout_status_required": "LOCKED_DO_NOT_RUN",
        "release_manifest_sha256": sha256(release),
        "validation_protocol_sha256": sha256(protocol_path),
        "run_metadata": metadata,
        "methodology_note": (
            "One-time independent validation result for frozen v0.28.0 r4. "
            "Do not rerun validation to improve metrics. The final holdout remains "
            "locked and requires a separate review/authorization step."
        ),
    }

    cmp.to_csv(run / "human_vs_judge.validation.csv", index=False, encoding="utf-8")
    matrix.to_csv(run / "confusion_matrix.validation.csv", encoding="utf-8")
    largest.to_csv(run / "largest_disagreements.validation.csv", index=False, encoding="utf-8")
    summary_file = run / "validation_summary.json"
    summary_file.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lock = load_json(validation_runner.VALIDATION_LOCK)
    if lock.get("status") not in {"INDEPENDENT_VALIDATION_STARTED", "INDEPENDENT_VALIDATION_COMPLETED"} or int(lock.get("judge_run_count", -1)) != 1:
        raise RuntimeError("Validation lock is not in the expected one-time started/completed state.")

    if lock.get("status") != "INDEPENDENT_VALIDATION_COMPLETED":
        lock["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    lock.update({
        "status": "INDEPENDENT_VALIDATION_COMPLETED",
        "judge_run_count": 1,
        "finality_marker_present": True,
        "run_directory": str(run.resolve()),
        "release_candidate_id": validation_runner.EXPECTED_RELEASE_CANDIDATE_ID,
        "all_validation_checks_pass": bool(all_pass),
        "validation_decision": decision,
        "validation_summary_sha256": sha256(summary_file),
        "instruction": (
            "Validation consumed exactly once. Do not rerun it. "
            "Final holdout remains LOCKED_DO_NOT_RUN pending separate review."
        ),
    })
    validation_runner.write_json_atomic(validation_runner.VALIDATION_LOCK, lock)

    # Re-check final holdout AFTER writing all validation outputs.
    validation_runner.validate_final_holdout_untouched()

    print("Frozen v0.28.0 r4 independent validation summary")
    print("-----------------------------------------------")
    print(f"Cases compared:            {EXPECTED_CASES}/{EXPECTED_CASES}")
    print(f"Exact agreement:           {exact:.1%}")
    print(f"Within ±1 agreement:       {within:.1%}")
    print(f"Mean absolute difference:  {mae:.3f}")
    print(f"Linear weighted kappa:     {linear_kappa:.3f}")
    print(f"Quadratic weighted kappa:  {quadratic_kappa:.3f}")
    print()
    print("Binary relevance (0 vs >0)")
    print("--------------------------")
    print(f"TP={tp} FP={fp} FN={fn} TN={tn}")
    print(f"Precision={precision:.1%} Recall={recall:.1%} F1={f1:.1%}")
    print()
    print("Error counts")
    print("------------")
    print(f"False zeros:               {false_zero}")
    print(f"False-positive relevance:  {false_positive}")
    print(f"|difference| >= 2:         {severe}")
    print(f"Over-promotions >= 2:      {over}")
    print(f"Under-promotions >= 2:     {under}")
    print()
    print("Pre-registered validation checks")
    print("--------------------------------")
    for name, passed in checks.items():
        print(f"{'PASS' if passed else 'FAIL':4}  {name}")
    print()
    print("Independent validation decision: " + decision)
    print("Validation lock transitioned to INDEPENDENT_VALIDATION_COMPLETED; judge_run_count remains 1.")
    print("Independent final holdout remains LOCKED_DO_NOT_RUN; judge_run_count remains 0.")
    print()
    print("Largest disagreements")
    print("---------------------")
    print(largest.head(15).to_string(index=False))


if __name__ == "__main__":
    main()
