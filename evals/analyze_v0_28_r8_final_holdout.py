"""Analyze the one-time independent final holdout for frozen v0.28.0 r8.

The analyzer never authorizes another judge run. Whether quality gates PASS or
REVIEW, a complete 30-case result is final evidence and the final-holdout lock
transitions permanently to FINAL_HOLDOUT_COMPLETED / judge_run_count=1.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import pandas as pd

import run_judge_v0_28_r8_final_holdout as final_runner

REPO_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_CASES = final_runner.EXPECTED_CASES


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def resolve(raw: str) -> Path:
    p = Path(raw)
    if not p.is_absolute():
        p = REPO_ROOT / p
    return p.resolve()


def safe_rate(n: int, d: int) -> float:
    return n / d if d else 0.0


def weighted_kappa(human: pd.Series, judge: pd.Series, mode: str) -> float:
    labels = [0, 1, 2, 3, 4]
    n = len(human)
    pos = {v: i for i, v in enumerate(labels)}
    k = len(labels)
    observed = [[0.0] * k for _ in range(k)]
    hc = [0.0] * k
    jc = [0.0] * k

    for h, j in zip(human.tolist(), judge.tolist()):
        hi, ji = pos[int(h)], pos[int(j)]
        observed[hi][ji] += 1.0
        hc[hi] += 1.0
        jc[ji] += 1.0

    observed_disagreement = 0.0
    expected_disagreement = 0.0
    for i in range(k):
        for j in range(k):
            distance = abs(i - j) / (k - 1)
            weight = distance if mode == "linear" else distance * distance
            observed_disagreement += weight * (observed[i][j] / n)
            expected_disagreement += weight * ((hc[i] / n) * (jc[j] / n))

    if expected_disagreement == 0:
        return 1.0 if observed_disagreement == 0 else 0.0
    return 1.0 - (observed_disagreement / expected_disagreement)


def evaluate_check(value, spec: dict) -> bool:
    op = spec["operator"]
    target = spec["threshold"]
    if op == ">=":
        return value >= target
    if op == "<=":
        return value <= target
    if op == "==":
        return value == target
    raise ValueError(f"Unsupported final-holdout check operator: {op}")


def confusion_matrix(human: pd.Series, judge: pd.Series) -> pd.DataFrame:
    labels = [0, 1, 2, 3, 4]
    matrix = []
    for h in labels:
        row = []
        for j in labels:
            row.append(int(((human == h) & (judge == j)).sum()))
        matrix.append(row)
    return pd.DataFrame(
        matrix,
        index=[f"human_{x}" for x in labels],
        columns=[f"judge_{x}" for x in labels],
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    args = ap.parse_args()

    run = resolve(args.run)
    metadata = load_json(run / "run_metadata.json")

    expected = {
        "judge_config_version": "0.28.0-r8",
        "evaluation_dataset_version": "4.0.0",
        "evaluation_dataset_role": final_runner.ROLE,
        "evaluation_scope": final_runner.SCOPE,
        "judge_behavior_frozen": True,
        "independent_final_holdout_one_time": True,
        "facet_spec_version": "0.9.13",
        "rubric_version": "0.1.0",
    }
    bad = [
        f"{k}: {metadata.get(k)!r} != {v!r}"
        for k, v in expected.items()
        if str(metadata.get(k)) != str(v)
    ]
    if bad:
        raise ValueError(
            "Not the frozen r8 independent final-holdout run:\n- " + "\n- ".join(bad)
        )

    dataset = resolve(str(metadata["evaluation_dataset_file"]))
    if sha256(dataset) != final_runner.EXPECTED_HOLDOUT_SHA256:
        raise ValueError("Final-holdout dataset no longer matches the frozen SHA.")
    if metadata.get("evaluation_dataset_sha256") != sha256(dataset):
        raise ValueError("Run metadata final-holdout SHA mismatch.")

    release = resolve(str(metadata["release_manifest_file"]))
    final_runner.validate_release_manifest(release)
    if metadata.get("release_manifest_sha256") != sha256(release):
        raise ValueError("Run metadata release-manifest SHA mismatch.")

    protocol_path = resolve(str(metadata["final_holdout_protocol_file"]))
    protocol = final_runner.validate_protocol(protocol_path)
    if metadata.get("final_holdout_protocol_sha256") != sha256(protocol_path):
        raise ValueError("Run metadata final-holdout protocol SHA mismatch.")

    final_runner.validate_resume_finality(run, dataset, release, protocol_path)
    final_runner.validate_historical_validation_lock()

    gold = pd.read_csv(
        dataset,
        dtype={"case_id": str, "query_id": str, "isbn13": str},
        encoding="utf-8-sig",
    )
    judged = pd.read_csv(
        run / "judge_results.csv",
        dtype={"case_id": str},
        encoding="utf-8",
    )

    if len(gold) != EXPECTED_CASES:
        raise ValueError(f"Expected {EXPECTED_CASES} final-holdout cases, found {len(gold)}.")
    if gold["case_id"].duplicated().any() or judged["case_id"].duplicated().any():
        raise ValueError("Duplicate case IDs in final-holdout evidence.")

    cmp = gold[
        ["case_id", "query_id", "query", "title", "human_score", "human_reason"]
    ].merge(
        judged,
        on="case_id",
        how="inner",
        validate="one_to_one",
    )
    if len(cmp) != EXPECTED_CASES:
        missing = sorted(set(gold["case_id"]) - set(judged["case_id"]))
        raise ValueError(
            f"Final holdout incomplete: {len(cmp)}/{EXPECTED_CASES}; missing={missing}"
        )

    cmp["human_score"] = cmp["human_score"].astype(int)
    cmp["judge_score"] = cmp["judge_score"].astype(int)
    cmp["signed_difference"] = cmp["judge_score"] - cmp["human_score"]
    cmp["absolute_difference"] = cmp["signed_difference"].abs()
    cmp["exact_match"] = cmp["absolute_difference"] == 0
    cmp["within_one"] = cmp["absolute_difference"] <= 1

    human = cmp["human_score"]
    judge = cmp["judge_score"]

    exact = float(cmp["exact_match"].mean())
    within = float(cmp["within_one"].mean())
    mae = float(cmp["absolute_difference"].mean())
    linear_kappa = weighted_kappa(human, judge, "linear")
    quadratic_kappa = weighted_kappa(human, judge, "quadratic")

    hp = human > 0
    jp = judge > 0
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
    false_zero = int(((human > 0) & (judge == 0)).sum())
    false_positive = int(((human == 0) & (judge > 0)).sum())

    cue_severe = 0
    if "deterministic_direct_cue_count" in cmp.columns:
        cue = (
            pd.to_numeric(cmp["deterministic_direct_cue_count"], errors="coerce")
            .fillna(0)
            .astype(int)
            > 0
        )
        cue_severe = int((cue & (human == 0) & (judge >= 2)).sum())

    metric_values = {
        "within_one_agreement": within,
        "exact_agreement": exact,
        "quadratic_weighted_kappa": quadratic_kappa,
        "linear_weighted_kappa": linear_kappa,
        "absolute_difference_ge_2": severe,
        "binary_precision": precision,
        "binary_recall": recall,
        "deterministic_cue_severe_false_positives": cue_severe,
    }

    checks = {}
    observations = {}
    for name, spec in protocol["preregistered_checks"].items():
        value = metric_values[spec["metric"]]
        passed = evaluate_check(value, spec)
        checks[name] = bool(passed)
        observations[name] = {
            "metric": spec["metric"],
            "observed": value,
            "operator": spec["operator"],
            "threshold": spec["threshold"],
            "passed": bool(passed),
        }

    all_pass = all(checks.values())
    decision = (
        protocol["decision_policy"]["all_checks_pass"]
        if all_pass
        else protocol["decision_policy"]["any_check_fails"]
    )

    matrix = confusion_matrix(human, judge)
    largest = cmp.sort_values(
        ["absolute_difference", "case_id"],
        ascending=[False, True],
    )[
        [
            "case_id", "query_id", "title", "human_score", "judge_score",
            "signed_difference", "absolute_difference",
        ]
    ]

    summary = {
        "evidence_role": "independent_final_holdout",
        "candidate_release": "semantic_relevance_v0.28.0-r8",
        "cases_compared": EXPECTED_CASES,
        "exact_agreement": exact,
        "within_one_agreement": within,
        "mean_absolute_difference": mae,
        "linear_weighted_kappa": linear_kappa,
        "quadratic_weighted_kappa": quadratic_kappa,
        "binary_relevance_0_vs_positive": {
            "true_positive": tp,
            "false_positive": fp,
            "false_negative": fn,
            "true_negative": tn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        },
        "error_counts": {
            "absolute_difference_ge_2": severe,
            "false_zeros": false_zero,
            "false_positive_relevance": false_positive,
            "over_promotions_by_2_or_more": over,
            "under_promotions_by_2_or_more": under,
        },
        "deterministic_cue_severe_false_positives": cue_severe,
        "preregistered_checks": observations,
        "all_final_holdout_checks_pass": bool(all_pass),
        "final_holdout_decision": decision,
        "release_manifest_sha256": sha256(release),
        "final_holdout_protocol_sha256": sha256(protocol_path),
        "run_metadata": metadata,
        "methodology_note": (
            "One-time independent final-holdout result for frozen v0.28.0 r8. "
            "This result is final evidence whether PASS or REVIEW. Do not retune, "
            "relabel, or rerun this holdout for the same candidate."
        ),
    }

    cmp.to_csv(
        run / "human_vs_judge.final_holdout.csv",
        index=False,
        encoding="utf-8",
    )
    matrix.to_csv(
        run / "confusion_matrix.final_holdout.csv",
        encoding="utf-8",
    )
    largest.to_csv(
        run / "largest_disagreements.final_holdout.csv",
        index=False,
        encoding="utf-8",
    )
    summary_path = run / "final_holdout_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    lock = load_json(final_runner.FINAL_LOCK)
    if (
        lock.get("status") not in {"FINAL_HOLDOUT_STARTED", "FINAL_HOLDOUT_COMPLETED"}
        or int(lock.get("judge_run_count", -1)) != 1
    ):
        raise RuntimeError(
            "Final-holdout lock is not in the expected one-time started/completed state."
        )
    if Path(str(lock.get("run_directory"))).resolve() != run.resolve():
        raise RuntimeError("Final-holdout lock run directory does not match analyzed run.")

    if lock.get("status") != "FINAL_HOLDOUT_COMPLETED":
        lock["completed_at_utc"] = datetime.now(timezone.utc).isoformat()

    lock.update({
        "status": "FINAL_HOLDOUT_COMPLETED",
        "judge_run_count": 1,
        "finality_marker_present": True,
        "run_directory": str(run.resolve()),
        "release_candidate_id": final_runner.EXPECTED_RELEASE_CANDIDATE_ID,
        "all_final_holdout_checks_pass": bool(all_pass),
        "final_holdout_decision": decision,
        "final_holdout_summary_sha256": sha256(summary_path),
        "instruction": (
            "Final holdout consumed exactly once. Do not rerun, relabel, or "
            "retune this same candidate using the final-holdout result."
        ),
    })
    final_runner.write_json_atomic(final_runner.FINAL_LOCK, lock)

    marker = load_json(final_runner.FINALITY_MARKER)
    marker.update({
        "completed_at_utc": lock.get("completed_at_utc"),
        "final_holdout_decision": decision,
        "final_holdout_summary_sha256": sha256(summary_path),
        "completed": True,
    })
    final_runner.write_json_atomic(final_runner.FINALITY_MARKER, marker)

    print("Frozen v0.28.0 r8 independent final-holdout summary")
    print("---------------------------------------------------")
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
    print("Pre-registered final-holdout checks")
    print("-----------------------------------")
    for name, passed in checks.items():
        print(f"{'PASS' if passed else 'FAIL':4}  {name}")
    print()
    print("Independent final-holdout decision: " + decision)
    print("Final-holdout lock transitioned to FINAL_HOLDOUT_COMPLETED; judge_run_count remains 1.")
    print("This 30-case holdout is now consumed final evidence and must not be rerun.")
    print()
    print("Largest disagreements")
    print("---------------------")
    print(largest.head(15).to_string(index=False))


if __name__ == "__main__":
    main()
