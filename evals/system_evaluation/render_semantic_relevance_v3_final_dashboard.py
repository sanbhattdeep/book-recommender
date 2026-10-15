from __future__ import annotations

import hashlib
import html
import json
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
EVALS = REPO_ROOT / "evals"
RUN_DIR = (
    EVALS
    / "runs"
    / "semantic_relevance_v3_holdout"
    / "20261007T124150Z_retrieval"
)

ANALYSIS_LOCK = RUN_DIR / "v3_final_holdout_analysis_lock.json"
V2_AGG = RUN_DIR / "v2_holdout_aggregate_metrics.json"
V3_AGG = RUN_DIR / "v3_holdout_aggregate_metrics.json"
V2_CI = RUN_DIR / "v2_holdout_bootstrap_confidence_intervals.json"
V3_CI = RUN_DIR / "v3_holdout_bootstrap_confidence_intervals.json"
V2_GATES = RUN_DIR / "v2_holdout_release_gate_result.json"
V3_GATES = RUN_DIR / "v3_holdout_release_gate_result.json"
PAIRED = RUN_DIR / "v2_vs_v3_holdout_paired_comparison.json"
DELTAS = RUN_DIR / "v2_vs_v3_holdout_query_deltas.csv"
V2_QM = RUN_DIR / "v2_holdout_query_metrics.csv"
V3_QM = RUN_DIR / "v3_holdout_query_metrics.csv"
FINAL_REPORT = RUN_DIR / "v3_final_unseen_holdout_report.md"
DASHBOARD = RUN_DIR / "v3_final_unseen_holdout_dashboard.html"

EXPECTED_ANALYSIS_LOCK_SHA256 = (
    "09b77bc0bd46b6ac88c79240657bb67f746c09959011f9bdea650d2d3bbd3a20"
)

EXPECTED_OUTPUTS = [
    "v2_holdout_query_metrics.csv",
    "v3_holdout_query_metrics.csv",
    "v2_holdout_aggregate_metrics.json",
    "v3_holdout_aggregate_metrics.json",
    "v2_holdout_bootstrap_confidence_intervals.json",
    "v3_holdout_bootstrap_confidence_intervals.json",
    "v2_holdout_release_gate_result.json",
    "v3_holdout_release_gate_result.json",
    "v2_vs_v3_holdout_query_deltas.csv",
    "v2_vs_v3_holdout_gate_transitions.csv",
    "v2_vs_v3_holdout_paired_comparison.json",
    "v3_final_unseen_holdout_report.md",
    "v3_final_holdout_analysis_metadata.json",
]

PERCENT_METRICS = {
    "clear_or_strong_rate_at_10",
    "irrelevant_rate_at_10",
    "hit_rate_at_5_clear_or_strong",
    "top1_clear_or_strong_rate",
    "catastrophic_query_rate",
}

DISPLAY_METRICS = [
    ("Mean relevance@10", "macro_mean_relevance_at_10"),
    ("Clear / strong @10", "clear_or_strong_rate_at_10"),
    ("Irrelevant @10", "irrelevant_rate_at_10"),
    ("HitRate@5", "hit_rate_at_5_clear_or_strong"),
    ("Top-1 clear / strong", "top1_clear_or_strong_rate"),
    ("MRR@10", "mrr_at_10_clear_or_strong"),
    ("nDCG@10", "ndcg_at_10"),
    ("Catastrophic query rate", "catastrophic_query_rate"),
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8-sig"))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def esc(value: Any) -> str:
    return html.escape(str(value))


def fmt_metric(name: str, value: float) -> str:
    if name in PERCENT_METRICS:
        return f"{float(value):.1%}"
    if name == "macro_mean_relevance_at_10":
        return f"{float(value):.3f} / 4"
    return f"{float(value):.3f}"


def fmt_delta(name: str, value: float) -> str:
    if name in PERCENT_METRICS:
        return f"{float(value) * 100:+.1f} pp"
    return f"{float(value):+.3f}"


def verdict_class(value: str) -> str:
    value = str(value).upper()
    if value == "PASS":
        return "pass"
    if value == "REVIEW":
        return "review"
    return "fail"


def svg_dual_bar_chart(
    labels: list[str],
    v2_values: list[float],
    v3_values: list[float],
    *,
    max_value: float,
    title: str,
    value_formatter,
    width: int = 980,
    row_height: int = 42,
) -> str:
    left = 215
    right = 155
    top = 54
    bottom = 30
    chart_width = width - left - right
    height = top + bottom + row_height * len(labels)

    parts = [
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{esc(title)}">',
        f'<text x="0" y="23" font-size="17" font-weight="700">{esc(title)}</text>',
        f'<text x="{left}" y="42" font-size="11">v2 / v3</text>',
    ]

    for i, (label, a, b) in enumerate(zip(labels, v2_values, v3_values)):
        y = top + i * row_height
        scale = chart_width / max_value if max_value else 0.0
        aw = max(0.0, min(float(a), max_value)) * scale
        bw = max(0.0, min(float(b), max_value)) * scale

        parts.append(f'<text x="0" y="{y+19}" font-size="12">{esc(label)}</text>')
        parts.append(
            f'<rect x="{left}" y="{y+4}" width="{aw:.1f}" height="12" '
            f'class="bar-v2"/>'
        )
        parts.append(
            f'<rect x="{left}" y="{y+21}" width="{bw:.1f}" height="12" '
            f'class="bar-v3"/>'
        )
        parts.append(
            f'<text x="{left+chart_width+10}" y="{y+14}" font-size="11">'
            f'v2 {esc(value_formatter(a))}</text>'
        )
        parts.append(
            f'<text x="{left+chart_width+10}" y="{y+31}" font-size="11">'
            f'v3 {esc(value_formatter(b))}</text>'
        )

    parts.append("</svg>")
    return "".join(parts)


def svg_query_delta_chart(rows: pd.DataFrame, *, width: int = 980) -> str:
    rows = rows.copy().sort_values("query_id", kind="stable")
    vals = rows["raw_delta_v3_minus_v2"].astype(float).tolist()
    labels = rows["query_id"].astype(str).tolist()

    max_abs = max(max(abs(v) for v in vals), 0.1)
    left = 80
    right = 75
    top = 54
    row_height = 26
    bottom = 28
    chart_width = width - left - right
    zero_x = left + chart_width / 2
    height = top + row_height * len(vals) + bottom
    scale = (chart_width / 2) / max_abs

    parts = [
        f'<svg viewBox="0 0 {width} {height}" role="img" '
        f'aria-label="Per-query mean relevance delta, v3 minus v2">',
        '<text x="0" y="23" font-size="17" font-weight="700">'
        'Per-query mean relevance Δ (v3 − v2)</text>',
        f'<line x1="{zero_x:.1f}" x2="{zero_x:.1f}" y1="{top-9}" '
        f'y2="{height-bottom}" class="zero-line"/>',
    ]

    for i, (label, value) in enumerate(zip(labels, vals)):
        y = top + i * row_height
        w = abs(value) * scale
        x = zero_x if value >= 0 else zero_x - w
        cls = "delta-positive" if value >= 0 else "delta-negative"
        parts.append(f'<text x="0" y="{y+14}" font-size="11">{esc(label)}</text>')
        parts.append(
            f'<rect x="{x:.1f}" y="{y+3}" width="{w:.1f}" height="13" '
            f'class="{cls}"/>'
        )
        tx = zero_x + (7 if value >= 0 else -7)
        anchor = "start" if value >= 0 else "end"
        parts.append(
            f'<text x="{tx:.1f}" y="{y+14}" font-size="10" '
            f'text-anchor="{anchor}">{value:+.2f}</text>'
        )

    parts.append("</svg>")
    return "".join(parts)


def main() -> None:
    print("Semantic relevance v3 - final visual layer")
    print("------------------------------------------")
    print("Judge execution: DISABLED")
    print("Metric recalculation: DISABLED")
    print("Consumes frozen final-analysis artifacts only")
    print()

    require(ANALYSIS_LOCK.exists(), "Final analysis lock not found.")
    actual_lock_sha = sha256(ANALYSIS_LOCK)
    require(
        actual_lock_sha == EXPECTED_ANALYSIS_LOCK_SHA256,
        f"Final analysis lock changed: expected={EXPECTED_ANALYSIS_LOCK_SHA256} "
        f"actual={actual_lock_sha}",
    )

    lock = load_json(ANALYSIS_LOCK)
    require(
        lock.get("status") == "V3_FINAL_UNSEEN_HOLDOUT_ANALYSIS_FROZEN",
        "Final analysis is not frozen.",
    )
    require(
        lock.get("paired_comparison_is_secondary") is True,
        "Paired comparison role changed.",
    )
    require(
        int(lock.get("judge_calls_performed_during_analysis", -1)) == 0,
        "Analysis lock does not record zero judge calls.",
    )

    locked_outputs = lock.get("outputs", {})
    for name in EXPECTED_OUTPUTS:
        require(name in locked_outputs, f"Final analysis lock missing output: {name}")
        path = RUN_DIR / name
        require(path.exists(), f"Frozen output missing: {name}")
        actual = sha256(path)
        require(
            actual == locked_outputs[name],
            f"Frozen analysis artifact changed: {name}",
        )

    print("PASS  frozen final analysis lock verified")
    print("PASS  all frozen analysis artifact hashes verified")

    v2_agg = load_json(V2_AGG)
    v3_agg = load_json(V3_AGG)
    v2_ci = load_json(V2_CI)
    v3_ci = load_json(V3_CI)
    v2_gates = load_json(V2_GATES)
    v3_gates = load_json(V3_GATES)
    paired = load_json(PAIRED)
    v2_qm = pd.read_csv(V2_QM, encoding="utf-8")
    v3_qm = pd.read_csv(V3_QM, encoding="utf-8")
    deltas = pd.read_csv(DELTAS, encoding="utf-8")

    require(
        paired.get("cannot_override_standalone_v3_release_decision") is True,
        "Paired comparison authority boundary changed.",
    )
    require(v3_gates.get("decision") == lock.get("v3_final_decision"), "v3 decision mismatch.")
    require(v2_gates.get("decision") == lock.get("v2_decision"), "v2 decision mismatch.")

    v2_decision = str(v2_gates["decision"])
    v3_decision = str(v3_gates["decision"])
    v3_passed = int(v3_gates["passed_gate_count"])
    v2_passed = int(v2_gates["passed_gate_count"])

    # All dashboard numbers below are presentation-only reads from frozen artifacts.
    kpis = [
        ("Final v3 decision", v3_decision, verdict_class(v3_decision)),
        ("v3 gates", f"{v3_passed}/8", verdict_class(v3_decision)),
        (
            "v3 mean relevance@10",
            f"{float(v3_agg['macro_mean_relevance_at_10']):.3f} / 4",
            "",
        ),
        (
            "v3 clear / strong",
            f"{float(v3_agg['clear_or_strong_rate_at_10']):.1%}",
            "",
        ),
        (
            "v3 irrelevant",
            f"{float(v3_agg['irrelevant_rate_at_10']):.1%}",
            "",
        ),
        (
            "v3 catastrophic queries",
            f"{float(v3_agg['catastrophic_query_rate']):.1%}",
            "",
        ),
    ]

    kpi_html = "".join(
        f'<div class="kpi {esc(cls)}">'
        f'<div class="kpi-label">{esc(label)}</div>'
        f'<div class="kpi-value">{esc(value)}</div></div>'
        for label, value, cls in kpis
    )

    metric_rows = []
    for label, metric in DISPLAY_METRICS:
        v2 = float(v2_agg[metric])
        v3 = float(v3_agg[metric])
        metric_rows.append(
            "<tr>"
            f"<td>{esc(label)}</td>"
            f"<td>{esc(fmt_metric(metric, v2))}</td>"
            f"<td>{esc(fmt_metric(metric, v3))}</td>"
            f"<td>{esc(fmt_delta(metric, v3-v2))}</td>"
            "</tr>"
        )

    # CI display, directly from frozen bootstrap artifacts.
    v2_ci_map = v2_ci.get("confidence_intervals", v2_ci)
    v3_ci_map = v3_ci.get("confidence_intervals", v3_ci)
    ci_rows = []
    for label, metric in DISPLAY_METRICS[:-1]:
        a = v2_ci_map[metric]
        b = v3_ci_map[metric]
        ci_rows.append(
            "<tr>"
            f"<td>{esc(label)}</td>"
            f"<td>{esc(fmt_metric(metric, v2_agg[metric]))} "
            f"[{esc(fmt_metric(metric, a['lower_95']))}, "
            f"{esc(fmt_metric(metric, a['upper_95']))}]</td>"
            f"<td>{esc(fmt_metric(metric, v3_agg[metric]))} "
            f"[{esc(fmt_metric(metric, b['lower_95']))}, "
            f"{esc(fmt_metric(metric, b['upper_95']))}]</td>"
            "</tr>"
        )

    pair_rows = []
    for metric, row in paired["metrics"].items():
        low = row.get("paired_ci_lower")
        high = row.get("paired_ci_upper")
        if low is None or high is None:
            nested = row.get("paired_bootstrap_95_ci_raw_delta", {})
            low = nested.get("lower")
            high = nested.get("upper")
        pair_rows.append(
            "<tr>"
            f"<td><code>{esc(metric)}</code></td>"
            f"<td>{esc(fmt_metric(metric, row['v2']))}</td>"
            f"<td>{esc(fmt_metric(metric, row['v3']))}</td>"
            f"<td>{esc(fmt_delta(metric, row['raw_delta_v3_minus_v2']))}</td>"
            f"<td>[{esc(fmt_delta(metric, low))}, "
            f"{esc(fmt_delta(metric, high))}]</td>"
            f"<td>{int(row['wins'])}/{int(row['ties'])}/{int(row['losses'])}</td>"
            "</tr>"
        )

    gate_rows = []
    for name, v3row in v3_gates["checks"].items():
        v2row = v2_gates["checks"][name]
        if not bool(v2row["passed"]) and bool(v3row["passed"]):
            transition = "FAIL → PASS"
        elif bool(v2row["passed"]) and bool(v3row["passed"]):
            transition = "PASS → PASS"
        elif not bool(v2row["passed"]) and not bool(v3row["passed"]):
            transition = "FAIL → FAIL"
        else:
            transition = "PASS → FAIL"
        gate_rows.append(
            "<tr>"
            f"<td><code>{esc(name)}</code></td>"
            f"<td>{'PASS' if v2row['passed'] else 'FAIL'}</td>"
            f"<td>{'PASS' if v3row['passed'] else 'FAIL'}</td>"
            f"<td>{esc(transition)}</td>"
            f"<td>{float(v3row['actual']):.4f}</td>"
            f"<td><code>{esc(v3row['operator'])} {esc(v3row['threshold'])}</code></td>"
            "</tr>"
        )

    # Query-level mean relevance comparison.
    q = v2_qm[
        ["query_id", "query", "stratum", "mean_relevance_at_10"]
    ].merge(
        v3_qm[["query_id", "mean_relevance_at_10"]],
        on="query_id",
        suffixes=("_v2", "_v3"),
        validate="one_to_one",
    )
    q = q.sort_values("query_id", kind="stable")

    query_table_rows = []
    for _, row in q.iterrows():
        delta = float(row["mean_relevance_at_10_v3"]) - float(
            row["mean_relevance_at_10_v2"]
        )
        query_table_rows.append(
            "<tr>"
            f"<td>{esc(row['query_id'])}</td>"
            f"<td>{esc(row['stratum'])}</td>"
            f"<td>{float(row['mean_relevance_at_10_v2']):.2f}</td>"
            f"<td>{float(row['mean_relevance_at_10_v3']):.2f}</td>"
            f"<td>{delta:+.2f}</td>"
            f"<td>{esc(row['query'])}</td>"
            "</tr>"
        )

    mean_delta_rows = deltas[
        deltas["metric"] == "macro_mean_relevance_at_10"
    ][["query_id", "raw_delta_v3_minus_v2"]].copy()

    chart = svg_dual_bar_chart(
        [label for label, _ in DISPLAY_METRICS[:-1]],
        [float(v2_agg[m]) for _, m in DISPLAY_METRICS[:-1]],
        [float(v3_agg[m]) for _, m in DISPLAY_METRICS[:-1]],
        max_value=4.0,
        title="v2 vs v3 headline metrics (shared display scale)",
        value_formatter=lambda x: f"{x:.2f}",
    )
    delta_chart = svg_query_delta_chart(mean_delta_rows)

    counts = paired["gate_transitions"]["counts"]
    transition_summary = (
        f"FAIL→PASS {counts['fail_to_pass']} · "
        f"PASS→PASS {counts['pass_to_pass']} · "
        f"FAIL→FAIL {counts['fail_to_fail']} · "
        f"PASS→FAIL {counts['pass_to_fail']}"
    )

    html_doc = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Semantic Relevance v3 — Final Unseen Holdout</title>
<style>
:root {{
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, Segoe UI, sans-serif;
  color-scheme: light dark;
}}
body {{
  max-width: 1320px;
  margin: 0 auto;
  padding: 28px;
  line-height: 1.45;
}}
header {{ margin-bottom: 24px; }}
h1, h2 {{ line-height: 1.16; }}
.subtle {{ opacity: .72; }}
.grid {{
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(195px, 1fr));
  gap: 12px;
  margin: 18px 0 28px;
}}
.kpi {{
  border: 1px solid currentColor;
  border-radius: 12px;
  padding: 14px;
}}
.kpi-label {{
  font-size: .84rem;
  opacity: .68;
  margin-bottom: 8px;
}}
.kpi-value {{ font-size: 1.5rem; font-weight: 750; }}
.kpi.fail {{ border-width: 2px; }}
.kpi.pass {{ border-width: 2px; }}
.kpi.review {{ border-width: 2px; }}
.panel {{
  border: 1px solid currentColor;
  border-radius: 12px;
  padding: 18px;
  margin: 18px 0;
  overflow-x: auto;
}}
.note {{
  border-left: 4px solid currentColor;
  padding: 10px 14px;
  opacity: .84;
}}
table {{
  border-collapse: collapse;
  width: 100%;
  font-size: .9rem;
}}
th, td {{
  border-bottom: 1px solid currentColor;
  padding: 8px 10px;
  text-align: left;
  vertical-align: top;
}}
th {{ font-weight: 700; }}
code {{ font-family: ui-monospace, SFMono-Regular, Consolas, monospace; }}
svg {{ width: 100%; min-width: 760px; height: auto; }}
.bar-v2 {{ fill: currentColor; opacity: .32; }}
.bar-v3 {{ fill: currentColor; opacity: .82; }}
.delta-positive {{ fill: currentColor; opacity: .80; }}
.delta-negative {{ fill: currentColor; opacity: .38; }}
.zero-line {{ stroke: currentColor; stroke-width: 1.5; opacity: .55; }}
.footer {{ margin-top: 28px; font-size: .84rem; opacity: .7; }}
</style>
</head>
<body>
<header>
<h1>Semantic Relevance v3 — Final Unseen Holdout</h1>
<div class="subtle">
24 unseen queries · selected frozen v3b RRF reranker · 322 unique displayed
pairs judged once · 5,000-query-bootstrap analysis
</div>
</header>

<section class="grid">{kpi_html}</section>

<section class="panel">
<h2>Decision boundary</h2>
<p class="note">
The authoritative decision is the <strong>standalone v3 unseen-holdout</strong>
result: <strong>{esc(v3_decision)}</strong>. The paired v2→v3 comparison is
secondary evidence and cannot override this decision.
</p>
<p>
v2 decision: <strong>{esc(v2_decision)}</strong> ({v2_passed}/8 gates) ·
v3 decision: <strong>{esc(v3_decision)}</strong> ({v3_passed}/8 gates).
</p>
</section>

<section class="panel">
<h2>Headline comparison</h2>
<table>
<thead><tr><th>Metric</th><th>v2</th><th>v3</th><th>Raw Δ v3−v2</th></tr></thead>
<tbody>{''.join(metric_rows)}</tbody>
</table>
</section>

<section class="panel">{chart}</section>

<section class="panel">
<h2>Standalone 95% query-cluster bootstrap intervals</h2>
<table>
<thead><tr><th>Metric</th><th>v2</th><th>v3</th></tr></thead>
<tbody>{''.join(ci_rows)}</tbody>
</table>
</section>

<section class="panel">
<h2>Paired v2 → v3 evidence</h2>
<p class="note">
Delta is v3−v2. For irrelevant-rate and catastrophic-query-rate metrics,
negative raw delta is improvement. W/T/L is query-level and
improvement-oriented.
</p>
<table>
<thead>
<tr><th>Metric</th><th>v2</th><th>v3</th><th>Δ</th>
<th>95% paired CI</th><th>W/T/L</th></tr>
</thead>
<tbody>{''.join(pair_rows)}</tbody>
</table>
</section>

<section class="panel">
{delta_chart}
</section>

<section class="panel">
<h2>Frozen release-gate transitions</h2>
<p class="subtle">{esc(transition_summary)}</p>
<table>
<thead>
<tr><th>Gate</th><th>v2</th><th>v3</th><th>Transition</th>
<th>v3 actual</th><th>Requirement</th></tr>
</thead>
<tbody>{''.join(gate_rows)}</tbody>
</table>
</section>

<section class="panel">
<h2>Query-level mean relevance</h2>
<table>
<thead>
<tr><th>Query</th><th>Stratum</th><th>v2 mean</th><th>v3 mean</th>
<th>Δ</th><th>Query text</th></tr>
</thead>
<tbody>{''.join(query_table_rows)}</tbody>
</table>
</section>

<section class="panel">
<h2>Interpretation</h2>
<p>
v3 improves over v2 on several paired metrics, but the final v3 standalone
unseen-holdout result remains the release authority. The visual layer does not
recompute metrics, rerun the judge, change gates, or make a new release
decision.
</p>
<p>
All displayed values come from artifacts frozen by
<code>v3_final_holdout_analysis_lock.json</code>.
</p>
</section>

<div class="footer">
Frozen analysis lock SHA-256:
<code>{esc(EXPECTED_ANALYSIS_LOCK_SHA256)}</code>.
Dashboard generation performs zero judge calls and zero metric recalculation.
</div>
</body>
</html>
"""

    DASHBOARD.write_text(html_doc, encoding="utf-8")

    dashboard_sha = sha256(DASHBOARD)
    print()
    print("V3 FINAL VISUAL LAYER: PASS")
    print(f"Final v3 decision displayed: {v3_decision}")
    print(f"v3 gates displayed: {v3_passed}/8")
    print("Judge calls performed: 0")
    print("Metrics recalculated: NO")
    print(f"Dashboard: {DASHBOARD}")
    print(f"Dashboard SHA-256: {dashboard_sha}")


if __name__ == "__main__":
    main()
