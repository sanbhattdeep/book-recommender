from __future__ import annotations

import html
import json
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
EVALS = REPO_ROOT / "evals"
RUN_DIR = (
    EVALS / "runs/semantic_relevance_system_eval_v2/"
    "20261004T140218Z_recommendations"
)

AGGREGATE_FILE = RUN_DIR / "aggregate_metrics.json"
BOOTSTRAP_FILE = RUN_DIR / "bootstrap_confidence_intervals.json"
GATES_FILE = RUN_DIR / "release_gate_result.json"
QUERY_METRICS_FILE = RUN_DIR / "query_metrics.csv"
FAILURES_FILE = RUN_DIR / "largest_quality_failures.csv"
METADATA_FILE = RUN_DIR / "analysis_metadata.json"
PAIRWISE_FILE = RUN_DIR / "v1_vs_v2_paired_comparison.json"
DASHBOARD_FILE = RUN_DIR / "system_evaluation_dashboard.html"


def load_json(path: Path):
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8-sig"))


def esc(value) -> str:
    return html.escape(str(value))


def pct(x: float) -> str:
    return f"{float(x):.1%}"


def fmt_metric(name: str, value: float) -> str:
    if name in {
        "clear_or_strong_rate_at_10",
        "irrelevant_rate_at_10",
        "hit_rate_at_5_clear_or_strong",
        "top1_clear_or_strong_rate",
        "catastrophic_query_rate",
    }:
        return f"{value:.1%}"
    return f"{value:.3f}"


def fmt_delta(name: str, value: float) -> str:
    if name in {
        "clear_or_strong_rate_at_10",
        "irrelevant_rate_at_10",
        "hit_rate_at_5_clear_or_strong",
        "top1_clear_or_strong_rate",
        "catastrophic_query_rate",
    }:
        return f"{value * 100:+.1f} pp"
    return f"{value:+.3f}"


def svg_bar_chart(
    labels,
    values,
    *,
    max_value,
    width=920,
    row_height=30,
    title="",
    value_formatter=lambda x: f"{x:.2f}",
    threshold=None,
):
    left = 170
    right = 90
    top = 46
    bottom = 28
    chart_width = width - left - right
    height = top + bottom + row_height * len(labels)

    parts = [
        f'<svg viewBox="0 0 {width} {height}" role="img" '
        f'aria-label="{esc(title)}">',
        f'<text x="0" y="22" font-size="16" '
        f'font-weight="600">{esc(title)}</text>',
    ]

    if threshold is not None:
        x = left + (float(threshold) / max_value) * chart_width
        parts.append(
            f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{top-8}" '
            f'y2="{height-bottom}" stroke="currentColor" '
            f'stroke-dasharray="4 4" opacity="0.55"/>'
        )
        parts.append(
            f'<text x="{x+4:.1f}" y="{top-12}" font-size="11">'
            f'threshold {esc(value_formatter(threshold))}</text>'
        )

    for i, (label, value) in enumerate(zip(labels, values)):
        y = top + i * row_height
        bar_w = max(0.0, min(float(value) / max_value, 1.0)) * chart_width
        parts.append(
            f'<text x="0" y="{y+17}" font-size="12">{esc(label)}</text>'
        )
        parts.append(
            f'<rect x="{left}" y="{y+4}" width="{chart_width}" '
            f'height="16" fill="none" stroke="currentColor" opacity="0.25"/>'
        )
        parts.append(
            f'<rect x="{left}" y="{y+4}" width="{bar_w:.1f}" '
            f'height="16" fill="currentColor" opacity="0.72"/>'
        )
        parts.append(
            f'<text x="{left+chart_width+8}" y="{y+17}" font-size="12">'
            f'{esc(value_formatter(value))}</text>'
        )

    parts.append("</svg>")
    return "".join(parts)


def svg_ci_chart(rows, *, width=920, row_height=34, title="95% confidence intervals"):
    left = 220
    right = 120
    top = 52
    bottom = 32
    chart_width = width - left - right
    height = top + bottom + row_height * len(rows)

    parts = [
        f'<svg viewBox="0 0 {width} {height}" role="img" '
        f'aria-label="{esc(title)}">',
        f'<text x="0" y="22" font-size="16" '
        f'font-weight="600">{esc(title)}</text>',
    ]

    for tick in [0, 0.25, 0.5, 0.75, 1.0]:
        x = left + tick * chart_width
        parts.append(
            f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{top-10}" '
            f'y2="{height-bottom}" stroke="currentColor" opacity="0.15"/>'
        )
        parts.append(
            f'<text x="{x-12:.1f}" y="{height-8}" '
            f'font-size="10">{tick:.0%}</text>'
        )

    for i, (label, value, low, high) in enumerate(rows):
        y = top + i * row_height + 12
        x1 = left + low * chart_width
        x2 = left + high * chart_width
        xv = left + value * chart_width
        parts.append(
            f'<text x="0" y="{y+4}" font-size="12">{esc(label)}</text>'
        )
        parts.append(
            f'<line x1="{x1:.1f}" x2="{x2:.1f}" y1="{y}" y2="{y}" '
            f'stroke="currentColor" stroke-width="4" opacity="0.55"/>'
        )
        parts.append(
            f'<circle cx="{xv:.1f}" cy="{y}" r="5" fill="currentColor"/>'
        )
        parts.append(
            f'<text x="{left+chart_width+8}" y="{y+4}" font-size="11">'
            f'{value:.1%} [{low:.1%}, {high:.1%}]</text>'
        )

    parts.append("</svg>")
    return "".join(parts)


def main() -> None:
    metadata = load_json(METADATA_FILE)
    if metadata.get("status") != "SYSTEM_EVALUATION_ANALYSIS_COMPLETE":
        raise ValueError("Analysis is not complete; dashboard will not be rendered.")

    aggregate = load_json(AGGREGATE_FILE)
    bootstrap = load_json(BOOTSTRAP_FILE)
    gates = load_json(GATES_FILE)
    paired = load_json(PAIRWISE_FILE)
    query_metrics = pd.read_csv(QUERY_METRICS_FILE, encoding="utf-8")
    failures = pd.read_csv(FAILURES_FILE, encoding="utf-8")

    ci = bootstrap["confidence_intervals"]
    decision = gates["decision"]

    kpis = [
        ("Decision", decision),
        ("Macro mean relevance@10", f"{aggregate['macro_mean_relevance_at_10']:.3f} / 4"),
        (
            "Mean relevance 95% CI",
            f"[{aggregate['macro_mean_relevance_at_10_ci_lower_95']:.3f}, "
            f"{aggregate['macro_mean_relevance_at_10_ci_upper_95']:.3f}]",
        ),
        ("Clear/strong rate@10", pct(aggregate["clear_or_strong_rate_at_10"])),
        ("Irrelevant rate@10", pct(aggregate["irrelevant_rate_at_10"])),
        ("HitRate@5", pct(aggregate["hit_rate_at_5_clear_or_strong"])),
        ("Top-1 clear/strong", pct(aggregate["top1_clear_or_strong_rate"])),
        ("nDCG@10", f"{aggregate['ndcg_at_10']:.3f}"),
    ]

    rate_ci_metrics = [
        ("Clear/strong@10", "clear_or_strong_rate_at_10"),
        ("Irrelevant@10", "irrelevant_rate_at_10"),
        ("HitRate@5", "hit_rate_at_5_clear_or_strong"),
        ("Top-1 clear/strong", "top1_clear_or_strong_rate"),
        ("MRR@10", "mrr_at_10_clear_or_strong"),
        ("nDCG@10", "ndcg_at_10"),
    ]
    ci_rows = [
        (
            label,
            float(aggregate[key]),
            float(ci[key]["lower_95"]),
            float(ci[key]["upper_95"]),
        )
        for label, key in rate_ci_metrics
    ]

    query_metrics = query_metrics.sort_values("query_id", kind="stable")
    query_chart = svg_bar_chart(
        query_metrics["query_id"].astype(str).tolist(),
        query_metrics["mean_relevance_at_10"].astype(float).tolist(),
        max_value=4.0,
        title="v2 mean semantic relevance@10 by query",
        value_formatter=lambda x: f"{x:.2f}",
        threshold=2.5,
    )

    dist = aggregate["score_distribution"]
    score_chart = svg_bar_chart(
        [f"Score {s}" for s in range(5)],
        [float(dist[str(s)]) for s in range(5)],
        max_value=max(float(v) for v in dist.values()) if dist else 1.0,
        title="v2 judge score distribution across 120 recommendations",
        value_formatter=lambda x: f"{int(round(x))}",
    )

    ci_chart = svg_ci_chart(ci_rows)

    gate_rows = []
    for name, row in gates["checks"].items():
        status = "PASS" if row["passed"] else "FAIL"
        gate_rows.append(
            "<tr>"
            f"<td><strong>{status}</strong></td>"
            f"<td><code>{esc(name)}</code></td>"
            f"<td>{float(row['actual']):.4f}</td>"
            f"<td><code>{esc(row['operator'])} {esc(row['threshold'])}</code></td>"
            "</tr>"
        )

    pair_rows = []
    for metric, row in paired["metrics"].items():
        ci_delta = row["paired_bootstrap_95_ci_raw_delta"]
        pair_rows.append(
            "<tr>"
            f"<td><code>{esc(metric)}</code></td>"
            f"<td>{esc(fmt_metric(metric, row['v1']))}</td>"
            f"<td>{esc(fmt_metric(metric, row['v2']))}</td>"
            f"<td>{esc(fmt_delta(metric, row['raw_delta_v2_minus_v1']))}</td>"
            f"<td>[{esc(fmt_delta(metric, ci_delta['lower']))}, "
            f"{esc(fmt_delta(metric, ci_delta['upper']))}]</td>"
            f"<td>{row['wins']}/{row['ties']}/{row['losses']}</td>"
            "</tr>"
        )

    mcnemar_rows = []
    for metric, row in paired["metrics"].items():
        m = row.get("mcnemar")
        if not m:
            continue
        mcnemar_rows.append(
            "<tr>"
            f"<td><code>{esc(metric)}</code></td>"
            f"<td>{m['improved_queries']}</td>"
            f"<td>{m['regressed_queries']}</td>"
            f"<td>{m['discordant_pairs']}</td>"
            f"<td>{m['exact_two_sided_p_value']:.4f}</td>"
            "</tr>"
        )

    worst = query_metrics.sort_values(
        ["mean_relevance_at_10", "ndcg_at_10"],
        ascending=[True, True],
        kind="stable",
    ).head(5)
    worst_rows = []
    for _, row in worst.iterrows():
        worst_rows.append(
            "<tr>"
            f"<td>{esc(row['query_id'])}</td>"
            f"<td>{float(row['mean_relevance_at_10']):.2f}</td>"
            f"<td>{float(row['clear_or_strong_rate_at_10']):.0%}</td>"
            f"<td>{float(row['irrelevant_rate_at_10']):.0%}</td>"
            f"<td>{float(row['ndcg_at_10']):.3f}</td>"
            f"<td>{esc(row['query'])}</td>"
            "</tr>"
        )

    failure_rows = []
    for _, row in failures.head(15).iterrows():
        failure_rows.append(
            "<tr>"
            f"<td>{esc(row['case_id'])}</td>"
            f"<td>{esc(row['query_id'])}</td>"
            f"<td>{esc(row['rank'])}</td>"
            f"<td>{esc(row['title'])}</td>"
            f"<td>{esc(row['judge_score'])}</td>"
            f"<td>{esc(row['match_level'])}</td>"
            "</tr>"
        )

    kpi_html = "".join(
        f'<div class="kpi"><div class="kpi-label">{esc(label)}</div>'
        f'<div class="kpi-value">{esc(value)}</div></div>'
        for label, value in kpis
    )

    html_doc = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Semantic Relevance System Evaluation v2</title>
<style>
:root {{
  color-scheme: light dark;
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, Segoe UI, sans-serif;
}}
body {{
  margin: 0;
  padding: 28px;
  max-width: 1280px;
  margin-inline: auto;
  line-height: 1.45;
}}
header {{ margin-bottom: 24px; }}
h1, h2 {{ line-height: 1.15; }}
.subtle {{ opacity: 0.72; }}
.grid {{
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
  gap: 12px;
  margin: 18px 0 28px;
}}
.kpi {{
  border: 1px solid currentColor;
  border-radius: 10px;
  padding: 14px;
  opacity: 0.95;
}}
.kpi-label {{ font-size: 0.84rem; opacity: 0.68; margin-bottom: 8px; }}
.kpi-value {{ font-size: 1.45rem; font-weight: 700; }}
.panel {{
  border: 1px solid currentColor;
  border-radius: 10px;
  padding: 18px;
  margin: 18px 0;
  overflow-x: auto;
}}
.note {{
  border-left: 4px solid currentColor;
  padding: 10px 14px;
  opacity: 0.82;
}}
svg {{ width: 100%; min-width: 720px; height: auto; }}
table {{ border-collapse: collapse; width: 100%; font-size: 0.92rem; }}
th, td {{
  border-bottom: 1px solid currentColor;
  padding: 8px 10px;
  text-align: left;
  vertical-align: top;
}}
th {{ font-weight: 700; }}
code {{ font-family: ui-monospace, SFMono-Regular, Consolas, monospace; }}
.footer {{ margin-top: 28px; font-size: 0.85rem; opacity: 0.7; }}
</style>
</head>
<body>
<header>
<h1>Semantic Relevance System Evaluation v2</h1>
<div class="subtle">
Corrected semantic-order recommender · 12 queries · 10 recommendations/query ·
120 judged pairs · query-cluster bootstrap 95% CIs
</div>
</header>

<section class="grid">{kpi_html}</section>

<section class="panel">
{ci_chart}
</section>

<section class="panel">
<h2>v1 → v2 paired comparison</h2>
<p class="note">
The release decision above is based only on the original frozen benchmark
gates. This paired section is secondary evidence about the magnitude and
consistency of the change.
</p>
<table>
<thead>
<tr>
<th>Metric</th><th>v1</th><th>v2</th><th>Δ v2-v1</th>
<th>95% paired bootstrap CI</th><th>W/T/L</th>
</tr>
</thead>
<tbody>{''.join(pair_rows)}</tbody>
</table>
</section>

<section class="panel">
<h2>Exact McNemar diagnostics — binary query outcomes</h2>
<p class="subtle">
McNemar is retained only for HitRate@5, Top-1 clear/strong, and catastrophic
query status. Exact p-values are exploratory diagnostics, not release gates.
</p>
<table>
<thead>
<tr>
<th>Metric</th><th>Improved queries</th><th>Regressed queries</th>
<th>Discordant</th><th>Exact two-sided p</th>
</tr>
</thead>
<tbody>{''.join(mcnemar_rows)}</tbody>
</table>
</section>

<section class="panel">
{query_chart}
</section>

<section class="panel">
{score_chart}
</section>

<section class="panel">
<h2>Preregistered release gates</h2>
<table>
<thead><tr><th>Result</th><th>Gate</th><th>Actual</th><th>Requirement</th></tr></thead>
<tbody>{''.join(gate_rows)}</tbody>
</table>
</section>

<section class="panel">
<h2>Five lowest-quality v2 queries</h2>
<table>
<thead>
<tr>
<th>Query</th><th>Mean@10</th><th>Score ≥3</th><th>Score 0</th>
<th>nDCG@10</th><th>Text</th>
</tr>
</thead>
<tbody>{''.join(worst_rows)}</tbody>
</table>
</section>

<section class="panel">
<h2>Lowest-scoring v2 recommendations</h2>
<table>
<thead>
<tr>
<th>Case</th><th>Query</th><th>Rank</th><th>Title</th>
<th>Score</th><th>Level</th>
</tr>
</thead>
<tbody>{''.join(failure_rows)}</tbody>
</table>
</section>

<section class="panel">
<h2>Interpretation</h2>
<p>
The standalone 95% confidence intervals use 5,000 query-level cluster
bootstrap replicates. Each query's complete top-10 list stays together.
</p>
<p>
The v1→v2 delta intervals also resample matched query IDs, keeping v1 and v2
paired in each draw. These intervals quantify variability across benchmark
queries, not judge run-to-run stochasticity.
</p>
<p>
Release thresholds were frozen before seeing the evaluation results. The
paired comparison and McNemar diagnostics do not change the release decision.
</p>
</section>

<div class="footer">
Source artifacts: <code>judge_scores.csv</code>, <code>query_metrics.csv</code>,
<code>aggregate_metrics.json</code>, <code>bootstrap_confidence_intervals.json</code>,
<code>release_gate_result.json</code>, and
<code>v1_vs_v2_paired_comparison.json</code>.
</div>
</body>
</html>
"""

    DASHBOARD_FILE.write_text(html_doc, encoding="utf-8")

    print("VISUAL DASHBOARD: PASS")
    print(f"Dashboard: {DASHBOARD_FILE}")


if __name__ == "__main__":
    main()
