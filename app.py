import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st


SNAPSHOT_PATH = Path("data/performance_snapshot.json")


def load_snapshot(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def format_kpi(value: float, style: str) -> str:
    if style in {"pct", "pct_monthly"}:
        return f"{value * 100:.2f}%"
    if style == "turnover_pct":
        return f"{value * 100:.2f}%"
    if style == "ratio":
        return f"{value:.2f}"
    if style == "weeks":
        return f"{int(value)} wks"
    return str(value)


def build_cards(kpis: dict, kpis_since_coverage: dict | None) -> list[dict]:
    sc = kpis_since_coverage or kpis
    return [
        {"label": "CAGR", "value": format_kpi(sc["cagr"], "pct"), "color": "green"},
        {"label": "SHARPE RATIO", "value": format_kpi(sc["sharpe"], "ratio"), "color": "green"},
        {"label": "SORTINO RATIO", "value": format_kpi(sc["sortino"], "ratio"), "color": "green"},
        {"label": "CALMAR RATIO", "value": format_kpi(sc["calmar"], "ratio"), "color": "green"},
        {"label": "MAX DRAWDOWN", "value": format_kpi(sc["max_drawdown"], "pct"), "color": "red"},
        {
            "label": "MAX DD DURATION",
            "value": format_kpi(sc["max_dd_duration_weeks"], "weeks"),
            "color": "default",
        },
        {"label": "CURRENT DRAWDOWN", "value": format_kpi(sc["current_drawdown"], "pct"), "color": "red"},
        {"label": "ANNUALIZED VOL", "value": format_kpi(sc["annualized_vol"], "pct"), "color": "default"},
        {"label": "BETA", "value": format_kpi(sc["beta"], "ratio"), "color": "blue"},
        {"label": "ALPHA", "value": format_kpi(sc["alpha"], "pct"), "color": "green"},
        {"label": "INFO RATIO", "value": format_kpi(sc["information_ratio"], "ratio"), "color": "green"},
        {"label": "WIN RATE", "value": format_kpi(sc["win_rate"], "pct"), "color": "green"},
        {
            "label": "AVG MONTHLY RETURN",
            "value": format_kpi(sc["avg_monthly_return"], "pct_monthly"),
            "color": "green",
        },
        {"label": "BEST MONTH", "value": format_kpi(sc["best_month"], "pct_monthly"), "color": "green"},
        {"label": "WORST MONTH", "value": format_kpi(sc["worst_month"], "pct_monthly"), "color": "red"},
        {"label": "AVG TURNOVER", "value": format_kpi(sc["avg_turnover"], "turnover_pct"), "color": "default"},
    ]


def rebase_since_coverage(df: pd.DataFrame, coverage_start_date: str | None) -> pd.DataFrame:
    if not coverage_start_date:
        return df

    filtered = df[df["date"] >= pd.Timestamp(coverage_start_date)].copy()
    if filtered.empty:
        return df

    base_portfolio = filtered.iloc[0]["portfolio"]
    base_benchmark = filtered.iloc[0]["benchmark"]
    filtered["portfolio"] = (1 + filtered["portfolio"]) / (1 + base_portfolio) - 1

    if pd.notna(base_benchmark):
        filtered["benchmark"] = filtered["benchmark"].apply(
            lambda x: ((1 + x) / (1 + base_benchmark) - 1) if pd.notna(x) else None
        )
    return filtered


def build_figure(chart_data: list[dict], coverage_start_date: str | None, since_coverage: bool) -> go.Figure:
    df = pd.DataFrame(chart_data)
    df["date"] = pd.to_datetime(df["date"])
    if since_coverage:
        df = rebase_since_coverage(df, coverage_start_date)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["date"],
            y=df["benchmark"],
            mode="lines",
            name="S&P 500",
            line={"color": "#6b7280", "width": 2, "dash": "dash"},
            hovertemplate="%{x|%Y-%m-%d}<br>S&P 500: %{y:.1%}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df["date"],
            y=df["portfolio"],
            mode="lines",
            name="Portfolio",
            line={"color": "#3b82f6", "width": 3},
            hovertemplate="%{x|%Y-%m-%d}<br>Portfolio: %{y:.1%}<extra></extra>",
        )
    )

    if not since_coverage and coverage_start_date:
        cov = pd.Timestamp(coverage_start_date)
        fig.add_vline(x=cov, line_color="#f59e0b", line_width=1.5, line_dash="dash")
        fig.add_annotation(
            x=cov,
            y=1.03,
            yref="paper",
            text="Full coverage",
            showarrow=False,
            font={"size": 11, "color": "#f59e0b"},
        )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=360,
        margin={"l": 52, "r": 20, "t": 8, "b": 58},
        hovermode="x unified",
        legend={
            "orientation": "h",
            "x": 0.5,
            "xanchor": "center",
            "y": -0.1,
            "font": {"size": 12, "color": "#9ca3af"},
        },
    )
    fig.update_xaxes(
        tickformat="%Y-%m",
        dtick="M2",
        tickfont={"size": 11, "color": "#9ca3af"},
        showgrid=True,
        gridcolor="#334155",
        gridwidth=1,
        zeroline=False,
        linecolor="#4b5563",
    )
    fig.update_yaxes(
        tickformat=".1%",
        tickfont={"size": 11, "color": "#9ca3af"},
        showgrid=True,
        gridcolor="#334155",
        gridwidth=1,
        zeroline=False,
        range=[-0.2, 0.6],
        linecolor="#4b5563",
    )
    return fig


def render_styles(button_active: bool) -> None:
    button_bg = "rgba(245, 158, 11, 0.18)" if button_active else "rgba(55, 65, 81, 0.5)"
    button_color = "#fbbf24" if button_active else "#9ca3af"
    button_border = "rgba(245, 158, 11, 0.45)" if button_active else "rgba(107, 114, 128, 0.5)"

    st.markdown(
        f"""
<style>
:root {{
  --app-bg: #030b1d;
  --panel-bg: #111b31;
  --panel-border: rgba(55, 65, 81, 0.45);
  --text-muted: #9ca3af;
  --text-default: #e5e7eb;
  --green: #34d399;
  --red: #f87171;
  --blue: #60a5fa;
}}

.stApp {{
  background: linear-gradient(180deg, #040c20 0%, #020917 100%);
}}

.block-container {{
  max-width: 1180px;
  padding-top: 0.85rem;
  padding-bottom: 1.0rem;
}}

div[data-testid="stHorizontalBlock"] {{
  gap: 0.75rem;
}}

.kpi-card {{
  background: linear-gradient(160deg, #141f37 0%, #101a2e 100%);
  border: 1px solid var(--panel-border);
  border-radius: 12px;
  padding: 11px 16px 10px 16px;
  min-height: 92px;
}}

.kpi-label {{
  color: var(--text-muted);
  font-size: 1.02rem;
  letter-spacing: 0.02em;
  text-transform: uppercase;
  margin-bottom: 6px;
  line-height: 1.1;
}}

.kpi-value {{
  font-size: 1.98rem;
  line-height: 1.05;
  font-weight: 600;
}}

.kpi-green {{ color: var(--green); }}
.kpi-red {{ color: var(--red); }}
.kpi-blue {{ color: var(--blue); }}
.kpi-default {{ color: var(--text-default); }}

.summary-line {{
  color: #6b7280;
  font-size: 0.84rem;
  margin: 12px 0 10px 0;
}}

.chart-title {{
  color: #d1d5db;
  font-size: 1.02rem;
  font-weight: 600;
  margin-top: 3px;
  margin-bottom: 6px;
}}

div[data-testid="stButton"] > button {{
  background: {button_bg};
  color: {button_color};
  border: 1px solid {button_border};
  border-radius: 7px;
  font-size: 0.74rem;
  font-weight: 600;
  padding: 0.26rem 0.65rem;
}}

div[data-testid="stButton"] > button:hover {{
  color: #fbbf24;
  border-color: rgba(245, 158, 11, 0.6);
}}

div[data-testid="stPlotlyChart"] {{
  background: linear-gradient(180deg, #111c33 0%, #0e1830 100%);
  border: 1px solid var(--panel-border);
  border-radius: 12px;
  padding: 6px 8px 0 8px;
}}

@media (max-width: 1200px) {{
  .block-container {{ max-width: 100%; }}
  .kpi-label {{ font-size: 0.88rem; }}
  .kpi-value {{ font-size: 1.55rem; }}
  .summary-line {{ font-size: 0.78rem; }}
  .chart-title {{ font-size: 0.95rem; }}
  div[data-testid="stButton"] > button {{ font-size: 0.72rem; }}
}}
</style>
""",
        unsafe_allow_html=True,
    )


def render_card(card: dict) -> None:
    st.markdown(
        f"""
<div class="kpi-card">
  <div class="kpi-label">{card["label"]}</div>
  <div class="kpi-value kpi-{card["color"]}">{card["value"]}</div>
</div>
""",
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(page_title="Portfolio Performance Showcase", layout="wide")
    if "since_coverage" not in st.session_state:
        st.session_state.since_coverage = True

    if not SNAPSHOT_PATH.exists():
        st.error(f"Missing snapshot file: {SNAPSHOT_PATH}")
        st.stop()

    snapshot = load_snapshot(SNAPSHOT_PATH)
    kpis = snapshot["kpis"]
    kpis_since_coverage = snapshot["kpis_since_coverage"]
    coverage_start_date = snapshot["coverage_start_date"]
    cards = build_cards(kpis, kpis_since_coverage)

    render_styles(button_active=st.session_state.since_coverage)

    for idx in range(0, len(cards), 4):
        cols = st.columns(4)
        for col, card in zip(cols, cards[idx : idx + 4]):
            with col:
                render_card(card)

    st.markdown(
        f"""
<div class="summary-line">
  {kpis["start_date"]} to {kpis["end_date"]} &nbsp;&nbsp;&nbsp;
  {kpis["n_weeks"]} weeks &nbsp;&nbsp;&nbsp;
  {kpis["n_months"]} months &nbsp;&nbsp;&nbsp;
  Total Return: {kpis["total_return"] * 100:.2f}% &nbsp;&nbsp;&nbsp;
  Full coverage from: {coverage_start_date}
</div>
""",
        unsafe_allow_html=True,
    )

    title_col, button_col = st.columns([8, 1.2], vertical_alignment="center")
    with title_col:
        st.markdown('<div class="chart-title">Cumulative Returns vs S&amp;P 500</div>', unsafe_allow_html=True)
    with button_col:
        label = "Since coverage" if st.session_state.since_coverage else "Full period"
        if st.button(label, use_container_width=True):
            st.session_state.since_coverage = not st.session_state.since_coverage
            st.rerun()

    figure = build_figure(
        chart_data=snapshot["charts"]["cumulative_returns"],
        coverage_start_date=coverage_start_date,
        since_coverage=st.session_state.since_coverage,
    )
    st.plotly_chart(figure, use_container_width=True, config={"displayModeBar": False})


if __name__ == "__main__":
    main()
