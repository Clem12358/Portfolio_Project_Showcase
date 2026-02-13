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


def build_figure(chart_data: list[dict], coverage_start_date: str | None) -> go.Figure:
    df = pd.DataFrame(chart_data)
    df["date"] = pd.to_datetime(df["date"])
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

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=320,
        margin={"l": 44, "r": 16, "t": 6, "b": 48},
        hovermode="x unified",
        legend={
            "orientation": "h",
            "x": 0.5,
            "xanchor": "center",
            "y": -0.1,
            "font": {"size": 11, "color": "#9ca3af"},
        },
    )
    fig.update_xaxes(
        tickformat="%Y-%m",
        dtick="M2",
        tickfont={"size": 10, "color": "#9ca3af"},
        showgrid=True,
        gridcolor="#334155",
        gridwidth=1,
        zeroline=False,
        linecolor="#4b5563",
    )
    fig.update_yaxes(
        tickformat=".1%",
        tickfont={"size": 10, "color": "#9ca3af"},
        showgrid=True,
        gridcolor="#334155",
        gridwidth=1,
        zeroline=False,
        range=[-0.2, 0.6],
        linecolor="#4b5563",
    )
    return fig


def build_drawdown_figure(cumulative_chart_data: list[dict], coverage_start_date: str | None) -> go.Figure:
    df = pd.DataFrame(cumulative_chart_data)
    df["date"] = pd.to_datetime(df["date"])
    df = rebase_since_coverage(df, coverage_start_date)

    wealth = 1.0 + df["portfolio"].astype(float)
    running_peak = wealth.cummax()
    drawdown = (wealth / running_peak) - 1.0

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["date"],
            y=drawdown,
            mode="lines",
            name="Drawdown",
            line={"color": "#ef4444", "width": 2},
            fill="tozeroy",
            fillcolor="rgba(239, 68, 68, 0.24)",
            hovertemplate="%{x|%Y-%m-%d}<br>Drawdown: %{y:.1%}<extra></extra>",
        )
    )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=230,
        margin={"l": 44, "r": 16, "t": 6, "b": 28},
        hovermode="x unified",
        showlegend=False,
    )
    fig.update_xaxes(
        tickformat="%Y-%m",
        dtick="M2",
        tickfont={"size": 10, "color": "#9ca3af"},
        showgrid=True,
        gridcolor="#334155",
        gridwidth=1,
        zeroline=False,
        linecolor="#4b5563",
    )
    fig.update_yaxes(
        tickformat=".1%",
        tickfont={"size": 10, "color": "#9ca3af"},
        showgrid=True,
        gridcolor="#334155",
        gridwidth=1,
        zeroline=False,
        range=[-0.12, 0.0],
        tickvals=[-0.12, -0.09, -0.06, -0.03, 0.0],
        linecolor="#4b5563",
    )
    return fig


def build_monthly_returns_figure(monthly_chart_data: list[dict], coverage_start_date: str | None) -> go.Figure:
    df = pd.DataFrame(monthly_chart_data)
    month_cutoff = coverage_start_date[:7] if coverage_start_date else None
    if month_cutoff:
        df = df[df["month"] >= month_cutoff].copy()

    df["month_date"] = pd.to_datetime(df["month"] + "-01")
    bar_colors = ["#10b981" if v >= 0 else "#ef4444" for v in df["portfolio"]]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=df["month_date"],
            y=df["portfolio"],
            marker_color=bar_colors,
            name="Portfolio",
            hovertemplate="%{x|%Y-%m}<br>Return: %{y:.1%}<extra></extra>",
        )
    )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=260,
        margin={"l": 44, "r": 16, "t": 6, "b": 36},
        hovermode="x unified",
        showlegend=False,
        bargap=0.18,
    )
    fig.update_xaxes(
        tickformat="%Y-%m",
        dtick="M3",
        tickfont={"size": 10, "color": "#9ca3af"},
        showgrid=True,
        gridcolor="#334155",
        gridwidth=1,
        zeroline=False,
        linecolor="#4b5563",
    )
    fig.update_yaxes(
        tickformat=".1%",
        tickfont={"size": 10, "color": "#9ca3af"},
        showgrid=True,
        gridcolor="#334155",
        gridwidth=1,
        zeroline=False,
        range=[-0.055, 0.165],
        tickvals=[-0.055, 0.0, 0.055, 0.11, 0.165],
        linecolor="#4b5563",
    )
    return fig


def render_styles() -> None:
    st.markdown(
        """
<style>
:root {
  --app-bg: #030b1d;
  --panel-bg: #111b31;
  --panel-border: rgba(55, 65, 81, 0.45);
  --text-muted: #9ca3af;
  --text-default: #e5e7eb;
  --green: #34d399;
  --red: #f87171;
  --blue: #60a5fa;
}

.stApp {
  background: linear-gradient(180deg, #040c20 0%, #020917 100%);
}

.block-container {
  max-width: 1030px;
  padding-top: 6.2rem;
  padding-bottom: 0.8rem;
}

div[data-testid="stHorizontalBlock"] {
  gap: 0.62rem;
}

.kpi-card {
  background: linear-gradient(160deg, #141f37 0%, #101a2e 100%);
  border: 1px solid var(--panel-border);
  border-radius: 10px;
  padding: 9px 13px 8px 13px;
  min-height: 76px;
  margin-bottom: 6px;
}

.kpi-label {
  color: var(--text-muted);
  font-size: 0.86rem;
  letter-spacing: 0.02em;
  text-transform: uppercase;
  margin-bottom: 4px;
  line-height: 1.1;
}

.kpi-value {
  font-size: 1.66rem;
  line-height: 1.05;
  font-weight: 600;
}

.kpi-green { color: var(--green); }
.kpi-red { color: var(--red); }
.kpi-blue { color: var(--blue); }
.kpi-default { color: var(--text-default); }

.summary-line {
  color: #6b7280;
  font-size: 0.72rem;
  margin: 9px 0 8px 0;
}

.nav-kicker {
  color: #cbd5e1;
  font-size: 0.8rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  margin-top: 0.2rem;
  margin-bottom: 0.2rem;
}

.nav-help {
  color: #9ca3af;
  font-size: 0.82rem;
  margin-bottom: 0.35rem;
}

.nav-current {
  color: #d1d5db;
  font-size: 0.76rem;
  font-weight: 600;
  margin-bottom: 0.45rem;
}

.nav-banner {
  color: #dbeafe;
  background: linear-gradient(120deg, rgba(29, 78, 216, 0.26) 0%, rgba(37, 99, 235, 0.16) 100%);
  border: 1px solid rgba(96, 165, 250, 0.5);
  border-radius: 10px;
  padding: 0.55rem 0.7rem;
  font-size: 0.8rem;
  line-height: 1.35;
}

.nav-footer-title {
  color: #d1d5db;
  font-size: 0.84rem;
  font-weight: 600;
  margin-top: 0.5rem;
  margin-bottom: 0.25rem;
}

.chart-title {
  color: #d1d5db;
  font-size: 0.9rem;
  font-weight: 600;
  margin-top: 1px;
  margin-bottom: 3px;
}

div[data-testid="stPageLink"] a {
  color: #d1d5db !important;
  background: rgba(30, 41, 59, 0.45) !important;
  border: 1px solid rgba(71, 85, 105, 0.65) !important;
  border-radius: 10px !important;
  min-height: 2.6rem !important;
}

div[data-testid="stPageLink"] a * {
  color: #d1d5db !important;
  opacity: 1 !important;
  font-weight: 600 !important;
}

div[data-testid="stPageLink"] a[aria-current="page"] {
  background: rgba(59, 130, 246, 0.2) !important;
  border-color: rgba(96, 165, 250, 0.8) !important;
  box-shadow: inset 0 -2px 0 rgba(147, 197, 253, 0.95) !important;
}

div[data-testid="stPageLink"] a:hover {
  border-color: rgba(59, 130, 246, 0.75) !important;
}

div[data-testid="stPageLink"] a[aria-current="page"] * {
  color: #f8fafc !important;
}

div[data-testid="stPlotlyChart"] {
  background: linear-gradient(180deg, #111c33 0%, #0e1830 100%);
  border: 1px solid var(--panel-border);
  border-radius: 10px;
  padding: 5px 6px 0 6px;
}

div[data-testid="stButton"] button {
  width: 100%;
  min-height: 2.35rem;
  border-radius: 10px;
  border: 1px solid rgba(71, 85, 105, 0.8);
  color: #e5e7eb;
  background: rgba(17, 24, 39, 0.88);
  font-weight: 600;
  font-size: 0.82rem;
}

div[data-testid="stButton"] button:hover {
  border-color: rgba(96, 165, 250, 0.85);
  color: #f8fafc;
}

@media (max-width: 1200px) {
  .block-container { max-width: 100%; }
  .kpi-label { font-size: 0.78rem; }
  .kpi-value { font-size: 1.38rem; }
  .summary-line { font-size: 0.7rem; }
  .chart-title { font-size: 0.84rem; }
}
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


def render_navigation_links() -> None:
    if not hasattr(st, "page_link"):
        st.warning("Page navigation is unavailable in this Streamlit version.")
        return

    st.markdown('<div class="nav-kicker">Navigation</div>', unsafe_allow_html=True)
    st.markdown('<div class="nav-help">Click a tab to switch section.</div>', unsafe_allow_html=True)
    st.markdown('<div class="nav-current">Current page: Performance</div>', unsafe_allow_html=True)

    nav_left, nav_right = st.columns(2)
    with nav_left:
        st.page_link("app.py", label="Open Performance")
    with nav_right:
        st.page_link("pages/2_Portfolio_Holdings.py", label="Open Portfolio Holdings")


def render_navigation_banner() -> None:
    if st.session_state.get("nav_tip_dismissed", False):
        return

    c1, c2 = st.columns([5.0, 1.0])
    with c1:
        st.markdown(
            '<div class="nav-banner">'
            "Use the navigation tabs to switch between Performance and Portfolio Holdings."
            "</div>",
            unsafe_allow_html=True,
        )
    with c2:
        if st.button("Dismiss tip", key="dismiss_nav_tip"):
            st.session_state["nav_tip_dismissed"] = True
            st.rerun()


def main() -> None:
    st.set_page_config(
        page_title="Portfolio Performance Showcase",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    if not SNAPSHOT_PATH.exists():
        st.error(f"Missing snapshot file: {SNAPSHOT_PATH}")
        st.stop()

    snapshot = load_snapshot(SNAPSHOT_PATH)
    kpis = snapshot["kpis"]
    kpis_since_coverage = snapshot["kpis_since_coverage"]
    coverage_start_date = snapshot["coverage_start_date"]
    cards = build_cards(kpis, kpis_since_coverage)

    render_styles()
    render_navigation_banner()
    render_navigation_links()

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

    st.markdown('<div class="chart-title">Cumulative Returns vs S&amp;P 500</div>', unsafe_allow_html=True)

    figure = build_figure(
        chart_data=snapshot["charts"]["cumulative_returns"],
        coverage_start_date=coverage_start_date,
    )
    st.plotly_chart(figure, use_container_width=True, config={"displayModeBar": False})

    st.markdown('<div class="chart-title">Drawdown</div>', unsafe_allow_html=True)
    drawdown_figure = build_drawdown_figure(
        cumulative_chart_data=snapshot["charts"]["cumulative_returns"],
        coverage_start_date=coverage_start_date,
    )
    st.plotly_chart(drawdown_figure, use_container_width=True, config={"displayModeBar": False})

    st.markdown('<div class="chart-title">Monthly Returns</div>', unsafe_allow_html=True)
    monthly_figure = build_monthly_returns_figure(
        monthly_chart_data=snapshot["charts"]["monthly_returns"],
        coverage_start_date=coverage_start_date,
    )
    st.plotly_chart(monthly_figure, use_container_width=True, config={"displayModeBar": False})

    st.markdown('<div class="nav-footer-title">Next section</div>', unsafe_allow_html=True)
    st.page_link(
        "pages/2_Portfolio_Holdings.py",
        label="Next: Open Portfolio Holdings",
    )


if __name__ == "__main__":
    main()
