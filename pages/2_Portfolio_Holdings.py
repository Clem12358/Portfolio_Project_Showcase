import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mobile_styles import get_mobile_css


REPO_ROOT = Path(__file__).resolve().parents[1]
HOLDINGS_SNAPSHOT_PATH = REPO_ROOT / "data" / "holdings_snapshot.json"
PERFORMANCE_SNAPSHOT_PATH = REPO_ROOT / "data" / "performance_snapshot.json"
SYMBOL_NAME_SNAPSHOT_PATH = REPO_ROOT / "data" / "symbol_name_snapshot.json"
COVARIANCE_SNAPSHOT_PATH = REPO_ROOT / "data" / "covariance_snapshot.json"

SECTOR_COLORS = {
    "Technology": "#3b82f6",
    "Healthcare": "#10b981",
    "Financial Services": "#f59e0b",
    "Consumer Cyclical": "#ef4444",
    "Consumer Defensive": "#f97316",
    "Industrials": "#8b5cf6",
    "Energy": "#06b6d4",
    "Basic Materials": "#84cc16",
    "Communication Services": "#ec4899",
    "Real Estate": "#14b8a6",
    "Utilities": "#a78bfa",
    "Unknown": "#6b7280",
}


@st.cache_data
def load_snapshots() -> tuple[pd.DataFrame, dict, dict, dict | None]:
    holdings_payload = json.loads(HOLDINGS_SNAPSHOT_PATH.read_text(encoding="utf-8"))
    performance_payload = json.loads(PERFORMANCE_SNAPSHOT_PATH.read_text(encoding="utf-8"))
    names_payload = (
        json.loads(SYMBOL_NAME_SNAPSHOT_PATH.read_text(encoding="utf-8"))
        if SYMBOL_NAME_SNAPSHOT_PATH.exists()
        else {"mapping": {}}
    )
    covariance_payload = (
        json.loads(COVARIANCE_SNAPSHOT_PATH.read_text(encoding="utf-8"))
        if COVARIANCE_SNAPSHOT_PATH.exists()
        else None
    )

    holdings_df = pd.DataFrame(holdings_payload["rows"])
    holdings_df["as_of_date"] = pd.to_datetime(holdings_df["as_of_date"])
    holdings_df["weight"] = pd.to_numeric(holdings_df["weight"], errors="coerce").fillna(0.0)
    holdings_df["expected_return"] = pd.to_numeric(holdings_df["expected_return"], errors="coerce")
    holdings_df["unrealized_pct"] = pd.to_numeric(holdings_df.get("unrealized_pct"), errors="coerce")
    symbol_name_map = names_payload.get("mapping", {})
    holdings_df["company_name"] = holdings_df["symbol"].map(symbol_name_map).fillna(holdings_df["symbol"])

    coverage_start = performance_payload.get("coverage_start_date")
    if coverage_start:
        holdings_df = holdings_df[holdings_df["as_of_date"] >= pd.Timestamp(coverage_start)].copy()
    # No upper-date filter: show all holdings including the latest rebalance
    # even if weekly return data hasn't caught up yet

    return holdings_df, holdings_payload, performance_payload, covariance_payload


def render_styles() -> None:
    st.markdown(
        """
<style>
.stApp {
  background: linear-gradient(180deg, #040c20 0%, #020917 100%);
}

header[data-testid="stHeader"],
div[data-testid="stAppHeader"] {
  background: #000000 !important;
}

header[data-testid="stHeader"] *,
div[data-testid="stAppHeader"] * {
  color: #e5e7eb !important;
}

div[data-testid="stToolbar"] button,
div[data-testid="stToolbar"] a {
  color: #e5e7eb !important;
}

div[data-testid="stToolbar"] svg,
div[data-testid="stToolbar"] svg * {
  fill: #e5e7eb !important;
  stroke: #e5e7eb !important;
}

.block-container {
  max-width: 1030px;
  padding-top: 6.2rem;
  padding-bottom: 0.9rem;
}

.page-title {
  color: #d1d5db;
  font-size: 1.15rem;
  font-weight: 600;
  margin-bottom: 8px;
}

.meta-line {
  color: #6b7280;
  font-size: 0.72rem;
  margin-bottom: 10px;
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

.nav-section-spacer {
  height: 1.15rem;
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

.panel {
  background: linear-gradient(160deg, #141f37 0%, #101a2e 100%);
  border: 1px solid rgba(55, 65, 81, 0.45);
  border-radius: 10px;
  padding: 8px 11px 6px 11px;
  margin-bottom: 8px;
}

.chart-title {
  color: #d1d5db;
  font-size: 0.9rem;
  font-weight: 600;
  margin-top: 0;
  margin-bottom: 4px;
}

.field-label {
  color: #9ca3af;
  font-size: 0.95rem;
  font-weight: 600;
  margin-bottom: 0.25rem;
}

.stat-card {
  background: linear-gradient(160deg, #141f37 0%, #101a2e 100%);
  border: 1px solid rgba(55, 65, 81, 0.45);
  border-radius: 10px;
  padding: 10px 12px 8px 12px;
  min-height: 90px;
}

.stat-label {
  color: #9ca3af;
  font-size: 0.9rem;
  font-weight: 600 !important;
  margin-bottom: 2px;
}

.stat-value {
  color: #f8fafc;
  font-size: 2.05rem;
  line-height: 1.05;
  font-weight: 600;
}

div[data-testid="stSelectbox"] label {
  color: #9ca3af !important;
  opacity: 1 !important;
}

div[data-testid="stSelectbox"] label p {
  color: #9ca3af !important;
  opacity: 1 !important;
  font-weight: 600 !important;
}

div[data-testid="stDataFrame"] {
  border: 1px solid rgba(55, 65, 81, 0.45);
  border-radius: 10px;
  overflow: hidden;
}

div[data-testid="stPlotlyChart"] {
  background: linear-gradient(180deg, #111c33 0%, #0e1830 100%);
  border: 1px solid rgba(55, 65, 81, 0.45);
  border-radius: 10px;
  padding: 4px 5px 0 5px;
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

.tx-panel {
  background: linear-gradient(160deg, #141f37 0%, #101a2e 100%);
  border: 1px solid rgba(55, 65, 81, 0.45);
  border-radius: 10px;
  padding: 10px 12px;
  margin-bottom: 8px;
  min-height: 120px;
}

.tx-header {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 6px;
}

.tx-title {
  color: #e5e7eb;
  font-size: 0.85rem;
  font-weight: 600;
}

.tx-badge {
  font-size: 0.7rem;
  padding: 1px 8px;
  border-radius: 999px;
  font-family: monospace;
}

.tx-badge-green { background: rgba(16,185,129,0.15); color: #34d399; }
.tx-badge-red { background: rgba(239,68,68,0.15); color: #f87171; }
.tx-badge-blue { background: rgba(59,130,246,0.15); color: #60a5fa; }
.tx-badge-orange { background: rgba(249,115,22,0.15); color: #fb923c; }

.tx-row {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 3px 0;
  border-bottom: 1px solid rgba(55,65,81,0.2);
  font-size: 0.78rem;
}

.tx-sym { color: #60a5fa; font-family: monospace; font-weight: 600; }
.tx-name { color: #6b7280; margin-left: 4px; }
.tx-weight { color: #d1d5db; font-family: monospace; }
.tx-none { color: #6b7280; font-size: 0.78rem; font-style: italic; }
"""
            + get_mobile_css()
            + """
</style>
""",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Sector Allocation Over Time (replaces pie chart)
# ---------------------------------------------------------------------------

def build_sector_allocation_figure(holdings_df: pd.DataFrame) -> go.Figure:
    sector_ts = (
        holdings_df.groupby(["as_of_date", "sector"])["weight"]
        .sum()
        .reset_index()
    )
    pivoted = sector_ts.pivot(index="as_of_date", columns="sector", values="weight").fillna(0)
    pivoted = pivoted.sort_index()

    fig = go.Figure()
    for sector in sorted(pivoted.columns):
        color = SECTOR_COLORS.get(sector, "#6b7280")
        fig.add_trace(go.Scatter(
            x=pivoted.index,
            y=pivoted[sector],
            name=sector,
            mode="lines",
            stackgroup="one",
            groupnorm="percent",
            line={"width": 0.5, "color": color},
            fillcolor=color,
            hovertemplate=f"{sector}<br>%{{x|%Y-%m}}: %{{y:.1f}}%<extra></extra>",
        ))

    fig.add_hline(
        y=25, line_dash="dot", line_color="rgba(239,68,68,0.5)", line_width=1,
        annotation_text="25%", annotation_position="right",
        annotation_font={"size": 9, "color": "#ef4444"},
    )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=360,
        margin={"l": 40, "r": 10, "t": 10, "b": 30},
        dragmode=False,
        showlegend=True,
        legend={
            "font": {"size": 9, "color": "#d1d5db"},
            "orientation": "h",
            "x": 0.5,
            "xanchor": "center",
            "y": -0.12,
            "yanchor": "top",
        },
    )
    fig.update_xaxes(
        tickformat="%Y-%m",
        dtick="M3",
        tickfont={"size": 9, "color": "#9ca3af"},
        showgrid=False,
        linecolor="#4b5563",
    )
    fig.update_yaxes(
        ticksuffix="%",
        tickfont={"size": 9, "color": "#9ca3af"},
        showgrid=True,
        gridcolor="#334155",
        linecolor="#4b5563",
        range=[0, 100],
    )
    return fig


# ---------------------------------------------------------------------------
# Covariance matrix heatmap
# ---------------------------------------------------------------------------

def build_correlation_figure(cov_data: dict) -> go.Figure:
    symbols = cov_data["symbols"]
    sectors = cov_data["sectors"]
    matrix = np.array(cov_data["matrix"])

    hover_text = []
    for i, s1 in enumerate(symbols):
        row_text = []
        for j, s2 in enumerate(symbols):
            row_text.append(f"{s1} vs {s2}<br>Corr: {matrix[i][j]:.3f}")
        hover_text.append(row_text)

    fig = go.Figure(data=go.Heatmap(
        z=matrix,
        x=symbols,
        y=symbols,
        colorscale="RdBu_r",
        zmid=0,
        zmin=-1,
        zmax=1,
        text=hover_text,
        hoverinfo="text",
        colorbar={
            "title": {"text": "Corr", "font": {"color": "#9ca3af", "size": 10}},
            "tickfont": {"color": "#9ca3af", "size": 9},
            "thickness": 12,
            "len": 0.6,
        },
    ))

    # Add sector divider lines
    prev_sector = sectors[0] if sectors else ""
    for i, sec in enumerate(sectors):
        if sec != prev_sector:
            fig.add_hline(y=i - 0.5, line_color="rgba(148,163,184,0.3)", line_width=1)
            fig.add_vline(x=i - 0.5, line_color="rgba(148,163,184,0.3)", line_width=1)
        prev_sector = sec

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=550,
        margin={"l": 60, "r": 20, "t": 10, "b": 60},
        dragmode=False,
    )
    fig.update_xaxes(
        tickfont={"size": 8, "color": "#9ca3af"},
        tickangle=45,
        side="bottom",
    )
    fig.update_yaxes(
        tickfont={"size": 8, "color": "#9ca3af"},
        autorange="reversed",
    )
    return fig


# ---------------------------------------------------------------------------
# Transaction panels (4 subpanels)
# ---------------------------------------------------------------------------

def _compute_transactions(current_df: pd.DataFrame, previous_df: pd.DataFrame) -> dict:
    prev = previous_df.set_index("symbol")
    curr = current_df.set_index("symbol")

    cur_syms = set(curr.index)
    prev_syms = set(prev.index)

    def _name(idx, frame):
        n = frame.at[idx, "company_name"] if idx in frame.index else idx
        return n if pd.notna(n) else idx

    new_positions = []
    for sym in sorted(cur_syms - prev_syms):
        new_positions.append({
            "symbol": sym,
            "company_name": _name(sym, curr),
            "new_weight": float(curr.at[sym, "weight"]),
        })
    new_positions.sort(key=lambda x: -x["new_weight"])

    closed_positions = []
    for sym in sorted(prev_syms - cur_syms):
        closed_positions.append({
            "symbol": sym,
            "company_name": _name(sym, prev),
            "old_weight": float(prev.at[sym, "weight"]),
        })
    closed_positions.sort(key=lambda x: -x["old_weight"])

    increased = []
    decreased = []
    for sym in sorted(cur_syms & prev_syms):
        old_w = float(prev.at[sym, "weight"])
        new_w = float(curr.at[sym, "weight"])
        delta = new_w - old_w
        if abs(delta) < 1e-7:
            continue
        entry = {
            "symbol": sym,
            "company_name": _name(sym, curr),
            "old_weight": old_w,
            "new_weight": new_w,
            "change": delta,
        }
        if delta > 0:
            increased.append(entry)
        else:
            decreased.append(entry)

    increased.sort(key=lambda x: -x["change"])
    decreased.sort(key=lambda x: x["change"])

    return {
        "new_positions": new_positions,
        "closed_positions": closed_positions,
        "increased": increased,
        "decreased": decreased,
    }


def _render_tx_panel(title: str, badge_class: str, items: list[dict], render_fn) -> None:
    count = len(items)
    badge_html = f'<span class="tx-badge {badge_class}">{count}</span>'
    html = f'<div class="tx-panel">'
    html += f'<div class="tx-header"><span class="tx-title">{title}</span>{badge_html}</div>'

    if not items:
        html += '<div class="tx-none">None</div>'
    else:
        for item in items:
            html += render_fn(item)

    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


def _render_new(item: dict) -> str:
    w = item["new_weight"] * 100
    return (
        f'<div class="tx-row">'
        f'<span><span class="tx-sym">{item["symbol"]}</span>'
        f'<span class="tx-name">{item["company_name"]}</span></span>'
        f'<span class="tx-weight">{w:.2f}%</span>'
        f'</div>'
    )


def _render_closed(item: dict) -> str:
    w = item["old_weight"] * 100
    return (
        f'<div class="tx-row">'
        f'<span><span class="tx-sym">{item["symbol"]}</span>'
        f'<span class="tx-name">{item["company_name"]}</span></span>'
        f'<span class="tx-weight">{w:.2f}%</span>'
        f'</div>'
    )


def _render_change(item: dict) -> str:
    old_w = item["old_weight"] * 100
    new_w = item["new_weight"] * 100
    delta = item["change"] * 100
    sign = "+" if delta > 0 else ""
    return (
        f'<div class="tx-row">'
        f'<span><span class="tx-sym">{item["symbol"]}</span>'
        f'<span class="tx-name">{item["company_name"]}</span></span>'
        f'<span class="tx-weight">{old_w:.2f}% &rarr; {new_w:.2f}% ({sign}{delta:.2f}%)</span>'
        f'</div>'
    )


def render_transactions_panels(
    month_df: pd.DataFrame,
    previous_df: pd.DataFrame | None,
    selected_month,
    previous_month,
) -> None:
    st.markdown(
        '<div class="chart-title">Monthly Transactions'
        + (f' <span style="color:#6b7280;font-size:0.75rem">vs {previous_month.strftime("%Y-%m")}</span>' if previous_month else "")
        + "</div>",
        unsafe_allow_html=True,
    )

    if previous_df is None or previous_df.empty:
        st.info("First rebalance — all positions are new entries.")
        return

    tx = _compute_transactions(month_df, previous_df)

    col1, col2 = st.columns(2)
    with col1:
        _render_tx_panel("New Positions", "tx-badge-green", tx["new_positions"], _render_new)
    with col2:
        _render_tx_panel("Closed Positions", "tx-badge-red", tx["closed_positions"], _render_closed)

    col3, col4 = st.columns(2)
    with col3:
        _render_tx_panel("Weight Increases", "tx-badge-blue", tx["increased"], _render_change)
    with col4:
        _render_tx_panel("Weight Decreases", "tx-badge-orange", tx["decreased"], _render_change)


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------

def render_stat_card(label: str, value: str) -> None:
    st.markdown(
        f"""
<div class="stat-card">
  <div class="stat-label">{label}</div>
  <div class="stat-value">{value}</div>
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
    st.markdown('<div class="nav-current">Current page: Portfolio Holdings</div>', unsafe_allow_html=True)

    nav_left, nav_middle, nav_right = st.columns(3)
    with nav_left:
        st.page_link("app.py", label="Open Performance")
    with nav_middle:
        st.page_link("pages/2_Portfolio_Holdings.py", label="Open Portfolio Holdings")
    with nav_right:
        st.page_link(
            "pages/3_Overview_of_our_modeling_method.py",
            label="Open Method Overview",
        )
    st.markdown('<div class="nav-section-spacer"></div>', unsafe_allow_html=True)


def render_navigation_banner() -> None:
    if st.session_state.get("nav_tip_dismissed", False):
        return

    c1, c2 = st.columns([5.0, 1.0])
    with c1:
        st.markdown(
            '<div class="nav-banner">'
            "Use the navigation tabs to switch between Performance, Portfolio Holdings, and Method Overview."
            "</div>",
            unsafe_allow_html=True,
        )
    with c2:
        if st.button("Dismiss tip", key="dismiss_nav_tip"):
            st.session_state["nav_tip_dismissed"] = True
            st.rerun()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(
        page_title="Portfolio Holdings",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    render_styles()
    render_navigation_banner()
    render_navigation_links()

    if not HOLDINGS_SNAPSHOT_PATH.exists():
        st.error(f"Missing holdings snapshot file: {HOLDINGS_SNAPSHOT_PATH}")
        st.stop()
    if not PERFORMANCE_SNAPSHOT_PATH.exists():
        st.error(f"Missing performance snapshot file: {PERFORMANCE_SNAPSHOT_PATH}")
        st.stop()

    holdings_df, holdings_payload, performance_payload, covariance_payload = load_snapshots()
    if holdings_df.empty:
        st.error("Holdings snapshot is empty for the current coverage window.")
        st.stop()

    month_options = sorted(holdings_df["as_of_date"].drop_duplicates(), reverse=True)

    st.markdown('<div class="page-title">Portfolio Holdings by Month</div>', unsafe_allow_html=True)
    st.markdown(
        (
            '<div class="meta-line">'
            f'Run: {holdings_payload["source_run_id"]} &nbsp;&nbsp;&nbsp; '
            f'Frozen at: {holdings_payload["frozen_at"]} &nbsp;&nbsp;&nbsp; '
            f'Coverage from: {performance_payload.get("coverage_start_date", "N/A")}'
            "</div>"
        ),
        unsafe_allow_html=True,
    )

    # --- Sector Allocation Over Time (full time series) ---
    st.markdown('<div class="chart-title">Sector Allocation Over Time</div>', unsafe_allow_html=True)
    st.plotly_chart(
        build_sector_allocation_figure(holdings_df),
        use_container_width=True,
        config={"displayModeBar": False, "scrollZoom": False, "doubleClick": False},
    )

    # --- Month selector ---
    st.markdown('<div class="field-label">Select month</div>', unsafe_allow_html=True)
    selected_month = st.selectbox(
        "select_month",
        options=month_options,
        index=0,
        label_visibility="collapsed",
        format_func=lambda d: d.strftime("%Y-%m"),
    )

    month_df = (
        holdings_df[holdings_df["as_of_date"] == selected_month]
        .sort_values("weight", ascending=False)
        .reset_index(drop=True)
    )

    c1, c2 = st.columns(2)
    with c1:
        render_stat_card("Month", selected_month.strftime("%Y-%m"))
    with c2:
        render_stat_card("Holdings", f"{len(month_df)}")

    # --- Holdings table with Unrealized column ---
    display_df = month_df.copy()
    display_df["weight_pct"] = display_df["weight"] * 100
    display_df["unrealized_display"] = display_df["unrealized_pct"] * 100

    display_df = display_df[["symbol", "company_name", "weight_pct", "sector", "unrealized_display"]]

    st.markdown('<div class="chart-title">All Holdings for Selected Month</div>', unsafe_allow_html=True)
    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "symbol": st.column_config.TextColumn("Symbol", width="small"),
            "company_name": st.column_config.TextColumn("Company Name"),
            "weight_pct": st.column_config.NumberColumn("Weight (%)", format="%.2f%%"),
            "sector": st.column_config.TextColumn("Sector"),
            "unrealized_display": st.column_config.NumberColumn("Unrealized (%)", format="%+.1f%%"),
        },
    )

    # --- Monthly Transactions (4 subpanels) ---
    months_asc = sorted(month_options)
    selected_idx = months_asc.index(selected_month)
    previous_month = months_asc[selected_idx - 1] if selected_idx > 0 else None

    if previous_month is None:
        render_transactions_panels(month_df, None, selected_month, None)
    else:
        previous_df = (
            holdings_df[holdings_df["as_of_date"] == previous_month]
            .sort_values("weight", ascending=False)
            .reset_index(drop=True)
        )
        render_transactions_panels(month_df, previous_df, selected_month, previous_month)

    # --- Covariance Matrix ---
    if covariance_payload and covariance_payload.get("symbols"):
        st.markdown(
            '<div class="chart-title">Correlation Matrix (252-day lookback)</div>',
            unsafe_allow_html=True,
        )
        st.plotly_chart(
            build_correlation_figure(covariance_payload),
            use_container_width=True,
            config={"displayModeBar": False, "scrollZoom": False, "doubleClick": False},
        )

    # --- Navigation footer ---
    st.markdown('<div class="nav-footer-title">Navigate</div>', unsafe_allow_html=True)
    nav_bottom_left, nav_bottom_right = st.columns(2)
    with nav_bottom_left:
        st.page_link(
            "app.py",
            label="Back: Open Performance",
        )
    with nav_bottom_right:
        st.page_link(
            "pages/3_Overview_of_our_modeling_method.py",
            label="Read: Method Overview",
        )


if __name__ == "__main__":
    main()
