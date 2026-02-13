import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st


REPO_ROOT = Path(__file__).resolve().parents[1]
HOLDINGS_SNAPSHOT_PATH = REPO_ROOT / "data" / "holdings_snapshot.json"
PERFORMANCE_SNAPSHOT_PATH = REPO_ROOT / "data" / "performance_snapshot.json"
SYMBOL_NAME_SNAPSHOT_PATH = REPO_ROOT / "data" / "symbol_name_snapshot.json"


@st.cache_data
def load_snapshots() -> tuple[pd.DataFrame, dict, dict]:
    holdings_payload = json.loads(HOLDINGS_SNAPSHOT_PATH.read_text(encoding="utf-8"))
    performance_payload = json.loads(PERFORMANCE_SNAPSHOT_PATH.read_text(encoding="utf-8"))
    names_payload = (
        json.loads(SYMBOL_NAME_SNAPSHOT_PATH.read_text(encoding="utf-8"))
        if SYMBOL_NAME_SNAPSHOT_PATH.exists()
        else {"mapping": {}}
    )

    holdings_df = pd.DataFrame(holdings_payload["rows"])
    holdings_df["as_of_date"] = pd.to_datetime(holdings_df["as_of_date"])
    holdings_df["weight"] = pd.to_numeric(holdings_df["weight"], errors="coerce").fillna(0.0)
    holdings_df["expected_return"] = pd.to_numeric(holdings_df["expected_return"], errors="coerce")
    symbol_name_map = names_payload.get("mapping", {})
    holdings_df["company_name"] = holdings_df["symbol"].map(symbol_name_map).fillna(holdings_df["symbol"])

    coverage_start = performance_payload.get("coverage_start_date")
    if coverage_start:
        holdings_df = holdings_df[holdings_df["as_of_date"] >= pd.Timestamp(coverage_start)].copy()
    kpis_end_date = performance_payload.get("kpis", {}).get("end_date")
    if kpis_end_date:
        end_month = pd.Timestamp(kpis_end_date).to_period("M").to_timestamp("M")
        holdings_df = holdings_df[holdings_df["as_of_date"] <= end_month].copy()

    return holdings_df, holdings_payload, performance_payload


def render_styles() -> None:
    st.markdown(
        """
<style>
.stApp {
  background: linear-gradient(180deg, #040c20 0%, #020917 100%);
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

div[data-testid="stMetric"] {
  background: linear-gradient(160deg, #141f37 0%, #101a2e 100%);
  border: 1px solid rgba(55, 65, 81, 0.45);
  border-radius: 10px;
  padding: 6px 8px;
}

div[data-testid="stMetricLabel"] {
  color: #9ca3af;
}

div[data-testid="stMetricValue"] {
  color: #e5e7eb;
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
</style>
""",
        unsafe_allow_html=True,
    )


def build_weights_figure(month_df: pd.DataFrame) -> go.Figure:
    top = month_df.nlargest(15, "weight").copy()
    labels = top["symbol"] + " - " + top["company_name"]
    values = top["weight"] * 100

    fig = go.Figure()
    fig.add_trace(
        go.Pie(
            labels=labels,
            values=values,
            hole=0.0,
            sort=False,
            textinfo="percent",
            textfont={"size": 10, "color": "#e5e7eb"},
            marker={"line": {"color": "#0f172a", "width": 1}},
            hovertemplate="%{label}<br>Weight: %{value:.2f}%<extra></extra>",
        )
    )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=330,
        margin={"l": 8, "r": 8, "t": 6, "b": 6},
        showlegend=True,
        legend={
            "font": {"size": 10, "color": "#d1d5db"},
            "orientation": "v",
            "x": 1.0,
            "xanchor": "left",
            "y": 0.5,
            "yanchor": "middle",
        },
    )
    return fig


def build_transactions_df(current_df: pd.DataFrame, previous_df: pd.DataFrame) -> pd.DataFrame:
    prev = previous_df.set_index("symbol")
    curr = current_df.set_index("symbol")
    symbols = sorted(set(prev.index).union(set(curr.index)))

    rows = []
    for symbol in symbols:
        prev_w = float(prev.at[symbol, "weight"]) if symbol in prev.index else 0.0
        curr_w = float(curr.at[symbol, "weight"]) if symbol in curr.index else 0.0
        delta = curr_w - prev_w
        if abs(delta) < 1e-10:
            continue

        company_name = None
        if symbol in curr.index:
            company_name = curr.at[symbol, "company_name"]
        elif symbol in prev.index:
            company_name = prev.at[symbol, "company_name"]

        sector = None
        if symbol in curr.index:
            sector = curr.at[symbol, "sector"]
        elif symbol in prev.index:
            sector = prev.at[symbol, "sector"]

        if prev_w == 0 and curr_w > 0:
            action = "BUY"
        elif curr_w == 0 and prev_w > 0:
            action = "SELL"
        elif delta > 0:
            action = "INCREASE"
        else:
            action = "DECREASE"

        rows.append(
            {
                "action": action,
                "symbol": symbol,
                "company_name": company_name if pd.notna(company_name) else symbol,
                "sector": sector if pd.notna(sector) else "",
                "prev_weight": prev_w,
                "curr_weight": curr_w,
                "delta_weight": delta,
                "abs_delta": abs(delta),
            }
        )

    tx_df = pd.DataFrame(rows).sort_values("abs_delta", ascending=False).reset_index(drop=True)
    return tx_df


def build_transactions_figure(tx_df: pd.DataFrame) -> go.Figure:
    top = tx_df.nlargest(20, "abs_delta").copy()
    top["delta_pct"] = top["delta_weight"] * 100
    top = top.sort_values("delta_pct")

    colors = ["#10b981" if v > 0 else "#ef4444" for v in top["delta_pct"]]
    labels = top["symbol"] + " - " + top["company_name"]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=top["delta_pct"],
            y=labels,
            orientation="h",
            marker_color=colors,
            hovertemplate="%{y}<br>Weight change: %{x:+.2f}%<extra></extra>",
        )
    )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=420,
        margin={"l": 58, "r": 18, "t": 6, "b": 30},
        showlegend=False,
    )
    fig.update_xaxes(
        title="Change in Weight (%)",
        title_font={"size": 10, "color": "#9ca3af"},
        tickfont={"size": 10, "color": "#9ca3af"},
        showgrid=True,
        gridcolor="#334155",
        linecolor="#4b5563",
        zeroline=True,
        zerolinecolor="#64748b",
    )
    fig.update_yaxes(
        tickfont={"size": 10, "color": "#d1d5db"},
        showgrid=False,
        linecolor="#4b5563",
    )
    return fig


def render_navigation_links() -> None:
    if not hasattr(st, "page_link"):
        st.warning("Page navigation is unavailable in this Streamlit version.")
        return

    nav_left, nav_right = st.columns(2)
    with nav_left:
        st.page_link("app.py", label="Performance")
    with nav_right:
        st.page_link("pages/2_Portfolio_Holdings.py", label="Portfolio Holdings")


def main() -> None:
    st.set_page_config(page_title="Portfolio Holdings", layout="wide")
    render_styles()
    render_navigation_links()

    if not HOLDINGS_SNAPSHOT_PATH.exists():
        st.error(f"Missing holdings snapshot file: {HOLDINGS_SNAPSHOT_PATH}")
        st.stop()
    if not PERFORMANCE_SNAPSHOT_PATH.exists():
        st.error(f"Missing performance snapshot file: {PERFORMANCE_SNAPSHOT_PATH}")
        st.stop()

    holdings_df, holdings_payload, performance_payload = load_snapshots()
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

    selected_month = st.selectbox(
        "Select month",
        options=month_options,
        index=0,
        format_func=lambda d: d.strftime("%Y-%m"),
    )

    month_df = (
        holdings_df[holdings_df["as_of_date"] == selected_month]
        .sort_values("weight", ascending=False)
        .reset_index(drop=True)
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Month", selected_month.strftime("%Y-%m"))
    c2.metric("Holdings", f"{len(month_df)}")
    c3.metric("Total Weight", f"{month_df['weight'].sum() * 100:.2f}%")

    st.markdown('<div class="chart-title">Top 15 Weights (Pie)</div>', unsafe_allow_html=True)
    st.plotly_chart(build_weights_figure(month_df), use_container_width=True, config={"displayModeBar": False})

    display_df = month_df.copy()
    display_df["weight_pct"] = display_df["weight"] * 100

    display_df = display_df[["symbol", "company_name", "weight_pct", "sector"]]

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
        },
    )

    months_asc = sorted(month_options)
    selected_idx = months_asc.index(selected_month)
    previous_month = months_asc[selected_idx - 1] if selected_idx > 0 else None

    st.markdown('<div class="chart-title">Transactions vs Previous Month</div>', unsafe_allow_html=True)
    if previous_month is None:
        st.info("No previous month available for comparison in the selected window.")
        return

    previous_df = (
        holdings_df[holdings_df["as_of_date"] == previous_month]
        .sort_values("weight", ascending=False)
        .reset_index(drop=True)
    )
    tx_df = build_transactions_df(month_df, previous_df)
    if tx_df.empty:
        st.info("No allocation changes detected between these two months.")
        return

    buy_weight = float(tx_df[tx_df["delta_weight"] > 0]["delta_weight"].sum())
    sell_weight = float((-tx_df[tx_df["delta_weight"] < 0]["delta_weight"]).sum())
    gross_change = float(tx_df["abs_delta"].sum())

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("From Month", previous_month.strftime("%Y-%m"))
    m2.metric("To Month", selected_month.strftime("%Y-%m"))
    m3.metric("Buy Weight", f"{buy_weight * 100:.2f}%")
    m4.metric("Sell Weight", f"{sell_weight * 100:.2f}%")
    st.caption(f"Gross reallocation: {gross_change * 100:.2f}%")

    st.markdown('<div class="chart-title">Top Weight Changes</div>', unsafe_allow_html=True)
    st.plotly_chart(build_transactions_figure(tx_df), use_container_width=True, config={"displayModeBar": False})

    tx_display = tx_df.copy()
    tx_display["prev_weight_pct"] = tx_display["prev_weight"] * 100
    tx_display["curr_weight_pct"] = tx_display["curr_weight"] * 100
    tx_display["delta_weight_pct"] = tx_display["delta_weight"] * 100
    tx_display = tx_display[
        ["action", "symbol", "company_name", "sector", "prev_weight_pct", "curr_weight_pct", "delta_weight_pct"]
    ]

    st.markdown('<div class="chart-title">Transaction Details</div>', unsafe_allow_html=True)
    st.dataframe(
        tx_display,
        use_container_width=True,
        hide_index=True,
        column_config={
            "action": st.column_config.TextColumn("Action", width="small"),
            "symbol": st.column_config.TextColumn("Symbol", width="small"),
            "company_name": st.column_config.TextColumn("Company Name"),
            "sector": st.column_config.TextColumn("Sector"),
            "prev_weight_pct": st.column_config.NumberColumn("Prev Weight (%)", format="%.2f%%"),
            "curr_weight_pct": st.column_config.NumberColumn("Curr Weight (%)", format="%.2f%%"),
            "delta_weight_pct": st.column_config.NumberColumn("Change (%)", format="%+.2f%%"),
        },
    )


if __name__ == "__main__":
    main()
