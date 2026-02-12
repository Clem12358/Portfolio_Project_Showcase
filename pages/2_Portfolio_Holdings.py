import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st


REPO_ROOT = Path(__file__).resolve().parents[1]
HOLDINGS_SNAPSHOT_PATH = REPO_ROOT / "data" / "holdings_snapshot.json"
PERFORMANCE_SNAPSHOT_PATH = REPO_ROOT / "data" / "performance_snapshot.json"


@st.cache_data
def load_snapshots() -> tuple[pd.DataFrame, dict, dict]:
    holdings_payload = json.loads(HOLDINGS_SNAPSHOT_PATH.read_text(encoding="utf-8"))
    performance_payload = json.loads(PERFORMANCE_SNAPSHOT_PATH.read_text(encoding="utf-8"))

    holdings_df = pd.DataFrame(holdings_payload["rows"])
    holdings_df["as_of_date"] = pd.to_datetime(holdings_df["as_of_date"])
    holdings_df["weight"] = pd.to_numeric(holdings_df["weight"], errors="coerce").fillna(0.0)
    holdings_df["expected_return"] = pd.to_numeric(holdings_df["expected_return"], errors="coerce")

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
    top = month_df.nlargest(15, "weight").sort_values("weight")

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=top["weight"] * 100,
            y=top["symbol"],
            orientation="h",
            marker={"color": "#3b82f6"},
            hovertemplate="%{y}<br>Weight: %{x:.2f}%<extra></extra>",
        )
    )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=330,
        margin={"l": 58, "r": 16, "t": 6, "b": 26},
        showlegend=False,
    )
    fig.update_xaxes(
        title="Weight (%)",
        title_font={"size": 10, "color": "#9ca3af"},
        tickfont={"size": 10, "color": "#9ca3af"},
        showgrid=True,
        gridcolor="#334155",
        linecolor="#4b5563",
        zeroline=False,
    )
    fig.update_yaxes(
        tickfont={"size": 10, "color": "#d1d5db"},
        showgrid=False,
        linecolor="#4b5563",
    )
    return fig


def main() -> None:
    st.set_page_config(page_title="Portfolio Holdings", layout="wide")
    render_styles()

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

    st.markdown('<div class="chart-title">Top 15 Weights</div>', unsafe_allow_html=True)
    st.plotly_chart(build_weights_figure(month_df), use_container_width=True, config={"displayModeBar": False})

    display_df = month_df.copy()
    display_df.insert(0, "rank", range(1, len(display_df) + 1))
    display_df["as_of_date"] = display_df["as_of_date"].dt.strftime("%Y-%m-%d")
    display_df["weight_pct"] = display_df["weight"] * 100
    display_df["expected_return_pct"] = display_df["expected_return"] * 100

    display_df = display_df[
        ["rank", "symbol", "weight_pct", "sector", "expected_return_pct", "optimizer", "as_of_date"]
    ]

    st.markdown('<div class="chart-title">All Holdings for Selected Month</div>', unsafe_allow_html=True)
    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "rank": st.column_config.NumberColumn("Rank", format="%d", width="small"),
            "symbol": st.column_config.TextColumn("Symbol", width="small"),
            "weight_pct": st.column_config.NumberColumn("Weight (%)", format="%.2f%%"),
            "sector": st.column_config.TextColumn("Sector"),
            "expected_return_pct": st.column_config.NumberColumn("Expected Return (%)", format="%.2f%%"),
            "optimizer": st.column_config.TextColumn("Optimizer"),
            "as_of_date": st.column_config.TextColumn("As Of"),
        },
    )


if __name__ == "__main__":
    main()
