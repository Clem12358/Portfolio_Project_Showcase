def get_mobile_css() -> str:
    """Return CSS media-query rules for mobile/tablet responsiveness."""
    return """
/* ========== MOBILE RESPONSIVE ========== */

@media (max-width: 768px) {
  .block-container {
    max-width: 100% !important;
    padding-top: 3.5rem !important;
    padding-left: 1rem !important;
    padding-right: 1rem !important;
  }

  /* Force all Streamlit columns to stack vertically */
  div[data-testid="stHorizontalBlock"] {
    flex-direction: column !important;
    gap: 0.5rem !important;
  }

  div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"] {
    width: 100% !important;
    flex: 1 1 100% !important;
    min-width: 100% !important;
  }

  /* KPI cards */
  .kpi-card { min-height: 64px; padding: 8px 12px 7px 12px; }
  .kpi-label { font-size: 0.82rem; }
  .kpi-value { font-size: 1.5rem; }

  /* Chart containers: horizontal scroll for wide content, disable pinch-to-zoom */
  div[data-testid="stPlotlyChart"] {
    overflow-x: auto;
    -webkit-overflow-scrolling: touch;
  }

  div[data-testid="stPlotlyChart"] * {
    touch-action: pan-y !important;
  }

  /* Dataframes: horizontal scroll */
  div[data-testid="stDataFrame"] {
    overflow-x: auto;
    -webkit-overflow-scrolling: touch;
  }

  /* LaTeX formulas: horizontal scroll */
  div[data-testid="stLatex"] {
    overflow-x: auto;
    -webkit-overflow-scrolling: touch;
  }

  /* Transaction panels */
  .tx-panel { min-height: auto; }

  /* Stat cards (page 2) */
  .stat-card { min-height: 72px; }
  .stat-value { font-size: 1.7rem; }

  /* Stage cards (page 3) */
  .stage-card { min-height: auto; padding: 0.5rem 0.6rem; }

  /* Navigation */
  .nav-banner { font-size: 0.78rem; }
  .summary-line { font-size: 0.68rem; word-break: break-word; }
  .chart-title { font-size: 0.85rem; }
  .signature-pill { font-size: 0.72rem; }

  /* Page link buttons: bigger tap targets */
  div[data-testid="stPageLink"] a {
    min-height: 2.8rem !important;
    font-size: 0.85rem !important;
  }

  /* Highlight banners (page 3) */
  .highlight-text { font-size: 0.9rem; }
  .automation-text { font-size: 0.82rem; }
}

@media (max-width: 480px) {
  .block-container {
    padding-top: 2.8rem !important;
    padding-left: 0.6rem !important;
    padding-right: 0.6rem !important;
  }

  /* KPI cards: compact */
  .kpi-card { min-height: 58px; padding: 6px 10px; margin-bottom: 4px; }
  .kpi-label { font-size: 0.74rem; }
  .kpi-value { font-size: 1.3rem; }

  /* Stage cards */
  .stage-title { font-size: 0.68rem; }
  .stage-text { font-size: 0.82rem; }

  /* Stat cards */
  .stat-value { font-size: 1.45rem; }
  .stat-label { font-size: 0.82rem; }

  /* Transaction rows */
  .tx-row { flex-wrap: wrap; font-size: 0.74rem; }
  .tx-title { font-size: 0.8rem; }

  /* Navigation */
  .nav-kicker { font-size: 0.74rem; }
  .nav-help { font-size: 0.76rem; }
  .nav-banner { font-size: 0.74rem; padding: 0.45rem 0.6rem; }

  /* Page title */
  .page-title { font-size: 1.0rem; }

  /* Charts */
  .chart-title { font-size: 0.8rem; }
  .summary-line { font-size: 0.64rem; }

  /* Buttons: bigger tap targets */
  div[data-testid="stButton"] button {
    min-height: 2.8rem;
    font-size: 0.8rem;
  }

  div[data-testid="stPageLink"] a {
    min-height: 3.0rem !important;
  }

  /* Highlight banners */
  .highlight-text { font-size: 0.84rem; }
  .highlight-title { font-size: 0.74rem; }

  /* Automation text */
  .automation-title { font-size: 0.72rem; }
  .automation-text { font-size: 0.78rem; }

  /* Signature */
  .signature-pill { font-size: 0.68rem; padding: 0.3rem 0.65rem; }
}
"""
