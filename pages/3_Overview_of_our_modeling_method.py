import streamlit as st


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
  padding-bottom: 1.0rem;
}

.page-title {
  color: #d1d5db;
  font-size: 1.2rem;
  font-weight: 700;
  margin-bottom: 0.25rem;
}

.page-subtitle {
  color: #94a3b8;
  font-size: 0.84rem;
  margin-bottom: 0.65rem;
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

.stage-card {
  background: linear-gradient(160deg, #141f37 0%, #101a2e 100%);
  border: 1px solid rgba(55, 65, 81, 0.45);
  border-radius: 10px;
  padding: 0.6rem 0.7rem;
  min-height: 70px;
}

.stage-title {
  color: #9ca3af;
  font-size: 0.72rem;
  text-transform: uppercase;
  letter-spacing: 0.04em;
}

.stage-text {
  color: #e5e7eb;
  font-size: 0.88rem;
  font-weight: 600;
  line-height: 1.2;
  margin-top: 0.16rem;
}

.nav-footer-title {
  color: #d1d5db;
  font-size: 0.84rem;
  font-weight: 600;
  margin-top: 0.55rem;
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

details[data-testid="stExpander"] {
  background: linear-gradient(160deg, #141f37 0%, #101a2e 100%);
  border: 1px solid rgba(55, 65, 81, 0.45);
  border-radius: 10px !important;
}

details[data-testid="stExpander"] summary p {
  color: #e5e7eb !important;
  font-weight: 700 !important;
}

div[data-testid="stMarkdownContainer"] p,
div[data-testid="stMarkdownContainer"] li {
  color: #cbd5e1;
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
</style>
""",
        unsafe_allow_html=True,
    )


def render_navigation_links() -> None:
    if not hasattr(st, "page_link"):
        st.warning("Page navigation is unavailable in this Streamlit version.")
        return

    st.markdown('<div class="nav-kicker">Navigation</div>', unsafe_allow_html=True)
    st.markdown('<div class="nav-help">Click a tab to switch section.</div>', unsafe_allow_html=True)
    st.markdown('<div class="nav-current">Current page: Method Overview</div>', unsafe_allow_html=True)

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


def render_stage_map() -> None:
    cols = st.columns(4)
    stages = [
        ("Valuation", "Multi-model valuation architecture"),
        ("Forecasting", "Revenue and cash-flow engine"),
        ("ML Signal", "Upside features and XGBoost alpha"),
        ("Optimization", "Mean-variance-sparsity allocation"),
    ]
    for col, (title, text) in zip(cols, stages):
        with col:
            st.markdown(
                f"""
<div class="stage-card">
  <div class="stage-title">{title}</div>
  <div class="stage-text">{text}</div>
</div>
""",
                unsafe_allow_html=True,
            )


def render_title_1() -> None:
    with st.expander("1. Multi-Model Valuation Architecture", expanded=True):
        st.markdown(
            "To ensure robustness across sectors, the model avoids a one-size-fits-all design "
            "and dynamically switches valuation frameworks by company type."
        )
        st.markdown(
            "- **Standard Industrials (Non-Financials):** EV/Sales, EV/EBITDA, EV/EBIT, and UFCF-DCF discounted with WACC."
        )
        st.markdown(
            "- **Financial Institutions (Banks/Insurance):** P/E, P/TBV, DDM, RIM, and LFCF-based DCF discounted with Cost of Equity (CoE)."
        )


def render_title_2() -> None:
    with st.expander("2. Stochastic Revenue Forecasting Engine", expanded=True):
        st.markdown(
            "Revenue is modeled as a convergence path between current momentum and sector long-run reality."
        )
        st.markdown(
            "1. **Base Growth ($g_0$):** 3-year adjusted CAGR, clamped to $[-50\\%, +50\\%]$."
        )
        st.latex(r"g_0 = \operatorname{clip}\!\left(\mathrm{CAGR}^{\mathrm{adj}}_{3y}, -0.50,\ 0.50\right)")
        st.markdown(
            "2. **Terminal Convergence ($g_{term}$):** sector long-run growth from mature stable peers, scaled by 0.35 and clamped to $[1\\%, 4\\%]$."
        )
        st.latex(
            r"g_{\mathrm{term}} = \operatorname{clip}\!\left(0.35 \times \bar{g}^{\mathrm{stable}}_{\mathrm{sector}},\ 0.01,\ 0.04\right)"
        )
        st.markdown(
            "3. **Fade Mechanism:** Years 1-5 keep $g_0$, years 6-10 fade linearly from $g_0$ to $g_{term}$."
        )
        st.latex(
            r"g_t = \begin{cases}"
            r"g_0, & t \in \{1,\dots,5\} \\"
            r"g_0 + \dfrac{t-5}{5}\left(g_{\mathrm{term}}-g_0\right), & t \in \{6,\dots,10\}"
            r"\end{cases}"
        )


def render_title_3() -> None:
    with st.expander("3. Operating Cash Flow Modeling (Non-Financials)", expanded=False):
        st.markdown("For standard companies, we project **Unlevered Free Cash Flow (UFCF)**.")
        st.markdown("- **Ratio Analysis:** trailing 3-year medians for EBIT margin, D&A, CapEx, and NWC ratios.")
        st.markdown(
            "- **Outlier Management (Shrink-on-Cap):** ratios outside sector percentile bands (5th-97th) are shrunk toward sector medians and clamped."
        )
        st.markdown(
            "- **Synthesis:** stabilized ratios are applied to forecast revenue to derive NOPAT and reinvestment."
        )
        st.latex(r"UFCF = EBIT \times (1 - Tax) + D\&A - CapEx - \Delta NWC")


def render_title_4() -> None:
    with st.expander("4. Financial Sector Cash Flow Modeling", expanded=False):
        st.markdown(
            "For financial institutions, EBITDA-based logic is not appropriate because interest expense is operational."
        )
        st.markdown("- **Methodology Shift:** bypass EBIT and forecast Net Income directly from median margin structure.")
        st.markdown("- **Levered Basis:** model **Levered Free Cash Flow (LFCF)** including capital retention constraints.")
        st.markdown("- **Discounting:** discount with **Cost of Equity (CoE)**, not WACC.")


def render_title_5() -> None:
    with st.expander("5. Advanced Peer Selection & Exit Multiples", expanded=False):
        st.markdown(
            "For terminal exit multiples, peer groups are selected dynamically via a K-Nearest Neighbors framework."
        )
        st.markdown(
            "- **Features:** Size (log market cap), Growth (NTM revenue), Profitability (EBITDA margin), Risk (beta), Leverage (Debt/EBITDA)."
        )
        st.markdown("- **Algorithm:** Euclidean distance in standardized feature space, selecting the 15 nearest peers.")
        st.markdown(
            "- **Aggregation:** harmonic mean of peer EV/EBITDA to reduce sensitivity to high-multiple outliers."
        )


def render_title_6() -> None:
    with st.expander("6. Monte Carlo Simulation & Uncertainty Quantification", expanded=False):
        st.markdown("Each asset runs **10,000 simulations** to quantify valuation uncertainty.")
        st.markdown("- **Stochastic Inputs:** WACC/CoE, terminal growth, and cash-flow multipliers.")
        st.latex(r"WACC,\ CoE \sim \mathcal{N}\!\left(\mu_{\mathrm{sector}},\ \sigma^2_{\mathrm{sector}}\right)")
        st.latex(r"m_t \sim \operatorname{LogNormal}(\mu_t,\ \sigma_t^2), \quad m_t > 0")
        st.markdown(
            "- **Time-Horizon Scaling:** uncertainty widens with horizon, increasing shock variance from Year 1 to Year 10."
        )
        st.latex(r"\sigma_t = \sigma_1 + \frac{t-1}{9}\left(\sigma_{10} - \sigma_1\right), \quad t \in \{1,\dots,10\}")


def render_title_7() -> None:
    with st.expander("7. Feature Extraction: From Valuation to Signal", expanded=False):
        st.markdown(
            "Valuation outputs are transformed into standardized **Upside Ratios** and distributional risk descriptors."
        )
        st.latex(r"Upside = \frac{\text{Implied Price}}{\text{Current Price}} - 1")
        st.markdown("- **Risk Features:** IQR% for uncertainty magnitude and skew for tail asymmetry.")


def render_title_8() -> None:
    with st.expander("8. Machine Learning Architecture (XGBoost)", expanded=False):
        st.markdown("An XGBoost regressor combines 23 features into a single alpha signal.")
        st.markdown(
            "- **Inputs:** valuation upside features, Monte Carlo uncertainty features, and meta-features (size, peer count, data richness)."
        )
        st.markdown("- **Target:** market-adjusted forward return (stock return minus market mean).")
        st.latex(r"y_{i,t+h} = r_{i,t+h} - \bar{r}_{m,t+h}")
        st.markdown(
            "- **Validation:** walk-forward expanding-window training to eliminate look-ahead bias."
        )


def render_title_9() -> None:
    with st.expander("9. Horizon Blending & Inference", expanded=False):
        st.markdown("The model is trained on 6-month and 12-month forward horizons in parallel.")
        st.markdown(
            "- **Rationale:** some repricing is rapid (multiples), some is slow (intrinsic DCF convergence)."
        )
        st.markdown("- **Inference:** query at blended horizon (0.75 years) for medium-term alpha.")
        st.latex(r"\hat{\alpha}_{i,t}(h=0.75) \approx \mathcal{F}_{\mathrm{XGB}}(x_{i,t}, h)")


def render_title_10() -> None:
    with st.expander("10. Portfolio Optimization (Mean-Variance-Sparsity)", expanded=True):
        st.markdown("Final portfolio weights are solved via quadratic programming.")
        st.markdown("- **Covariance:** Ledoit-Wolf shrinkage for stable risk estimation.")
        st.markdown("- **Objective:** maximize expected return while penalizing risk and excess dispersion.")
        st.latex(
            r"\max_w \quad w^\top\mu - \frac{\gamma}{2}\,w^\top\Sigma w - \lambda \lVert w \rVert_1"
        )
        st.markdown(
            "This produces a concentrated, risk-aware portfolio and helps filter high-upside but high-uncertainty value traps."
        )


def main() -> None:
    st.set_page_config(
        page_title="Overview of Our Modeling Method",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    render_styles()
    render_navigation_banner()
    render_navigation_links()

    st.markdown('<div class="page-title">Overview of Our Modeling Method</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-subtitle">'
        "End-to-end architecture from valuation modeling to machine learning signal generation and portfolio construction."
        "</div>",
        unsafe_allow_html=True,
    )

    render_stage_map()

    tab_1, tab_2, tab_3, tab_4 = st.tabs(
        [
            "Foundation",
            "Forecasting & Cash Flows",
            "Uncertainty & ML",
            "Portfolio Construction",
        ]
    )

    with tab_1:
        render_title_1()
        render_title_2()

    with tab_2:
        render_title_3()
        render_title_4()
        render_title_5()

    with tab_3:
        render_title_6()
        render_title_7()
        render_title_8()
        render_title_9()

    with tab_4:
        render_title_10()

    st.markdown('<div class="nav-footer-title">Navigate</div>', unsafe_allow_html=True)
    nav_bottom_left, nav_bottom_right = st.columns(2)
    with nav_bottom_left:
        st.page_link("app.py", label="Back: Open Performance")
    with nav_bottom_right:
        st.page_link("pages/2_Portfolio_Holdings.py", label="Back: Open Portfolio Holdings")


if __name__ == "__main__":
    main()
