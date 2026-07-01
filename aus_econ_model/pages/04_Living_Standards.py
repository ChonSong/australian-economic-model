"""
Page 4: Living Standards Dashboard — Final Integration
========================================================
Pulls together all sub-models (Keen, SFC, Housing, Government, Resources)
into a comprehensive living standards assessment.

Central question: Will Australian living standards improve or decline?
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from aus_econ_model.models.keen_model import KeenParams, simulate_keen
from aus_econ_model.models.sfc_model import ExtendedKeenParams, simulate_extended
from aus_econ_model.models.housing_model import HousingParams, simulate_housing
from aus_econ_model.models.govt_model import GovtParams, simulate_fiscal
from aus_econ_model.models.resource_model import ResourceParams, simulate_resources

st.set_page_config(page_title="Living Standards", page_icon="📏", layout="wide")

st.title("📏 Living Standards Assessment")
st.markdown(
    "**The central question of this project:** *will Australian living standards "
    "improve or decline?* This page integrates all sub-models into a comprehensive "
    "composite index with scenario analysis."
)

# ── Framework ────────────────────────────────────────────────────────────────
with st.expander("How Living Standards Are Measured Here", expanded=False):
    st.markdown("""
    | Component | Weight | Source | What It Captures |
    |-----------|--------|--------|-----------------|
    | **Real Income** | 20% | Keen model ω (wage share) × α (productivity) | Disposable household income |
    | **Employment** | 20% | Keen model λ (employment rate) | Access to paid work |
    | **Debt Burden** | 15% | Keen model d (debt ratio), inverted | Financial fragility & debt service |
    | **Housing Affordability** | 15% | Housing model P/I ratio, inverted | Cost of shelter |
    | **Fiscal Sustainability** | 10% | Govt model debt/GDP trajectory | Future tax burden & public services |
    | **Natural Wealth** | 10% | Resource model depletion-adjusted revenue | Intergenerational resource equity |
    | **Inflation** | 5% | SFC model π_e (expected inflation), inverted | Purchasing power erosion |
    | **Financial Stability** | 5% | SFC model bank capital ratio (BCR) | Crisis risk |
    
    **Index interpretation:** 100 = today's level. Components normalised so
    higher = better. The composite tells you *whether* standards change; the
    components tell you *why*.
    """)

# ── Sidebar Controls ─────────────────────────────────────────────────────────
st.sidebar.header("Scenario Parameters")

projection_years = st.sidebar.slider("Years to project", 10, 60, 40, 5)

with st.sidebar.expander("Demographics & Productivity", expanded=True):
    beta_val = st.slider("Population growth (% p.a.)", 0.0, 3.0, 2.0, 0.1)
    alpha_val = st.slider("Productivity growth (% p.a.)", 0.0, 4.0, 2.0, 0.1)

with st.sidebar.expander("Private Debt & Credit", expanded=True):
    kappa1_val = st.slider("κ₁ — Investment response to profit", 0.1, 1.5, 0.5, 0.1)
    r_val = st.slider("Real interest rate (%)", 0.0, 8.0, 3.0, 0.5)

with st.sidebar.expander("Fiscal Policy", expanded=True):
    tax_w = st.slider("Tax rate — wages (%)", 10, 50, 30, 1) / 100
    tax_pi = st.slider("Tax rate — profits (%)", 10, 50, 35, 1) / 100
    g0 = st.slider("Autonomous govt spending (% GDP)", 5, 30, 16, 1) / 100

with st.sidebar.expander("Resource & Energy", expanded=True):
    resource_scenario = st.selectbox(
        "Resource scenario",
        ["Business As Usual", "Accelerated Depletion", "Energy Transition (Net Zero 2050)"]
    )

run_btn = st.button("▶️ **Run Comprehensive Analysis**", type="primary", width='stretch')

# ── Run Full Integration ─────────────────────────────────────────────────────
if run_btn:
    with st.spinner("Running integrated multi-sector model..."):
        # --- 1. Core Keen model ---
        params = KeenParams(
            alpha=alpha_val / 100,
            beta=beta_val / 100,
            r=r_val / 100,
            kappa1=kappa1_val,
        )
        sol = simulate_keen(params, t_max=projection_years, t_steps=projection_years * 20)
        if not sol.success:
            st.error("Core model failed. Try different parameters.")
            st.stop()

        # --- 2. SFC Extended model ---
        sfcp = ExtendedKeenParams(
            alpha=alpha_val / 100,
            beta=beta_val / 100,
            r=r_val / 100,
            kappa1=kappa1_val,
            tax_rate_wages=tax_w,
            tax_rate_profits=tax_pi,
            g0=g0,
        )
        sfc_sol = simulate_extended(sfcp, t_max=projection_years, t_steps=projection_years * 20)

        # --- 3. Housing model ---
        hp = HousingParams()
        hs = simulate_housing(sfc_sol, hp)

        # --- 4. Government fiscal model ---
        gp = GovtParams()
        gs = simulate_fiscal(sfc_sol, gp)

        # --- 5. Resource model ---
        rp = ResourceParams()
        scenario_map = {
            "Business As Usual": "BAU",
            "Accelerated Depletion": "accelerated_depletion",
            "Energy Transition (Net Zero 2050)": "net_zero_2050",
        }
        rp.scenario = scenario_map[resource_scenario]
        rs = simulate_resources(sfc_sol, rp)

    # ── Build Component Indices ─────────────────────────────────────────────

    # 1. Real Income (20%) — wage share × productivity growth
    real_income_idx = sol.omega * (1.0 + alpha_val / 100) ** sol.t
    real_income_idx = real_income_idx / real_income_idx[0] * 100

    # 2. Employment (20%)
    employment_idx = sol.lam / sol.lam[0] * 100

    # 3. Debt Burden (15%) — inverted: lower service = better
    peak_burden = max(sol.debt_service_ratio.max(), 0.15)
    debt_burden_idx = 100 * (1.0 - sol.debt_service_ratio / peak_burden)
    debt_burden_idx = np.clip(debt_burden_idx, 0, 100)

    # 4. Housing Affordability (15%) — inverted: lower P/I = better
    pi_ratio = hs['price_to_income']
    pi_start = pi_ratio[0]
    housing_idx = 100 * pi_start / np.clip(pi_ratio, pi_start * 0.5, pi_start * 3.0)

    # 5. Fiscal Sustainability (10%) — inverted: lower govt debt = better
    fed_debt = gs['fed_net_debt_gdp']
    fed_debt_0 = max(fed_debt[0], 0.01)
    fiscal_idx = 100 * (2.0 * fed_debt_0) / np.clip(fed_debt + fed_debt_0, fed_debt_0, fed_debt_0 * 3.0)
    fiscal_idx = np.clip(fiscal_idx, 10, 200)

    # 6. Natural Wealth (10%) — resource depletion-adjusted
    tot_rev = rs['total_export_revenue']
    natural_idx = tot_rev / tot_rev[0] * 100

    # 7. Inflation (5%) — inverted: lower inflation = better
    if hasattr(sfc_sol, 'pi_e') and sfc_sol.pi_e is not None:
        infl = sfc_sol.pi_e
        max_infl = max(infl.max(), 0.10)
        inflation_idx = 100 * (1.0 - infl / max_infl)
    else:
        inflation_idx = np.full_like(sol.t, 100.0)

    # 8. Financial Stability (5%) — bank capital ratio: higher = better
    if hasattr(sfc_sol, 'bcr') and sfc_sol.bcr is not None:
        bcr = sfc_sol.bcr
        bcr_0 = bcr[0]
        fin_stab_idx = 100 * bcr / bcr_0
    else:
        fin_stab_idx = np.full_like(sol.t, 100.0)

    # ── Composite Index ─────────────────────────────────────────────────────
    weights = {
        'real_income': 0.20,
        'employment': 0.20,
        'debt_burden': 0.15,
        'housing': 0.15,
        'fiscal': 0.10,
        'natural_wealth': 0.10,
        'inflation': 0.05,
        'financial_stability': 0.05,
    }

    composite = (
        weights['real_income'] * real_income_idx
        + weights['employment'] * employment_idx
        + weights['debt_burden'] * debt_burden_idx
        + weights['housing'] * housing_idx
        + weights['fiscal'] * fiscal_idx
        + weights['natural_wealth'] * natural_idx
        + weights['inflation'] * inflation_idx
        + weights['financial_stability'] * fin_stab_idx
    )

    # ── Display Results ─────────────────────────────────────────────────────

    final_composite = composite[-1]
    delta = final_composite - 100

    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        st.metric("Composite Index (today)", "100", "= baseline")
    with col2:
        st.metric(f"Composite Index in {projection_years} years",
                  f"{final_composite:.0f}",
                  delta=f"{delta:+.1f}",
                  delta_color="inverse" if delta < 0 else "normal")
    with col3:
        if delta > 10:
            verdict = "🟢 Improving"
        elif delta > 3:
            verdict = "🟡 Marginally Improving"
        elif delta > -3:
            verdict = "🟠 Stable / Mixed"
        elif delta > -10:
            verdict = "🔴 Declining"
        else:
            verdict = "⛔ Significantly Declining"
        st.metric("Verdict", verdict)

    # Main composite chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=sol.t, y=composite, mode="lines",
        name="Composite Living Standards Index",
        line=dict(color="#2ECC71", width=3),
        fill="tozeroy",
        fillcolor="rgba(46, 204, 113, 0.08)",
    ))
    fig.add_hline(y=100, line_color="rgba(255,255,255,0.3)", line_width=1, line_dash="dash")
    fig.update_layout(
        title="📊 Composite Living Standards Index (100 = today)",
        xaxis_title="Years from now",
        yaxis_title="Index",
        height=400, hovermode="x unified",
    )
    st.plotly_chart(fig, width='stretch')

    # ── Component Breakdown ─────────────────────────────────────────────────
    st.subheader("📈 Component Breakdown")

    components = [
        ("Real Income", real_income_idx, 0.20, "#E74C3C"),
        ("Employment", employment_idx, 0.20, "#3498DB"),
        ("Debt Burden", debt_burden_idx, 0.15, "#F39C12"),
        ("Housing Affordability", housing_idx, 0.15, "#9B59B6"),
        ("Fiscal Sustainability", fiscal_idx, 0.10, "#1ABC9C"),
        ("Natural Wealth", natural_idx, 0.10, "#27AE60"),
        ("Inflation", inflation_idx, 0.05, "#E67E22"),
        ("Financial Stability", fin_stab_idx, 0.05, "#8E44AD"),
    ]

    fig2 = make_subplots(rows=2, cols=4,
                         subplot_titles=[c[0] for c in components],
                         vertical_spacing=0.08, horizontal_spacing=0.04)
    for i, (name, values, weight, colour) in enumerate(components):
        row = i // 4 + 1
        col = i % 4 + 1
        fig2.add_trace(
            go.Scatter(x=sol.t, y=values, name=name,
                       line=dict(color=colour, width=2),
                       showlegend=False),
            row=row, col=col,
        )
        fig2.add_hline(y=100, line_color="gray", line_width=1,
                       line_dash="dash", row=row, col=col)
        # Annotate final value
        fig2.add_annotation(
            x=sol.t[-1], y=values[-1],
            text=f"{values[-1]:.0f}",
            showarrow=False, xshift=10,
            font=dict(size=10, color=colour),
            row=row, col=col,
        )
    fig2.update_layout(height=450, margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig2, width='stretch')

    # ── Trade-off Table ─────────────────────────────────────────────────────
    st.subheader("⚖️ What's Driving the Result")

    # Find the largest positive and negative contributors
    contributions = {}
    for name, values, weight, _ in components:
        contr = weight * (values[-1] - 100)
        contributions[name] = contr

    strongest = max(contributions, key=contributions.get)
    weakest = min(contributions, key=contributions.get)

    col1, col2 = st.columns(2)
    with col1:
        if contributions[strongest] > 0:
            st.success(f"**Strongest positive**: {strongest} (+{contributions[strongest]:+.1f} pts)")
    with col2:
        if contributions[weakest] < 0:
            st.error(f"**Strongest negative**: {weakest} ({contributions[weakest]:+.1f} pts)")

    df_contrib = pd.DataFrame({
        "Component": list(contributions.keys()),
        "Weight": [f"{w:.0%}" for _, _, w, _ in components],
        "Final Index": [f"{values[-1]:.0f}" for _, values, _, _ in components],
        "Contribution (pts)": [f"{c:+.1f}" for c in contributions.values()],
        "Status": ["📈" if c > 0 else "📉" if c < 0 else "➡️" for c in contributions.values()],
    })
    st.dataframe(df_contrib, width='stretch', hide_index=True)

    # ── Multi-Scenario Comparison ─────────────────────────────────────────
    st.subheader("🔀 How Different Futures Compare")

    # Run alternative scenarios
    scenarios = {
        "Baseline": (params, hp, gp, rp),
        "High Immigration": (
            KeenParams(alpha=alpha_val/100, beta=0.03, r=r_val/100, kappa1=kappa1_val),
            hp, gp, rp,
        ),
        "Rate Hike (+200bp)": (
            KeenParams(alpha=alpha_val/100, beta=beta_val/100, r=r_val/100 + 0.02, kappa1=kappa1_val),
            hp, gp, rp,
        ),
        "Credit Boom (high κ₁)": (
            KeenParams(alpha=alpha_val/100, beta=beta_val/100, r=r_val/100, kappa1=min(kappa1_val * 1.5, 1.5)),
            hp, gp, rp,
        ),
    }

    scenario_composites = {}
    with st.spinner("Running scenario comparisons..."):
        for sc_name, (sc_params, sc_hp, sc_gp, sc_rp) in scenarios.items():
            sc_sol = simulate_keen(sc_params, t_max=projection_years, t_steps=projection_years * 20)
            sc_sfc = simulate_extended(
                ExtendedKeenParams(
                    alpha=sc_params.alpha, beta=sc_params.beta,
                    r=sc_params.r, kappa1=sc_params.kappa1,
                ),
                t_max=projection_years, t_steps=projection_years * 20,
            )
            sc_hs = simulate_housing(sc_sfc, sc_hp)
            sc_gs = simulate_fiscal(sc_sfc, sc_gp)
            sc_rs = simulate_resources(sc_sfc, sc_rp)

            sc_ri = sch = sc_sol.omega * (1.0 + sc_params.alpha) ** sc_sol.t
            sc_ri = sc_ri / sc_ri[0] * 100
            sc_empl = sc_sol.lam / sc_sol.lam[0] * 100
            pk_d = max(sc_sol.debt_service_ratio.max(), 0.15)
            sc_db = 100 * (1.0 - sc_sol.debt_service_ratio / pk_d)
            sc_hou = 100 * sc_hs['price_to_income'][0] / np.clip(sc_hs['price_to_income'], sc_hs['price_to_income'][0]*0.5, sc_hs['price_to_income'][0]*3.0)
            sc_fis = 100 * (2*sc_gs['fed_net_debt_gdp'][0]) / np.clip(sc_gs['fed_net_debt_gdp'] + sc_gs['fed_net_debt_gdp'][0], sc_gs['fed_net_debt_gdp'][0], sc_gs['fed_net_debt_gdp'][0]*3.0)
            sc_nat_val = sc_rs['total_export_revenue']
            sc_nat = sc_nat_val / sc_nat_val[0] * 100
            sc_comp = (
                0.20*sc_ri + 0.20*sc_empl + 0.15*sc_db + 0.15*sc_hou
                + 0.10*sc_fis + 0.10*sc_nat + 0.05*inflation_idx[:len(sc_ri)] + 0.05*fin_stab_idx[:len(sc_ri)]
            )
            scenario_composites[sc_name] = sc_comp

    fig3 = go.Figure()
    colors = {"Baseline": "#2ECC71", "High Immigration": "#3498DB", "Rate Hike (+200bp)": "#E74C3C", "Credit Boom (high κ₁)": "#F39C12"}
    for sc_name, sc_comp in scenario_composites.items():
        fig3.add_trace(go.Scatter(
            x=sol.t[:len(sc_comp)], y=sc_comp, mode="lines",
            name=sc_name,
            line=dict(color=colors.get(sc_name, "#888"), width=2,
                      dash="dash" if sc_name != "Baseline" else "solid"),
        ))
    fig3.add_hline(y=100, line_color="rgba(255,255,255,0.3)", line_width=1, line_dash="dash")
    fig3.update_layout(
        title="Scenario Comparison: Composite Living Standards Index",
        xaxis_title="Years from now", yaxis_title="Index (100 = today)",
        height=400, hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig3, width='stretch')

    # Scenario summary table
    st.markdown("#### Scenario Outcome Summary")
    scenario_summary = []
    for sc_name, sc_comp in scenario_composites.items():
        final_sc = sc_comp[-1]
        sc_delta = final_sc - 100
        scenario_summary.append({
            "Scenario": sc_name,
            f"Index in {projection_years}yr": f"{final_sc:.0f}",
            "Change": f"{sc_delta:+.1f}",
            "Direction": "🟢" if sc_delta > 5 else "🟡" if sc_delta > -5 else "🔴",
        })
    st.dataframe(pd.DataFrame(scenario_summary), width='stretch', hide_index=True)

    # ── Detailed Sector Analysis ──────────────────────────────────────────
    st.subheader("🔍 Sector-by-Sector Detail")

    tab1, tab2, tab3, tab4 = st.tabs(["Labour & Credit", "Housing", "Fiscal", "Resources"])

    with tab1:
        fig_lc = make_subplots(rows=1, cols=3,
                               subplot_titles=("Employment Rate", "Wage Share", "Debt Ratio"))
        fig_lc.add_trace(go.Scatter(x=sol.t, y=sol.lam, name="λ"), row=1, col=1)
        fig_lc.add_trace(go.Scatter(x=sol.t, y=sol.omega, name="ω"), row=1, col=2)
        fig_lc.add_trace(go.Scatter(x=sol.t, y=sol.d, name="d"), row=1, col=3)
        fig_lc.update_layout(height=300, showlegend=False)
        st.plotly_chart(fig_lc, width='stretch')

    with tab2:
        fig_h = make_subplots(rows=1, cols=3,
                              subplot_titles=("Price / Income", "Housing Wealth/GDP", "Mortgage Service"))
        fig_h.add_trace(go.Scatter(x=hs['t'], y=hs['price_to_income'], name="P/I"), row=1, col=1)
        fig_h.add_trace(go.Scatter(x=hs['t'], y=hs['housing_wealth_gdp'], name="Wealth/GDP"), row=1, col=2)
        fig_h.add_trace(go.Scatter(x=hs['t'], y=hs['mortgage_service'], name="Service"), row=1, col=3)
        fig_h.update_layout(height=300, showlegend=False)
        st.plotly_chart(fig_h, width='stretch')

    with tab3:
        fig_g = make_subplots(rows=1, cols=3,
                              subplot_titles=("Federal Revenue/Spending", "Federal Debt/GDP", "Fiscal Sustainability"))
        fig_g.add_trace(go.Scatter(x=sol.t, y=gs['total_revenue_pct'], name="Revenue"), row=1, col=1)
        fig_g.add_trace(go.Scatter(x=sol.t, y=gs['total_spending_pct'], name="Spending"), row=1, col=1)
        fig_g.add_trace(go.Scatter(x=sol.t, y=gs['fed_net_debt_gdp'], name="Fed Net Debt/GDP"), row=1, col=2)
        fig_g.add_trace(go.Scatter(x=sol.t, y=gs['consolidated_net_debt_gdp'], name="Total Govt Debt/GDP"), row=1, col=2)
        fig_g.add_trace(go.Scatter(x=sol.t, y=gs['fed_interest_revenue_ratio'], name="Interest/Revenue"), row=1, col=3)
        fig_g.update_layout(height=300, showlegend=False)
        st.plotly_chart(fig_g, width='stretch')

    with tab4:
        fig_r = make_subplots(rows=1, cols=3,
                              subplot_titles=("Total Export Revenue ($M)", "Terms of Trade", "Resource GDP Share"))
        fig_r.add_trace(go.Scatter(x=sol.t, y=rs['total_export_revenue'], name="Export Rev"), row=1, col=1)
        if hasattr(rs, 'terms_of_trade_index'):
            fig_r.add_trace(go.Scatter(x=sol.t, y=rs['terms_of_trade_index'], name="ToT"), row=1, col=2)
        if hasattr(rs, 'resource_gdp_share'):
            fig_r.add_trace(go.Scatter(x=sol.t, y=rs['resource_gdp_share'], name="% GDP"), row=1, col=3)
        fig_r.update_layout(height=300, showlegend=False)
        st.plotly_chart(fig_r, width='stretch')

    # ── Final Verdict ─────────────────────────────────────────────────────
    st.divider()
    st.subheader("📋 Summary Assessment")

    # Count improving vs declining components
    improving = sum(1 for _, values, _, _ in components if values[-1] > 100)
    declining = sum(1 for _, values, _, _ in components if values[-1] < 100)

    if delta > 5:
        st.success(f"""
        ### 🟢 Verdict: **Improving** (+{delta:+.1f} points)
        
        {improving} of 8 components are above baseline. The primary driver is
        **{strongest}**.
        """)
    elif delta > 0:
        st.info(f"""
        ### 🟡 Verdict: **Marginal Improvement** (+{delta:+.1f} points)
        
        The composite is slightly positive, but {declining} of 8 components are
        declining. **{strongest}** is the main support; **{weakest}** is the main drag.
        """)
    elif delta > -5:
        st.warning(f"""
        ### 🟠 Verdict: **Mild Decline** ({delta:+.1f} points)
        
        {declining} of 8 components are below baseline. The main drag is
        **{weakest}**. This path suggests gradual erosion of living standards
        through the {weakest.lower()} channel.
        """)
    else:
        st.error(f"""
        ### 🔴 Verdict: **Significant Decline** ({delta:+.1f} points)
        
        {declining} of 8 components are below baseline. The primary drag is
        **{weakest}**. If Australia follows this trajectory, living standards
        will materially decline over the projection period.
        """)

    with st.expander("Why This Verdict — Methodological Notes"):
        st.markdown("""
        This assessment follows **Steve Keen's debt-centred framework**: aggregate
        demand = income + change in debt. When debt growth slows or reverses,
        the economy must contract to restore balance. The living standards index
        captures this by:
        
        1. **Centring debt burden** as a direct drag on disposable income
        2. **Modelling housing as a credit-driven asset** — rising prices boost
           wealth but also increase debt service and financial fragility
        3. **Tracking fiscal sustainability** — high government debt constrains
           future public spending
        4. **Adjusting for resource depletion** — today's export revenue is
           partly financed by drawing down natural capital
        
        **Limitations:**
        - No distributional data (top-decile vs bottom-decile dynamics)
        - Simplified policy responses (no endogenous RBA reaction function)
        - National aggregates only (no state/territory breakdown)
        - No climate change impacts beyond commodity transition
        - All sub-models are simplified representations
        """)

else:
    # Pre-run explanation
    st.info("👈 Set scenario parameters in the sidebar and click **Run Comprehensive Analysis**.")

    st.markdown("""
    ### How to Use This Page

    1. **Set scenario parameters** in the left sidebar — demographics, productivity,
       credit sensitivity, interest rates, fiscal policy, and resource outlook
    2. Click **Run Comprehensive Analysis** to simulate all sub-models together
    3. **The Composite Index** shows overall living standards trajectory (100 = today)
    4. **Component Breakdown** reveals which factors are driving the result
    5. **Scenario Comparison** shows how different assumptions change the outcome
    6. **Sector tabs** let you dive into labour/credit, housing, fiscal, and resources

    ### The Keen Framework Applied to Living Standards

    Standard living standards measures (GDP/capita, HDI) ignore **private debt**.
    This model follows Keen's insight that debt is central:

    - **Rising debt** temporarily boosts demand but creates future repayment burden
    - **Housing wealth** from rising prices is illusory if offset by higher debt
    - **Fiscal sustainability** depends on both government debt AND private debt
      (private debt crises become public debt burdens via bailouts)
    - **Resource wealth** must be adjusted for depletion — selling non-renewable
      assets is not genuine income
    """)

st.divider()
st.caption("Built following Prof. Steve Keen's debt-centred macroeconomic framework. "
           "Composite Living Standards Index adapted from the Stiglitz-Sen-Fitoussi Commission "
           "and Keen's debt-adjusted welfare approach.")
