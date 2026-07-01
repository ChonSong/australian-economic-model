"""
Page 8: Natural Resources & Commodity Prices Dashboard
=======================================================
Interactive dashboard for the natural resources sub-model.
Shows per-commodity depletion trajectories, price dynamics,
export revenue projections, energy transition scenarios,
and fiscal impacts (royalties, mining taxes).
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from aus_econ_model.models.resource_model import (
    ResourceParams, simulate_resources, run_scenario_comparison,
    COMMODITY_DATA, SCENARIOS, BUSINESS_AS_USUAL,
    get_commodity_summary,
)
from aus_econ_model.models.keen_model import KeenParams, simulate_keen
from aus_econ_model.components.charts import THEME, _apply_theme

st.set_page_config(page_title="Resources & Commodities", page_icon="⛏️", layout="wide")

# ── Page Title ────────────────────────────────────────────────────────────────

st.title("⛏️ Natural Resources & Commodity Prices")
st.markdown(
    "Australia's resource sector supplies ~60% of exports by value. "
    "This sub-model tracks depletion, prices, revenues, and the energy transition "
    "for five major commodities: iron ore, coal, LNG, gold, and lithium."
)

# ── Run Model ─────────────────────────────────────────────────────────────────

with st.spinner("Running resource sub-model..."):
    # Run core model first
    core = simulate_keen(KeenParams(), t_max=60, t_steps=1200)

    # Run BAU resource model
    bau_params = ResourceParams(scenario_name="bau")
    bau = simulate_resources(core, bau_params)

    # Run all scenarios for comparison
    all_scenarios = run_scenario_comparison(core)

# ── Key Metrics Cards ─────────────────────────────────────────────────────────

st.subheader("📊 Key Aggregates")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    rev = bau.total_export_revenue[0] / 1_000
    st.metric("Initial Export Revenue", f"${rev:.0f}B", help="Total resource export revenue at start of simulation")

with col2:
    rev_last = bau.total_export_revenue[-1] / 1_000
    delta = ((rev_last / rev) - 1) * 100 if rev > 0 else 0
    st.metric("Final Export Revenue (BAU)", f"${rev_last:.0f}B", f"{delta:+.0f}%", help="End-of-simulation (60yr) BAU")

with col3:
    tot = bau.terms_of_trade_index[-1]
    st.metric("Terms of Trade Index", f"{tot:.3f}", help="Weighted commodity price index (1.0 = start)")

with col4:
    share = bau.resource_gdp_share[-1]
    st.metric("Resource GDP Share", f"{share:.1%}", help="Resource value-added as share of GDP (end of simulation)")

with col5:
    gdp_share_start = bau.resource_gdp_share[0]
    st.metric("Resource GDP Share (Start)", f"{gdp_share_start:.1%}", help="Initial resource value-added share of GDP")

st.divider()

# ── Scenario Selector ─────────────────────────────────────────────────────────

st.subheader("🔮 Scenario Selector")

scenario_names = list(SCENARIOS.keys())
scenario_labels = [SCENARIOS[s].label for s in scenario_names]
selected_scenario = st.selectbox(
    "Choose a scenario to explore:",
    options=scenario_names,
    format_func=lambda x: SCENARIOS[x].label,
    index=0,
    help="Each scenario modifies commodity decay rates, discovery rates, and world GDP assumptions.",
)

sel_params = ResourceParams(scenario_name=selected_scenario)
sel = simulate_resources(core, sel_params) if selected_scenario == "bau" else all_scenarios[selected_scenario]

st.info(SCENARIOS[selected_scenario].description)

# ── Commodity Selection ───────────────────────────────────────────────────────

st.divider()
st.subheader("🔍 Per-Commodity Analysis")

commodity_keys = list(COMMODITY_DATA.keys())
commodity_labels = [COMMODITY_DATA[k]["name"] for k in commodity_keys]
selected_commodity = st.selectbox(
    "Select commodity:",
    options=commodity_keys,
    format_func=lambda x: COMMODITY_DATA[x]["name"],
    index=0,
)

cdata = COMMODITY_DATA[selected_commodity]
ctraj = sel.commodities[selected_commodity]

# ── Commodity Detail Charts ───────────────────────────────────────────────────

col1, col2 = st.columns(2)

with col1:
    # Depletion clock
    st.subheader("🕒 Depletion Clock")
    init_reserves = cdata["reserves"]
    current_reserves = ctraj.reserves[-1]
    remaining_yrs = ctraj.remaining_years[-1] if ctraj.remaining_years[-1] < 900 else float("inf")

    dep_col1, dep_col2, dep_col3 = st.columns(3)
    with dep_col1:
        st.metric("Initial Reserves", f"{init_reserves:,.0f} {cdata['unit']}")
    with dep_col2:
        if remaining_yrs < 900:
            st.metric("Remaining Reserves", f"{current_reserves:,.0f} {cdata['unit']}")
        else:
            st.metric("Remaining Reserves", f"{current_reserves:,.0f} {cdata['unit']}")
    with dep_col3:
        if remaining_yrs == float("inf"):
            st.metric("Years Remaining", "> 900")
        else:
            # Color code: green > 50yr, yellow 20-50yr, red < 20yr
            color = "🟢" if remaining_yrs > 50 else ("🟡" if remaining_yrs > 20 else "🔴")
            st.metric("Years Remaining", f"{remaining_yrs:.0f}", help=color)

    # Reserves chart
    st.subheader("Reserves Depletion Trajectory")
    fig_res = go.Figure()
    fig_res.add_trace(go.Scatter(x=ctraj.t, y=ctraj.reserves, mode="lines",
                                  name="Reserves", line=dict(color=THEME["font_color"], width=2)))
    fig_res.update_layout(height=250, margin=dict(l=10, r=10, t=10, b=10),
                           xaxis_title="Years", yaxis_title="Reserves")
    st.plotly_chart(_apply_theme(fig_res), width='stretch')

with col2:
    # Price dynamics
    st.subheader("💰 Price & Revenue Dynamics")

    price_col1, price_col2 = st.columns(2)
    with price_col1:
        st.metric(
            "Current Price",
            f"${ctraj.price_aud[-1]:,.0f} AUD/{cdata['unit']}",
            f"{((ctraj.price_aud[-1] / ctraj.price_aud[0]) - 1) * 100:+.1f}%"
        )
    with price_col2:
        st.metric(
            "Export Revenue",
            f"${ctraj.export_revenue[-1] / 1_000:,.0f} B AUD",
            f"{((ctraj.export_revenue[-1] / max(ctraj.export_revenue[0], 1)) - 1) * 100:+.1f}%"
        )

    # Price chart
    fig_price = go.Figure()
    fig_price.add_trace(go.Scatter(x=ctraj.t, y=ctraj.price_aud, mode="lines",
                                    name="Price (AUD/unit)", line=dict(width=2)))
    fig_price.add_trace(go.Scatter(x=ctraj.t, y=ctraj.export_revenue / 1_000, mode="lines",
                                    name="Export Revenue (B AUD)",
                                    line=dict(width=2, dash="dot"), yaxis="y2"))
    fig_price.update_layout(height=250, margin=dict(l=10, r=10, t=10, b=10),
                             xaxis_title="Years",
                             yaxis=dict(title="AUD/unit"),
                             yaxis2=dict(title="B AUD", overlaying="y", side="right"))
    st.plotly_chart(_apply_theme(fig_price), width='stretch')

# ── Production & Capacity Utilisation ─────────────────────────────────────────

col1, col2 = st.columns(2)

with col1:
    st.subheader("🏭 Production & Capacity")
    fig_prod = go.Figure()
    fig_prod.add_trace(go.Scatter(x=ctraj.t, y=ctraj.production, mode="lines",
                                   name="Production", line=dict(width=2)))
    fig_prod.add_trace(go.Scatter(x=ctraj.t, y=ctraj.capacity_utilisation, mode="lines",
                                   name="Capacity Utilisation",
                                   line=dict(width=2, dash="dot"), yaxis="y2"))
    fig_prod.update_layout(height=220, margin=dict(l=10, r=10, t=10, b=10),
                            xaxis_title="Years",
                            yaxis=dict(title="Production"),
                            yaxis2=dict(title="Utilisation", overlaying="y", side="right",
                                        tickformat=".0%"))
    st.plotly_chart(_apply_theme(fig_prod), width='stretch')

with col2:
    st.subheader("💼 Royalties & Mining Tax")
    fig_fisc = go.Figure()
    fig_fisc.add_trace(go.Scatter(x=ctraj.t, y=ctraj.royalty_revenue / 1_000, mode="lines",
                                   name="Royalties (B AUD)", line=dict(width=2)))
    fig_fisc.add_trace(go.Scatter(x=ctraj.t, y=ctraj.mining_tax / 1_000, mode="lines",
                                   name="Mining Tax (B AUD)", line=dict(width=2, dash="dot")))
    fig_fisc.update_layout(height=220, margin=dict(l=10, r=10, t=10, b=10),
                            xaxis_title="Years", yaxis_title="B AUD")
    st.plotly_chart(_apply_theme(fig_fisc), width='stretch')

# ── Aggregates Charts ─────────────────────────────────────────────────────────

st.divider()
st.subheader("📈 Scenario Comparison")

# Export revenue by scenario
scenario_revenue = {}
for sname, sres in all_scenarios.items():
    scenario_revenue[SCENARIOS[sname].label] = sres.total_export_revenue / 1_000

scenario_df = pd.DataFrame(scenario_revenue, index=bau.t)
fig_rev = go.Figure()
for col in scenario_df.columns:
    fig_rev.add_trace(go.Scatter(x=scenario_df.index, y=scenario_df[col],
                                  mode="lines", name=col, line=dict(width=2)))
fig_rev.update_layout(height=350, xaxis_title="Years", yaxis_title="Export Revenue (B AUD)")
st.plotly_chart(_apply_theme(fig_rev), width='stretch')

# Terms of trade comparison
st.subheader("Terms of Trade Index by Scenario")
tot_df = {}
for sname, sres in all_scenarios.items():
    tot_df[SCENARIOS[sname].label] = sres.terms_of_trade_index
tot_plot = pd.DataFrame(tot_df, index=bau.t)
fig_tot = go.Figure()
for col in tot_plot.columns:
    fig_tot.add_trace(go.Scatter(x=tot_plot.index, y=tot_plot[col],
                                  mode="lines", name=col, line=dict(width=2)))
fig_tot.update_layout(height=300, xaxis_title="Years", yaxis_title="Index")
st.plotly_chart(_apply_theme(fig_tot), width='stretch')

# ── Commodity Stacked Revenue ─────────────────────────────────────────────────

st.divider()
st.subheader("🏦 Resource Export Revenue Breakdown (BAU Scenario)")

comm_rev = {}
for key, ctraj in bau.commodities.items():
    comm_rev[COMMODITY_DATA[key]["name"]] = ctraj.export_revenue / 1_000

stack_df = pd.DataFrame(comm_rev, index=bau.t)
fig_stack = go.Figure()
for col in stack_df.columns:
    fig_stack.add_trace(go.Scatter(x=stack_df.index, y=stack_df[col],
                                    mode="lines", name=col,
                                    stackgroup="one", line=dict(width=0.5)))
fig_stack.update_layout(height=400, xaxis_title="Years", yaxis_title="Export Revenue (B AUD)")
st.plotly_chart(_apply_theme(fig_stack), width='stretch')

# ── Summary Table ─────────────────────────────────────────────────────────────

st.divider()
st.subheader("📋 Commodity Summary")

summary_time = st.slider(
    "Time (years from start):",
    min_value=0.0,
    max_value=float(bau.t[-1]),
    value=0.0,
    step=1.0,
    format="%.0f years",
)

time_idx = np.argmin(np.abs(bau.t - summary_time))

col1, col2 = st.columns([2, 1])

with col1:
    df = get_commodity_summary(sel, time_idx=time_idx)
    st.dataframe(df, width='stretch', hide_index=True)

with col2:
    # Mini aggregate summary at this time
    rev_at_time = sel.total_export_revenue[time_idx] / 1_000
    tot_at_time = sel.terms_of_trade_index[time_idx]
    share_at_time = sel.resource_gdp_share[time_idx]
    st.metric("Total Export Revenue", f"${rev_at_time:.0f}B AUD")
    st.metric("Terms of Trade Index", f"{tot_at_time:.3f}")
    st.metric("Resource GDP Share", f"{share_at_time:.1%}")

# ── Scenario Summary Cards ────────────────────────────────────────────────────

st.divider()
st.subheader("🏁 End-of-Simulation Scenario Comparison")

scenario_summaries = []
for sname, sres in all_scenarios.items():
    s = sres.scenario_summary()
    scenario_summaries.append({
        "Scenario": SCENARIOS[sname].label,
        "Export Rev (B AUD)": f"{s['total_export_revenue_AUD_B']:.0f}",
        "Royalties (B AUD)": f"{s['total_royalty_revenue_AUD_B']:.1f}",
        "Mining Tax (B AUD)": f"{s['total_mining_tax_AUD_B']:.1f}",
        "Inv (B AUD)": f"{s['total_mining_investment_AUD_B']:.1f}",
        "ToT Index": f"{s['terms_of_trade_index']:.3f}",
        "Vol Index": f"{s['export_volume_index']:.2f}",
        "GDP Share": f"{s['resource_gdp_share']:.1%}",
    })

st.dataframe(pd.DataFrame(scenario_summaries), width='stretch', hide_index=True)

# ── Background Information ────────────────────────────────────────────────────

st.divider()
with st.expander("📖 About the Resource Sub-Model", expanded=False):
    st.markdown("""
    ### How It Works

    The resource sub-model tracks five major commodity groups individually,
    simulating their physical depletion, price dynamics, and contributions
    to Australia's export revenue and fiscal position.

    **Physical Model:**
    - Each commodity starts with known reserves (Geoscience Australia AIMR 2024)
    - Production adjusts to price signals and energy transition trends
    - New discoveries add to reserves at scenario-dependent rates
    - Depletion clock = remaining reserves / annual production

    **Price Model:**
    - World GDP growth drives commodity demand (income elasticity varies by commodity)
    - Supply scarcity from depletion puts upward pressure on prices
    - Energy transition scenarios affect coal (declining), LNG (plateau), and lithium (growing)
    - China-specific demand for iron ore (35% weight in global demand)

    **Fiscal Linkages:**
    - State royalties = production × price × royalty rate (varies by commodity/state)
    - Company tax = mining profit × corporate tax rate (30%)
    - PRRT applies to LNG (Petroleum Resource Rent Tax)

    **Mining Investment:**
    - Goodwin-type cycle: high prices → investment boom → 5-7 year lag → supply comes online → prices fall
    - Investment responds to trailing average commodity prices

    **Scenarios:**
    1. **Business As Usual**: Current trends continue
    2. **Accelerated Depletion**: Higher demand, fewer discoveries, faster drawdown
    3. **New Discoveries**: Major finds expand reserves, sustaining production
    4. **Energy Transition (Net Zero 2050)**: Coal phases down, lithium booms, LNG plateaus

    ### Data Sources
    - **Geoscience Australia**: Australian Identified Mineral Resources (AIMR) — annual
    - **Office of the Chief Economist**: Resources and Energy Quarterly (REQ) — quarterly
    - **ABS International Trade**: Export volumes and values
    - **Department of Industry**: Energy commodity reports

    ### Key Calibrations
    | Commodity | Initial Reserves | Annual Production | Price (AUD) | Royalty Rate |
    |-----------|:----------------:|:-----------------:|:-----------:|:------------:|
    | Iron Ore  | 50,000 Mt        | 960 Mt            | $150/t      | 7.5% (WA)    |
    | Coal      | 75,000 Mt        | 460 Mt            | $250/t      | 8% (NSW/QLD) |
    | LNG       | 3,200 Mt         | 80 Mt             | $750/t      | 10% (PRRT)   |
    | Gold      | 385,800 koz      | 10,606 koz        | $3,500/oz   | 2.5%         |
    | Lithium   | 62,000 kt LCE    | 420 kt LCE        | $25,000/t   | 5% (WA)      |
    """)

# ── Footer ────────────────────────────────────────────────────────────────────

st.caption(
    "Data sources: Geoscience Australia AIMR (2024), Office of the Chief Economist REQ. "
    "Model projections are illustrative — not forecasts. "
    "Version 0.1 — June 2026."
)
