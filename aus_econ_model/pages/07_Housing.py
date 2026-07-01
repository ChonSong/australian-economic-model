"""Housing Sector Analysis — Page 7"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from aus_econ_model.models.keen_model import KeenParams, simulate_keen
from aus_econ_model.models.housing_model import HousingParams, simulate_housing
from aus_econ_model.components.charts import _apply_theme

st.set_page_config(page_title="Housing Sector", page_icon="🏘️", layout="wide")

st.title("🏘️ Australian Housing Sector Model")
st.markdown(
    "A Minsky-style housing sub-model driven by credit flows, supply constraints, "
    "and demographic pressures. Housing prices are primarily determined by the **flow "
    "of new credit** into the market, with structural trends from tax policy (negative "
    "gearing, CGT discount) and supply limitations."
)

# ── Sidebar Controls ─────────────────────────────────────────────────
st.sidebar.header("Housing Parameters")

hp = HousingParams()
hp.credit_flow_elasticity = st.sidebar.slider(
    "Credit Flow Elasticity", 0.5, 5.0, 2.0, 0.1,
    help="How strongly P/I responds to credit growth. Higher = more volatile."
)
hp.fundamental_price_income = st.sidebar.slider(
    "Fundamental P/I Ratio", 3.0, 8.0, 5.0, 0.5,
    help="Long-run equilibrium price-to-income ratio (before structural drift)."
)
hp.structural_trend = st.sidebar.slider(
    "Structural Trend (%/yr)", 0.0, 1.5, 0.5, 0.1,
    help="Annual structural drift from supply constraints + tax policy."
)
hp.mean_reversion_strength = st.sidebar.slider(
    "Mean Reversion Speed", 0.0, 0.02, 0.003, 0.001,
    help="Speed of reversion to fundamental value. AU data suggests very slow."
)
hp.initial_price_income = st.sidebar.slider(
    "Initial P/I Ratio", 4.0, 10.0, 6.5, 0.5
)

st.sidebar.subheader("Mortgage & LVR")
hp.mortgage_share_of_debt = st.sidebar.slider("Mortgage Share of Debt", 0.2, 0.6, 0.45, 0.05)
hp.initial_lvr = st.sidebar.slider("Initial LVR", 0.3, 0.8, 0.60, 0.05)
hp.lvr_cycle_sensitivity = st.sidebar.slider("LVR Cycle Sensitivity", 0.0, 0.3, 0.10, 0.02)

st.sidebar.subheader("Construction")
hp.base_construction_gdp = st.sidebar.slider("Base Construction (% GDP)", 3, 10, 6, 1) / 100
hp.construction_elasticity = st.sidebar.slider("Construction Price Response", 0.0, 0.5, 0.20, 0.05)
hp.construction_lag = st.sidebar.slider("Construction Lag (years)", 0.5, 5.0, 2.0, 0.5)

st.sidebar.subheader("Core Model")
kappa1 = st.sidebar.slider("κ₁ (Investment Response)", 0.1, 1.5, 0.5, 0.1)
r = st.sidebar.slider("Interest Rate (r)", 0.01, 0.08, 0.03, 0.005)

# ── Run Simulation ─────────────────────────────────────────────────
with st.spinner("Running housing model simulation..."):
    params = KeenParams(kappa1=kappa1, r=r)
    sol = simulate_keen(params, t_max=50)
    hs = simulate_housing(sol, hp)

# ── Key Metrics ─────────────────────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    st.metric("Final P/I Ratio", f"{hs['price_to_income'][-1]:.1f}×",
              f"{(hs['price_to_income'][-1]/hs['price_to_income'][0]-1):+.1%}")
with c2:
    st.metric("P/I Range", f"{hs['price_to_income'].min():.1f}–{hs['price_to_income'].max():.1f}×",
              f"{(hs['price_to_income'].max()/hs['price_to_income'].min()-1):+.1%} spread")
with c3:
    st.metric("Mortgage/GDP", f"{hs['mortgage_gdp'][-1]:.0%}",
              f"{(hs['mortgage_gdp'][-1]/hs['mortgage_gdp'][0]-1):+.1%}")
with c4:
    st.metric("Rental Yield", f"{hs['rental_yield'][-1]:.1%}",
              f"{(hs['rental_yield'][-1]/hs['rental_yield'][0]-1):+.1%}")
with c5:
    st.metric("Housing Wealth/GDP", f"{hs['housing_wealth_gdp'][-1]:.1f}×",
              f"{(hs['housing_wealth_gdp'][-1]/hs['housing_wealth_gdp'][0]-1):+.1%}")

# ── Main Charts ─────────────────────────────────────────────────────
st.subheader("Housing Market Dynamics")

fig = make_subplots(
    rows=3, cols=3,
    subplot_titles=("Price-to-Income Ratio", "Mortgage Debt / GDP", "Loan-to-Value Ratio",
                    "Rent-to-Income Ratio", "Rental Yield", "Construction / GDP",
                    "Housing Wealth / GDP", "Mortgage Service Ratio", "Dwelling Stock Ratio"),
    vertical_spacing=0.08, horizontal_spacing=0.06
)

# Row 1
fig.add_trace(go.Scatter(x=hs['t'], y=hs['price_to_income'], name="P/I"), row=1, col=1)
fig.add_hline(y=hp.fundamental_price_income, line_dash="dash", line_color="gray",
              annotation_text="Fundamental", row=1, col=1)
fig.add_trace(go.Scatter(x=hs['t'], y=hs['mortgage_gdp'], name="Mortgage/GDP"), row=1, col=2)
fig.add_trace(go.Scatter(x=hs['t'], y=hs['lvr'], name="LVR"), row=1, col=3)
fig.add_hline(y=0.80, line_dash="dash", line_color="red", annotation_text="LVR cap", row=1, col=3)

# Row 2
fig.add_trace(go.Scatter(x=hs['t'], y=hs['rent_to_income'], name="Rent/Income"), row=2, col=1)
fig.add_trace(go.Scatter(x=hs['t'], y=hs['rental_yield'], name="Rental Yield"), row=2, col=2)
fig.add_hline(y=hp.rental_yield_target, line_dash="dash", line_color="gray",
              annotation_text="Target yield", row=2, col=2)
fig.add_trace(go.Scatter(x=hs['t'], y=hs['construction_gdp'], name="Construction"), row=2, col=3)
fig.add_hline(y=0.06, line_dash="dash", line_color="gray",
              annotation_text="Base", row=2, col=3)

# Row 3
fig.add_trace(go.Scatter(x=hs['t'], y=hs['housing_wealth_gdp'], name="Wealth/GDP"), row=3, col=1)
fig.add_trace(go.Scatter(x=hs['t'], y=hs['mortgage_service'], name="Service Ratio", fill='tozeroy'), row=3, col=2)
fig.add_trace(go.Scatter(x=hs['t'], y=hs['dwelling_stock'], name="Dwelling Stock"), row=3, col=3)
fig.add_hline(y=1.0, line_dash="dash", line_color="green",
              annotation_text="Adequate", row=3, col=3)

fig.update_layout(height=700, showlegend=False, margin=dict(l=20, r=20, t=40, b=20))
fig.update_xaxes(title_text="Years", row=3, col=1)
fig.update_xaxes(title_text="Years", row=3, col=2)
fig.update_xaxes(title_text="Years", row=3, col=3)
st.plotly_chart(_apply_theme(fig), width='stretch')

# ── Scenario Comparison ─────────────────────────────────────────────
st.subheader("Scenario Analysis")
scenario = st.radio(
    "Select scenario", ["Baseline", "Credit Boom", "Supply Shock", "Rate Hike", "Immigration Surge"],
    horizontal=True
)

if scenario == "Credit Boom":
    test_hp = HousingParams(credit_flow_elasticity=hp.credit_flow_elasticity * 2,
                            fundamental_price_income=hp.fundamental_price_income)
    label = "Credit Boom (2× elasticity)"
elif scenario == "Supply Shock":
    test_hp = HousingParams(structural_trend=hp.structural_trend + 0.005,
                            construction_elasticity=hp.construction_elasticity * 0.5)
    label = "Supply Shock (+0.5%/yr structural, -50% construction response)"
elif scenario == "Rate Hike":
    test_params = KeenParams(kappa1=kappa1, r=r + 0.03)
    sol_rate = simulate_keen(test_params, t_max=50)
    hs_rate = simulate_housing(sol_rate, hp)
    label = "Rate Hike (+300bp)"
elif scenario == "Immigration Surge":
    test_params = KeenParams(kappa1=kappa1, r=r, beta=0.030)
    sol_pop = simulate_keen(test_params, t_max=50)
    hs_pop = simulate_housing(sol_pop, hp)
    label = "Immigration Surge (3%/yr pop growth)"

if scenario == "Credit Boom":
    test_params = KeenParams(kappa1=kappa1 * 1.5, r=r)
    sol_test = simulate_keen(test_params, t_max=50)
    hs_test = simulate_housing(sol_test, test_hp)
elif scenario == "Supply Shock":
    test_params = KeenParams(kappa1=kappa1, r=r)
    sol_test = simulate_keen(test_params, t_max=50)
    hs_test = simulate_housing(sol_test, test_hp)
elif scenario in ("Rate Hike", "Immigration Surge"):
    pass  # already computed above

if scenario != "Baseline":
    if scenario in ("Rate Hike",):
        hs_test = hs_rate
        sol_test = sol_rate
    elif scenario in ("Immigration Surge",):
        hs_test = hs_pop
        sol_test = sol_pop

if scenario != "Baseline":
    fig2 = make_subplots(rows=1, cols=3,
                         subplot_titles=("Price-to-Income", "Mortgage/GDP", "Housing Wealth/GDP"))
    fig2.add_trace(go.Scatter(x=hs['t'], y=hs['price_to_income'], name="Baseline"), row=1, col=1)
    fig2.add_trace(go.Scatter(x=hs_test['t'], y=hs_test['price_to_income'], name=label, line=dict(dash='dash')), row=1, col=1)
    fig2.add_trace(go.Scatter(x=hs['t'], y=hs['mortgage_gdp'], name="Baseline"), row=1, col=2)
    fig2.add_trace(go.Scatter(x=hs_test['t'], y=hs_test['mortgage_gdp'], name=label, line=dict(dash='dash')), row=1, col=2)
    fig2.add_trace(go.Scatter(x=hs['t'], y=hs['housing_wealth_gdp'], name="Baseline"), row=1, col=3)
    fig2.add_trace(go.Scatter(x=hs_test['t'], y=hs_test['housing_wealth_gdp'], name=label, line=dict(dash='dash')), row=1, col=3)
    fig2.update_layout(height=350, margin=dict(l=20, r=20, t=30, b=20))
    st.plotly_chart(_apply_theme(fig2), width='stretch')

# ── Data Summary ────────────────────────────────────────────────────
with st.expander("Data Table (10-year snapshots)"):
    idx = np.arange(0, len(hs['t']), max(1, len(hs['t']) // 10))
    df = pd.DataFrame({
        "Year": hs['t'][idx].astype(int),
        "P/I Ratio": [f"{hs['price_to_income'][i]:.1f}" for i in idx],
        "Mortgage/GDP": [f"{hs['mortgage_gdp'][i]:.0%}" for i in idx],
        "LVR": [f"{hs['lvr'][i]:.0%}" for i in idx],
        "Rent/Income": [f"{hs['rent_to_income'][i]:.0%}" for i in idx],
        "Rental Yield": [f"{hs['rental_yield'][i]:.1%}" for i in idx],
        "Construction/GDP": [f"{hs['construction_gdp'][i]:.0%}" for i in idx],
        "Housing Wealth/GDP": [f"{hs['housing_wealth_gdp'][i]:.1f}" for i in idx],
    })
    st.dataframe(df, width='stretch')

# ── Methodological Notes ────────────────────────────────────────────
with st.expander("Model Methodology"):
    st.markdown("""
    ### Housing Sub-Model Methodology (Minsky-Style)
    
    **Framework:** The housing market is modelled as a credit-driven asset price system
    following Minsky's financial instability hypothesis:
    
    - **Prices are driven primarily by credit flows** (changes in debt), not by
      fundamentals. New credit entering the housing market pushes prices up through
      increased purchasing power.
    - **Mean reversion is very weak** — Australian data shows housing prices can remain
      far above fundamental value for decades (unlike equities).
    - **Structural trend** captures supply constraints (planning restrictions, land
      release policies), tax advantages (negative gearing, CGT discount), and foreign
      investment — factors that create a rising floor under prices.
    
    **Key equations:**
    
    - Δ(P/I) = f(credit_flow, income_growth, supply_gap, mean_reversion, demographics)
    - P(t) = P(t-1) × (1 + credit_push + income_effect + supply_effect + reversion)
    - Mortgage debt grows with total private debt + equity withdrawal when prices rise
    - LVR = LVR_prev - price_growth × cycle_sensitivity (banks tighten during booms)
    
    **Limitations:**
    - No explicit spatial/geographic disaggregation (national average)
    - No first-home buyer vs investor distinction
    - Rental market is simplified (no vacancy rate data feedback)
    - No explicit auction/transaction volume effects
    """)

st.caption("Sources: RBA Statistical Tables (D2, E1, G1), ABS National Accounts, CoreLogic data patterns, Professor Steve Keen's credit-driven asset pricing framework.")
