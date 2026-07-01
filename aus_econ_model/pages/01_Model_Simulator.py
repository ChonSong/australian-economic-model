"""
Page 1: Interactive Keen Model Simulator
========================================
The core of the app — run the differential equations model interactively,
adjust parameters, see the results in real time.
"""

import streamlit as st
import numpy as np
import time

from aus_econ_model.models.keen_model import (
    KeenParams, simulate_keen, simulate_keen_housing, KeenHousingParams, equilibrium_profit,
)
from aus_econ_model.components.charts import (
    plot_time_series, plot_phase_diagram, plot_debt_dynamics,
    plot_conceptual_diagram, plot_housing_submodel,
)
from aus_econ_model.components.explainers import (
    MODEL_EXPLAINER, KEY_EQUATIONS, AUSTRALIA_CONTEXT,
    model_explanation_section,
)

st.set_page_config(page_title="Model Simulator", page_icon="📈", layout="wide")

st.title("📈 Model Simulator")
st.markdown(
    "Adjust parameters and see how the Keen model responds. "
    "The system of three differential equations captures the core dynamics "
    "of a credit-driven economy."
)

# ── Parameters Panel ────────────────────────────────────────────────────────

with st.sidebar:
    st.header("🎛️ Model Parameters")

    # Initial Conditions
    with st.expander("Initial Conditions", expanded=True):
        omega0 = st.slider(
            "Wage Share (ω₀)", 0.30, 0.85, 0.62, 0.01,
            help="Labour's share of national income. Australia was ~55% in 2025, ~62% in 1990s."
        )
        lambda0 = st.slider(
            "Employment Rate (λ₀)", 0.80, 0.99, 0.94, 0.001,
            help="Fraction of labour force employed. 0.94 = 6% unemployment."
        )
        d0 = st.slider(
            "Private Debt / GDP (d₀)", 0.0, 4.0, 0.8, 0.05,
            help="Total private debt as ratio of GDP. Australia ~200% (2.0) now, ~80% in 1990s."
        )

    # Real Economy
    with st.expander("Real Economy", expanded=True):
        alpha = st.slider(
            "Productivity Growth (α)", 0.0, 0.05, 0.02, 0.005,
            help=model_explanation_section("alpha"),
        )
        beta = st.slider(
            "Labour Force Growth (β)", 0.0, 0.03, 0.02, 0.005,
            help=model_explanation_section("beta"),
        )
        delta = st.slider(
            "Depreciation (δ)", 0.0, 0.05, 0.02, 0.005,
            help=model_explanation_section("delta"),
        )
        nu = st.slider(
            "Capital/Output Ratio (ν)", 2.0, 8.0, 5.0, 0.5,
            help=model_explanation_section("nu"),
        )
        r = st.slider(
            "Real Interest Rate (r)", 0.0, 0.10, 0.03, 0.005,
            help=model_explanation_section("r"),
        )

    # Wage Phillips Curve
    with st.expander("Wage Bargaining (Phillips Curve)", expanded=False):
        phi_min = st.slider("Φ_min — Minimum wage floor", 0.10, 0.50, 0.25, 0.01,
                           help=model_explanation_section("phi_min"))
        phi_max = st.slider("Φ_max — Maximum wage at full employment", 0.60, 0.95, 0.80, 0.01,
                           help=model_explanation_section("phi_max"))
        phi_n = st.slider("Φ_n — Nonlinearity exponent", 1, 10, 4, 1,
                         help=model_explanation_section("phi_n"))

    # Investment Function
    with st.expander("Investment (Animal Spirits)", expanded=False):
        kappa1 = st.slider("κ₁ — Investment sensitivity to profit", 0.2, 0.95, 0.70, 0.05,
                          help=model_explanation_section("kappa1"))

    # Derived equilibrium display
    pi_star = nu * (alpha + beta + delta)
    st.caption(f"Equilibrium profit share π* = {pi_star:.0%} (implied by growth + ν)")

    # Simulation settings
    with st.expander("Simulation Settings", expanded=False):
        t_max = st.slider("Years to simulate", 20, 150, 60, 5)
        t_steps = st.select_slider("Output resolution",
                                   options=[500, 1000, 2000, 4000], value=2000)

    include_housing = st.toggle("🏠 Include Housing Sub-Model", value=False)

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        run_btn = st.button("▶️ Run Simulation", type="primary", width='stretch')
    with col2:
        reset_btn = st.button("↺ Reset to Defaults", width='stretch')

if reset_btn:
    st.rerun()

# ── Build Parameters & Run ─────────────────────────────────────────────────

if not run_btn:
    st.info("👈 **Set parameters in the sidebar and click 'Run Simulation'**")
    st.markdown("### How the model works")
    st.markdown(MODEL_EXPLAINER)

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(plot_conceptual_diagram(), width='stretch')
    with col2:
        st.markdown(KEY_EQUATIONS)
    st.stop()

# Build params
params = KeenParams(
    alpha=alpha, beta=beta, delta=delta, nu=nu, r=r,
    phi_min=phi_min, phi_max=phi_max, phi_n=phi_n,
    kappa1=kappa1,
    omega0=omega0, lambda0=lambda0, d0=d0,
)

with st.spinner("Simulating the model..."):
    start = time.time()

    if include_housing:
        housing_params = KeenHousingParams(
            alpha=alpha, beta=beta, delta=delta, nu=nu, r=r,
            phi_min=phi_min, phi_max=phi_max, phi_n=phi_n,
            kappa1=kappa1,
            omega0=omega0, lambda0=lambda0, d0=d0,
        )
        sol, housing = simulate_keen_housing(housing_params, t_max=t_max, t_steps=t_steps)
    else:
        sol = simulate_keen(params, t_max=t_max, t_steps=t_steps)
        housing = None

    elapsed = time.time() - start

# ── Results Display ─────────────────────────────────────────────────────────

if not sol.success:
    st.error(f"❌ Simulation failed: {sol.message}")
    st.stop()

st.success(f"✅ Simulation complete ({elapsed:.2f}s) — {len(sol.t)} time steps")

# Key metrics
st.subheader("📊 Key Results")
col1, col2, col3, col4 = st.columns(4)

with col1:
    final_omega = sol.omega[-1]
    delta_omega = (sol.omega[-1] - sol.omega[0]) / sol.omega[0] * 100
    st.metric("Final Wage Share", f"{final_omega:.1%}",
              delta=f"{delta_omega:+.1f}%")

with col2:
    final_lam = sol.lam[-1]
    delta_lam = (sol.lam[-1] - sol.lam[0]) / sol.lam[0] * 100
    st.metric("Final Employment", f"{final_lam:.1%}",
              delta=f"{delta_lam:+.1f}%")

with col3:
    final_d = sol.d[-1]
    delta_d = (sol.d[-1] - sol.d[0]) / sol.d[0] * 100
    st.metric("Final Debt/GDP", f"{final_d:.2f}",
              delta=f"{delta_d:+.1f}%", delta_color="inverse")

with col4:
    # Check for collapse (debt explodes or employment crashes)
    if final_d > 5 or final_lam < 0.5:
        st.metric("System State", "⚠️ COLLAPSE", delta="Unstable")
    elif final_d > sol.d[0] * 1.5:
        st.metric("System State", "⚠️ Deteriorating", delta="Debt rising")
    elif final_lam > sol.lam[0] * 0.95 and final_omega > sol.omega[0] * 0.95:
        st.metric("System State", "✅ Stable", delta="Balanced")
    else:
        st.metric("System State", "👀 Transition", delta="Watch")

# Time series plot
st.subheader("📈 Time Series")
st.plotly_chart(plot_time_series(sol, height=500), width='stretch')

# Debt dynamics panel
st.subheader("💰 Debt Dynamics")
col1, col2 = st.columns(2)
with col1:
    st.plotly_chart(plot_debt_dynamics(sol), width='stretch')
with col2:
    ds_ratio = sol.debt_service_ratio[-1]
    inv_ratio = sol.investment_share[-1]
    profit_ratio = sol.profit_share[-1]

    st.markdown("### What to look for")
    st.markdown(f"""
    - **Debt service** (r × d) = **{ds_ratio:.1%}** of GDP
    - **Investment** κ(π) = **{inv_ratio:.1%}** of GDP  
    - **Profit share** π = **{profit_ratio:.1%}** of GDP
    - **Equilibrium π*** = **{params.pi_star:.1%}** of GDP

    The model is stable when profit share oscillates around its equilibrium
    value π*. When debt service grows rapidly, it squeezes profits, which
    reduces investment and employment — a Minskyan downturn.

    **If debt service exceeds investment**, the system is in dangerous
    territory — interest payments consume the funds needed for capital
    formation.
    """)

    if ds_ratio > inv_ratio:
        st.warning("⚠️ **Debt service exceeds investment** — Minsky moment risk")
    if profit_ratio < 0:
        st.error("❌ **Negative profits** — systemic collapse imminent")
    if sol.d[-1] > 3.0:
        st.warning(f"⚠️ **Debt ratio ({sol.d[-1]:.1f})** is at crisis levels")

# Phase diagram
st.subheader("🌀 Phase Space")
col1, col2 = st.columns([3, 1])
with col1:
    st.plotly_chart(plot_phase_diagram(sol, height=500), width='stretch')
with col2:
    st.markdown("""
    ### Reading the phase diagram

    The 3D trajectory shows how the three state variables evolve together.

    - **X-axis**: Debt ratio (d)
    - **Y-axis**: Wage share (ω)
    - **Z-axis**: Employment (λ)
    - **Colour gradient**: Light → dark = earlier → later
    - **Green dot**: Start
    - **Red cross**: End

    **What to look for:**
    - A **stable spiral** converging to a point → equilibrium
    - A **limit cycle** → business cycle oscillations
    - **Explosive divergence** → crisis/collapse
    """)

# Housing sub-model
if include_housing and housing:
    st.subheader("🏠 Housing Sub-Model")
    st.plotly_chart(plot_housing_submodel(sol, housing), width='stretch')

    aff = housing["affordability"][-1]
    prv = housing["price_to_income"][-1]
    st.markdown(f"""
    - Final price-to-income ratio: **{prv:.1f}x**
    - Affordability index: **{aff:.2f}** (higher = more affordable)
    - Price growth rate (final): **{housing['price_growth'][-1]:.1f}%** p.a.
    """)

# Parameter exploration
st.subheader("🔬 Sensitivity Explorer")
st.markdown("Change one parameter at a time to see its effect.")

with st.expander("Run sensitivity comparison", expanded=False):
    sens_param = st.selectbox(
        "Parameter to vary",
        ["r (interest rate)", "alpha (productivity)", "beta (pop growth)",
         "kappa1 (investment sensitivity)", "nu (capital/output)",
         "phi_n (Phillips curvature)", "phi_min (wage floor)"],
    )

    sens_range = st.slider("Variation range (±%)", 10, 100, 50, 10)

    param_map = {
        "r (interest rate)": ("r", 0.03),
        "alpha (productivity)": ("alpha", 0.02),
        "beta (pop growth)": ("beta", 0.02),
        "kappa1 (investment sensitivity)": ("kappa1", 0.70),
        "nu (capital/output)": ("nu", 5.0),
        "phi_n (Phillips curvature)": ("phi_n", 4),
        "phi_min (wage floor)": ("phi_min", 0.25),
    }

    attr, base_val = param_map[sens_param]
    low_val = max(0.001, base_val * (1 - sens_range / 100))
    high_val = base_val * (1 + sens_range / 100)

    cols = st.columns(3)
    results = []
    for i, (label, val) in enumerate(zip(
        ["Low", "Baseline", "High"],
        [low_val, base_val, high_val]
    )):
        p = KeenParams(
            alpha=alpha, beta=beta, delta=delta, nu=nu, r=r,
            phi_min=phi_min, phi_max=phi_max, phi_n=phi_n,
            kappa1=kappa1,
            omega0=omega0, lambda0=lambda0, d0=d0,
        )
        setattr(p, attr, val)
        p._label = f"{label} ({val:.3f})"
        sol_sens = simulate_keen(p, t_max=t_max, t_steps=t_steps)
        results.append((p, sol_sens))

        with cols[i]:
            if sol_sens.success:
                st.metric(
                    f"{label}: {attr} = {val:.3f}",
                    f"d={sol_sens.d[-1]:.2f}, λ={sol_sens.lam[-1]:.1%}",
                )
            else:
                st.metric(f"{label}: {attr} = {val:.3f}", "❌ Collapsed")

    if len(results) == 3:
        from aus_econ_model.components.charts import plot_parameter_sensitivity
        st.plotly_chart(
            plot_parameter_sensitivity(results[1][1], [r[0] for r in results], [r[1] for r in results]),
            width='stretch',
        )

# Technical details
with st.expander("📐 Technical Notes", expanded=False):
    st.markdown(f"""
    - **Solver**: RK45 (adaptive step), max_step=0.5yr
    - **Time horizon**: {t_max} years, {t_steps} output points
    - **Success**: {sol.success}
    - **Final state**: ω={sol.omega[-1]:.3f}, λ={sol.lam[-1]:.4f}, d={sol.d[-1]:.3f}
    - **Equilibrium π*** = {params.pi_star:.1%}
    - **Investment slope κ₁**: {params.kappa1}

    The model is calibrated so that the equilibrium profit share
    π* = ν(α+β+δ) {params.pi_star:.1%} is consistent with the capital
    accumulation needed for steady-state growth. When π > π* the debt
    ratio falls; when π < π* it rises.

    **Numerical safeguards**: State variables are clipped to prevent
    values below zero. The solver uses tight tolerances (rtol=1e-8, atol=1e-10).
    """)

st.caption("Model based on Keen (1995, 2013). Implementation: Python + SciPy + Streamlit.")
