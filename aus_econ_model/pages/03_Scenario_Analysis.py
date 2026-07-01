"""
Page 3: Scenario Analysis
=========================
Test what-if scenarios: policy changes, parameter shocks, and structural shifts.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from aus_econ_model.models.keen_model import KeenParams, simulate_keen
from aus_econ_model.components.charts import (
    plot_time_series, plot_parameter_sensitivity, THEME, COLOURS, _apply_theme,
)

st.set_page_config(page_title="Scenario Analysis", page_icon="🔬", layout="wide")

st.title("🔬 Scenario Analysis")
st.markdown(
    "Test how different policy or structural changes affect the model's "
    "trajectory. Each scenario modifies one or more parameters relative "
    "to a baseline."
)


# ── Helpers (defined before use — Streamlit reruns top-to-bottom) ───────────

def _make_comparison_chart(results, attr, title, colour):
    fig = go.Figure()
    for name, r in results.items():
        sol = r["solution"]
        if not sol.success:
            continue
        values = getattr(sol, attr)
        fig.add_trace(go.Scatter(
            x=sol.t, y=values, mode="lines",
            name=name, line=dict(width=2),
        ))
    fig.update_layout(title=title, height=400, hovermode="x unified")
    fig.update_xaxes(title_text="Years")
    fig.update_yaxes(title_text=title)
    return _apply_theme(fig)


# ── Baseline ────────────────────────────────────────────────────────────────

BASE_PARAMS = KeenParams()  # Uses new defaults: alpha=0.02, beta=0.02, nu=5.0, etc.

with st.sidebar:
    st.header("🎛️ Baseline Parameters")
    st.markdown("These are the starting conditions for all scenarios.")

    with st.expander("Initial Conditions", expanded=True):
        omega0 = st.slider("Wage Share (ω₀)", 0.30, 0.85, 0.62, 0.01)
        lambda0 = st.slider("Employment (λ₀)", 0.80, 0.99, 0.94, 0.005)
        d0 = st.slider("Debt/GDP (d₀)", 0.0, 4.0, 0.80, 0.05)

    with st.expander("Real Economy", expanded=False):
        alpha = st.slider("Productivity (α)", 0.0, 0.05, 0.02, 0.005)
        beta = st.slider("Pop growth (β)", 0.0, 0.03, 0.02, 0.005)
        delta = st.slider("Depreciation (δ)", 0.0, 0.05, 0.02, 0.005)
        nu = st.slider("Capital/output (ν)", 2.0, 8.0, 5.0, 0.5)
        r = st.slider("Interest (r)", 0.0, 0.10, 0.03, 0.005)

    t_max = st.slider("Simulation years", 20, 100, 50, 5)

# ── Scenario Definitions ────────────────────────────────────────────────────

st.subheader("Choose Scenarios to Compare")

scenarios_config = {
    "Baseline": {
        "params": {},
        "description": "Current trends continued — no policy change.",
    },
    "Immigration Cap": {
        "params": {"beta": 0.005},
        "description": "Reduce population growth from 2% to 0.5% (tight immigration cap). "
                       "Slows labour force growth but also reduces housing demand.",
    },
    "Rate Hike": {
        "params": {"r": 0.06},
        "description": "RBA raises real rates to 6% to fight inflation. "
                       "Increases debt service burden significantly.",
    },
    "Wage Recovery": {
        "params": {"phi_min": 0.35, "phi_max": 0.85},
        "description": "Stronger union bargaining + tighter labour market. "
                       "Wage share floor rises, giving workers more bargaining power.",
    },
    "Productivity Boom": {
        "params": {"alpha": 0.035},
        "description": "AI/automation drives productivity growth to 3.5% p.a. "
                       "Can raise living standards but may reduce employment.",
    },
    "Credit Crunch": {
        "params": {"kappa1": 0.35},
        "description": "Banks tighten lending — investment less responsive "
                       "to profitability. Lower debt accumulation but slower growth.",
    },
    "Debt Jubilee": {
        "params": {"d0": 1.0},
        "description": "Starting from much lower debt (100% of GDP). "
                       "Simulates a debt write-down or restructuring event.",
    },
    "High-Inflation Erosion": {
        "params": {"r": 0.01},
        "description": "Real rates near zero — inflation erodes debt values. "
                       "Boon for debtors but punishes savers.",
    },
    "Stagflation": {
        "params": {"alpha": 0.005, "kappa1": 0.35},
        "description": "Low productivity + collapsed investment. Stagflation scenario.",
    },
}

selected_scenarios = []
cols = st.columns(3)
for i, (name, cfg) in enumerate(scenarios_config.items()):
    with cols[i % 3]:
        on = st.checkbox(name, value=(name == "Baseline"),
                         help=cfg["description"])
        if on:
            selected_scenarios.append(name)

if not selected_scenarios:
    st.warning("Select at least one scenario (Baseline is selected by default).")
    st.stop()

# ── Run Scenarios ───────────────────────────────────────────────────────────

if st.button("▶️ Run Scenarios", type="primary", width='stretch'):

    results = {}
    for name in selected_scenarios:
        cfg = scenarios_config[name]

        p = KeenParams(
            omega0=omega0, lambda0=lambda0, d0=d0,
            alpha=alpha, beta=beta, delta=delta, nu=nu, r=r,
        )
        for attr, val in cfg["params"].items():
            setattr(p, attr, val)
        p._label = name

        sol = simulate_keen(p, t_max=t_max, t_steps=2000)
        results[name] = {"params": p, "solution": sol}

    st.success(f"Ran {len(results)} scenarios")

    # ── Comparison Dashboard ────────────────────────────────────────────────

    st.subheader("📊 Scenario Comparison")

    # Metrics table
    metrics_data = []
    for name, r in results.items():
        sol = r["solution"]
        if sol.success:
            metrics_data.append({
                "Scenario": name,
                "Final ω": f"{sol.omega[-1]:.1%}",
                "Final λ": f"{sol.lam[-1]:.1%}",
                "Final d": f"{sol.d[-1]:.2f}",
                "Avg Growth": f"{np.mean(sol.growth_rate[sol.t > 5]):.1%} p.a.",
                "Debt Service": f"{sol.debt_service_ratio[-1]:.1%}",
                "Status": "✅" if sol.lam[-1] > 0.80 and sol.d[-1] < 4 else "⚠️",
            })
        else:
            metrics_data.append({
                "Scenario": name,
                "Final ω": "❌", "Final λ": "❌", "Final d": "❌",
                "Avg Growth": "❌", "Debt Service": "❌",
                "Status": "💀 Collapsed",
            })

    df_metrics = pd.DataFrame(metrics_data)
    st.dataframe(df_metrics, width='stretch', hide_index=True)

    # ── Comparison Charts ───────────────────────────────────────────────────

    st.subheader("📈 Debt Ratio Over Time")
    fig_debt = _make_comparison_chart(results, "d", "Debt / GDP", "#F39C12")
    st.plotly_chart(fig_debt, width='stretch')

    st.subheader("📈 Employment Rate Over Time")
    fig_emp = _make_comparison_chart(results, "lam", "Employment Rate", "#3498DB")
    st.plotly_chart(fig_emp, width='stretch')

    st.subheader("📈 Wage Share Over Time")
    fig_wage = _make_comparison_chart(results, "omega", "Wage Share", "#E74C3C")
    st.plotly_chart(fig_wage, width='stretch')

    # ── Summary interpretation ──────────────────────────────────────────────

    st.subheader("📝 Interpretation")

    stable_results = {n: r for n, r in results.items() if r["solution"].success}
    if stable_results:
        best_debt = min(stable_results.items(), key=lambda x: x[1]["solution"].d[-1])
        best_emp = max(stable_results.items(), key=lambda x: x[1]["solution"].lam[-1])

        st.markdown(f"""
        - **Best debt outcome**: *{best_debt[0]}* — final debt ratio = **{best_debt[1]['solution'].d[-1]:.2f}**
        - **Best employment outcome**: *{best_emp[0]}* — final employment = **{best_emp[1]['solution'].lam[-1]:.1%}**

        **Key insight**: In the Keen model, there is no single "good" scenario
        without trade-offs. Lower debt usually means lower growth. Higher
        employment usually means rising wage share and compressed profits.
        The question is which trade-off is sustainable.
        """)

        for name, r in results.items():
            sol = r["solution"]
            if sol.success:
                min_profit = np.min(sol.profit_share)
                if min_profit < 0:
                    st.warning(f"⚠️ **{name}**: Profits turned negative at some point "
                               f"(min π = {min_profit:.1%}). This would cause a crisis "
                               f"in a full model.")

    # ── Detailed per-scenario ───────────────────────────────────────────────

    st.subheader("🔍 Individual Scenario Details")

    tabs = st.tabs(list(results.keys()))
    for i, (name, r) in enumerate(results.items()):
        with tabs[i]:
            sol = r["solution"]
            if sol.success:
                st.plotly_chart(plot_time_series(sol, height=350), width='stretch')

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Final Wage Share", f"{sol.omega[-1]:.1%}")
                with col2:
                    st.metric("Final Employment", f"{sol.lam[-1]:.1%}")
                with col3:
                    st.metric("Final Debt/GDP", f"{sol.d[-1]:.2f}")

                st.markdown(f"**Description**: {scenarios_config[name]['description']}")

                param_changes = scenarios_config[name]["params"]
                if param_changes:
                    st.markdown("**Parameters changed:**")
                    for attr, val in param_changes.items():
                        base_val = getattr(BASE_PARAMS, attr, "?")
                        st.markdown(f"- `{attr}`: {base_val} → {val}")
            else:
                st.error(f"Scenario collapsed: {sol.message}")

else:
    st.info("Select scenarios above and click **Run Scenarios** to begin.")


# ── Helper ──────────────────────────────────────────────────────────────────


st.caption("Scenarios are comparative — they show relative dynamics, not predictions.")
