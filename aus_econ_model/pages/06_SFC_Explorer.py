"""
Page 6: SFC Explorer — Extended Stock-Flow Consistent Model
=============================================================
Clean rewrite after corruption.
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from aus_econ_model.models.sfc_model import (
    ExtendedKeenParams, simulate_extended,
    build_flow_of_funds, build_balance_sheet,
)
from aus_econ_model.models.keen_model import KeenParams, simulate_keen
from aus_econ_model.components.charts import THEME, COLOURS, _apply_theme, plot_time_series

st.set_page_config(page_title="SFC Explorer", page_icon="🏦", layout="wide")

st.title("🏦 SFC Explorer — Extended Stock-Flow Consistent Model")
st.markdown(
    "Extends the Keen model with **government**, **external**, and **banking** "
    "sectors in a stock-flow consistent (SFC) framework. "
    "Adjust parameters in the sidebar and explore sectoral interactions."
)

# ── Causal diagram (defined before first use) ─────────────────────────────
def _plot_sfc_diagram() -> go.Figure:
    nodes = {
        "Fiscal\nPolicy": (0, 2), "Gov\nDebt": (0, 1),
        "Gov\nSpending": (1, 1.5), "Taxes": (0, 0.5),
        "Private\nDebt": (2, 1.5), "Aggregate\nDemand": (3, 1),
        "Credit\nConstraint": (2, 0.5), "Employment": (4, 0.5),
        "Wage\nShare": (5, 1), "Profit\nShare": (4, 1.5),
        "Bank\nCapital": (2.5, 0), "Foreign\nDebt": (4, 0),
        "Trade\nBalance": (3.5, 0.5), "Inflation\nExpectations": (5.5, 0.5),
    }
    edges = [
        ("Private\nDebt", "Aggregate\nDemand", "+", "Borrowing adds to demand"),
        ("Aggregate\nDemand", "Employment", "+", "More demand → more hiring"),
        ("Employment", "Wage\nShare", "+", "Tight labour → higher wages"),
        ("Wage\nShare", "Profit\nShare", "−", "Wages squeeze profits"),
        ("Profit\nShare", "Aggregate\nDemand", "+", "Profits fund investment"),
        ("Profit\nShare", "Private\nDebt", "+", "Expectations drive borrowing"),
        ("Fiscal\nPolicy", "Gov\nSpending", "+", "Policy determines spending"),
        ("Fiscal\nPolicy", "Taxes", "+", "Policy determines tax rates"),
        ("Gov\nSpending", "Aggregate\nDemand", "+", "Gov spending adds to AD"),
        ("Taxes", "Aggregate\nDemand", "−", "Taxes reduce disposable income"),
        ("Gov\nDebt", "Fiscal\nPolicy", "−", "Debt constrains fiscal space"),
        ("Gov\nSpending", "Gov\nDebt", "+", "Deficit adds to debt"),
        ("Bank\nCapital", "Credit\nConstraint", "−", "Weak capital → crunch"),
        ("Credit\nConstraint", "Private\nDebt", "−", "Crunch reduces lending"),
        ("Credit\nConstraint", "Aggregate\nDemand", "−", "Crunch reduces AD"),
        ("Profit\nShare", "Bank\nCapital", "+", "Profits boost bank capital"),
        ("Trade\nBalance", "Foreign\nDebt", "−", "Deficit adds to foreign debt"),
        ("Aggregate\nDemand", "Trade\nBalance", "−", "More demand → more imports"),
        ("Foreign\nDebt", "Aggregate\nDemand", "−", "Debt service leaks abroad"),
        ("Wage\nShare", "Inflation\nExpectations", "+", "High wages → inflation"),
        ("Inflation\nExpectations", "Wage\nShare", "+", "Indexation spiral"),
        ("Employment", "Inflation\nExpectations", "+", "Full emp → inflation"),
    ]

    fig = go.Figure()
    nx = [v[0] for v in nodes.values()]
    ny = [v[1] for v in nodes.values()]
    labels = list(nodes.keys())

    fig.add_trace(go.Scatter(
        x=nx, y=ny, mode="markers+text",
        marker=dict(size=28, color="#2C3E50", line=dict(color="#3498DB", width=2)),
        text=labels, textposition="middle center",
        textfont=dict(size=9, color="#FAFAFA"), hoverinfo="text",
    ))

    for src, dst, sign, tooltip in edges:
        if src not in nodes or dst not in nodes:
            continue
        sx, sy = nodes[src]
        dx, dy = nodes[dst]
        mx, my = (sx + dx) / 2, (sy + dy) / 2
        fig.add_annotation(
            x=dx, y=dy, ax=sx, ay=sy, xref="x", yref="y", axref="x", ayref="y",
            showarrow=True, arrowhead=2, arrowsize=1.2, arrowwidth=1.5,
            arrowcolor="#7F8C8D",
        )
        fig.add_annotation(
            x=mx + 0.1, y=my + 0.15,
            text=f"<b>{sign}</b>", showarrow=False,
            font=dict(size=14, color="#E74C3C" if sign == "−" else "#2ECC71"),
        )

    fig.update_layout(
        xaxis=dict(visible=False, range=[-0.5, 6.5]),
        yaxis=dict(visible=False, range=[-0.5, 2.8]),
        height=500, title="Extended SFC Model — Causal Structure", showlegend=False,
    )
    return _apply_theme(fig)


st.plotly_chart(_plot_sfc_diagram(), width='stretch')

# ── Sidebar: Parameters ────────────────────────────────────────────────────
with st.sidebar:
    st.header("SFC Model Parameters")

    with st.expander("Initial Conditions", expanded=True):
        omega0 = st.slider("Wage Share (ω₀)", 0.30, 0.85, 0.62, 0.01)
        lambda0 = st.slider("Employment Rate (λ₀)", 0.80, 0.99, 0.94, 0.005)
        d0 = st.slider("Private Debt/GDP (d₀)", 0.0, 4.0, 0.80, 0.05)
        d_g0 = st.slider("Gov Debt/GDP (d_g₀)", 0.0, 2.0, 0.55, 0.05)
        d_f0 = st.slider("Foreign Debt/GDP (d_f₀)", 0.0, 2.0, 0.60, 0.05)
        pi_e0 = st.slider("Expected Inflation (π_e₀)", 0.0, 0.10, 0.025, 0.005)
        bcr0 = st.slider("Bank Capital Ratio (BCR₀)", 0.05, 0.25, 0.12, 0.01)

    with st.expander("Real Economy", expanded=True):
        alpha = st.slider("Productivity Growth (α)", 0.0, 0.05, 0.02, 0.005)
        beta = st.slider("Labour Force Growth (β)", 0.0, 0.03, 0.02, 0.005)
        delta = st.slider("Depreciation (δ)", 0.0, 0.05, 0.02, 0.005)
        nu = st.slider("Capital/Output (ν)", 2.0, 8.0, 5.0, 0.5)

    with st.expander("🏛️ Government", expanded=False):
        tax_rate_wages = st.slider("Tax Rate — Wages (τ_w)", 0.10, 0.50, 0.30, 0.01)
        tax_rate_profits = st.slider("Tax Rate — Profits (τ_π)", 0.10, 0.50, 0.35, 0.01)
        g0 = st.slider("Autonomous Spending (g₀)", 0.05, 0.35, 0.16, 0.01)
        g1 = st.slider("Proportional Spending (g₁)", 0.05, 0.30, 0.16, 0.01)
        g2 = st.slider("Counter-cyclical (g₂)", 0.0, 0.30, 0.12, 0.01)
        g3 = st.slider("Debt Stabilisation (g₃)", 0.0, 0.10, 0.03, 0.005)
        d_g_target = st.slider("Target Gov Debt/GDP", 0.20, 1.50, 0.50, 0.05)
        r_g = st.slider("Gov Bond Rate (r_g)", 0.01, 0.08, 0.035, 0.005)

    with st.expander("🌏 External Sector", expanded=False):
        export_share = st.slider("Export Share of GDP", 0.05, 0.40, 0.20, 0.01)
        import_share = st.slider("Import Share of GDP", 0.10, 0.50, 0.22, 0.01)
        r_f = st.slider("Foreign Interest Rate (r_f)", 0.01, 0.10, 0.04, 0.005)
        net_income_abroad = st.slider("Net Income Abroad (% GDP)", -0.05, 0.05, -0.01, 0.005)

    with st.expander("🏦 Banking Sector", expanded=False):
        bcr_target = st.slider("Target Bank Capital Ratio (BCR)", 0.08, 0.25, 0.12, 0.01)
        min_lending_rate = st.slider("Min Lending Rate Spread", 0.01, 0.08, 0.03, 0.005)

    with st.expander("Extended Phillips Curve", expanded=False):
        inflation_sensitivity = st.slider("Inflation Sensitivity (κ_π)", 0.1, 2.0, 0.8, 0.1)
        wage_indexation = st.slider("Wage Indexation (κ_w)", 0.0, 1.0, 0.5, 0.05)

    with st.expander("Simulation Settings", expanded=False):
        sim_years = st.slider("Years to simulate", 10, 100, 50, 5)
        compare_base = st.checkbox("Compare with base Keen model", value=True)

# ── Run simulation ─────────────────────────────────────────────────────────
run_btn = st.button("▶️ Run SFC Simulation", type="primary")

if not run_btn:
    st.info("Set parameters in the sidebar and click **Run SFC Simulation**.")
    st.stop()

with st.spinner("Running SFC model..."):
    params = ExtendedKeenParams(
        alpha=alpha, beta=beta, delta=delta, nu=nu,
        omega0=omega0, lambda0=lambda0, d0=d0,
        d_g0=d_g0, d_f0=d_f0, pi_e0=pi_e0, bcr0=bcr0,
        phi_min=min_lending_rate, phi_max=min_lending_rate + 0.05, phi_n=1,
        kappa1=0.5,
        # Government
        tax_rate_wages=tax_rate_wages, tax_rate_profits=tax_rate_profits,
        g0=g0, g1=g1, g2=g2, g3=g3,
        d_g_target=d_g_target, r_g=r_g,
        # External
        export_share=export_share, import_share=import_share,
        r_f=r_f, net_income_abroad=net_income_abroad,
        # Banking
        bcr_target=bcr_target,
        # Inflation
        inflation_sensitivity=inflation_sensitivity,
        wage_indexation=wage_indexation,
    )
    sfc_sol = simulate_extended(params, t_max=sim_years)

# ── Key indicators ─────────────────────────────────────────────────────────
st.subheader("📊 Key SFC Indicators")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Final Wage Share", f"{sfc_sol.omega[-1]:.1%}", f"{sfc_sol.omega[-1]-sfc_sol.omega[0]:+.1%}")
c2.metric("Final Employment", f"{sfc_sol.lam[-1]:.1%}", f"{sfc_sol.lam[-1]-sfc_sol.lam[0]:+.1%}")
c3.metric("Private Debt/GDP", f"{sfc_sol.d[-1]:.2f}x", f"{sfc_sol.d[-1]-sfc_sol.d[0]:+.2f}")
c4.metric("Gov Debt/GDP", f"{sfc_sol.d_g[-1]:.1%}", f"{sfc_sol.d_g[-1]-sfc_sol.d_g[0]:+.1%}")

# ── Time series ────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["Core Variables", "Sectoral Balances", "Banking & Inflation"])

with tab1:
    fig = make_subplots(rows=2, cols=2,
                        subplot_titles=("Wage Share ω", "Employment λ",
                                        "Private Debt/GDP d", "Gov Debt/GDP d_g"))
    fig.add_trace(go.Scatter(x=sfc_sol.t, y=sfc_sol.omega, name="ω", line=dict(color=COLOURS[0])), row=1, col=1)
    fig.add_trace(go.Scatter(x=sfc_sol.t, y=sfc_sol.lam, name="λ", line=dict(color=COLOURS[1])), row=1, col=2)
    fig.add_trace(go.Scatter(x=sfc_sol.t, y=sfc_sol.d, name="d", line=dict(color=COLOURS[2])), row=2, col=1)
    fig.add_trace(go.Scatter(x=sfc_sol.t, y=sfc_sol.d_g, name="d_g", line=dict(color=COLOURS[3])), row=2, col=2)
    for r in range(1, 3):
        for c in range(1, 3):
            fig.update_xaxes(title_text="Years" if r == 2 else "", row=r, col=c)
    fig.update_layout(height=500, showlegend=False, hovermode="x unified")
    st.plotly_chart(_apply_theme(fig), width='stretch')

    base_sol = None
    if compare_base:
        base_p = KeenParams(alpha=alpha, beta=beta, delta=delta, nu=nu,
                            omega0=omega0, lambda0=lambda0, d0=d0)
        base_sol = simulate_keen(base_p, t_max=sim_years)
        fig2 = make_subplots(rows=2, cols=2,
                             subplot_titles=("Wage Share: SFC vs Base", "Employment: SFC vs Base",
                                             "Private Debt: SFC vs Base", "Gov Debt"))
        fig2.add_trace(go.Scatter(x=sfc_sol.t, y=sfc_sol.omega, name="SFC ω", line=dict(color=COLOURS[0])), row=1, col=1)
        fig2.add_trace(go.Scatter(x=base_sol.t, y=base_sol.omega, name="Base ω", line=dict(color=COLOURS[0], dash="dash")), row=1, col=1)
        fig2.add_trace(go.Scatter(x=sfc_sol.t, y=sfc_sol.lam, name="SFC λ", line=dict(color=COLOURS[1])), row=1, col=2)
        fig2.add_trace(go.Scatter(x=base_sol.t, y=base_sol.lam, name="Base λ", line=dict(color=COLOURS[1], dash="dash")), row=1, col=2)
        fig2.add_trace(go.Scatter(x=sfc_sol.t, y=sfc_sol.d, name="SFC d", line=dict(color=COLOURS[2])), row=2, col=1)
        fig2.add_trace(go.Scatter(x=base_sol.t, y=base_sol.d, name="Base d", line=dict(color=COLOURS[2], dash="dash")), row=2, col=1)
        fig2.add_trace(go.Scatter(x=sfc_sol.t, y=sfc_sol.d_g, name="Gov Debt d_g", line=dict(color=COLOURS[3])), row=2, col=2)
        for r in range(1, 3):
            for c in range(1, 3):
                fig2.update_xaxes(title_text="Years" if r == 2 else "", row=r, col=c)
        fig2.update_layout(height=500, showlegend=True, hovermode="x unified")
        st.plotly_chart(_apply_theme(fig2), width='stretch')

with tab2:
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Fiscal Dynamics")
        st.markdown(f"""
        - **Primary deficit**: {sfc_sol.gov_primary_deficit[-1]:.1%} of GDP
        - **Total deficit**: {sfc_sol.gov_total_deficit[-1]:.1%} of GDP
        - **Tax revenue**: {sfc_sol.tax_revenue[-1]:.1%} of GDP
        - **Gov spending**: {sfc_sol.gov_spending_primary[-1]:.1%} of GDP
        - **Debt service**: {params.r_g * sfc_sol.d_g[-1]:.1%} of GDP
        - **Debt dynamics**: Δd_g = G−T−g·d_g = {sfc_sol.d_g[-1] - sfc_sol.d_g[0]:+.2f}
        """)
        impulse = sfc_sol.gov_spending_primary - sfc_sol.tax_revenue
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=sfc_sol.t, y=impulse * 100, mode="lines",
                                   name="Fiscal Impulse", fill="tozeroy",
                                   line=dict(color=COLOURS[4])))
        fig3.update_layout(height=350, title="Fiscal Impulse (% GDP)",
                           xaxis_title="Years", yaxis_title="% GDP")
        st.plotly_chart(_apply_theme(fig3), width='stretch')

    with col2:
        st.markdown("### Trade & External Balance")
        st.markdown(f"""
        - **Export share**: {sfc_sol.export_share[-1]:.1%} of GDP
        - **Import share**: {sfc_sol.import_share[-1]:.1%} of GDP
        - **Trade balance**: {sfc_sol.trade_balance[-1]:.1%} of GDP
        - **Current account**: {sfc_sol.current_account[-1]:.1%} of GDP
        - **Foreign interest**: {params.r_f * sfc_sol.d_f[-1]:.1%} of GDP
        - **Debt dynamics**: Δd_f = M−X+r_f·d_f − g·d_f = {sfc_sol.d_f[-1] - sfc_sol.d_f[0]:+.2f}
        """)
        fig4 = go.Figure()
        fig4.add_trace(go.Scatter(x=sfc_sol.t, y=sfc_sol.trade_balance * 100, mode="lines",
                                   name="Trade Balance", line=dict(color=COLOURS[5])))
        fig4.add_trace(go.Scatter(x=sfc_sol.t, y=sfc_sol.current_account * 100, mode="lines",
                                   name="Current Account", line=dict(color=COLOURS[6])))
        fig4.add_hline(y=0, line_color="white", line_dash="dot")
        fig4.update_layout(height=350, title="External Sector (% GDP)",
                           xaxis_title="Years", yaxis_title="% GDP")
        st.plotly_chart(_apply_theme(fig4), width='stretch')

with tab3:
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Credit & Banking Conditions")
        st.markdown(f"""
        - **Bank capital ratio**: {sfc_sol.bcr[-1]:.1%} (target: {params.bcr_target:.1%})
        - **Private debt/GDP**: {sfc_sol.d[-1]:.2f}
        - **Debt service ratio**: {sfc_sol.debt_service_ratio[-1]:.1%}
        - **Credit constraint**: {'Active' if sfc_sol.bcr[-1] < params.bcr_target else 'Relaxed'}
        """)
        fig5 = go.Figure()
        fig5.add_trace(go.Scatter(x=sfc_sol.t, y=sfc_sol.bcr * 100, mode="lines",
                                   name="Bank Capital Ratio", line=dict(color=COLOURS[7])))
        fig5.add_hline(y=params.bcr_target * 100, line_dash="dash",
                       annotation_text=f"Target {params.bcr_target:.0%}")
        fig5.update_layout(height=300, title="Bank Capital Ratio (%)",
                           xaxis_title="Years", yaxis_title="%")
        st.plotly_chart(_apply_theme(fig5), width='stretch')

    with col2:
        st.markdown("### Inflation Expectations")
        st.markdown(f"""
        - **Expected inflation**: {sfc_sol.pi_e[-1]:.1%}
        - **Inflation sensitivity**: {params.inflation_sensitivity:.1f}
        - **Wage indexation**: {params.wage_indexation:.0%}
        """)
        fig6 = go.Figure()
        fig6.add_trace(go.Scatter(x=sfc_sol.t, y=sfc_sol.pi_e * 100, mode="lines",
                                   name="Expected Inflation", line=dict(color=COLOURS[8])))
        fig6.add_trace(go.Scatter(x=sfc_sol.t, y=sfc_sol.omega * 100, mode="lines",
                                   name="Wage Share", line=dict(color=COLOURS[0], dash="dot")))
        fig6.update_layout(height=300, title="Inflation & Wage Share",
                           xaxis_title="Years", yaxis_title="%")
        st.plotly_chart(_apply_theme(fig6), width='stretch')

# ── Flow of funds matrix ───────────────────────────────────────────────────
st.divider()
st.subheader("💹 Flow-of-Funds Matrix")
with st.expander("Show flow-of-funds table"):
    fof = build_flow_of_funds(sfc_sol)
    st.dataframe(fof.head(20), width='stretch')

# ── Balance sheet ──────────────────────────────────────────────────────────
st.subheader("🏦 Balance Sheet Snapshot (t=final)")
with st.expander("Show balance sheet"):
    bs = build_balance_sheet(sfc_sol)
    st.dataframe(bs, width='stretch')

# ── Methodological notes ───────────────────────────────────────────────────
with st.expander("📝 SFC Methodology Notes"):
    st.markdown("""
    ### Seven state variables in the extended model

    | Variable | Symbol | Meaning |
    |----------|--------|---------|
    | Wage share | ω | Labour's share of GDP |
    | Employment | λ | Fraction of labour force employed |
    | Private debt | d | Private debt / GDP |
    | Gov debt | d_g | Government net debt / GDP |
    | Foreign debt | d_f | Net foreign liabilities / GDP |
    | Inflation exp. | π_e | Expected inflation rate |
    | Bank capital | bcr | Bank capital ratio |

    ### Key assumptions
    - **Government**: fiscal rule with autonomous + proportional + counter-cyclical + debt-stabilisation components
    - **External**: fixed export/import shares, constant world interest rate
    - **Banking**: capital ratio target constrains lending via Minsky mechanism
    - **Inflation**: adaptive expectations with wage-price indexation
    """)
