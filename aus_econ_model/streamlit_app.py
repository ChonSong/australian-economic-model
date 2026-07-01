# aus_econ_model/streamlit_app.py
import streamlit as st

st.set_page_config(
    page_title="Australia Econ Model — Keen Approach",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar ─────────────────────────────────────────────────────────────────

st.sidebar.markdown("# Australia Economic Model")
st.sidebar.markdown("**Keen–Goodwin–Minsky Framework**")
st.sidebar.divider()

st.sidebar.markdown("""
### How to use

- **Model Simulator** — interactively explore the Keen model dynamics
- **Data Explorer** — visualise Australian economic time series
- **Scenario Analysis** — test what-if policy scenarios
- **Living Standards** — composite welfare metrics (integrates all sub-models)
- **SFC Explorer** — stock-flow consistent extended model
- **Housing** — credit-driven housing market sub-model
- **Resources** — natural resource depletion & export revenue model
- **About** — methodology, data sources, Steve Keen's approach
""")

st.sidebar.divider()
st.sidebar.markdown(
    "Built following Prof. Steve Keen's methodology — "
    "private debt as the driver of aggregate demand."
)
st.sidebar.markdown("⚠️ **Version 0.1** — prototype. Data lives in cache, model is simplified.")

# ── Main App ────────────────────────────────────────────────────────────────

st.title("🏛️ Australian Economic Model")

st.markdown("""
### Will Australian living standards improve or decline?

This app builds a **stock-flow consistent macroeconomic model** of Australia,
following the approach of **Professor Steve Keen**.

The model starts from a simple premise: **private debt matters**. Most mainstream
models ignore it, treat it as a side-effect, or assume it's always benign.
Keen's work — and the empirical record — says it's the main event.

#### Three convictions that drive this project:

1. **Aggregate Demand = Income + ΔDebt** — when borrowing exceeds earnings,
   the economy grows; when it doesn't, it contracts
2. **The financial system is endogenous** — banks create money by lending;
   the size of the banking system is the size of private debt
3. **Trend is not destiny** — projecting current trends forward without
   understanding underlying dynamics gives false certainty

#### What you can do here:

| Explore | Description |
|---------|-------------|
| **Model Simulator** | Run the Keen model with adjustable parameters. See how wage share, employment, and debt interact over decades. |
| **Data Explorer** | Browse actual Australian data on private debt, housing, wages, and government finances. |
| **Scenario Analysis** | Test policies: immigration caps, interest rate changes, productivity shocks, housing policy. |
| **Living Standards** | Composite welfare metrics combining income, housing, debt service, employment, fiscal sustainability, and natural wealth. |
| **SFC Explorer** | Stock-flow consistent extension with banking sector, inflation dynamics, and interlocking balance sheets. |
| **Housing** | Minsky-style credit-driven housing sub-model with rental yields, LVR cycles, and construction lags. |
| **Resources** | Natural resource depletion, commodity prices, export revenue, and royalty/fiscal flows. |
""")

with st.expander("⚠️ **Important caveat**", expanded=False):
    st.markdown("""
    - This is a **mechanical model**, not a forecast. It shows dynamics inherent in
      the structure, not a prediction of what will happen.
    - Real economies have stabilising mechanisms (policy responses, innovation,
      institutional change) that the model doesn't capture.
    - The model is deliberately **simplified** — to show causal structure, not
      to simulate every detail.
    - Steve Keen's own models are more complex. This is an educational
      implementation of his core insights.
    """)

st.divider()

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Household Debt/GDP (AUS)", "~120%", delta="Highest in developed world")
with col2:
    st.metric("Private Debt/GDP", "~200%", delta="Approaching crisis territory")
with col3:
    st.metric("Housing Price/Income", "~8x Sydney", delta="4x is long-run norm")
with col4:
    st.metric("Wage Share of GDP", "~55%", delta="Down from 65% in 1970s")

st.info("👈 **Navigate using the sidebar** to explore the model, data, and scenarios.")
