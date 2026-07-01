"""
Page 5: About — Methodology, Sources, and Background
"""

import streamlit as st

st.set_page_config(page_title="About", page_icon="ℹ️", layout="wide")

st.title("ℹ️ About This Project")

# ── Why This Model ──────────────────────────────────────────────────────────

st.header("Why This Model? Why Steve Keen's Approach?")

col1, col2 = st.columns([3, 2])

with col1:
    st.markdown("""
    Most macroeconomic models used by central banks and treasuries
    (DSGE models — Dynamic Stochastic General Equilibrium) assume:

    1. The economy tends toward equilibrium
    2. Financial markets are efficient
    3. Money is neutral (doesn't affect real activity in the long run)
    4. Private debt is a side-effect, not a driver

    **Steve Keen argues these assumptions are wrong**, and the empirical
    evidence supports his critique:

    - The 2008 Global Financial Crisis was a debt crisis that DSGE models
      could not predict and cannot explain
    - Australia's household debt to GDP ratio is ~120% — higher than the
      US before the GFC, and still rising
    - Private debt has predicted every major financial crisis of the last
      50 years (Kaminsky-Reinhart 1999, Schularick-Taylor 2012)

    ### Keen's Alternative

    Builds on three intellectual traditions:

    1. **Hyman Minsky (Financial Instability Hypothesis)** — stability
       breeds instability. The longer the boom, the more debt accumulates,
       the more fragile the system becomes.

    2. **Wynne Godley (Stock-Flow Consistency)** — every financial flow
       must be someone else's offsetting flow. You cannot have borrowing
       without lending, and total financial assets must equal total
       financial liabilities. This is enforced by double-entry bookkeeping.

    3. **Richard Goodwin (Growth Cycle Model)** — the economy oscillates
       because of the conflict between wages and profits. Add debt to this
       and you get explosive cycles.
    """)

with col2:
    st.markdown("""
    ### Key Publications

    **"Deeper in Debt: Australia's Addiction to Borrowed Money"** (2007)
    Centre for Policy Development.
    [Read PDF](https://cpd.org.au/wp-content/uploads/2007/09/KeenCPD_DeeperInDebt_FullDoc_1.pdf)

    Predicted Australia's debt trajectory would lead to crisis. Ignored by
    mainstream economists. Vindicated by the GFC.

    **"A Monetary Minsky Model of the Great Moderation and the Great
    Recession"** (2013)
    *Journal of Economic Behavior & Organization*, 86, 221-235.

    The formal mathematical model that this app implements.

    **"The New Economics: A Manifesto"** (2021)
    Polity Press.

    His most accessible book-length treatment.

    **"The Keen Model"** (ongoing)
    [Minsky Software](https://github.com/highperformancecoder/minsky)
    Open-source system dynamics program with Godley Tables.
    """)

# ── Key Differences from Mainstream ─────────────────────────────────────────

st.header("🔬 Key Differences from Mainstream Economic Models")

diff_data = {
    "Feature": [
        "View of the economy",
        "Role of banks",
        "Private debt",
        "Money creation",
        "Equilibrium",
        "Policy implication",
    ],
    "Mainstream DSGE": [
        "Self-correcting, tends to equilibrium",
        "Intermediaries between savers and borrowers",
        "A side-effect, not a driver of demand",
        "Exogenous (controlled by central bank)",
        "The natural state",
        "Deregulate, minimise government intervention",
    ],
    "Keen (SFC/Minsky)": [
        "Inherently unstable, debt-driven cycles",
        "Creators of money through lending",
        "The primary driver of aggregate demand",
        "Endogenous (created by bank lending)",
        "A special case, not the default",
        "Manage private debt, use fiscal policy actively",
    ],
}

import pandas as pd
st.dataframe(pd.DataFrame(diff_data), width='stretch', hide_index=True)

# ── Data Sources ────────────────────────────────────────────────────────────

st.header("📊 Data Sources")

sources_data = {
    "Source": [
        "Australian Bureau of Statistics (ABS)",
        "Reserve Bank of Australia (RBA)",
        "Parliamentary Budget Office (PBO)",
        "Geoscience Australia",
        "Office of the Chief Economist",
        "Australian Office of Financial Management (AOFM)",
        "Commonwealth Grants Commission (CGC)",
        "World Bank / IMF",
    ],
    "What": [
        "GDP, CPI, employment, wages, population, housing finance, government finance",
        "Interest rates, lending aggregates, balance sheet, exchange rates, forecasts",
        "Fiscal projections (medium & long-term), sustainability analysis",
        "Mineral & energy resource stocks (annual assessment)",
        "Resources & Energy Quarterly — production & export forecasts",
        "Commonwealth government debt issuance & management",
        "GST distribution methodology & state fiscal capacity",
        "International comparisons, terms of trade data",
    ],
    "Access": [
        "SDMX API (api.data.abs.gov.au) or Indicator API",
        "CSV direct download (rba.gov.au/statistics/tables/csv/)",
        "Data portal (pbo.gov.au) + publications",
        "GA website (ga.gov.au), data.gov.au",
        "Industry.gov.au — REQ publications",
        "AOFM website — debt issuance data",
        "CGC website — reports & data",
        "API / bulk download",
    ],
}
st.dataframe(pd.DataFrame(sources_data), width='stretch', hide_index=True)

# ── Tools Used ──────────────────────────────────────────────────────────────

st.header("🛠️ Technical Stack")

st.markdown("""
| Tool | Purpose |
|------|---------|
| **Python** (NumPy, SciPy) | ODE solver, numerical computation |
| **Minsky** (optional) | Full SFC models with Godley Tables (visual) |
| **Streamlit** | Interactive web frontend |
| **Plotly** | Interactive charts |
| **Requests / SDMX** | Data API calls (ABS, RBA) |
| **R / readrba** (optional) | Alternative RBA data pull |

All open source. The model is deliberately kept simple to show causal
structure, not to maximise forecasting accuracy.
""")

# ── Project Roadmap ─────────────────────────────────────────────────────────

st.header("🗺️ Full Project Roadmap")

st.markdown("""
The app you see is **Phase 0-1 of a 6-phase project**. Here's the full plan:

| Phase | What | Status |
|-------|------|--------|
| **0** | Foundation — tools, data pipeline, validation | ✅ **Done** (prototype) |
| **1** | Full data inventory — all time series catalogued | 🔶 Partially done |
| **2** | Core Keen-Goodwin-Minsky model — calibrated to Australia | ✅ **Done** (simplified) |
| **3** | Housing & household sector sub-model | 🔶 Partial (this app) |
| **4** | Government & fiscal sector (Commonwealth + states) | ❌ Not started |
| **5** | Natural resource stocks & external sector | ❌ Not started |
| **6** | Integrated living standards assessment | ✅ **Done** (prototype framework) |

### What Comes Next

1. **Full data pipeline** — automated daily pulls from ABS/RBA APIs into
   a local time-series database
2. **Minsky integration** — export Python data → Minsky model → back to
   Streamlit for visualisation
3. **Housing sector** — proper HST (Housing Stock Turnover) model
4. **State-by-state model** — each state economy with its own fiscal position
5. **Resource depletion** — integrate Geoscience Australia stock estimates
6. **Publication** — documented methodology, reproducible via Docker
""")

# ── Limitations ─────────────────────────────────────────────────────────────

st.header("⚠️ Important Limitations")

st.markdown("""
1. **This is not a forecast**. It shows dynamics inherent in the model
   structure, not a prediction. Real economies have policy responses
   (fiscal stimulus, regulation, innovation) that a 3-equation model
   cannot capture.

2. **The model is simplified** relative to Keen's own work. He uses
   Minsky with full Godley Table accounting. This Python version has
   basic stock-flow consistency but not the full double-entry framework.

3. **Parameter uncertainty** is large. Small changes in investment
   sensitivity (κ₂) or wage bargaining power (Φ₁) produce wildly
   different outcomes. The sensitivity explorer is there for a reason.

4. **No distributional data**. Averages conceal enormous variation.
   The top 10% experience a completely different economy from the
   bottom 10%.

5. **No financial sector breakdown**. Banks, shadow banks, foreign
   lenders, and superannuation funds all behave differently. A single
   interest rate and debt ratio collapses them.

6. **No climate change**. The physical impacts of climate change on
   productivity, housing, and public finances are not included.
""")

# ── About the Author / Project ──────────────────────────────────────────────

st.header("👤 About This Project")

st.markdown("""
Built following the methodology of **Professor Steve Keen** (University of
Western Sydney, Kingston University London, author of *Debunking Economics*,
*The New Economics: A Manifesto*).

This implementation is an independent, open-source educational project.
It is not affiliated with Prof. Keen, though it attempts to faithfully
represent his modelling approach.

**Source code**: Available at this workspace.
**License**: MIT — free to use, modify, and distribute.

### Further Reading

- Keen, S. (2021). *The New Economics: A Manifesto*. Polity Press.
- Keen, S. (2011). *Debunking Economics* (Revised ed.). Zed Books.
- Godley, W. & Lavoie, M. (2007). *Monetary Economics*. Palgrave Macmillan.
- Minsky, H. (1986). *Stabilizing an Unstable Economy*. Yale University Press.
- Goodwin, R. M. (1967). "A Growth Cycle". In *Socialism, Capitalism and
  Economic Growth*, Cambridge University Press.
""")

st.caption("Version 0.1 — June 2026. Built with Python, Streamlit, SciPy, and Plotly.")
