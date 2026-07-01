"""
Page 2: Data Explorer
=====================
Browse actual Australian economic data from ABS, RBA, and other sources.
"""

import streamlit as st
import pandas as pd
import numpy as np

from aus_econ_model.models.data_manager import get_manager
from aus_econ_model.components.charts import plot_time_series_simple

st.set_page_config(page_title="Data Explorer", page_icon="📊", layout="wide")

st.title("📊 Australian Economic Data Explorer")
st.markdown(
    "Browse time series pulled directly from the RBA Statistical Tables "
    "and ABS APIs. Data is cached locally for 6 hours."
)

# ── Data Status ─────────────────────────────────────────────────────────────

with st.status("Checking data sources...", expanded=False) as status:
    st.write("Connecting to RBA statistical tables...")
    dm = get_manager()
    summary = dm.data_summary()
    st.write(f"Data manager ready: {len(summary)} datasets cached/available")

    if summary:
        status.update(label="Data sources available", state="complete")
    else:
        status.update(label="⚠️ Could not reach data sources. Is internet working?", state="error")

if not summary:
    st.warning("""
    Could not fetch live data. The data sources may be unavailable.

    **Alternative data sources to try:**
    - RBA Statistical Tables: https://www.rba.gov.au/statistics/tables/
    - ABS Data API: https://api.data.abs.gov.au
    - Public Ledger: https://publicledger.au
    """)

# ── Data Browser ────────────────────────────────────────────────────────────

st.subheader("📋 Available RBA Tables")
RBA_TABLES = {
    "A1": "Reserve Bank Balance Sheet",
    "D2": "Lending and credit aggregates (private debt)",
    "F1": "Interest rates (cash rate, bond yields)",
    "G1": "Consumer price inflation",
    "G3": "Wage and labour statistics",
    "H2": "Housing finance and building approvals",
    "I2": "International investment position",
}

col1, col2 = st.columns([1, 2])

with col1:
    selected_table = st.selectbox(
        "Select an RBA table to explore:",
        list(RBA_TABLES.keys()),
        format_func=lambda k: f"{k} — {RBA_TABLES[k]}",
    )

    fetch_btn = st.button("📥 Fetch Table", type="primary", width='stretch')

if fetch_btn:
    with st.spinner(f"Fetching RBA table {selected_table}..."):
        try:
            dm = get_manager()
            df = dm.fetch_rba_table(selected_table)
        except Exception as e:
            df = None
            st.error(f"Fetch failed: {e}")

    if df is not None and not df.empty:
        st.success(f"Loaded table {selected_table}: {len(df)} rows × {len(df.columns)} columns")

        # Show columns
        st.subheader("Columns")
        st.write(list(df.columns[:15]))

        # Auto-detect numeric/data columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        date_cols = [c for c in df.columns if 'date' in c.lower() or 'period' in c.lower()]

        # Preview
        with st.expander("Raw data preview (first 20 rows)", expanded=False):
            st.dataframe(df.head(20), width='stretch')

        # Simple time series plot
        if numeric_cols:
            st.subheader("Quick Plot")
            chart_cols = st.multiselect(
                "Select columns to plot", numeric_cols,
                default=numeric_cols[:min(3, len(numeric_cols))],
            )
            if chart_cols:
                # Try to find a date column for x-axis
                x_col = None
                for dc in date_cols:
                    if dc in df.columns:
                        x_col = dc
                        break

                fig = plot_time_series_simple(df, x_col, chart_cols,
                                               title=f"RBA {selected_table} — {RBA_TABLES[selected_table]}")
                st.plotly_chart(fig, width='stretch')

        # Show head/tail stats
        st.subheader("Summary Statistics")
        st.dataframe(df.describe(), width='stretch')

    else:
        st.error(f"Could not load table {selected_table}. Try another table.")
else:
    st.info("Select a table and click **Fetch Table** to view data.")

# ── Pre-built Charts ────────────────────────────────────────────────────────

st.divider()
st.subheader("📈 Key Indicator Charts")

chart_type = st.selectbox(
    "Choose a pre-built chart:",
    ["Private Debt (D2)", "Cash Rate (F1)", "CPI Inflation (G1)",
     "Housing Finance (H2)", "Balance Sheet (A1)"],
)

if st.button("🔍 Generate Chart"):
    table_map = {
        "Private Debt (D2)": "D2",
        "Cash Rate (F1)": "F1",
        "CPI Inflation (G1)": "G1",
        "Housing Finance (H2)": "H2",
        "Balance Sheet (A1)": "A1",
    }

    table_code = table_map[chart_type]
    try:
        dm = get_manager()
        df = dm.fetch_rba_table(table_code)
    except Exception as e:
        df = None
        st.error(f"Fetch failed: {e}")

    if df is not None and not df.empty:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        with st.spinner("Building chart..."):
            fig = plot_time_series_simple(
                df, None,
                numeric_cols[:min(4, len(numeric_cols))],
                title=f"{chart_type} — RBA {table_code}",
                height=450,
            )
            st.plotly_chart(fig, width='stretch')
    else:
        st.warning(f"Could not fetch data for {chart_type}")

# ── Data Quality Notes ──────────────────────────────────────────────────────

with st.expander("📝 Data Quality Notes", expanded=False):
    st.markdown("""
    ### Data quality considerations for the model

    | Issue | Impact | Mitigation |
    |-------|--------|------------|
    | **Series breaks** — ABS/RBA periodically change methodology | Discontinuities in long time series | Flag break points; use footnoted data |
    | **Seasonal adjustment** — RSA vs original | Different signals for short vs long run | Prefer seasonally adjusted for quarterly, original for annual |
    | **Chain volume vs current price** — real vs nominal GDP | Real is what you want for growth, nominal for debt ratios | Debt ratios use nominal GDP denominator |
    | **Consolidated vs unconsolidated debt** | Off by ~$1T for government debt | Need to check footnotes |
    | **Coverage changes** — e.g. superannuation inclusion in household wealth | Non-comparable across decades | Use consistent definitions where possible |

    **Golden rule**: Always read the methodological notes before using a series
    in a model. ABS catalogue numbers have detailed documentation.
    """)

st.caption("Data sourced from RBA (rba.gov.au/statistics) and ABS (abs.gov.au). Cache clears every 6 hours.")
