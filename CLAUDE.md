# Australian Economic Model — Autonomous Development Guide

## Overview
Interactive Streamlit app modelling the Australian economy using Steve Keen's Minskyan debt-dynamics framework. Features Keen-Goodwin-Minsky core model with housing, SFC (government/external/banking), and resource sub-models.

**Live:** https://aus.codeovertcp.com
**Stack:** Python 3.11, Streamlit, Plotly, NumPy, SciPy, Pandas

## Architecture

```
aus_econ_model/
├── streamlit_app.py          # Entry point (redirects to Model Simulator)
├── run.sh                    # Dev launch script
├── .streamlit/               # (missing — needs config.toml)
├── components/
│   ├── charts.py             # Shared Plotly chart configs, theming, COLOURS dict
│   └── explainers.py         # LaTeX + math explainers
├── models/
│   ├── keen_model.py         # Core Keen (wage share, employment, debt) ODE system
│   ├── sfc_model.py          # Extended SFC (government, external, banking sectors)
│   ├── govt_model.py         # Federal + State fiscal disaggregation
│   ├── housing_model.py      # Minsky-style housing sub-model (credit-driven)
│   └── resource_model.py     # Commodity depletion, prices, energy transition
├── pages/
│   ├── 01_Model_Simulator.py  # Main time series + parameter sensitivity
│   ├── 02_Data_Explorer.py    # RBA/ABS live data + charts
│   ├── 03_Scenario_Analysis.py# What-if scenario comparison
│   ├── 04_Living_Standards.py # Welfare, fiscal, housing impacts
│   ├── 05_About.py            # Methodology & documentation
│   ├── 06_SFC_Explorer.py     # Extended SFC model (CRASHES — see Issues #1, #2)
│   ├── 07_Housing.py          # Housing market dynamics
│   └── 08_Resources.py        # Commodity dashboards
└── data/                      # Parquet cache dir (created at runtime)
```

## Running Locally

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Deployed via systemd user service behind nginx/caddy.

## Priority Issues

| # | Priority | Area | Description |
|---|----------|------|-------------|
| 1 | **CRITICAL** | `charts.py` | `_apply_theme()` crashes on figures without all subplot axes (PlotlyKeyError). Fix: use `.get()` |
| 2 | **CRITICAL** | `sfc_explorer.py` | `COLOURS` dict used as indexed list — KeyError on chart render |
| 3 | MEDIUM | `housing_model.py` | `housing_wealth_gdp` = `housing_wealth` (same array, not divided by GDP) |
| 4 | MEDIUM | `tests/` | No test suite — add pytest coverage |
| 5 | LOW | `housing_model.py` | P/I ratio hits 15.0x clip ceiling every run |
| 6 | LOW | `01_Model_Simulator.py` | Unused `equilibrium_profit` import |
| 7 | LOW | `run.sh` | Hardcoded venv path |
| 8 | LOW | `.streamlit/` | Missing config.toml |

## Approach Rules

1. **Investigate before building** — when assigned an issue, first confirm it's reproducible and understand the root cause
2. **Fix the root cause, not the symptom** — patches that mask a deeper problem will be rejected
3. **Add tests alongside fixes** — any bug fix should include a test that would have caught it
4. **One issue per PR** — keep changes scoped and reviewable
5. **Model changes require visual verification** — run the page and confirm charts look right after any model parameter change
6. **`_apply_theme` must handle any subplot count** — don't add more hardcoded axis indices
7. **Colours should be list-based if indexed, dict-based if named** — don't mix

## Testing

```bash
pip install pytest pytest-mock
python -m pytest tests/ -v
```

Test files go in `tests/` mirroring the source structure:
- `tests/test_keen_model.py`
- `tests/test_data_manager.py`
- `tests/test_charts.py` (smoke test that `_apply_theme` works on single-axis figures)
- `tests/test_pages.py` (smoke tests that each page can import)
