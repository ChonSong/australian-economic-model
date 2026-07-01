# Australia Economic Model — Keen Approach

An interactive Streamlit app that models the Australian economy following
**Professor Steve Keen's** methodology (Minskyan debt dynamics, stock-flow
consistent modelling).

## Quick Start

```bash
cd /home/sc/workspace/aus-econ-model
./run.sh
```

Opens at http://localhost:8501

## What It Does

| Page | Purpose |
|------|---------|
| **Model Simulator** | Interactively run the Keen-Goodwin-Minsky differential equations model. Adjust parameters for wage share, employment, private debt, interest rates, and investment. |
| **Data Explorer** | Pull and visualise actual Australian data from RBA and ABS APIs (private debt, CPI, housing finance, interest rates). |
| **Scenario Analysis** | Compare multiple scenarios: immigration caps, rate hikes, wage recovery, credit crunch, productivity boom. |
| **Living Standards** | Composite welfare index combining income, employment, debt burden, housing affordability, and growth. |
| **About** | Methodology, data sources, Steve Keen's framework, roadmap. |

## Architecture

```
aus_econ_model/
├── streamlit_app.py          # Entry point + home page
├── pages/
│   ├── 01_Model_Simulator.py # Interactive Keen model
│   ├── 02_Data_Explorer.py   # ABS/RBA data browser
│   ├── 03_Scenario_Analysis.py # What-if policy scenarios
│   ├── 04_Living_Standards.py # Composite welfare index
│   └── 05_About.py           # Methodology + sources
├── models/
│   ├── keen_model.py         # ODE model (Keen 1995 + housing extension)
│   └── data_manager.py       # RBA CSV + ABS SDMX data pull
├── components/
│   ├── charts.py             # Plotly visualisation components
│   └── explainers.py         # Educational text + parameter docs
└── data/
    └── cache/                # Cached API responses
```

## The Model

Implements the Keen (1995) three-equation system:

- **dω/dt** = (Φ(λ) − ω) × (g + α) — Wage share dynamics
- **dλ/dt** = (g − α − β) × λ — Employment dynamics  
- **dd/dt** = κ(π) − π — Private debt dynamics

Where private debt (d) drives aggregate demand, and debt service
(r × d) eventually squeezes profits and triggers a Minsky crisis.

## Full Project Roadmap

1. **Phase 0**: ✅ Foundation — tools, data pipeline, model validation
2. **Phase 1**: 🔶 Data inventory — catalog all time series
3. **Phase 2**: ✅ Core model — Keen-Goodwin-Minsky calibrated to Australia
4. **Phase 3**: 🔶 Housing & household sector sub-model
5. **Phase 4**: ❌ Government & fiscal (Commonwealth + states)
6. **Phase 5**: ❌ Natural resource stocks & external sector
7. **Phase 6**: ✅ Living standards assessment framework (prototype)

## Data Sources

- **RBA**: Statistical tables (CSV direct from rba.gov.au)
- **ABS**: Indicator API + Data API (SDMX-JSON)
- **PBO**: Fiscal projections (pbo.gov.au)
- **Geoscience Australia**: Mineral & energy resource stocks
- **Office of the Chief Economist**: Resources & Energy Quarterly

## Tech Stack

- Python + SciPy (ODE solver)
- Streamlit (UI)
- Plotly (interactive charts)
- Requests + pandas (data ingestion)
