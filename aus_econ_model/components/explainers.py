"""
Explainers — Clear, non-technical explanations of the Keen model concepts.
"""

MODEL_EXPLAINER = r"""
## The Core Idea

Most economic models assume the economy is a self-correcting system that tends
towards equilibrium. **Steve Keen's models disagree.**

The Keen model is built on three observations:

### 1. Debt Drives Demand

When someone takes out a loan and spends it, that spending becomes someone else's
income. **Aggregate Demand = Income + Change in Debt.**

In a growing economy, ever-increasing debt is what keeps demand growing
faster than wages alone would allow. This is how Australia has sustained
growth despite stagnant real wages — by borrowing against housing.

### 2. Debt Has Limits

Debt must be serviced. The interest payments drain income away from spending
on goods and services. When the debt ratio gets high enough, the drag from
debt service exceeds the boost from new borrowing, and the system destabilises.

This is **Minsky's Financial Instability Hypothesis**: stability breeds
instability — the longer a boom goes on, the more debt accumulates, and the
more fragile the system becomes.

### 3. The Three State Variables

| Variable | Symbol | What it means |
|----------|--------|---------------|
| Wage share | ω (omega) | Labour's cut of national income |
| Employment rate | λ (lambda) | Fraction of people who want work and have it |
| Private debt ratio | d | Total private debt as a multiple of GDP |

These three numbers capture the essential dynamics. When you see how they
interact, you can understand why growth can turn into crisis.

### The Feedback Loops

```
Credit → Debt → Aggregate Demand → Employment → Wages → Profits → Credit (again)
          ↑                                                    |
          └─────────── Debt Service ───────────────────────────┘
```

Two loops work in opposition:
- **Virtuous cycle**: More credit → more demand → more hiring → higher wages
  → more demand (this fuels the boom)
- **Vicious cycle**: More debt → more interest payments → lower profits
  → less investment → less hiring → lower wages (this causes the bust)

The model's central question: **which loop dominates, and when?**
"""

KEY_EQUATIONS = r"""
## The Model in Equations (Simplified)

### Wage Share Change
$$\frac{d\omega}{dt} = (\Phi(\lambda) - \omega) \times (g + \alpha)$$

The wage share (ω) moves towards what workers can bargain for ($\Phi(\lambda)$).
Workers bargain harder when employment (λ) is high — that's the Phillips curve.
The speed of adjustment depends on how fast the economy grows (g) and
productivity (α).

### Employment Change
$$\frac{d\lambda}{dt} = (g - \alpha - \beta) \times \lambda$$

Employment rises when the economy grows faster than productivity + population
growth. This is the "Okun's law" relationship.

### Debt Ratio Change
$$\frac{dd}{dt} = \kappa(\pi) - \pi$$

The debt ratio rises when investment ($\kappa$) exceeds profits ($\pi$).
Firms can only invest more than they earn by borrowing. The gap between
investment and retained profits *is* the change in private debt.

### Profit Share
$$\pi = 1 - \omega - r \times d$$

Profits (π) are what's left after paying wages ($\omega$) and interest ($r \times d$).
Higher debt → higher interest → lower profit → less investment → slower growth.
"""

AUSTRALIA_CONTEXT = r"""
## Why Australia Specifically

Australia has some of the **highest household debt in the world**:

| Metric | Australia | Comparable |
|--------|-----------|------------|
| Household debt / GDP | ~120% | US ~75%, UK ~85% |
| Household debt / disposable income | ~190% | US ~100% |
| Housing price / income | ~8× Sydney/Melbourne | US ~4× |

Keen has been warning about Australia's debt trajectory since 2007
("Deeper in Debt: Australia's Addiction to Borrowed Money").

### The Australian Dilemma

1. **Housing is the banks' main asset** — most private debt is mortgages
2. **Population growth (immigration)** pumps housing demand
3. **Supply constraints** (planning, geography) limit new building
4. **Foreign capital** funds the gap between savings and lending
5. **Resource exports** (iron ore, coal, LNG) prop up national income,
   masking domestic weakness

The model can show what happens when any of these supports weakens.
"""

DATA_SOURCES = r"""
## Data Sources We Use

| Data | Source | Frequency |
|------|--------|-----------|
| Private debt | RBA D2 (Lending & credit aggregates) | Monthly |
| National Accounts (GDP) | ABS 5206.0 | Quarterly |
| Wage share | ABS National Accounts | Quarterly |
| Employment/unemployment | ABS Labour Force (6202.0) | Monthly |
| CPI / Inflation | ABS 6401.0 | Monthly |
| Housing finance | ABS 5609.0 | Monthly |
| Building approvals | ABS 8731.0 | Monthly |
| Government finances | ABS Government Finance Stats | Quarterly |
| Resource reserves | Geoscience Australia AIMR/AECR | Annual |
| Resource export forecasts | Office of the Chief Economist REQ | Quarterly |
| Fiscal projections | Parliamentary Budget Office | Annual |
| Population | ABS Population Projections | Annual |

All data is fetched via public APIs (ABS SDMX, RBA CSV, data.gov.au).
"""


def model_explanation_section(param_name: str) -> str:
    """Return specific explanation for a parameter."""
    explanations = {
        "alpha": "**Productivity growth (α)**: How fast worker output rises each year. Higher productivity allows higher wages without inflation — but in the Keen model, it also means fewer workers needed for the same output (capital-biased technical change).",
        "beta": "**Labour force growth (β)**: Population + participation growth. Australia's is ~2% due to immigration. More workers = more potential output, but also more people competing for jobs and housing.",
        "r": "**Real interest rate (r)**: The cost of borrowing after inflation. Determines how much of income is consumed by debt service. At high debt levels, even modest rate rises can crush the economy.",
        "phi_min": "**Wage floor (Φ_min)**: The minimum wage share workers can achieve — the floor below which wages cannot fall. Set by minimum wage laws, union power, and social norms (~25%).",
        "phi_max": "**Maximum wage (Φ_max)**: The wage share workers can bargain for at full employment. In practice ~80%, as some profits are always needed to sustain investment.",
        "phi_n": "**Phillips curvature (Φ_n)**: Controls how non-linear the wage-employment relationship is. Higher values mean wages stay flat until employment is very tight, then rise sharply.",
        "kappa1": "**Investment sensitivity (κ₁)**: How strongly investment responds to profit. κ₁ < 1 is stabilising (debt falls when profit is high, rises when profit is low). Higher values mean more aggressive borrowing in booms.",
        "nu": "**Capital-output ratio (ν)**: How much capital is needed to produce one unit of output. ~5 means $5 of capital produces $1 of annual output. Higher = more capital-intensive economy.",
        "delta": "**Depreciation (δ)**: How fast capital wears out. 0.02 = 2% per year. Higher depreciation means more investment is needed just to keep the capital stock from shrinking.",
        "omega0": "**Initial wage share**: Starting value for wage share. Australia's was ~55% in 2025 — low by historical standards (was ~65% in 1970s).",
        "lambda0": "**Initial employment rate**: 1 − unemployment rate. At 6% unemployment → 94% employment rate.",
        "d0": "**Initial private debt ratio**: Total private debt / GDP. Australia ~200% (household ~120%, corporate ~80%). 1990s baseline ~80%.",
    }
    return explanations.get(param_name, "")
