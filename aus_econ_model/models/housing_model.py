"""
Australian Housing Sector Sub-Model (v4 — Minsky credit-cycle driven)
======================================================================
Housing prices driven primarily by credit flows (change in debt),
with structural supply constraints creating a rising platform.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class HousingParams:
    """Parameters for Australian housing sub-model, Minsky-style."""

    # Credit-driven pricing: how much new credit flows into housing
    credit_flow_elasticity: float = 2.0  # P/I response to credit growth rate
    credit_flow_persistence: float = 0.85  # Momentum of price appreciation

    # Structural anchors
    fundamental_price_income: float = 5.0
    initial_price_income: float = 6.5
    structural_trend: float = 0.004  # Supply constraint + policy drift

    # Mean reversion (weak — AU housing has very slow reversion to fair value)
    mean_reversion_strength: float = 0.008

    # Mortgage
    mortgage_share_of_debt: float = 0.45
    mortgage_rate_premium: float = 0.015

    # Rental
    initial_rent_to_income: float = 0.25
    vacancy_adjustment_speed: float = 0.30
    rental_yield_target: float = 0.035

    # Construction
    base_construction_gdp: float = 0.06
    construction_lag: float = 2.0
    construction_elasticity: float = 0.20
    dwelling_depreciation: float = 0.01
    dwelling_stock_initial: float = 0.80

    # LVR
    initial_lvr: float = 0.60
    lvr_cycle_sensitivity: float = 0.10

    # Wealth effect
    housing_collateral_effect: float = 0.25


def simulate_housing(core_sol, params: Optional[HousingParams] = None) -> dict:
    if params is None:
        params = HousingParams()

    t = core_sol.t
    n = len(t)
    omega = core_sol.omega
    lam = core_sol.lam
    d = core_sol.d
    r = core_sol.params.r
    beta = core_sol.params.beta
    income = omega
    mortgage_rate = r + params.mortgage_rate_premium

    # State arrays
    pti = np.full(n, params.initial_price_income)
    mg = np.full(n, d[0] * params.mortgage_share_of_debt)
    lvr_a = np.full(n, params.initial_lvr)
    rti = np.full(n, params.initial_rent_to_income)
    ry = np.full(n, params.initial_rent_to_income / params.initial_price_income)
    cgdp = np.full(n, params.base_construction_gdp)
    ds = np.full(n, params.dwelling_stock_initial)
    hw = np.full(n, params.initial_price_income * income[0])
    col = np.full(n, 0.0)
    ms = np.full(n, 0.0)
    # GDP index (reconstructed from growth rates)
    gdp_idx = np.ones(n)

    lag_buf = np.full(
        int(max(params.construction_lag * 5, 5)), params.base_construction_gdp
    )

    # Running annualised credit growth for persistence
    credit_growth_smoothed = 0.0

    for i in range(1, n):
        dt = t[i] - t[i - 1]

        # --- Credit flow (change in debt drives prices) ---
        d_growth = np.clip((d[i] - d[i - 1]) / max(d[i - 1], 0.01), -0.5, 0.5)
        credit_growth_smoothed = (
            1.0 - params.credit_flow_persistence
        ) * d_growth + params.credit_flow_persistence * credit_growth_smoothed

        # Income growth
        inc_growth = np.clip(
            (income[i] - income[i - 1]) / max(income[i - 1], 0.01), -0.3, 0.3
        )

        # Fundamental value (drifts upward)
        fund_val = (
            params.fundamental_price_income * (1.0 + params.structural_trend) ** t[i]
        )

        # Mean reversion
        reversion = params.mean_reversion_strength * (fund_val - pti[i - 1]) * dt

        # Credit cycle effect (this is the main driver)
        credit_push = params.credit_flow_elasticity * credit_growth_smoothed * dt

        # Supply constraint effect
        supply_gap = np.clip(1.0 - ds[i - 1], -0.2, 0.4)
        supply_effect = supply_gap * 0.01 * dt

        # Population pressure
        pop_effect = beta * 0.15 * dt

        # Employment confidence
        emp_eff = max(lam[i] - 0.90, 0) * 0.02 * dt

        # Total price change
        dp = reversion + credit_push + supply_effect + pop_effect + emp_eff
        dp = np.clip(dp, -0.12, 0.15)
        pti[i] = np.clip(pti[i - 1] * (1.0 + dp), 1.5, 30.0)

        # --- Mortgage debt ---
        hp_g = pti[i] / max(pti[i - 1], 0.01) - 1.0
        base_mg = d[i] * params.mortgage_share_of_debt
        eq_wd = max(hp_g, 0) * params.housing_collateral_effect * mg[i - 1]
        mg[i] = max(base_mg + eq_wd, 0.01)

        # --- LVR ---
        lvr_chg = -hp_g * params.lvr_cycle_sensitivity
        if hp_g < -0.02:
            lvr_chg -= params.lvr_cycle_sensitivity * 0.3
        lvr_a[i] = np.clip(lvr_a[i - 1] + lvr_chg, 0.20, 0.95)

        # --- Mortgage service ---
        ms[i] = mortgage_rate * mg[i]

        # --- Rental ---
        vac = np.clip(1.0 - ds[i - 1], -0.3, 0.5)
        rent_adj = params.vacancy_adjustment_speed * (0.05 - vac) * dt
        rent_inc_adj = inc_growth * 0.4
        rti[i] = np.clip(rti[i - 1] * (1.0 + rent_adj + rent_inc_adj), 0.15, 0.50)
        ry[i] = rti[i] / max(pti[i], 0.01)

        # --- Construction ---
        sig = np.clip((pti[i] / fund_val - 1.0), -0.5, 1.0)
        dc = params.base_construction_gdp * (1.0 + params.construction_elasticity * sig)
        dc = np.clip(dc, 0.02, 0.12)
        lag_buf = np.roll(lag_buf, 1)
        lag_buf[0] = dc
        li = min(int(params.construction_lag * 5), len(lag_buf) - 1)
        cgdp[i] = lag_buf[li]

        # --- Dwelling stock ---
        nd = cgdp[i] * 0.15
        ds[i] = np.clip(
            ds[i - 1] * (1.0 - params.dwelling_depreciation * dt) + nd * dt, 0.5, 2.0
        )

        # --- Wealth ---
        hw[i] = pti[i] * income[i]
        col[i] = max(pti[i] / pti[0] - 1.0, 0) * 0.3

        # --- GDP index (from model growth rate) ---
        pi_i = 1.0 - omega[i - 1] - r * d[i - 1]
        kap_i = core_sol.params.pi_star + core_sol.params.kappa1 * (
            pi_i - core_sol.params.pi_star
        )
        g_i = kap_i / core_sol.params.nu - core_sol.params.delta
        gdp_idx[i] = gdp_idx[i - 1] * (1.0 + g_i * dt)

    neg_eq = np.maximum(0.0, 1.0 - 1.0 / (lvr_a + 0.001))
    return {
        "t": t,
        "price_to_income": pti,
        "mortgage_gdp": mg,
        "lvr": lvr_a,
        "mortgage_service": ms,
        "rent_to_income": rti,
        "rental_yield": ry,
        "construction_gdp": cgdp,
        "dwelling_stock": ds,
        "housing_wealth": hw * gdp_idx,
        "housing_wealth_gdp": hw,
        "collateral_effect": col,
        "negative_equity_risk": neg_eq,
        "forced_sale_risk": neg_eq * 0.05,
        "mortgage_rate": np.full(n, mortgage_rate),
        "price_appreciation": np.diff(pti, prepend=pti[0]) / pti[0],
    }
