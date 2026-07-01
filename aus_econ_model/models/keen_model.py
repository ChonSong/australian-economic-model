"""
Keen-Goodwin-Minsky Model Implementation
=========================================
Implements the core differential equations from Keen (1995) and Keen (2013).

State variables:
    ω (wage share)   — labour's share of national income
    λ (employment)   — fraction of labour force employed
    d  (debt ratio)  — private debt as fraction of GDP

Key insight: Aggregate Demand = Income + ΔDebt
"""

import numpy as np
from scipy.integrate import solve_ivp
from dataclasses import dataclass
from typing import Optional


# ── Equilibrium profit share ────────────────────────────────────────────────
# π* = ν·(α + β + δ) — the profit share needed to sustain steady-state growth
# At equilibrium: κ(π*) = π* (firms invest all profits), g* = α + β
# This quantity anchors the investment function calibration.
def equilibrium_profit(nu, alpha, beta, delta):
    return nu * (alpha + beta + delta)


@dataclass
class KeenParams:
    """Parameters for the Keen model, calibrated near Australian ranges.

    Equilibrium relationships:
        π*  = ν·(α+β+δ)           — steady-state profit share
        ω*  = 1 − π* − r·d*       — steady-state wage share
        Φ(λ*) = ω*                 — steady-state employment from Phillips curve
        κ(π*) = π*                 — investment = profit at steady state
    """
    # Real economy
    alpha: float = 0.02          # Labour productivity growth (2% p.a.)
    beta: float = 0.02           # Labour force growth (2% p.a. incl. immigration)
    delta: float = 0.02          # Depreciation rate (2% p.a.)
    nu: float = 5.0              # Capital-to-output ratio (Australia ~5)
    r: float = 0.03              # Real interest rate (3%)

    # Real wage Phillips curve: Φ(λ) = ω_min + (ω_max−ω_min)·λ^n
    # At very low employment, wage share → ω_min (subsistence/minimum wage floor)
    # At full employment, wage share → ω_max
    phi_min: float = 0.25        # ω_min — minimum wage share floor
    phi_max: float = 0.80        # ω_max — maximum wage share at full employment
    phi_n: float = 4.0           # Nonlinearity exponent (higher = steeper near full emp.)

    # Investment function: κ(π) = π* + κ₁·(π − π*)
    # Linearised around equilibrium profit share π*
    # At π=π*: κ=π* → dd/dt=0
    # Slope κ₁ < 1 for local stability (debt falls when profit is high)
    kappa1: float = 0.70         # Slope: how investment responds to profit (0<κ₁<1)

    # Initial conditions
    # Australia 1990s baseline: moderate debt, normal wage share
    omega0: float = 0.62         # Initial wage share (62% of GDP)
    lambda0: float = 0.94        # Initial employment rate (94%)
    d0: float = 0.80             # Initial private debt ratio (80% of GDP)

    @property
    def pi_star(self) -> float:
        """Equilibrium profit share."""
        return self.nu * (self.alpha + self.beta + self.delta)

    def investment(self, pi: np.ndarray) -> np.ndarray:
        """κ(π): Investment share of output — linearised around π*."""
        return self.pi_star + self.kappa1 * (pi - self.pi_star)

    def phillips(self, lam: np.ndarray) -> np.ndarray:
        """Φ(λ): Real wage curve — workers bargain harder near full employment."""
        return self.phi_min + (self.phi_max - self.phi_min) * np.clip(lam, 0, 1) ** self.phi_n


@dataclass
class KeenSolution:
    """Container for model simulation results."""
    t: np.ndarray
    omega: np.ndarray     # Wage share
    lam: np.ndarray       # Employment rate
    d: np.ndarray         # Debt ratio
    params: KeenParams
    success: bool
    message: str = ""

    @property
    def profit_share(self) -> np.ndarray:
        """π = 1 − ω − r·d — Profit share after interest payments."""
        return 1.0 - self.omega - self.params.r * self.d

    @property
    def investment_share(self) -> np.ndarray:
        """κ(π): Investment as share of output."""
        return self.params.investment(self.profit_share)

    @property
    def growth_rate(self) -> np.ndarray:
        """Capital accumulation rate g = I/K = κ/ν − δ."""
        return self.investment_share / self.params.nu - self.params.delta

    @property
    def debt_service_ratio(self) -> np.ndarray:
        """r·d — Debt service as share of GDP."""
        return self.params.r * self.d

    @property
    def real_wage(self) -> np.ndarray:
        """Φ(λ) — Bargained real wage at current employment."""
        return self.params.phillips(self.lam)


def keen_ode(t: float, y: np.ndarray, p: KeenParams) -> list:
    """
    The Keen (1995) model ODEs.

    y[0] = ω (wage share)
    y[1] = λ (employment rate)
    y[2] = d (private debt / GDP)
    """
    omega, lam, d = y

    # Safety: prevent negative or extreme values
    omega = max(omega, 0.001)
    lam = max(lam, 0.001)
    d = max(d, 0.0)

    # Derived quantities
    pi = 1.0 - omega - p.r * d                 # Profit share after interest
    kap = p.investment(np.array([pi]))[0]       # Investment share
    g = kap / p.nu - p.delta                    # Growth rate of capital
    phi = p.phillips(np.array([lam]))[0]        # Real wage from Phillips curve

    # ODEs
    domega_dt = (phi - omega) * (g + p.alpha)   # Wage share dynamics
    dlam_dt = (g - p.alpha - p.beta) * lam       # Employment dynamics
    dd_dt = kap - pi                             # Debt ratio dynamics

    return [domega_dt, dlam_dt, dd_dt]


def simulate_keen(
    params: Optional[KeenParams] = None,
    t_max: float = 100.0,
    t_steps: int = 2000,
    omega0: float = None,
    lambda0: float = None,
    d0: float = None,
    events: list = None,
    max_step: float = 0.5,
) -> KeenSolution:
    """Run the Keen model simulation."""
    if params is None:
        params = KeenParams()

    p = params
    omega_ic = omega0 if omega0 is not None else p.omega0
    lam_ic = lambda0 if lambda0 is not None else p.lambda0
    d_ic = d0 if d0 is not None else p.d0

    y0 = [omega_ic, lam_ic, d_ic]
    t_eval = np.linspace(0, t_max, t_steps)

    try:
        sol = solve_ivp(
            keen_ode,
            (0, t_max),
            y0,
            args=(p,),
            t_eval=t_eval,
            method='RK45',
            dense_output=True,
            max_step=max_step,
            events=events,
            rtol=1e-8,
            atol=1e-10,
        )
        success = sol.success
        message = sol.message
    except Exception as e:
        return KeenSolution(
            t=np.array([0, t_max]),
            omega=np.array([omega_ic, omega_ic]),
            lam=np.array([lam_ic, lam_ic]),
            d=np.array([d_ic, d_ic]),
            params=p,
            success=False,
            message=str(e),
        )

    omega_sol = sol.y[0]
    lam_sol = sol.y[1]
    d_sol = sol.y[2]

    # Clip to sensible ranges for display
    omega_sol = np.clip(omega_sol, 0, 2)
    lam_sol = np.clip(lam_sol, 0, 1.05)
    d_sol = np.clip(d_sol, 0, 10)

    return KeenSolution(
        t=sol.t,
        omega=omega_sol,
        lam=lam_sol,
        d=d_sol,
        params=p,
        success=success,
        message=message,
    )


# ─── Extended model with housing (simplified) ──────────────────────────────


@dataclass
class KeenHousingParams(KeenParams):
    """Extended parameters for housing sub-model."""
    h_supply_elas: float = 0.15       # Housing supply elasticity
    h_loan_to_income: float = 5.0     # Max LTI ratio banks will lend
    h_pop_growth: float = 0.015       # Population growth (drives housing demand)
    h_depreciation: float = 0.005     # Housing stock depreciation
    h_rental_yield: float = 0.035     # Rental yield (3.5%)
    h_initial_price: float = 5.0      # Initial price/income ratio (Sydney ~8-10)
    h_initial_stock: float = 1.0      # Normalised housing stock


def simulate_keen_housing(
    params: Optional[KeenHousingParams] = None,
    t_max: float = 100.0,
    t_steps: int = 2000,
) -> tuple:
    """
    Extended Keen model with a simplified housing sector.
    Returns (KeenSolution, housing_series_dict).
    """
    if params is None:
        params = KeenHousingParams()

    core = simulate_keen(params, t_max, t_steps)
    if not core.success:
        return core, {}

    h_price = np.full_like(core.t, params.h_initial_price)
    h_stock = np.full_like(core.t, params.h_initial_stock)
    constr_cost = np.full_like(core.t, 1.0)   # Normalised construction cost = 1

    for i in range(1, len(core.t)):
        dt = core.t[i] - core.t[i - 1]

        # Housing demand: population plus credit availability
        pop_demand = 1 + params.h_pop_growth * dt
        credit_boost = 1 + 0.3 * max(0, core.d[i - 1] - 0.8) * dt
        demand_index = pop_demand * credit_boost

        # Slow supply response
        # Price above construction cost → more building
        profit_margin = h_price[i - 1] / max(constr_cost[i - 1], 0.1) - 1.0
        new_supply = params.h_supply_elas * max(0, profit_margin) * dt
        h_stock[i] = h_stock[i - 1] * (1 - params.h_depreciation * dt) + new_supply

        # Price adjustment: ratio of demand pressure to supply
        if h_stock[i] > 0.01:
            pressure = demand_index / h_stock[i]
        else:
            pressure = demand_index

        # Price = previous_price × (pressure_ratio) + speculative boost from debt
        h_price[i] = h_price[i - 1] * (0.4 + 0.6 * pressure)
        h_price[i] += 0.02 * max(0, core.d[i] - 0.5) * dt
        h_price[i] = max(h_price[i], 0.5)

    housing = {
        'price_to_income': h_price,
        'housing_stock': h_stock,
        'price_growth': np.gradient(h_price) / np.maximum(h_price, 0.01) * 100,
        'affordability': params.h_loan_to_income / np.clip(h_price, 0.5, None),
    }

    return core, housing
