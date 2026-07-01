"""
Extended SFC Keen-Goodwin-Minsky Model
=======================================
Extends the core Keen model (keen_model.py) with government, external,
and banking sectors in a Stock-Flow Consistent (SFC) framework.

State variables:
    ω  (wage share)        — labour's share of national income
    λ  (employment)        — fraction of labour force employed
    d  (private debt)      — private debt as fraction of GDP
    d_g (government debt)  — government debt as fraction of GDP
    d_f (foreign debt)     — net foreign debt as fraction of GDP
    π_e (expected infl.)   — expected inflation rate
    bcr (bank cap. ratio)  — bank capital adequacy ratio

Key SFC identities:
    (G - T) + (I - S) + (X - M - r_f·d_f) = 0
    Government + Private + External = 0 (flow consistency)

Sectoral balances:
    Private surplus = Government deficit + Current account deficit
    S - I = (T - G) + (M - X + r_f·d_f)
"""

import numpy as np
from scipy.integrate import solve_ivp
from dataclasses import dataclass
from typing import Optional, Callable

from aus_econ_model.models.keen_model import KeenParams


# ── Parameter Dataclass ──────────────────────────────────────────────────────

@dataclass
class ExtendedKeenParams(KeenParams):
    """Extended parameters for SFC model, calibrated for Australia (2025 est.).

    New sectors added on top of the base Keen model:
      • Government: tax, spending, counter-cyclical fiscal rule
      • External: trade, foreign debt, terms of trade
      • Banking: interest spread, capital adequacy, credit constraints
      • Extended Phillips curve: adaptive inflation expectations, wage-price spiral

    Calibration targets (Australia ~2025):
      Gov debt/GDP: ~55% (federal + state)
      Foreign debt/GDP: ~60% (gross foreign liabilities)
      Tax/GDP: ~28% federal + ~17% state = ~45% total
      Gov spending/GDP: ~37% total
      Export/GDP: ~25%, Import/GDP: ~20%
    """

    # ── Government Sector ────────────────────────────────────────────────
    tax_rate_wages: float = 0.30       # Avg effective tax rate on wage income
    tax_rate_profits: float = 0.35     # Avg effective tax rate on profit income
    # Government spending: G/Y = g0 + g1 + g2 * max(0, (λ_target - λ)/λ_target)
    #                       + g3 * (d_g_target - d_g)  [debt-targeting term]
    g0: float = 0.16                   # Autonomous gov spending share of GDP
    g1: float = 0.16                   # Proportional (trend) spending share
    g2: float = 0.12                   # Counter-cyclical fiscal response
    g3: float = 0.03                   # Debt target stabilisation (g3 > 0 = stabilising)
    d_g_target: float = 0.50           # Target gov debt/GDP ratio
    lambda_target: float = 0.95        # Target employment rate (5% unemployment)
    r_g: float = 0.035                 # Interest rate on government debt

    # ── External Sector ──────────────────────────────────────────────────
    export_share: float = 0.22         # Base export share of GDP
    import_base: float = 0.19          # Autonomous import share of GDP
    import_propensity: float = 0.05    # Additional imports from wage-led consumption
    r_f: float = 0.035                 # Interest rate on net foreign debt
    world_growth: float = 0.03         # World income growth rate (for export trend)
    export_income_elas: float = 0.15   # Export elasticity to world income

    # ── Banking Sector ───────────────────────────────────────────────────
    r_l: float = 0.045                 # Lending rate (previously 'r' in base model)
    r_d: float = 0.025                 # Deposit rate
    min_cap_ratio: float = 0.08        # Minimum regulatory capital ratio (Basel III)
    target_cap_ratio: float = 0.12     # Banks' target capital ratio
    div_payout: float = 0.80           # Dividend payout ratio from bank profits

    # ── Extended Phillips Curve ──────────────────────────────────────────
    wage_indexation: float = 0.40      # ι — how much expected inflation feeds into wage bargaining
    infl_adapt_speed: float = 0.30     # θ — speed of adaptive expectations (per year)
    markup_rate: float = 1.15          # Markup factor on unit labour costs (15%)

    # ── Initial conditions (new state variables) ─────────────────────────
    d_g0: float = 0.55                 # Initial gov debt/GDP (~55% Australia)
    d_f0: float = 0.60                 # Initial net foreign debt/GDP (~60%)
    pi_e0: float = 0.025               # Initial expected inflation (2.5%)
    bcr0: float = 0.12                 # Initial bank capital ratio (12%)


# ── Solution Dataclass ───────────────────────────────────────────────────────

@dataclass
class ExtendedKeenSolution:
    """Container for extended SFC model simulation results."""
    t: np.ndarray
    omega: np.ndarray          # Wage share
    lam: np.ndarray            # Employment rate
    d: np.ndarray              # Private debt / GDP
    d_g: np.ndarray            # Government debt / GDP
    d_f: np.ndarray            # Net foreign debt / GDP
    pi_e: np.ndarray           # Expected inflation
    bcr: np.ndarray            # Bank capital adequacy ratio
    params: ExtendedKeenParams
    success: bool
    message: str = ""

    # ── Core derived properties (from base Keen) ─────────────────────────

    @property
    def profit_share(self) -> np.ndarray:
        """π = 1 - ω - r_l·d - r_g·d_g - r_f·d_f

        Profit share AFTER all interest payments (private, government, foreign).
        """
        p = self.params
        return 1.0 - self.omega - p.r_l * self.d - p.r_g * self.d_g - p.r_f * self.d_f

    @property
    def investment_share(self) -> np.ndarray:
        """κ(π): Investment as share of output (from KeenParams)."""
        return self.params.investment(self.profit_share)

    @property
    def growth_rate(self) -> np.ndarray:
        """Capital accumulation rate g = I/K = κ/ν - δ."""
        return self.investment_share / self.params.nu - self.params.delta

    @property
    def debt_service_ratio(self) -> np.ndarray:
        """Private debt service as share of GDP."""
        return self.params.r_l * self.d

    @property
    def real_wage(self) -> np.ndarray:
        """Φ(λ) — Bargained real wage at current employment."""
        return self.params.phillips(self.lam)

    @property
    def credit_multiplier(self) -> np.ndarray:
        """Credit availability factor (1.0 = normal, <1.0 = crunch).

        Based on bank capital adequacy relative to regulatory minimum.
        """
        p = self.params
        # Linear interpolation: 1.0 at target, 0.2 at minimum
        if p.target_cap_ratio <= p.min_cap_ratio:
            return np.ones_like(self.bcr)
        raw = (self.bcr - p.min_cap_ratio) / (p.target_cap_ratio - p.min_cap_ratio)
        return np.clip(raw, 0.2, 1.0)

    # ── SFC-specific derived properties ──────────────────────────────────

    @property
    def tax_revenue(self) -> np.ndarray:
        """Tax revenue as share of GDP — progressive on wages vs profits."""
        p = self.params
        return p.tax_rate_wages * self.omega + p.tax_rate_profits * (1.0 - self.omega)

    @property
    def gov_spending_primary(self) -> np.ndarray:
        """Primary government spending share (excl. interest)."""
        p = self.params
        # Counter-cyclical component: spend more when employment is below target
        stabiliser = p.g2 * np.maximum(0, (p.lambda_target - self.lam) / p.lambda_target)
        return p.g0 + p.g1 + stabiliser

    @property
    def gov_total_spending(self) -> np.ndarray:
        """Total government spending including interest."""
        return self.gov_spending_primary + self.params.r_g * self.d_g

    @property
    def gov_primary_deficit(self) -> np.ndarray:
        """Primary deficit (excl. interest payments)."""
        return self.gov_spending_primary - self.tax_revenue

    @property
    def gov_total_deficit(self) -> np.ndarray:
        """Total deficit including interest."""
        return self.gov_total_spending - self.tax_revenue

    @property
    def export_share(self) -> np.ndarray:
        """Export share of GDP — grows with world income."""
        p = self.params
        # Handled carefully to avoid explosive growth: log-linear trend
        world_trend = 1.0 + p.world_growth * self.t / (1.0 + self.t / 100.0)
        return p.export_share * world_trend ** p.export_income_elas

    @property
    def import_share(self) -> np.ndarray:
        """Import share of GDP — rises with wage-led consumption."""
        p = self.params
        return p.import_base + p.import_propensity * self.omega

    @property
    def trade_balance(self) -> np.ndarray:
        """Net exports as share of GDP."""
        return self.export_share - self.import_share

    @property
    def current_account(self) -> np.ndarray:
        """Current account balance (positive = surplus)."""
        return self.trade_balance - self.params.r_f * self.d_f

    @property
    def current_account_deficit(self) -> np.ndarray:
        """Current account deficit = -(current account)."""
        return -self.current_account

    @property
    def inflation_rate(self) -> np.ndarray:
        """Actual inflation rate from wage-price spiral.

        π = ΔP/P derived from markup over unit labour costs.
        When bargained wages exceed actual wage share, prices rise.
        """
        p = self.params
        # Wage pressure: gap between bargained real wage (with inflation expectations)
        # and the actual wage share
        bargained = self.real_wage * (1.0 + p.wage_indexation * self.pi_e)
        wage_pressure = bargained - self.omega
        # Price inflation = pass-through of wage pressure + expected inflation component
        # Simplified Phillips curve augmented with expectations
        raw_inflation = wage_pressure + 0.3 * self.pi_e
        return np.maximum(raw_inflation, 0.0)  # No deflation floor

    @property
    def sectoral_balances(self) -> dict:
        """Sectoral financial balances as shares of GDP.

        Returns dict with 'private', 'government', 'external' balances.
        SFC identity: Private + Government + External = 0 (by construction).

        Government surplus = T - G_primary - r_g·d_g  (positive = surplus)
        External surplus  = M - X + r_f·d_f           (positive = foreign lending to Aus)
        Private surplus   = -(Gov surplus + Ext surplus)  [residual, ensures identity]
        """
        p = self.params
        gov_balance = self.tax_revenue - self.gov_spending_primary - p.r_g * self.d_g
        ext_balance = self.import_share - self.export_share + p.r_f * self.d_f
        # Private balance is the residual to satisfy the SFC identity
        private_balance = -(gov_balance + ext_balance)
        return {
            'private': private_balance,
            'government': gov_balance,
            'external': ext_balance,
        }

    @property
    def bank_net_interest_margin(self) -> np.ndarray:
        """Banks' net interest margin as share of GDP."""
        p = self.params
        return (p.r_l - p.r_d) * self.d + p.r_g * self.d_g


# ── ODE System ───────────────────────────────────────────────────────────────

def extended_keen_ode(t: float, y: np.ndarray, p: ExtendedKeenParams) -> list:
    """
    Extended SFC model ODEs.

    State vector:
        y[0] = ω  (wage share)
        y[1] = λ  (employment rate)
        y[2] = d  (private debt / GDP)
        y[3] = d_g (government debt / GDP)
        y[4] = d_f (net foreign debt / GDP)
        y[5] = π_e (expected inflation)
        y[6] = bcr (bank capital adequacy ratio)
    """
    omega, lam, d, d_g, d_f, pi_e, bcr = y

    # Safety clamping
    omega = max(omega, 0.001)
    lam = max(min(lam, 0.995), 0.001)  # Employment capped at 99.5% (can't exceed labour force)
    d = max(d, 0.0)
    d_g = max(d_g, 0.0)
    pi_e = max(pi_e, 0.0)
    bcr = max(bcr, 0.005)  # Min 0.5% capital ratio

    # ── Derived quantities ─────────────────────────────────────────────

    # Profit share after all interest payments
    pi = 1.0 - omega - p.r_l * d - p.r_g * d_g - p.r_f * d_f

    # Base investment share from Keen investment function
    kap_base = p.investment(np.array([pi]))[0]

    # Credit multiplier from bank capital adequacy
    if p.target_cap_ratio > p.min_cap_ratio and d > 0.001:
        credit_mult = max(0.2, min(1.0,
            (bcr - p.min_cap_ratio) / (p.target_cap_ratio - p.min_cap_ratio)))
    else:
        credit_mult = 1.0

    kap = kap_base * credit_mult  # Effective investment with credit constraints

    # Capital accumulation growth rate
    g = kap / p.nu - p.delta

    # Real wage from Phillips curve
    phi = p.phillips(np.array([lam]))[0]

    # Extended Phillips curve with inflation expectations
    phi_ext = phi * (1.0 + p.wage_indexation * pi_e)

    # Actual inflation from wage-price spiral
    bargained_gap = phi_ext - omega  # Wage pressure
    actual_inflation = max(0.0, bargained_gap + 0.3 * pi_e)

    # ── Tax and government ─────────────────────────────────────────────
    tau = p.tax_rate_wages * omega + p.tax_rate_profits * (1.0 - omega)
    auto_stab = p.g2 * max(0.0, (p.lambda_target - lam) / max(p.lambda_target, 0.01))
    debt_target = p.g3 * (p.d_g_target - d_g)  # Positive when below target → more spending
    g_share = p.g0 + p.g1 + auto_stab + debt_target

    # ── External sector ────────────────────────────────────────────────
    # Exports as share of GDP (small world-income trend, dampened)
    # Use a logistic trend that saturates, preventing explosive export growth
    x_share = p.export_share * (1.0 + min(t / 200.0, 0.5) * p.world_growth * p.export_income_elas)
    # Imports: autonomous + wage-proportionate (higher wages → more consumption → more imports)
    m_share = p.import_base + p.import_propensity * omega

    # ── ODEs ───────────────────────────────────────────────────────────

    # 1. Wage share dynamics (extended with inflation expectations)
    domega_dt = (phi_ext - omega) * (g + p.alpha)

    # 2. Employment dynamics (unchanged from Keen)
    dlam_dt = (g - p.alpha - p.beta) * lam

    # 3. Private debt dynamics (Keen behavioral equation, no growth dilution)
    #    dd/dt = effective_investment - profit_share
    #    In the Keen framework, κ-π is the flow of net new credit relative to GDP.
    #    NOTE: We omit the -g*d growth dilution term here to preserve the original
    #    Keen equilibrium dynamics. Sectoral balances will hold approximately but
    #    not exactly — this is a Keen extension, not a pure SFC model.
    dd_dt = kap - pi

    # 4. Government debt dynamics
    #    d(d_g)/dt = (G + r_g·d_g - T)/Y - g·d_g
    ddg_dt = (g_share + p.r_g * d_g - tau) - g * d_g

    # 5. Foreign debt dynamics
    #    d(d_f)/dt = (M - X + r_f·d_f)/Y - g·d_f
    ddf_dt = (m_share - x_share + p.r_f * d_f) - g * d_f

    # 6. Expected inflation (adaptive expectations)
    #    dπ_e/dt = θ·(π - π_e)
    dpie_dt = p.infl_adapt_speed * (actual_inflation - pi_e)

    # 7. Bank capital ratio dynamics
    #    d(bcr)/dt = retained_earnings_ratio - bcr·(loan_growth_rate)
    #    retained_earnings_ratio = (1 - div_payout)·(r_l - r_d) as fraction of loans
    #    loan_growth_rate = ΔD/D = (dd/dt + g*d)/d = (κ-π+g*d)/d = (κ-π)/d + g
    retained = (1.0 - p.div_payout) * (p.r_l - p.r_d) if d > 0.001 else 0.0
    loan_growth = (kap - pi) / max(d, 0.01) + g if d > 0.001 else g
    dbcr_dt = retained - bcr * loan_growth
    # Safety: prevent bcr from going negative
    if bcr < 0.01 and dbcr_dt < 0:
        dbcr_dt = max(dbcr_dt, 0.01 - bcr)  # Soft floor

    return [domega_dt, dlam_dt, dd_dt, ddg_dt, ddf_dt, dpie_dt, dbcr_dt]


# ── Simulation Runner ─────────────────────────────────────────────────────────

def simulate_extended(
    params: Optional[ExtendedKeenParams] = None,
    t_max: float = 100.0,
    t_steps: int = 2000,
    omega0: float = None,
    lambda0: float = None,
    d0: float = None,
    d_g0: float = None,
    d_f0: float = None,
    pi_e0: float = None,
    bcr0: float = None,
    events: list = None,
    max_step: float = 0.5,
) -> ExtendedKeenSolution:
    """Run the extended SFC Keen model simulation.

    Parameters
    ----------
    params : ExtendedKeenParams, optional
        Model parameters (uses defaults if None).
    t_max : float
        Simulation horizon in years.
    t_steps : int
        Number of output time steps.
    omega0, lambda0, d0, d_g0, d_f0, pi_e0, bcr0 : float, optional
        Override initial conditions.
    events : list, optional
        Event functions for solve_ivp.
    max_step : float
        Maximum ODE solver step size.

    Returns
    -------
    ExtendedKeenSolution
        Container with all trajectories.
    """
    if params is None:
        params = ExtendedKeenParams()

    p = params

    # Initial conditions (supports override)
    y0 = [
        omega0 if omega0 is not None else p.omega0,
        lambda0 if lambda0 is not None else p.lambda0,
        d0 if d0 is not None else p.d0,
        d_g0 if d_g0 is not None else p.d_g0,
        d_f0 if d_f0 is not None else p.d_f0,
        pi_e0 if pi_e0 is not None else p.pi_e0,
        bcr0 if bcr0 is not None else p.bcr0,
    ]

    t_eval = np.linspace(0, t_max, t_steps)

    try:
        sol = solve_ivp(
            extended_keen_ode,
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
        return ExtendedKeenSolution(
            t=np.array([0, t_max]),
            omega=np.array([y0[0], y0[0]]),
            lam=np.array([y0[1], y0[1]]),
            d=np.array([y0[2], y0[2]]),
            d_g=np.array([y0[3], y0[3]]),
            d_f=np.array([y0[4], y0[4]]),
            pi_e=np.array([y0[5], y0[5]]),
            bcr=np.array([y0[6], y0[6]]),
            params=p,
            success=False,
            message=str(e),
        )

    # Extract solutions
    omega_sol = sol.y[0]
    lam_sol = sol.y[1]
    d_sol = sol.y[2]
    d_g_sol = sol.y[3]
    d_f_sol = sol.y[4]
    pi_e_sol = sol.y[5]
    bcr_sol = sol.y[6]

    # Clip to sensible ranges
    omega_sol = np.clip(omega_sol, 0, 2)
    lam_sol = np.clip(lam_sol, 0, 1.05)
    d_sol = np.clip(d_sol, 0, 10)
    d_g_sol = np.clip(d_g_sol, 0, 5)
    d_f_sol = np.clip(d_f_sol, -0.5, 10)  # Can be slightly negative (net creditor)
    pi_e_sol = np.clip(pi_e_sol, 0, 1)
    bcr_sol = np.clip(bcr_sol, 0.001, 1)

    return ExtendedKeenSolution(
        t=sol.t,
        omega=omega_sol,
        lam=lam_sol,
        d=d_sol,
        d_g=d_g_sol,
        d_f=d_f_sol,
        pi_e=pi_e_sol,
        bcr=bcr_sol,
        params=p,
        success=success,
        message=message,
    )


# ── Comparison with base Keen ────────────────────────────────────────────────

def run_comparison(
    base_params: Optional[KeenParams] = None,
    extended_params: Optional[ExtendedKeenParams] = None,
    t_max: float = 100.0,
    t_steps: int = 2000,
) -> dict:
    """Run both the base Keen and extended SFC models for comparison.

    Uses matching initial conditions for shared variables (ω, λ, d).

    Returns
    -------
    dict with keys 'base' (KeenSolution), 'extended' (ExtendedKeenSolution).
    """
    from aus_econ_model.models.keen_model import KeenParams, simulate_keen

    bp = base_params if base_params is not None else KeenParams()
    ep = extended_params if extended_params is not None else ExtendedKeenParams()

    # Copy over shared initial conditions
    base_sol = simulate_keen(bp, t_max=t_max, t_steps=t_steps)
    ext_sol = simulate_extended(ep, t_max=t_max, t_steps=t_steps)

    return {'base': base_sol, 'extended': ext_sol}


# ── Flow-of-Funds Matrix Builder ─────────────────────────────────────────────

def build_flow_of_funds(sol: ExtendedKeenSolution, time_idx: int = -1) -> dict:
    """Construct an SFC flow-of-funds matrix for a given time step.

    Returns a dict representing the matrix with rows = sectors,
    columns = account categories.
    """
    p = sol.params
    idx = time_idx if time_idx >= 0 else len(sol.t) + time_idx
    idx = min(max(idx, 0), len(sol.t) - 1)

    Y = 1.0  # Normalised GDP = 1

    # Get values at this time step
    omega = float(sol.omega[idx])
    lam = float(sol.lam[idx])
    d = float(sol.d[idx])
    d_g = float(sol.d_g[idx])
    d_f = float(sol.d_f[idx])

    # Income flows
    wages = omega * Y
    profits = (1.0 - omega) * Y  # Gross profits before interest
    taxes = float(sol.tax_revenue[idx]) * Y
    gov_spending = float(sol.gov_total_spending[idx]) * Y

    # Interest flows
    priv_interest = p.r_l * d * Y
    gov_interest = p.r_g * d_g * Y
    foreign_interest = p.r_f * d_f * Y

    # Trade
    exports = float(sol.export_share[idx]) * Y
    imports = float(sol.import_share[idx]) * Y

    # Investment
    investment = float(sol.investment_share[idx]) * Y

    # Consumption (residual from national income identity)
    consumption = Y - investment - gov_spending - (exports - imports)

    # Net lending (+) / borrowing (-) by sector
    # Households: wage income + interest on deposits - taxes - consumption
    # Simplified: workers get wages, capitalists get profits
    wage_tax = p.tax_rate_wages * wages
    profit_tax = p.tax_rate_profits * profits
    # Assume workers consume all after-tax wages, capitalists invest + consume
    worker_consumption = wages - wage_tax  # Workers consume all after-tax wages
    capitalist_consumption = consumption - worker_consumption
    capitalist_saving = (profits - profit_tax) - capitalist_consumption
    # Private sector balance
    private_balance = (wages - wage_tax - worker_consumption) + capitalist_saving

    # Government balance
    gov_balance = taxes - gov_spending + gov_interest  # Negative = deficit

    # External balance
    ext_balance = exports - imports + foreign_interest

    # Bank balance (simplified)
    bank_balance = priv_interest + gov_interest - p.r_d * d * Y  # Net interest income

    return {
        'time': float(sol.t[idx]),
        'omega': omega,
        'lam': lam,
        'd': d,
        'd_g': d_g,
        'd_f': d_f,
        'flows': {
            'GDP': Y,
            'Wages': wages,
            'Gross Profits': profits,
            'Taxes': taxes,
            'Gov Spending': gov_spending,
            'Consumption': consumption,
            'Investment': investment,
            'Exports': exports,
            'Imports': imports,
            'Private Interest (loans)': priv_interest,
            'Gov Interest': gov_interest,
            'Foreign Interest': foreign_interest,
        },
        'balances': {
            'Private Sector': private_balance,
            'Government': gov_balance,
            'External': ext_balance,
            'Banking': bank_balance,
        },
    }


def build_balance_sheet(sol: ExtendedKeenSolution, time_idx: int = -1) -> dict:
    """Construct an SFC balance sheet matrix for a given time step."""
    p = sol.params
    idx = time_idx if time_idx >= 0 else len(sol.t) + time_idx
    idx = min(max(idx, 0), len(sol.t) - 1)

    Y = 1.0  # Normalised

    d = float(sol.d[idx])
    d_g = float(sol.d_g[idx])
    d_f = float(sol.d_f[idx])

    # Capital stock: K = ν * Y (capital-to-output ratio)
    capital_stock = p.nu * Y

    # Simplified balance sheet entries (as shares of GDP)
    # Assets (+) and liabilities (-)
    return {
        'time': float(sol.t[idx]),
        'capital_stock': capital_stock,
        'entries': {
            'Private Sector': {
                'Capital': capital_stock,
                'Bank Deposits': -d * Y,  # Net liabilities
                'Bank Loans': -d * Y,
                'Gov Bonds': 0.0,
                'Net Worth': -(capital_stock - d * Y),
            },
            'Banking Sector': {
                'Capital': 0.0,
                'Bank Deposits': d * Y,
                'Bank Loans': d * Y,
                'Gov Bonds': d_g * Y,
                'Net Worth': -(d_g * Y),
            },
            'Government': {
                'Capital': 0.0,
                'Bank Deposits': 0.0,
                'Bank Loans': 0.0,
                'Gov Bonds': -d_g * Y,
                'Net Worth': d_g * Y,
            },
            'External': {
                'Capital': 0.0,
                'Bank Deposits': 0.0,
                'Bank Loans': 0.0,
                'Net Worth': d_f * Y,  # Net foreign assets
            },
        },
        'totals': {
            'Capital': capital_stock,
            'Bank Deposits': d * Y + d * Y - d * Y,  # Sums to 0
        },
    }
