"""
Federal and State Government Fiscal Model
============================================
Detailed fiscal disaggregation model that builds ON TOP of the core
Keen or SFC model solution. Takes a simulated trajectory (wage share,
employment rate, debt ratios, etc.) and computes:

  • Federal government: revenue (PIT, company tax, GST, excise) and
    spending (welfare, health, education, defence, NDIS, interest)
  • State governments: revenue (payroll tax, stamp duty, land tax,
    mining royalties, GST grants) and spending (health, education,
    transport, other)
  • Intergovernmental transfers (GST distribution, specific purpose
    payments)
  • Demographic pressures (ageing, NDIS cost growth)
  • Fiscal sustainability metrics (net debt/GDP, interest/revenue,
    tax gap, fiscal impulse)

Works with both KeenSolution (base model) and ExtendedKeenSolution
(SFC model). For the base model, government debt dynamics are
simulated from first principles. For the SFC model, existing aggregate
fiscal figures are decomposed into federal/state components.

All quantities are expressed as shares of GDP unless otherwise noted.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Union, Any

# ── Parameter Dataclass ──────────────────────────────────────────────────────


@dataclass
class GovtParams:
    """Federal and State government fiscal parameters.

    Calibrated to approximate Australian fiscal settings (2024–25)
    with intergenerational and NDIS projections built in.
    """

    # ═════════════════════════════════════════════════════════════════════
    # FEDERAL GOVERNMENT — Revenue
    # ═════════════════════════════════════════════════════════════════════

    # Personal Income Tax (progressive brackets — 2024-25 rates)
    tax_free_threshold: float = 18200        # AUD (for distribution model)
    pit_threshold_1: float = 45000           # 19c bracket starts
    pit_threshold_2: float = 120000          # 32.5c bracket starts
    pit_threshold_3: float = 180000          # 37c bracket starts
    pit_rate_1: float = 0.19                 # 19%
    pit_rate_2: float = 0.325                # 32.5%
    pit_rate_3: float = 0.37                 # 37%
    pit_rate_4: float = 0.45                 # 45%
    medicare_levy: float = 0.02              # 2% Medicare levy
    # Effective PIT rate at reference (ω₀, λ₀) for the aggregate model
    pit_base_effective_rate: float = 0.22
    # Semi-elasticity of effective rate to average wage index (bracket creep)
    pit_bracket_creep_elas: float = 0.35

    # Company Tax
    company_tax_rate: float = 0.25           # Large business (25%)
    company_tax_rate_sme: float = 0.275      # SME (27.5% — phased)
    sme_share_of_profits: float = 0.35       # Share of profits from SMEs
    # Share of total profit share π that is company profit (vs. unincorporated)
    company_profit_share: float = 0.60

    # Indirect Tax
    gst_rate: float = 0.10                   # 10% GST
    gst_base_coverage: float = 0.55          # Share of consumption in GST base
    excise_petrol_share: float = 0.015       # ~1.5% of GDP
    other_indirect_share: float = 0.020      # ~2.0% of GDP (tobacco, alcohol, tariffs)

    # ═════════════════════════════════════════════════════════════════════
    # FEDERAL GOVERNMENT — Spending
    # ═════════════════════════════════════════════════════════════════════

    # Welfare (as share of GDP at reference)
    age_pension_rate: float = 0.028          # ~2.8% of GDP
    jobseeker_base_rate: float = 0.005       # ~0.5% of GDP at reference employment
    jobseeker_cyclical_sensitivity: float = 0.5  # Increases when employment falls
    family_benefits_rate: float = 0.012      # ~1.2% of GDP
    other_welfare_rate: float = 0.020        # ~2.0% of GDP (DSP, carer allowance, etc.)
    # Welfare indexation
    welfare_indexation_rate: float = 0.015    # Real indexation rate (slower than GDP + wages)

    # Health & Human Services
    health_spending_rate: float = 0.050      # ~5.0% of GDP (Medicare + public hospitals)
    health_cost_growth_premium: float = 0.02 # Health costs grow 2% faster than GDP

    # Education (federal share)
    federal_education_rate: float = 0.020    # ~2.0% of GDP (universities, schools funding)

    # Defence
    defence_rate: float = 0.020              # ~2.0% of GDP

    # NDIS
    ndis_base_rate: float = 0.015            # ~1.5% of GDP (2024–25)
    ndis_real_growth: float = 0.08           # 8% p.a. real growth (projected)
    ndis_max_rate: float = 0.035             # Cap at ~3.5% of GDP (logistic saturation)
    ndis_saturation_speed: float = 0.12      # Speed of saturation (higher = faster to max)

    # Other federal spending
    other_federal_spending_rate: float = 0.080  # ~8.0% of GDP (general govt, housing,
                                                 #       immigration, foreign aid, GST grants to states,
                                                 #       other program expenses)

    # Debt service
    r_g_federal: float = 0.040               # Interest rate on federal government debt
    federal_debt_share: float = 0.70          # Federal share of total gross govt debt

    # ═════════════════════════════════════════════════════════════════════
    # STATE GOVERNMENTS — Revenue
    # ═════════════════════════════════════════════════════════════════════

    payroll_tax_rate: float = 0.055          # Average effective rate ~5.5% of wages
    payroll_tax_threshold_share: float = 0.85  # Share of wages above threshold

    # Property taxes
    stamp_duty_rate: float = 0.040           # ~4% of property turnover value
    property_turnover_share: float = 0.06    # Annual turnover as share of housing stock
    property_value_to_gdp: float = 4.5       # Residential property value / GDP (~4.5x)
    land_tax_rate: float = 0.015             # ~1.5% on assessed value
    land_tax_coverage: float = 0.30          # Share of property value covered by land tax

    # Mining royalties
    mining_royalty_rate: float = 0.085       # Average royalty rate ~8.5%
    mining_share_of_gdp: float = 0.085       # Mining ~8.5% of GDP
    # Resource price sensitivity: when employment is high, commodity prices tend higher
    resource_cycle_elasticity: float = 0.5

    # Other state revenue
    state_other_revenue_rate: float = 0.020  # ~2% of GDP (fines, fees, public enterprises)

    # GST distribution from federal to states
    gst_distribution_share: float = 1.0      # All GST goes to states (after admin)
    # Share of state revenue from Commonwealth grants
    commonwealth_grants_share_of_state_rev: float = 0.45

    # ═════════════════════════════════════════════════════════════════════
    # STATE GOVERNMENTS — Spending
    # ═════════════════════════════════════════════════════════════════════

    # State spending as share of GDP
    state_spending_base: float = 0.16        # ~16% of GDP total state spending
    state_health_share: float = 0.30         # Health = 30% of state spending
    state_education_share: float = 0.25      # Education = 25% of state spending
    state_transport_share: float = 0.10      # Transport/infrastructure = 10%
    state_other_spending_share: float = 0.35 # Other (police, courts, admin, etc.)

    # State debt
    r_g_state: float = 0.040                 # Interest rate on state government debt
    state_debt_share: float = 0.30           # State share of total gross govt debt
    state_initial_debt_to_gdp: float = 0.15  # Initial state net debt (~15% GDP aggregate)

    # ═════════════════════════════════════════════════════════════════════
    # DEMOGRAPHIC / INTERGENERATIONAL PRESSURES
    # ═════════════════════════════════════════════════════════════════════

    old_age_dependency_ratio_base: float = 0.25   # Current ~25% (65+/15-64)
    old_age_dependency_growth: float = 0.008      # ~0.8% p.a. growth (slower than historical)
    health_cost_growth_premium: float = 0.02      # 2% p.a. faster than GDP

    # ═════════════════════════════════════════════════════════════════════
    # INITIAL CONDITIONS
    # ═════════════════════════════════════════════════════════════════════

    # Federal debt
    federal_debt0: float = 0.28               # Federal net debt ~28% GDP (2025)
    # (Total gross debt ~55% → split 70/30 → federal gross ~38.5%, net ~28%)

    # Fiscal rule parameters
    fiscal_rule_active: bool = True
    fed_debt_anchor: float = 0.35             # Net debt target ~35% GDP (2030 PBO)
    fiscal_adjustment_speed: float = 0.05     # Speed of fiscal adjustment per year

    # Reference values for effective tax rate calculation
    omega_ref: float = 0.55                   # Reference wage share
    lam_ref: float = 0.94                     # Reference employment rate


# ── Solution Dataclass ────────────────────────────────────────────────────────


@dataclass
class GovtSolution:
    """Container for detailed government fiscal model results.

    All arrays are time series matching the core solution trajectory.
    All monetary values expressed as shares of GDP unless noted.

    Supports dict-style subscript access: gs['total_revenue_pct']
    """
    t: np.ndarray
    success: bool
    message: str = ""

    # ── Federal Revenue ─────────────────────────────────────────────────
    fed_pit: np.ndarray = field(default_factory=lambda: np.array([]))          # Personal income tax
    fed_company_tax: np.ndarray = field(default_factory=lambda: np.array([]))   # Company tax
    fed_gst: np.ndarray = field(default_factory=lambda: np.array([]))           # GST
    fed_excise: np.ndarray = field(default_factory=lambda: np.array([]))        # Excise & other indirect
    fed_other_revenue: np.ndarray = field(default_factory=lambda: np.array([])) # Other federal revenue
    fed_total_revenue: np.ndarray = field(default_factory=lambda: np.array([])) # Total federal revenue

    # ── Federal Spending ────────────────────────────────────────────────
    fed_age_pension: np.ndarray = field(default_factory=lambda: np.array([]))
    fed_jobseeker: np.ndarray = field(default_factory=lambda: np.array([]))
    fed_family_benefits: np.ndarray = field(default_factory=lambda: np.array([]))
    fed_other_welfare: np.ndarray = field(default_factory=lambda: np.array([]))
    fed_total_welfare: np.ndarray = field(default_factory=lambda: np.array([]))
    fed_health: np.ndarray = field(default_factory=lambda: np.array([]))
    fed_education: np.ndarray = field(default_factory=lambda: np.array([]))
    fed_defence: np.ndarray = field(default_factory=lambda: np.array([]))
    fed_ndis: np.ndarray = field(default_factory=lambda: np.array([]))
    fed_other_spending: np.ndarray = field(default_factory=lambda: np.array([]))
    fed_primary_spending: np.ndarray = field(default_factory=lambda: np.array([]))
    fed_interest: np.ndarray = field(default_factory=lambda: np.array([]))     # Gross interest
    fed_total_spending: np.ndarray = field(default_factory=lambda: np.array([]))

    # ── Federal Balances ────────────────────────────────────────────────
    fed_primary_balance: np.ndarray = field(default_factory=lambda: np.array([]))
    fed_total_balance: np.ndarray = field(default_factory=lambda: np.array([]))
    fed_debt: np.ndarray = field(default_factory=lambda: np.array([]))          # Federal debt / GDP

    # ── State Revenue ───────────────────────────────────────────────────
    state_payroll_tax: np.ndarray = field(default_factory=lambda: np.array([]))
    state_stamp_duty: np.ndarray = field(default_factory=lambda: np.array([]))
    state_land_tax: np.ndarray = field(default_factory=lambda: np.array([]))
    state_mining_royalties: np.ndarray = field(default_factory=lambda: np.array([]))
    state_gst_grants: np.ndarray = field(default_factory=lambda: np.array([]))  # GST distribution
    state_spp: np.ndarray = field(default_factory=lambda: np.array([]))         # Specific purpose payments
    state_other_revenue: np.ndarray = field(default_factory=lambda: np.array([]))
    state_total_revenue: np.ndarray = field(default_factory=lambda: np.array([]))

    # ── State Spending ──────────────────────────────────────────────────
    state_health: np.ndarray = field(default_factory=lambda: np.array([]))
    state_education: np.ndarray = field(default_factory=lambda: np.array([]))
    state_transport: np.ndarray = field(default_factory=lambda: np.array([]))
    state_other_spending: np.ndarray = field(default_factory=lambda: np.array([]))
    state_primary_spending: np.ndarray = field(default_factory=lambda: np.array([]))
    state_interest: np.ndarray = field(default_factory=lambda: np.array([]))
    state_total_spending: np.ndarray = field(default_factory=lambda: np.array([]))

    # ── State Balances ──────────────────────────────────────────────────
    state_primary_balance: np.ndarray = field(default_factory=lambda: np.array([]))
    state_total_balance: np.ndarray = field(default_factory=lambda: np.array([]))
    state_debt: np.ndarray = field(default_factory=lambda: np.array([]))        # State debt / GDP

    # ── Intergovernmental Transfers ─────────────────────────────────────
    total_gst_collected: np.ndarray = field(default_factory=lambda: np.array([]))
    total_spp: np.ndarray = field(default_factory=lambda: np.array([]))
    net_federal_to_states: np.ndarray = field(default_factory=lambda: np.array([]))

    # ── Consolidated Government ─────────────────────────────────────────
    total_revenue_pct: np.ndarray = field(default_factory=lambda: np.array([]))
    total_spending_pct: np.ndarray = field(default_factory=lambda: np.array([]))
    total_primary_spending: np.ndarray = field(default_factory=lambda: np.array([]))
    total_balance_pct: np.ndarray = field(default_factory=lambda: np.array([]))
    total_primary_balance: np.ndarray = field(default_factory=lambda: np.array([]))
    total_gross_debt: np.ndarray = field(default_factory=lambda: np.array([]))
    total_net_debt: np.ndarray = field(default_factory=lambda: np.array([]))

    # ── Fiscal Sustainability Metrics ───────────────────────────────────
    fed_net_debt_gdp: np.ndarray = field(default_factory=lambda: np.array([]))
    state_net_debt_gdp: np.ndarray = field(default_factory=lambda: np.array([]))
    consolidated_net_debt_gdp: np.ndarray = field(default_factory=lambda: np.array([]))
    fed_interest_revenue_ratio: np.ndarray = field(default_factory=lambda: np.array([]))
    state_interest_revenue_ratio: np.ndarray = field(default_factory=lambda: np.array([]))
    total_interest_revenue_ratio: np.ndarray = field(default_factory=lambda: np.array([]))
    tax_gap: np.ndarray = field(default_factory=lambda: np.array([]))           # Revenue shortfall vs benchmark
    fiscal_impulse: np.ndarray = field(default_factory=lambda: np.array([]))    # Δ structural balance

    # ── Fiscal Rule Adjustment ──────────────────────────────────────────
    fed_fiscal_rule_adjustment: np.ndarray = field(default_factory=lambda: np.array([]))
    fed_fiscal_rule_primary_balance: np.ndarray = field(default_factory=lambda: np.array([]))
    fed_fiscal_rule_total_balance: np.ndarray = field(default_factory=lambda: np.array([]))
    state_fiscal_rule_adjustment: np.ndarray = field(default_factory=lambda: np.array([]))
    state_fiscal_rule_primary_balance: np.ndarray = field(default_factory=lambda: np.array([]))
    state_fiscal_rule_total_balance: np.ndarray = field(default_factory=lambda: np.array([]))

    # ── Demographic Indicators ──────────────────────────────────────────
    old_age_dependency: np.ndarray = field(default_factory=lambda: np.array([]))
    age_pension_to_gdp: np.ndarray = field(default_factory=lambda: np.array([]))
    health_spending_to_gdp: np.ndarray = field(default_factory=lambda: np.array([]))
    ndis_to_gdp: np.ndarray = field(default_factory=lambda: np.array([]))

    # ── Revenue Composition (for visualisation) ─────────────────────────
    fed_revenue_breakdown: dict = field(default_factory=dict)
    fed_spending_breakdown: dict = field(default_factory=dict)
    state_revenue_breakdown: dict = field(default_factory=dict)
    state_spending_breakdown: dict = field(default_factory=dict)
    consolidated_composition: dict = field(default_factory=dict)

    def __getitem__(self, key: str) -> Any:
        """Allow dict-style subscript access (e.g. gs['total_revenue_pct'])."""
        return getattr(self, key)


# ── Core Simulation Function ──────────────────────────────────────────────────


def simulate_fiscal(
    core_solution: Any,
    params: Optional[GovtParams] = None,
) -> GovtSolution:
    """Post-process a core model solution to compute detailed fiscal breakdowns.

    Works with both ``KeenSolution`` (base model) and ``ExtendedKeenSolution``
    (SFC model).

    Parameters
    ----------
    core_solution : KeenSolution or ExtendedKeenSolution
        The core model simulation result. Must have at minimum:
        ``.t``, ``.omega``, ``.lam``, ``.d``, ``.profit_share``,
        ``.investment_share``, ``.growth_rate``.
        If available, also uses: ``.d_g``, ``.tax_revenue``, ``.gov_spending_primary``,
        ``.trade_balance``, ``.export_share``, ``.import_share``.
    params : GovtParams, optional
        Fiscal parameters (defaults used if None).

    Returns
    -------
    GovtSolution
        Dataclass with all revenue, spending, debt, and sustainability metrics.
    """
    if params is None:
        params = GovtParams()

    p = params
    sol = core_solution

    # ── Extract core time series ────────────────────────────────────────────
    t = np.asarray(sol.t, dtype=float)
    n = len(t)
    omega = np.asarray(sol.omega, dtype=float)
    lam = np.asarray(sol.lam, dtype=float)
    d = np.asarray(sol.d, dtype=float)
    pi = np.asarray(sol.profit_share, dtype=float)

    # Growth rate for debt dynamics
    if hasattr(sol, 'growth_rate') and sol.growth_rate is not None:
        g = np.asarray(sol.growth_rate, dtype=float)
    else:
        g = np.full(n, 0.03)  # Default 3% growth

    # Investment share (needed to estimate consumption)
    if hasattr(sol, 'investment_share') and sol.investment_share is not None:
        inv_share = np.asarray(sol.investment_share, dtype=float)
    else:
        # Approximate from profit share
        inv_share = pi.copy()

    # ── Government sector from SFC model (if available) ─────────────────────
    has_sfc = hasattr(sol, 'd_g') and sol.d_g is not None

    if has_sfc:
        d_g = np.asarray(sol.d_g, dtype=float)
        # Aggregate gov spending from SFC model
        if hasattr(sol, 'gov_spending_primary') and sol.gov_spending_primary is not None:
            g_primary_agg = np.asarray(sol.gov_spending_primary, dtype=float)
        else:
            g_primary_agg = None
        if hasattr(sol, 'gov_total_spending') and sol.gov_total_spending is not None:
            g_total_agg = np.asarray(sol.gov_total_spending, dtype=float)
        else:
            g_total_agg = None
        if hasattr(sol, 'tax_revenue') and sol.tax_revenue is not None:
            tax_agg = np.asarray(sol.tax_revenue, dtype=float)
        else:
            tax_agg = None
        # Trade data for consumption estimate
        if hasattr(sol, 'trade_balance') and sol.trade_balance is not None:
            nx = np.asarray(sol.trade_balance, dtype=float)
        else:
            nx = np.zeros(n)
        if hasattr(sol, 'export_share') and sol.export_share is not None:
            x_share = np.asarray(sol.export_share, dtype=float)
        else:
            x_share = np.full(n, 0.22)
        if hasattr(sol, 'import_share') and sol.import_share is not None:
            m_share = np.asarray(sol.import_share, dtype=float)
        else:
            m_share = np.full(n, 0.20)
    else:
        d_g = np.full(n, 0.55)  # Default gross debt estimate
        g_primary_agg = None
        g_total_agg = None
        tax_agg = None
        nx = np.zeros(n)
        x_share = np.full(n, 0.22)
        m_share = np.full(n, 0.20)

    # ── Consumption share of GDP (C/Y) ──────────────────────────────────────
    # Y = C + I + G + NX  →  C/Y = 1 - I/Y - G/Y - NX/Y
    if g_total_agg is not None:
        c_share = 1.0 - inv_share - g_total_agg - nx
    else:
        # Estimate: G/Y ≈ sum of federal + state primary spending before debt dynamics
        # Use a rough constant initially, then refine
        c_share = 1.0 - inv_share - 0.32 - nx  # ~32% total gov spending estimate
    c_share = np.clip(c_share, 0.3, 0.9)  # Plausible range

    # ── Reference values (t=0) ──────────────────────────────────────────────
    omega_0 = float(omega[0])
    lam_0 = float(lam[0])

    # ── Initialise output arrays ────────────────────────────────────────────
    def _arr(init_val=0.0):
        return np.full(n, init_val, dtype=float)

    # ═══════════════════════════════════════════════════════════════════════
    # FEDERAL REVENUE
    # ═══════════════════════════════════════════════════════════════════════

    # 1. Personal Income Tax — progressive with bracket creep
    #    Average wage index: higher ω and λ → higher avg wage → bracket creep
    wage_index = np.ones(n)
    safe_lam = np.maximum(lam, 0.01)
    safe_lam_0 = max(lam_0, 0.01)
    wage_index = (omega / safe_lam) / (omega_0 / safe_lam_0)

    # Effective PIT rate (captures bracket creep progressively)
    pit_eff_rate = p.pit_base_effective_rate * (wage_index ** p.pit_bracket_creep_elas)
    pit_eff_rate = pit_eff_rate + p.medicare_levy * np.minimum(1.0, wage_index / 0.8)
    pit_eff_rate = np.minimum(pit_eff_rate, 0.50)  # Cap

    fed_pit = pit_eff_rate * omega  # PIT as share of GDP

    # 2. Company Tax
    #    Company profits as share of total profits π
    company_profits = p.company_profit_share * pi
    #    Large business portion
    large_biz_profits = (1.0 - p.sme_share_of_profits) * company_profits
    sme_profits = p.sme_share_of_profits * company_profits
    fed_company_tax = p.company_tax_rate * large_biz_profits + p.company_tax_rate_sme * sme_profits

    # 3. GST — 10% on taxable consumption
    fed_gst = p.gst_rate * p.gst_base_coverage * c_share

    # 4. Excise and other indirect taxes
    fed_excise = np.full(n, p.excise_petrol_share)
    fed_other_revenue = np.full(n, p.other_indirect_share)

    fed_total_revenue = fed_pit + fed_company_tax + fed_gst + fed_excise + fed_other_revenue

    # ═══════════════════════════════════════════════════════════════════════
    # FEDERAL SPENDING
    # ═══════════════════════════════════════════════════════════════════════

    time_years = t.copy()

    # ── Welfare ──
    # Age pension — driven by old-age dependency ratio, with saturating growth
    odr = p.old_age_dependency_ratio_base * np.exp(p.old_age_dependency_growth * time_years)
    odr = np.minimum(odr, 0.40)  # Cap at 40%
    # Age pension spending grows with dependency ratio but slower than 1:1
    age_pension = p.age_pension_rate * (1.0 + 0.6 * (odr / p.old_age_dependency_ratio_base - 1.0))
    # Modest real indexation
    pen_idx = 1.0 + 0.3 * (1.0 - np.exp(-0.04 * time_years))
    age_pension = age_pension * pen_idx

    # JobSeeker — counter-cyclical (rises when employment falls)
    unemp_gap = np.maximum(0, (p.lam_ref - lam) / max(p.lam_ref, 0.01))
    jobseeker = p.jobseeker_base_rate * (1.0 + p.jobseeker_cyclical_sensitivity * unemp_gap)
    # Partial indexation with saturation
    js_idx = 1.0 + 0.2 * (1.0 - np.exp(-0.04 * time_years))
    jobseeker = jobseeker * js_idx

    # Family benefits & other welfare (constant share of GDP)
    family_benefits = np.full(n, p.family_benefits_rate)
    other_welfare = np.full(n, p.other_welfare_rate)

    fed_total_welfare = age_pension + jobseeker + family_benefits + other_welfare

    # ── Health — growing faster than GDP (ageing + technology), saturating ──
    health_growth_factor = 1.0 + p.health_cost_growth_premium * (1.0 - np.exp(-0.06 * time_years))
    fed_health = p.health_spending_rate * health_growth_factor

    # ── Education (stable share of GDP) ──
    fed_education = np.full(n, p.federal_education_rate)

    # ── Defence (stable share of GDP) ──
    fed_defence = np.full(n, p.defence_rate)

    # ── NDIS — logistic growth that saturates at ndis_max_rate ──
    # Logistic: ndis(t) = L / (1 + (L/K0 - 1) * exp(-r * t))
    # where L = max_rate, K0 = base_rate, r = saturation_speed
    L = p.ndis_max_rate
    K0 = p.ndis_base_rate
    r = p.ndis_saturation_speed
    ndis = L / (1.0 + (L / max(K0, 0.001) - 1.0) * np.exp(-r * time_years))
    # Ensure monotonic growth from base rate
    ndis = np.maximum(ndis, K0)

    # ── Other federal spending ──
    fed_other = np.full(n, p.other_federal_spending_rate)

    # ── Total federal primary spending ──
    fed_primary_spending = fed_total_welfare + fed_health + fed_education + fed_defence + ndis + fed_other

    # ═══════════════════════════════════════════════════════════════════════
    # FEDERAL DEBT DYNAMICS
    # ═══════════════════════════════════════════════════════════════════════

    # Split total gov debt into federal and state components
    fed_debt_share = p.federal_debt_share
    fed_debt = d_g * fed_debt_share

    # Default interest (will recompute if dynamic debt is simulated)
    fed_interest = p.r_g_federal * fed_debt

    # If we have an SFC solution, use the d_g trajectory directly.
    # Otherwise, simulate federal debt dynamics from initial condition
    # using raw balances. The fiscal rule adjustment is reported separately.
    if not has_sfc:
        fed_debt_dynamic = np.full(n, p.federal_debt0)
        for i in range(1, n):
            dt = t[i] - t[i - 1] if i > 0 else 0.0
            if dt > 0:
                # Raw primary balance (no fiscal rule adjustment)
                primary_balance = fed_total_revenue[i] - fed_primary_spending[i]
                # Interest on existing debt
                interest_flow = p.r_g_federal * fed_debt_dynamic[i - 1]
                # Total deficit (negative = deficit/borrowing)
                total_deficit = primary_balance - interest_flow
                # Debt dynamics: d(debt)/dt = -total_deficit - g*debt
                dd = -total_deficit - g[i] * fed_debt_dynamic[i - 1]
                fed_debt_dynamic[i] = fed_debt_dynamic[i - 1] + dd * dt
                fed_debt_dynamic[i] = max(fed_debt_dynamic[i], 0.0)
            else:
                fed_debt_dynamic[i] = fed_debt_dynamic[i - 1]

        fed_debt = fed_debt_dynamic

        # Recompute interest and total spending with dynamic debt
        fed_interest = p.r_g_federal * fed_debt
        fed_total_spending = fed_primary_spending + fed_interest

        # Raw balances (what actually happens without fiscal rule adjustment)
        fed_primary_balance = fed_total_revenue - fed_primary_spending
        fed_total_balance = fed_total_revenue - fed_total_spending

        # Fiscal rule adjustment (reported as separate metric)
        if p.fiscal_rule_active:
            debt_gap = fed_debt - p.fed_debt_anchor
            fiscal_adj = p.fiscal_adjustment_speed * debt_gap
            fed_fiscal_rule_adjustment = fiscal_adj
            fed_fiscal_rule_primary_balance = fed_primary_balance + fiscal_adj
            fed_fiscal_rule_total_balance = fed_total_balance + fiscal_adj
        else:
            fed_fiscal_rule_adjustment = np.zeros(n)
            fed_fiscal_rule_primary_balance = fed_primary_balance
            fed_fiscal_rule_total_balance = fed_total_balance
    else:
        # SFC model provides d_g trajectory — compute derived fiscal values
        fed_interest = p.r_g_federal * fed_debt
        fed_total_spending = fed_primary_spending + fed_interest
        fed_primary_balance = fed_total_revenue - fed_primary_spending
        fed_total_balance = fed_total_revenue - fed_total_spending
        fed_fiscal_rule_adjustment = np.zeros(n)
        fed_fiscal_rule_primary_balance = fed_primary_balance
        fed_fiscal_rule_total_balance = fed_total_balance

    # ═══════════════════════════════════════════════════════════════════════
    # STATE REVENUE
    # ═══════════════════════════════════════════════════════════════════════

    # 1. Payroll tax — on wages above threshold
    taxable_wages = omega * p.payroll_tax_threshold_share
    state_payroll_tax = p.payroll_tax_rate * taxable_wages

    # 2. Stamp duty — on property turnover
    #    Property value relative to GDP
    prop_value = p.property_value_to_gdp
    #    Property turnover (transactions as share of stock)
    turnover_val = prop_value * p.property_turnover_share
    state_stamp_duty = p.stamp_duty_rate * turnover_val * np.ones(n)
    #    Stamp duty is cyclical — boost when economy strong
    state_stamp_duty = state_stamp_duty * (0.5 + 0.5 * lam / p.lam_ref)

    # 3. Land tax — on investment properties
    taxable_land = prop_value * p.land_tax_coverage
    state_land_tax = p.land_tax_rate * taxable_land * np.ones(n)
    #    Land tax grows with property values (proxied by wage share + employment)
    state_land_tax = state_land_tax * (wage_index ** 0.5)

    # 4. Mining royalties — on resource extraction value
    #    Resource revenue depends on global demand (proxied by employment)
    resource_revenue = p.mining_share_of_gdp * (1.0 + p.resource_cycle_elasticity * (lam - p.lam_ref) / p.lam_ref)
    resource_revenue = np.maximum(resource_revenue, p.mining_share_of_gdp * 0.5)  # Floor
    state_mining_royalties = p.mining_royalty_rate * resource_revenue

    # 5. GST grants from Commonwealth
    state_gst_grants = fed_gst * p.gst_distribution_share

    # 6. Specific purpose payments (federal → states for health, education, housing)
    #    Roughly 30% of federal health + education spending flows to states as SPP
    state_spp = 0.30 * (fed_health + fed_education)

    # 7. Other state revenue
    state_other_revenue = np.full(n, p.state_other_revenue_rate)

    state_total_revenue = (state_payroll_tax + state_stamp_duty + state_land_tax
                           + state_mining_royalties + state_gst_grants
                           + state_spp + state_other_revenue)

    # ═══════════════════════════════════════════════════════════════════════
    # STATE SPENDING
    # ═══════════════════════════════════════════════════════════════════════

    # Total state primary spending base (stable share of GDP)
    state_spending_base = np.full(n, p.state_spending_base)

    # Decompose into functional shares
    state_health = state_spending_base * p.state_health_share
    state_education = state_spending_base * p.state_education_share
    state_transport = state_spending_base * p.state_transport_share
    state_other_spending = state_spending_base * p.state_other_spending_share

    state_primary_spending = state_health + state_education + state_transport + state_other_spending

    # State debt dynamics
    state_debt_share = p.state_debt_share
    state_debt = d_g * state_debt_share if has_sfc else np.full(n, p.state_initial_debt_to_gdp)

    # State interest (will recompute if dynamic)
    state_interest = p.r_g_state * state_debt

    if not has_sfc:
        # Simulate state debt from initial value with fiscal rule (as separate metric)
        state_debt_dynamic = np.full(n, p.state_initial_debt_to_gdp)
        for i in range(1, n):
            dt = t[i] - t[i - 1] if i > 0 else 0.0
            if dt > 0:
                # Raw primary balance (no fiscal rule)
                primary_balance = state_total_revenue[i] - state_primary_spending[i]
                interest_flow = p.r_g_state * state_debt_dynamic[i - 1]
                total_deficit = primary_balance - interest_flow
                dd = -total_deficit - g[i] * state_debt_dynamic[i - 1]
                state_debt_dynamic[i] = state_debt_dynamic[i - 1] + dd * dt
                state_debt_dynamic[i] = max(state_debt_dynamic[i], 0.0)
            else:
                state_debt_dynamic[i] = state_debt_dynamic[i - 1]

        state_debt = state_debt_dynamic

        # Recompute interest and spending
        state_interest = p.r_g_state * state_debt
        state_total_spending = state_primary_spending + state_interest

        # Raw balances
        state_primary_balance = state_total_revenue - state_primary_spending
        state_total_balance = state_total_revenue - state_total_spending

        # State fiscal rule adjustment (separate metric)
        if p.fiscal_rule_active:
            state_debt_target = p.state_initial_debt_to_gdp * 1.5
            debt_gap = state_debt - state_debt_target
            fiscal_adj = p.fiscal_adjustment_speed * 0.5 * debt_gap
            state_fiscal_rule_adjustment = fiscal_adj
            state_fiscal_rule_primary_balance = state_primary_balance + fiscal_adj
            state_fiscal_rule_total_balance = state_total_balance + fiscal_adj
        else:
            state_fiscal_rule_adjustment = np.zeros(n)
            state_fiscal_rule_primary_balance = state_primary_balance
            state_fiscal_rule_total_balance = state_total_balance
    else:
        # SFC model provides d_g trajectory — compute derived values
        state_interest = p.r_g_state * state_debt
        state_total_spending = state_primary_spending + state_interest
        state_primary_balance = state_total_revenue - state_primary_spending
        state_total_balance = state_total_revenue - state_total_spending
        state_fiscal_rule_adjustment = np.zeros(n)
        state_fiscal_rule_primary_balance = state_primary_balance
        state_fiscal_rule_total_balance = state_total_balance

    # ═══════════════════════════════════════════════════════════════════════
    # INTERGOVERNMENTAL TRANSFERS
    # ═══════════════════════════════════════════════════════════════════════

    total_gst_collected = fed_gst.copy()
    total_spp = state_spp.copy()
    net_federal_to_states = state_gst_grants + state_spp - 0.0  # Net flow: federal → states

    # ═══════════════════════════════════════════════════════════════════════
    # CONSOLIDATED GOVERNMENT
    # ═══════════════════════════════════════════════════════════════════════

    # Consolidated = federal + state, removing intergovernmental transfers
    intergov_transfers = state_gst_grants + state_spp
    total_revenue = fed_total_revenue + state_total_revenue - intergov_transfers
    total_primary_spending = fed_primary_spending + state_primary_spending - intergov_transfers
    total_spending = fed_total_spending + state_total_spending - intergov_transfers
    total_balance = total_revenue - total_spending
    total_primary_balance = total_revenue - total_primary_spending
    total_gross_debt = fed_debt + state_debt

    # Net debt: gross debt minus financial assets (approximate with a haircut)
    # Federal government has ~30% of gross debt offset by assets
    fed_net_debt = fed_debt * 0.70
    state_net_debt = state_debt * 0.80  # States have fewer financial assets
    total_net_debt = fed_net_debt + state_net_debt

    # ═══════════════════════════════════════════════════════════════════════
    # FISCAL SUSTAINABILITY METRICS
    # ═══════════════════════════════════════════════════════════════════════

    # Net debt / GDP (already in ratio form)
    fed_net_debt_gdp = fed_net_debt.copy()
    state_net_debt_gdp = state_net_debt.copy()
    consolidated_net_debt_gdp = total_net_debt.copy()

    # Interest / revenue ratio
    fed_interest_revenue = np.divide(fed_interest, fed_total_revenue,
                                     out=np.zeros(n), where=fed_total_revenue > 0.001)
    state_interest_revenue = np.divide(state_interest, state_total_revenue,
                                       out=np.zeros(n), where=state_total_revenue > 0.001)
    # Consolidated: total interest / (total revenue - intergov transfers received by states)
    cons_rev_net = total_revenue.copy()
    total_interest = fed_interest + state_interest
    total_interest_revenue = np.divide(total_interest, cons_rev_net,
                                       out=np.zeros(n), where=cons_rev_net > 0.001)

    # Tax gap — difference between actual total tax revenue and a benchmark
    # Benchmark: OECD average tax-to-GDP ratio (~34%)
    benchmark_tax_gdp = 0.34
    total_tax = fed_total_revenue + state_payroll_tax + state_stamp_duty + state_land_tax + state_mining_royalties + state_other_revenue
    tax_gap = (total_tax - benchmark_tax_gdp) / benchmark_tax_gdp  # Positive = above benchmark

    # Fiscal impulse — year-on-year change in structural primary balance
    # Structural = cyclically adjusted (remove the output gap effect)
    potential_output_gap = (lam - p.lam_ref) / max(p.lam_ref, 0.01)
    # Cyclically adjusted primary balance
    cyc_adj_primary_bal = total_primary_balance - 0.3 * potential_output_gap  # ~0.3 automatic stabiliser
    fiscal_impulse = np.zeros(n)
    dt_vals = np.diff(t, prepend=t[0])
    fiscal_impulse[1:] = (cyc_adj_primary_bal[1:] - cyc_adj_primary_bal[:-1]) / np.maximum(dt_vals[1:], 0.01)
    fiscal_impulse[0] = fiscal_impulse[1]

    # ═══════════════════════════════════════════════════════════════════════
    # DEMOGRAPHIC INDICATORS
    # ═══════════════════════════════════════════════════════════════════════

    old_age_dependency = odr
    age_pension_to_gdp = age_pension
    health_spending_to_gdp = fed_health + state_health
    ndis_to_gdp = ndis

    # ═══════════════════════════════════════════════════════════════════════
    # COMPOSITION BREAKDOWNS (final time step for quick reference)
    # ═══════════════════════════════════════════════════════════════════════

    fed_revenue_breakdown = {
        'Personal Income Tax': fed_pit,
        'Company Tax': fed_company_tax,
        'GST': fed_gst,
        'Excise & Indirect': fed_excise + fed_other_revenue,
    }
    fed_spending_breakdown = {
        'Age Pension': age_pension,
        'JobSeeker & Welfare': jobseeker + family_benefits + other_welfare,
        'Health': fed_health,
        'Education': fed_education,
        'Defence': fed_defence,
        'NDIS': ndis,
        'Other Federal': fed_other,
        'Interest': fed_interest,
    }
    state_revenue_breakdown = {
        'Payroll Tax': state_payroll_tax,
        'Stamp Duty': state_stamp_duty,
        'Land Tax': state_land_tax,
        'Mining Royalties': state_mining_royalties,
        'GST Grants': state_gst_grants,
        'SPP': state_spp,
        'Other State': state_other_revenue,
    }
    state_spending_breakdown = {
        'Health': state_health,
        'Education': state_education,
        'Transport': state_transport,
        'Other State': state_other_spending,
        'Interest': state_interest,
    }
    consolidated_composition = {
        'Federal Revenue': fed_total_revenue,
        'State Revenue (own)': state_total_revenue - state_gst_grants - state_spp,
        'Federal Spending': fed_total_spending,
        'State Spending': state_total_spending,
        'Intergov Transfers': state_gst_grants + state_spp,
    }

    # ═══════════════════════════════════════════════════════════════════════
    # BUILD SOLUTION
    # ═══════════════════════════════════════════════════════════════════════

    return GovtSolution(
        t=t,
        success=True,
        message="Government fiscal model computed successfully",

        # Federal revenue
        fed_pit=fed_pit,
        fed_company_tax=fed_company_tax,
        fed_gst=fed_gst,
        fed_excise=fed_excise,
        fed_other_revenue=fed_other_revenue,
        fed_total_revenue=fed_total_revenue,

        # Federal spending
        fed_age_pension=age_pension,
        fed_jobseeker=jobseeker,
        fed_family_benefits=family_benefits,
        fed_other_welfare=other_welfare,
        fed_total_welfare=fed_total_welfare,
        fed_health=fed_health,
        fed_education=fed_education,
        fed_defence=fed_defence,
        fed_ndis=ndis,
        fed_other_spending=fed_other,
        fed_primary_spending=fed_primary_spending,
        fed_interest=fed_interest,
        fed_total_spending=fed_total_spending,

        # Federal balances
        fed_primary_balance=fed_primary_balance,
        fed_total_balance=fed_total_balance,
        fed_debt=fed_debt,

        # State revenue
        state_payroll_tax=state_payroll_tax,
        state_stamp_duty=state_stamp_duty,
        state_land_tax=state_land_tax,
        state_mining_royalties=state_mining_royalties,
        state_gst_grants=state_gst_grants,
        state_spp=state_spp,
        state_other_revenue=state_other_revenue,
        state_total_revenue=state_total_revenue,

        # State spending
        state_health=state_health,
        state_education=state_education,
        state_transport=state_transport,
        state_other_spending=state_other_spending,
        state_primary_spending=state_primary_spending,
        state_interest=state_interest,
        state_total_spending=state_total_spending,

        # State balances
        state_primary_balance=state_primary_balance,
        state_total_balance=state_total_balance,
        state_debt=state_debt,

        # Intergovernmental
        total_gst_collected=total_gst_collected,
        total_spp=total_spp,
        net_federal_to_states=net_federal_to_states,

        # Consolidated
        total_revenue_pct=total_revenue,
        total_spending_pct=total_spending,
        total_primary_spending=total_primary_spending,
        total_balance_pct=total_balance,
        total_primary_balance=total_primary_balance,
        total_gross_debt=total_gross_debt,
        total_net_debt=total_net_debt,

        # Sustainability
        fed_net_debt_gdp=fed_net_debt_gdp,
        state_net_debt_gdp=state_net_debt_gdp,
        consolidated_net_debt_gdp=consolidated_net_debt_gdp,
        fed_interest_revenue_ratio=fed_interest_revenue,
        state_interest_revenue_ratio=state_interest_revenue,
        total_interest_revenue_ratio=total_interest_revenue,
        tax_gap=tax_gap,
        fiscal_impulse=fiscal_impulse,

        # Fiscal rule
        fed_fiscal_rule_adjustment=fed_fiscal_rule_adjustment,
        fed_fiscal_rule_primary_balance=fed_fiscal_rule_primary_balance,
        fed_fiscal_rule_total_balance=fed_fiscal_rule_total_balance,
        state_fiscal_rule_adjustment=state_fiscal_rule_adjustment,
        state_fiscal_rule_primary_balance=state_fiscal_rule_primary_balance,
        state_fiscal_rule_total_balance=state_fiscal_rule_total_balance,

        # Demographic
        old_age_dependency=old_age_dependency,
        age_pension_to_gdp=age_pension_to_gdp,
        health_spending_to_gdp=health_spending_to_gdp,
        ndis_to_gdp=ndis_to_gdp,

        # Composition
        fed_revenue_breakdown=fed_revenue_breakdown,
        fed_spending_breakdown=fed_spending_breakdown,
        state_revenue_breakdown=state_revenue_breakdown,
        state_spending_breakdown=state_spending_breakdown,
        consolidated_composition=consolidated_composition,
    )


# ── Scenario helper ───────────────────────────────────────────────────────────


def fiscal_projection_scenario(
    core_solution: Any,
    base_params: GovtParams = None,
    scenario_overrides: dict = None,
) -> tuple:
    """Run the fiscal model with parameter scenario overrides.

    Parameters
    ----------
    core_solution : KeenSolution or ExtendedKeenSolution
        The core model simulation.
    base_params : GovtParams, optional
        Baseline fiscal parameters.
    scenario_overrides : dict, optional
        Dict of parameter overrides (e.g. ``{'company_tax_rate': 0.30}``).

    Returns
    -------
    tuple of (GovtSolution, GovtParams)
        Scenario results and the (possibly modified) parameters used.
    """
    if base_params is None:
        base_params = GovtParams()
    if scenario_overrides is None:
        scenario_overrides = {}

    # Create modified parameters
    import copy
    scenario_params = copy.deepcopy(base_params)
    for key, val in scenario_overrides.items():
        if hasattr(scenario_params, key):
            setattr(scenario_params, key, val)

    return simulate_fiscal(core_solution, scenario_params), scenario_params


# ── Summary Statistics ────────────────────────────────────────────────────────


def fiscal_summary(gs: GovtSolution) -> dict:
    """Compute summary statistics for the fiscal simulation.

    Parameters
    ----------
    gs : GovtSolution
        Government model solution.

    Returns
    -------
    dict
        Summary statistics (initial, final, min, max for key metrics).
    """
    def _stats(arr):
        return {
            'initial': float(arr[0]),
            'final': float(arr[-1]),
            'min': float(np.min(arr)),
            'max': float(np.max(arr)),
            'mean': float(np.mean(arr)),
        }

    return {
        'federal_revenue': _stats(gs.fed_total_revenue),
        'federal_spending': _stats(gs.fed_total_spending),
        'federal_debt': _stats(gs.fed_debt),
        'state_revenue': _stats(gs.state_total_revenue),
        'state_spending': _stats(gs.state_total_spending),
        'state_debt': _stats(gs.state_debt),
        'consolidated_revenue': _stats(gs.total_revenue_pct),
        'consolidated_spending': _stats(gs.total_spending_pct),
        'consolidated_balance': _stats(gs.total_balance_pct),
        'consolidated_net_debt': _stats(gs.consolidated_net_debt_gdp),
        'fed_interest_revenue': _stats(gs.fed_interest_revenue_ratio),
        'total_interest_revenue': _stats(gs.total_interest_revenue_ratio),
        'tax_gap': _stats(gs.tax_gap),
        'fiscal_impulse': _stats(gs.fiscal_impulse),
        'age_pension': _stats(gs.age_pension_to_gdp),
        'health_spending': _stats(gs.health_spending_to_gdp),
        'ndis': _stats(gs.ndis_to_gdp),
    }
