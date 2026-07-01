"""
Natural Resources & Commodity Price Sub-Model
==============================================
Connects to the extended SFC Keen model to provide:
  • Resource depletion trajectories per commodity
  • Commodity price dynamics (world GDP, supply constraints, energy transition)
  • Fiscal linkages (royalties, mining company tax)
  • Mining investment (commodity-price-driven with 5-7yr lag)
  • Energy transition scenario (coal decline, critical minerals rise)

Key facts modelled:
  - Resources = ~60% of Australia's exports by value
  - Iron ore (~$150B/yr), Coal (~$100B/yr), LNG (~$70B/yr), Gold (~$30B/yr)
  - Resources = ~10% of GDP directly, ~25% including supply chain
  - Mining investment boom-bust cycle (recorded wave 2011-2014, currently subdued)
  - Resource rents generate ~$20B/yr in company tax and royalties
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional, Dict, Callable, Literal

# ── Known Australian Commodity Data (Geoscience Australia AIMR 2024 estimates) ──
# Units are chosen for reasonable numerical behaviour in the model.
# For detailed official data, see Geoscience Australia's AIMR and the OCE REQ.

COMMODITY_DATA = {
    "iron_ore": {
        "name": "Iron Ore",
        "unit": "Mt",
        "reserves": 50_000.0,            # Mt EDR (Geoscience Australia 2024)
        "production": 960.0,             # Mt/yr (2023-24)
        "price_aud": 150.0,              # AUD/t (~$100 USD/t × 1.5 AUD/USD)
        "export_share": 0.80,             # 80% of production exported (rest domestic)
        "revenue_to_M_AUD": 1.0,         # Mt × AUD/t = M AUD (million AUD) natively
        "price_income_elas": 1.2,        # Very sensitive to world GDP (China)
        "price_supply_elas": -0.3,
        "decay_rate": 0.0,               # No energy transition decay for iron ore
        "royalty_rate": 0.075,           # ~7.5% royalty (WA rate)
        "mine_tax_rate": 0.30,           # Corporate tax rate
        "capital_intensity": 0.25,
    },
    "coal": {
        "name": "Coal",
        "unit": "Mt",
        "reserves": 75_000.0,            # Mt EDR (black coal)
        "production": 460.0,             # Mt/yr (2022-23)
        "price_aud": 250.0,              # AUD/t (blend met+thermal, ~$160 USD/t)
        "export_share": 0.75,
        "revenue_to_M_AUD": 1.0,         # Mt × AUD/t = M AUD
        "price_income_elas": 0.8,
        "price_supply_elas": -0.4,
        "decay_rate": 0.025,             # ~2.5% p.a. decline from energy transition
        "royalty_rate": 0.08,            # ~8% (NSW/QLD rates)
        "mine_tax_rate": 0.30,
        "capital_intensity": 0.30,
    },
    "lng": {
        "name": "LNG",
        "unit": "Mt",
        "reserves": 3_200.0,             # Mt LNG equivalent (~85,000 PJ / 50 GJ/t)
        "production": 80.0,              # Mt/yr (2023-24)
        "price_aud": 750.0,              # AUD/t (~$500 USD/t, ~$12/mmbtu)
        "export_share": 0.90,
        "revenue_to_M_AUD": 1.0,         # Mt × AUD/t = M AUD
        "price_income_elas": 1.0,
        "price_supply_elas": -0.35,
        "decay_rate": 0.01,              # Mild decline as transition progresses
        "royalty_rate": 0.10,            # Petroleum resource rent tax (PRRT) + royalties
        "mine_tax_rate": 0.30,
        "capital_intensity": 0.40,
    },
    "gold": {
        "name": "Gold",
        "unit": "koz",
        "reserves": 385_800.0,           # koz EDR (12,000 t × 32.15 oz/t)
        "production": 10_606.0,          # koz/yr (330 t/yr × 32.15 oz/t)
        "price_aud": 3_500.0,            # AUD/oz (~$2,300 USD/oz × 1.5 AUD/USD)
        "export_share": 0.95,
        "revenue_to_M_AUD": 0.001,       # koz × AUD/oz = kAUD → ÷1000 → M AUD
        "price_income_elas": 0.3,        # Gold is a safe haven, less GDP-sensitive
        "price_supply_elas": -0.1,
        "decay_rate": 0.0,
        "royalty_rate": 0.025,           # ~2.5% (varies by state)
        "mine_tax_rate": 0.30,
        "capital_intensity": 0.15,
    },
    "lithium": {
        "name": "Lithium",
        "unit": "kt LCE",
        "reserves": 62_000.0,            # kt LCE (6.2 Mt LCE)
        "production": 420.0,             # kt LCE/yr (2023-24)
        "price_aud": 25_000.0,           # AUD/t LCE (~$16,000 USD/t)
        "export_share": 0.95,
        "revenue_to_M_AUD": 0.001,       # kt × AUD/t = kAUD → ÷1000 → M AUD
        "price_income_elas": 1.5,        # Very sensitive to EV adoption
        "price_supply_elas": -0.15,
        "decay_rate": -0.04,             # NEGATIVE = 4% annual growth (BAU)
        "royalty_rate": 0.05,            # WA royalty rate for lithium
        "mine_tax_rate": 0.30,
        "capital_intensity": 0.20,
    },
}


# ── Scenario Definitions ─────────────────────────────────────────────────────

@dataclass
class ScenarioConfig:
    """Configuration for a resource scenario — modifies commodity parameters."""
    name: str
    label: str
    description: str
    multiplier_decay_rate: Dict[str, float]  # commodity -> multiplier on base decay
    multiplier_world_gdp: float              # multiplier on world GDP growth
    multiplier_supply_elasticity: float      # multiplier on supply constraints
    discovery_rate: Dict[str, float]         # commodity -> annual reserve addition rate
    investment_sensitivity: float            # how much investment responds to prices


# Pre-built scenarios
BUSINESS_AS_USUAL = ScenarioConfig(
    name="bau",
    label="Business As Usual",
    description="Current trends continue — moderate energy transition, stable discoveries, normal investment cycles.",
    multiplier_decay_rate={"iron_ore": 0.0, "coal": 1.0, "lng": 1.0, "gold": 0.0, "lithium": 1.0},
    multiplier_world_gdp=1.0,
    multiplier_supply_elasticity=1.0,
    discovery_rate={"iron_ore": 0.01, "coal": 0.005, "lng": 0.01, "gold": 0.005, "lithium": 0.02},
    investment_sensitivity=1.0,
)

ACCELERATED_DEPLETION = ScenarioConfig(
    name="accelerated",
    label="Accelerated Depletion",
    description="Higher production rates from strong demand, fewer new discoveries. Faster reserve drawdown.",
    multiplier_decay_rate={"iron_ore": 0.0, "coal": 1.5, "lng": 1.5, "gold": 0.0, "lithium": 2.0},
    multiplier_world_gdp=1.3,
    multiplier_supply_elasticity=0.7,
    discovery_rate={"iron_ore": 0.005, "coal": 0.003, "lng": 0.005, "gold": 0.003, "lithium": 0.01},
    investment_sensitivity=1.3,
)

NEW_DISCOVERIES = ScenarioConfig(
    name="discoveries",
    label="New Discoveries",
    description="Major new resource discoveries expand reserves. Production can be sustained or increased.",
    multiplier_decay_rate={"iron_ore": 0.0, "coal": 0.8, "lng": 0.8, "gold": 0.0, "lithium": 1.2},
    multiplier_world_gdp=1.0,
    multiplier_supply_elasticity=1.2,
    discovery_rate={"iron_ore": 0.03, "coal": 0.015, "lng": 0.03, "gold": 0.015, "lithium": 0.05},
    investment_sensitivity=1.0,
)

ENERGY_TRANSITION = ScenarioConfig(
    name="net_zero",
    label="Energy Transition (Net Zero 2050)",
    description="Aggressive decarbonisation: coal phases down fast, LNG plateaus then declines, lithium/critical minerals boom.",
    multiplier_decay_rate={"iron_ore": 0.0, "coal": 4.0, "lng": 2.0, "gold": 0.0, "lithium": 3.0},
    multiplier_world_gdp=0.9,
    multiplier_supply_elasticity=0.8,
    discovery_rate={"iron_ore": 0.01, "coal": 0.0, "lng": 0.005, "gold": 0.005, "lithium": 0.06},
    investment_sensitivity=1.2,
)

SCENARIOS = {
    s.name: s for s in [BUSINESS_AS_USUAL, ACCELERATED_DEPLETION, NEW_DISCOVERIES, ENERGY_TRANSITION]
}


# ── Parameter Dataclass ──────────────────────────────────────────────────────

@dataclass
class ResourceParams:
    """Parameters for the natural resources sub-model.

    Connects to the SFC model via:
      • export_share → enhanced with commodity-specific volumes and prices
      • mining_investment → component of business investment (κ)
      • royalty_revenue → state government revenue stream
      • mining_tax → federal company tax revenue
    """
    # World economy
    world_gdp_trend: float = 0.030           # Trend world GDP growth rate (3% p.a.)
    china_gdp_weight: float = 0.35            # China's share of global commodity demand
    china_gdp_trend: float = 0.045            # China GDP trend (4.5%, slowing from 6%+)

    # Energy transition
    net_zero_year: float = 2050.0             # Net zero target year
    transition_start_year: float = 2025.0     # Year significant transition policy begins
    ev_adoption_speed: float = 0.12           # Annual increase in EV share of new sales
    carbon_price_trend: float = 0.05          # Annual carbon price increase (USD/t CO2)

    # Mining investment (Goodwin cycle in mining)
    # Investment follows commodity prices with a lag (feasibility → construction → production)
    mining_investment_lag: float = 6.0        # Years from investment decision to production
    mining_investment_slope: float = 0.4      # Sensitivity of investment share to commodity price index
    mining_investment_base: float = 0.03      # Base mining investment as share of GDP (3%)
    mining_investment_mean_revert: float = 0.2  # Speed of mean reversion

    # Fiscal
    royalty_rate_base: float = 0.075          # Average state royalty rate (7.5%)
    prrt_rate: float = 0.40                   # Petroleum Resource Rent Tax rate (40%)
    mining_corp_tax_rate: float = 0.30        # Company tax rate for mining sector

    # Initial conditions at t=0 (relative to model start year ~2025)
    initial_world_gdp_gap: float = 0.0        # Output gap in world economy (0 = at trend)
    initial_mining_investment: float = 0.03   # Mining investment as share of GDP

    # Scenario selection
    scenario_name: str = "bau"

    @property
    def scenario(self) -> ScenarioConfig:
        return SCENARIOS.get(self.scenario_name, BUSINESS_AS_USUAL)


# ── Commodity State Container ────────────────────────────────────────────────

@dataclass
class CommodityTrajectory:
    """Per-commodity state through time."""
    commodity_key: str
    name: str
    unit: str
    t: np.ndarray                        # Time array (years)

    # Physical
    reserves: np.ndarray                 # Remaining reserves (units: Mt, t, kt LCE)
    production: np.ndarray               # Annual production (units/year)
    depletion_rate: np.ndarray           # Production / reserves (fraction per year)
    remaining_years: np.ndarray          # Reserves / production at current rate

    # Economic
    price_aud: np.ndarray               # Price in AUD per unit
    export_volume: np.ndarray            # Exported quantity (units/year)
    export_revenue: np.ndarray           # Export revenue (AUD, annual)
    royalty_revenue: np.ndarray          # State royalty revenue (AUD)
    mining_tax: np.ndarray               # Company tax from this commodity (AUD)

    # Investment
    investment: np.ndarray               # Mining investment for this commodity (AUD)
    capacity_utilisation: np.ndarray    # Capacity utilisation rate

    def to_dict(self) -> dict:
        return {
            "commodity": self.commodity_key,
            "name": self.name,
            "unit": self.unit,
            "reserves": self.reserves,
            "production": self.production,
            "depletion_rate": self.depletion_rate,
            "remaining_years": self.remaining_years,
            "price_aud": self.price_aud,
            "export_volume": self.export_volume,
            "export_revenue": self.export_revenue,
            "royalty_revenue": self.royalty_revenue,
            "mining_tax": self.mining_tax,
            "investment": self.investment,
            "capacity_utilisation": self.capacity_utilisation,
        }


# ── Resource Solution Container ──────────────────────────────────────────────

@dataclass
class ResourceSolution:
    """Container for complete resource sub-model output."""
    t: np.ndarray
    params: ResourceParams
    commodities: Dict[str, CommodityTrajectory]
    success: bool
    message: str = ""

    def __getitem__(self, key: str):
        return getattr(self, key)

    # Aggregates
    @property
    def total_export_revenue(self) -> np.ndarray:
        """Total resource export revenue (AUD)."""
        if not self.commodities:
            return np.zeros_like(self.t)
        total = np.zeros_like(self.t)
        for c in self.commodities.values():
            total += c.export_revenue
        return total

    @property
    def total_royalty_revenue(self) -> np.ndarray:
        """Total state royalty revenue (AUD)."""
        if not self.commodities:
            return np.zeros_like(self.t)
        total = np.zeros_like(self.t)
        for c in self.commodities.values():
            total += c.royalty_revenue
        return total

    @property
    def total_mining_tax(self) -> np.ndarray:
        """Total federal mining company tax (AUD)."""
        if not self.commodities:
            return np.zeros_like(self.t)
        total = np.zeros_like(self.t)
        for c in self.commodities.values():
            total += c.mining_tax
        return total

    @property
    def total_mining_investment(self) -> np.ndarray:
        """Total mining investment (AUD)."""
        if not self.commodities:
            return np.zeros_like(self.t)
        total = np.zeros_like(self.t)
        for c in self.commodities.values():
            total += c.investment
        return total

    @property
    def total_export_volume_index(self) -> np.ndarray:
        """Volume index of total resource exports (base=1.0 at t=0)."""
        if not self.commodities:
            return np.ones_like(self.t)
        # Weight by initial revenue share
        initial_revenue = {}
        total_initial = 0.0
        for key, c in self.commodities.items():
            rev = c.export_revenue[0] if len(c.export_revenue) > 0 else 1.0
            initial_revenue[key] = rev
            total_initial += rev
        if total_initial == 0:
            return np.ones_like(self.t)
        index = np.zeros_like(self.t)
        for key, c in self.commodities.items():
            weight = initial_revenue[key] / total_initial
            index += weight * c.export_volume / max(c.export_volume[0], 1e-6)
        return index

    @property
    def terms_of_trade_index(self) -> np.ndarray:
        """Terms of trade index for resources (base=1.0 at t=0).

        = weighted average of commodity prices, weighted by initial export revenue.
        """
        if not self.commodities:
            return np.ones_like(self.t)
        initial_revenue = {}
        total_initial = 0.0
        for key, c in self.commodities.items():
            rev = c.export_revenue[0] if len(c.export_revenue) > 0 else 1.0
            initial_revenue[key] = rev
            total_initial += rev
        if total_initial == 0:
            return np.ones_like(self.t)
        index = np.zeros_like(self.t)
        for key, c in self.commodities.items():
            weight = initial_revenue[key] / total_initial
            index += weight * c.price_aud / max(c.price_aud[0], 1e-6)
        return index

    @property
    def resource_gdp_share(self) -> np.ndarray:
        """Resource sector value-added as share of GDP.

        Export revenue scaled by value-added ratio (~0.6 for mining)
        divided by Australian GDP (~2,500,000 M AUD).
        """
        gdp_maud = 2_500_000  # Australia's GDP in millions of AUD (~$2.5T)
        value_added_ratio = 0.6  # Export revenue → value-added conversion
        return np.clip(self.total_export_revenue * value_added_ratio / gdp_maud, 0.05, 0.20)

    def scenario_summary(self) -> dict:
        """Return a summary of key aggregates at the final time step."""
        idx = -1
        return {
            "scenario": self.params.scenario_name,
            "total_export_revenue_AUD_B": self.total_export_revenue[idx] / 1_000,
            "total_royalty_revenue_AUD_B": self.total_royalty_revenue[idx] / 1_000,
            "total_mining_tax_AUD_B": self.total_mining_tax[idx] / 1_000,
            "total_mining_investment_AUD_B": self.total_mining_investment[idx] / 1_000,
            "terms_of_trade_index": self.terms_of_trade_index[idx],
            "export_volume_index": self.total_export_volume_index[idx],
            "resource_gdp_share": self.resource_gdp_share[idx],
        }


# ── Transfer Function: Resource → SFC Model ─────────────────────────────────

def compute_resource_export_share(resource_solution: ResourceSolution) -> np.ndarray:
    """Compute enhanced export share of GDP from resource model.

    Returns export_share as fraction of GDP, to be used in the SFC model's
    export sector instead of the simple trend-based estimate.
    """
    # Total resource export revenue (AUD) / estimated GDP
    # Australia GDP ~2.5T AUD → 2,500,000 M AUD for normalisation
    gdp_estimate = 2_500_000  # AUD in millions (~$2.5T)
    export_gdp_share = resource_solution.total_export_revenue / gdp_estimate
    return np.clip(export_gdp_share, 0.08, 0.40)


def compute_mining_investment_share(resource_solution: ResourceSolution) -> np.ndarray:
    """Mining investment as share of GDP for the SFC model.

    This feeds into κ (investment share) in the Keen model.
    """
    gdp_estimate = 2_500_000  # AUD in millions
    mining_inv_share = resource_solution.total_mining_investment / gdp_estimate
    return np.clip(mining_inv_share, 0.005, 0.08)


# ── Core Simulation ──────────────────────────────────────────────────────────

def simulate_resources(
    core_solution,
    params: Optional[ResourceParams] = None,
) -> ResourceSolution:
    """Run the natural resources sub-model alongside the core Keen/SFC solution.

    Parameters
    ----------
    core_solution : object
        Should have attributes:
        - t : np.ndarray (time array in years)
        - lam : np.ndarray (employment rate)
        - growth_rate : np.ndarray (GDP growth rate)
        - params : object (with .alpha, .beta etc.)
        Can be a KeenSolution or ExtendedKeenSolution.
    params : ResourceParams, optional
        Resource sub-model parameters. Uses defaults if None.

    Returns
    -------
    ResourceSolution
        Container with all commodity trajectories and aggregates.
    """
    if params is None:
        params = ResourceParams()

    # Time array from core solution
    t = core_solution.t
    n = len(t)
    dt = np.diff(t, prepend=t[0])  # Time step sizes

    # GDP growth rate from core solution
    if hasattr(core_solution, 'growth_rate'):
        gdp_growth = core_solution.growth_rate
    else:
        # Approximate from GDP-like growth
        gdp_growth = np.full_like(t, params.world_gdp_trend)

    # Employment rate (used for demand-side effects)
    if hasattr(core_solution, 'lam'):
        employment = core_solution.lam
    else:
        employment = np.full_like(t, 0.94)

    # Smooth GDP growth for commodity demand calculation (avoid short-term noise)
    gdp_smooth = _smooth_series(gdp_growth, window=20)

    # Scenario
    scenario = params.scenario

    # Build commodity trajectories
    commodities = {}

    # Initial mining investment state
    mining_investment_state = params.initial_mining_investment

    # Commodity price index (for mining investment)
    # Use a single index based on aggregated price movements
    commodity_price_index = np.ones(n)

    # ── World GDP index ──────────────────────────────────────────────────
    # World GDP relative to initial (index = 1 at t=0)
    world_gdp_trend = params.world_gdp_trend * scenario.multiplier_world_gdp
    world_gdp_index = np.exp(np.cumsum(world_gdp_trend * dt))

    # China-specific GDP index (for iron ore demand)
    china_gdp_index = np.exp(np.cumsum(params.china_gdp_trend * dt))

    # Energy transition pressure (0 = none, 1 = full net zero)
    transition_pressure = np.clip(
        (t - (params.transition_start_year - 2025)) / (params.net_zero_year - params.transition_start_year),
        0, 1.2
    )

    # ── Per-commodity simulation ─────────────────────────────────────────
    for key, cdata in COMMODITY_DATA.items():
        name = cdata["name"]
        unit = cdata["unit"]

        # Allocate arrays
        reserves = np.full(n, cdata["reserves"])
        production = np.full(n, cdata["production"])
        price_aud = np.full(n, cdata["price_aud"])
        export_revenue = np.zeros(n)
        royalty_revenue = np.zeros(n)
        mining_tax_out = np.zeros(n)
        investment = np.zeros(n)
        capacity_utilisation = np.ones(n)
        export_volume = np.zeros(n)

        # Initial revenue scale factor
        rev_scale = cdata["revenue_to_M_AUD"]

        # Set t=0 initial values
        export_volume[0] = cdata["production"] * cdata["export_share"]
        export_revenue[0] = export_volume[0] * cdata["price_aud"] * rev_scale
        # Royalties at t=0
        if key == "lng":
            prrt0 = max(0, export_revenue[0] * 0.1)
            royalty_revenue[0] = export_revenue[0] * 0.025 + prrt0
        else:
            royalty_revenue[0] = export_revenue[0] * cdata["royalty_rate"]
        # Mining tax at t=0
        profit0 = export_revenue[0] * 0.40
        taxable0 = max(0, profit0 - royalty_revenue[0])
        mining_tax_out[0] = taxable0 * cdata["mine_tax_rate"]
        # Mining investment at t=0
        investment[0] = params.mining_investment_base * cdata["capital_intensity"] * 2_500_000

        # Apply scenario modifiers
        decay_mod = scenario.multiplier_decay_rate.get(key, 1.0)
        discovery_mod = scenario.discovery_rate.get(key, 0.0)
        supply_elas_mod = scenario.multiplier_supply_elasticity
        invest_sensitivity = scenario.investment_sensitivity

        # Energy transition decay factor (positive = declining, negative = growing)
        base_decay = cdata["decay_rate"]
        effective_decay = base_decay * decay_mod

        # Initial revenue weight for price index (in M AUD)
        rev_scale = cdata["revenue_to_M_AUD"]
        initial_rev = cdata["production"] * cdata["price_aud"] * cdata["export_share"] * rev_scale

        # Commodity-specific price index tracking
        price_index_base = 1.0

        for i in range(1, n):
            dt_i = dt[i]

            # ── Demand-side: World GDP effect on price ─────────────────────
            # Commodity demand grows/declines with world GDP
            income_elas = cdata["price_income_elas"]
            # World GDP growth affects demand pressure
            demand_pressure = world_gdp_index[i] ** income_elas - 1.0

            # ── Supply-side: Reserves depletion effect on price ────────────
            # As reserves deplete, extraction costs rise → price increase
            depletion_ratio = max(reserves[i - 1] / max(cdata["reserves"], 1), 0.1)
            supply_scarcity = (1.0 - depletion_ratio) * 0.5 * supply_elas_mod

            # ── Energy transition price effect ─────────────────────────────
            # Coal/LNG: transition reduces demand → lower price
            # Lithium: transition increases demand → higher price
            # effective_decay > 0 → decline, so trans_effect < 0 → price down
            # effective_decay < 0 → growth, so trans_effect > 0 → price up
            trans_effect = effective_decay * transition_pressure[i] * (-1.0) * 0.3

            # ── Production adjustment ─────────────────────────────────────
            # Production expands/contracts with price signals
            # If price > initial price, expand; if lower, contract
            price_ratio = price_aud[i - 1] / cdata["price_aud"]
            # effective_decay: positive = decline (coal), negative = growth (lithium)
            # So we SUBTRACT effective_decay from the price-led growth rate
            production_growth_rate = 0.05 * (price_ratio - 1.0) - effective_decay
            # Capacity utilisation constraints
            if capacity_utilisation[i - 1] > 0.95 and production_growth_rate > 0:
                production_growth_rate *= 0.5  # Capacity constrained
            production[i] = production[i - 1] * (1.0 + production_growth_rate * dt_i)
            production[i] = max(production[i], cdata["production"] * 0.1)  # Floor at 10% of initial

            # ── Reserve depletion ──────────────────────────────────────────
            # New discoveries add to reserves (as fraction of current reserves)
            discovery = discovery_mod * reserves[i - 1] * dt_i
            # Production depletes reserves
            reserves[i] = reserves[i - 1] + discovery - production[i] * dt_i
            # Soft floor: reserves cannot go below 1% of initial
            min_reserves = cdata["reserves"] * 0.01
            if reserves[i] < min_reserves:
                # Production constrained by remaining reserves
                production[i] = max(production[i], 0)
                production[i] = min(production[i], reserves[i - 1] / max(dt_i, 0.01))
                reserves[i] = max(reserves[i - 1] - production[i] * dt_i, 0)

            # ── Capacity utilisation ───────────────────────────────────────
            # Tracks how hard existing mines are running
            target_util = 0.5 + 0.5 * min(price_ratio, 2.0)
            capacity_utilisation[i] = capacity_utilisation[i - 1] + 0.2 * (target_util - capacity_utilisation[i - 1]) * dt_i
            capacity_utilisation[i] = np.clip(capacity_utilisation[i], 0.2, 1.0)

            # ── Price dynamics ────────────────────────────────────────────
            # Price change = f(demand pressure, supply scarcity, transition, random)
            # Using a partial adjustment model
            price_target = cdata["price_aud"] * (
                1.0
                + demand_pressure * 0.3 * dt_i
                + supply_scarcity * dt_i
                + trans_effect * dt_i
            )
            # Add a small mean-reversion component
            price_aud[i] = price_aud[i - 1] + 0.3 * (price_target - price_aud[i - 1]) * dt_i
            # Floor to prevent negative prices
            price_aud[i] = max(price_aud[i], cdata["price_aud"] * 0.1)

            # ── Export volume ─────────────────────────────────────────────
            export_share_ratio = cdata["export_share"]
            export_volume[i] = production[i] * export_share_ratio

            # ── Export revenue (in M AUD) ─────────────────────────────────
            export_revenue[i] = export_volume[i] * price_aud[i] * rev_scale

            # ── Royalty revenue (state government, in M AUD) ─────────────
            royalty_rate = cdata["royalty_rate"]
            # Adjust royalty for LNG (PRRT applies above a threshold)
            if key == "lng":
                # PRRT: 40% of profits above threshold, simplified as rate on revenue
                prrt_component = max(0, export_revenue[i] * 0.1)  # Simplified PRRT
                royalty_revenue[i] = export_revenue[i] * 0.025 + prrt_component  # Base royalty + PRRT
            else:
                royalty_revenue[i] = export_revenue[i] * royalty_rate

            # ── Mining company tax (federal) ─────────────────────────────
            # Approximate: mine profit margin ~40% of revenue
            profit_margin = 0.40  # Industry average operating margin
            operating_profit = export_revenue[i] * profit_margin
            # Deduct royalties
            taxable_profit = max(0, operating_profit - royalty_revenue[i])
            mining_tax_out[i] = taxable_profit * cdata["mine_tax_rate"]

            # ── Mining investment (Goodwin cycle with lag) ───────────────
            # Investment responds to price with a lag
            capital_intensity = cdata["capital_intensity"]
            # Price momentum: trailing average price relative to initial
            lookback = max(1, int(params.mining_investment_lag / np.mean(dt)))
            if i >= lookback:
                trailing_price = np.mean(price_aud[i - lookback:i])
            else:
                trailing_price = price_aud[i]
            price_momentum = max(0, (trailing_price / cdata["price_aud"]) - 0.8)
            # Investment = base + sensitivity * price_momentum * capital_intensity
            base_inv = params.mining_investment_base * (
                production[i] / cdata["production"]
            ) * capital_intensity * 2_500_000  # Scale to AUD
            cyclic_inv = (
                params.mining_investment_slope
                * invest_sensitivity
                * price_momentum
                * capital_intensity
                * export_revenue[i]
            )
            investment[i] = max(0, base_inv + cyclic_inv)

        # ── Derived quantities ────────────────────────────────────────────
        depletion_rate = np.zeros(n)
        remaining_years = np.full(n, 999.0)
        for i in range(n):
            if reserves[i] > 0 and production[i] > 0:
                depletion_rate[i] = production[i] / reserves[i]
                remaining_years[i] = reserves[i] / production[i]
            else:
                depletion_rate[i] = 0.0
                remaining_years[i] = 999.0

        commodities[key] = CommodityTrajectory(
            commodity_key=key,
            name=name,
            unit=unit,
            t=t.copy(),
            reserves=reserves,
            production=production,
            depletion_rate=depletion_rate,
            remaining_years=remaining_years,
            price_aud=price_aud,
            export_volume=export_volume,
            export_revenue=export_revenue,
            royalty_revenue=royalty_revenue,
            mining_tax=mining_tax_out,
            investment=investment,
            capacity_utilisation=capacity_utilisation,
        )

        # Update aggregate price index
        weight = initial_rev / max(
            sum(COMMODITY_DATA[k]["production"] * COMMODITY_DATA[k]["price_aud"]
                * COMMODITY_DATA[k]["export_share"]
                * COMMODITY_DATA[k]["revenue_to_M_AUD"] for k in COMMODITY_DATA), 1
        )
        commodity_price_index += weight * (price_aud / cdata["price_aud"] - 1.0)

    return ResourceSolution(
        t=t.copy(),
        params=params,
        commodities=commodities,
        success=True,
        message="Resource sub-model completed successfully.",
    )


# ── Scenario Comparison Runner ────────────────────────────────────────────────

def run_scenario_comparison(
    core_solution,
    scenarios: list[str] = None,
) -> Dict[str, ResourceSolution]:
    """Run the resource model under multiple scenarios and return all results.

    Parameters
    ----------
    core_solution : object
        Core Keen/SFC model solution (same for all scenarios).
    scenarios : list of str, optional
        Scenario names to run. Default: all four scenarios.

    Returns
    -------
    dict[str, ResourceSolution]
        Mapping from scenario name to ResourceSolution.
    """
    if scenarios is None:
        scenarios = list(SCENARIOS.keys())

    results = {}
    for s_name in scenarios:
        if s_name not in SCENARIOS:
            continue
        params = ResourceParams(scenario_name=s_name)
        results[s_name] = simulate_resources(core_solution, params)

    return results


# ── Helper Functions ─────────────────────────────────────────────────────────

def _smooth_series(x: np.ndarray, window: int = 10) -> np.ndarray:
    """Simple moving average smoother."""
    if len(x) <= window:
        return x
    cumsum = np.cumsum(np.insert(x, 0, 0))
    smoothed = (cumsum[window:] - cumsum[:-window]) / window
    # Pad edges
    pad = (len(x) - len(smoothed)) // 2
    smoothed = np.concatenate([
        np.full(pad, smoothed[0]),
        smoothed,
        np.full(len(x) - pad - len(smoothed), smoothed[-1]),
    ])
    return smoothed[:len(x)]


def get_commodity_summary(resource_solution: ResourceSolution, time_idx: int = -1) -> "pd.DataFrame":
    """Return a pandas DataFrame summarising all commodities at a point in time."""
    rows = []
    for key, c in resource_solution.commodities.items():
        idx = time_idx if time_idx >= 0 else len(c.t) + time_idx
        idx = min(max(idx, 0), len(c.t) - 1)
        rows.append({
            "Commodity": c.name,
            "Unit": c.unit,
            "Reserves": f"{c.reserves[idx]:.0f}",
            "Production": f"{c.production[idx]:.1f}",
            "Price (AUD/unit)": f"${c.price_aud[idx]:.0f}",
            "Export Revenue (AUD B)": f"{c.export_revenue[idx] / 1_000:.1f}",
            "Remaining Years": f"{c.remaining_years[idx]:.0f}",
            "Royalties (AUD B)": f"{c.royalty_revenue[idx] / 1_000:.2f}",
            "Mining Tax (AUD B)": f"{c.mining_tax[idx] / 1_000:.2f}",
            "Cap. Utilisation": f"{c.capacity_utilisation[idx]:.0%}",
        })
    return pd.DataFrame(rows)


# Export the key scenario configs for easy import
__all__ = [
    "ResourceParams", "ResourceSolution", "CommodityTrajectory",
    "COMMODITY_DATA", "SCENARIOS",
    "BUSINESS_AS_USUAL", "ACCELERATED_DEPLETION", "NEW_DISCOVERIES", "ENERGY_TRANSITION",
    "simulate_resources", "run_scenario_comparison",
    "compute_resource_export_share", "compute_mining_investment_share",
    "get_commodity_summary",
]
