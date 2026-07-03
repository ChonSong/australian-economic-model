"""
Tests for the core Keen model (keen_model.py).

Covers:
  - Parameter defaults and equilibrium relationships
  - ODE solver convergence and stability
  - Edge cases (near-zero values, extreme parameters)
  - Housing extension model
"""

import numpy as np
import pytest
from aus_econ_model.models.keen_model import (
    KeenParams,
    KeenSolution,
    simulate_keen,
    simulate_keen_housing,
    KeenHousingParams,
    equilibrium_profit,
    keen_ode,
)


class TestEquilibriumProfit:
    """Test the equilibrium_profit helper function."""

    def test_basic_calculation(self):
        """Equilibrium profit = nu * (alpha + beta + delta)."""
        result = equilibrium_profit(nu=5.0, alpha=0.02, beta=0.02, delta=0.02)
        expected = 5.0 * (0.02 + 0.02 + 0.02)  # 0.30
        assert abs(result - expected) < 1e-10

    def test_default_params(self):
        """Default KeenParams should match equilibrium_profit."""
        p = KeenParams()
        expected = p.nu * (p.alpha + p.beta + p.delta)
        assert (
            abs(equilibrium_profit(p.nu, p.alpha, p.beta, p.delta) - expected) < 1e-10
        )
        assert abs(p.pi_star - expected) < 1e-10


class TestKeenParams:
    """Test KeenParams dataclass."""

    def test_default_values_in_range(self):
        """Default parameter values should be internally consistent."""
        p = KeenParams()
        assert 0 < p.pi_star < 1  # Profit share must be between 0 and 1
        assert p.phi_min < p.phi_max  # Wage curve bounds
        assert 0 < p.kappa1 < 1  # Investment slope in stable range
        assert p.omega0 + p.r * p.d0 < 1  # Initial wage + debt service < GDP

    def test_investment_function(self):
        """Investment equals pi* when profit equals pi*."""
        p = KeenParams()
        pi_star = p.pi_star
        assert abs(p.investment(np.array([pi_star]))[0] - pi_star) < 1e-10

    def test_investment_linear(self):
        """Investment should be linear in profit."""
        p = KeenParams()
        pi_vals = np.array([0.1, 0.2, 0.3, 0.4])
        inv_vals = p.investment(pi_vals)
        # Slope should be kappa1
        slopes = np.diff(inv_vals) / np.diff(pi_vals)
        assert all(abs(s - p.kappa1) < 1e-10 for s in slopes)

    def test_phillips_curve_bounds(self):
        """Phillips curve should be bounded by phi_min and phi_max."""
        p = KeenParams()
        # At λ=0, wage should be at floor
        assert abs(p.phillips(np.array([0.0]))[0] - p.phi_min) < 1e-10
        # At λ=1, wage should be near ceiling (but not exceed phi_max)
        assert p.phillips(np.array([1.0]))[0] <= p.phi_max
        # Monotonic
        lam_vals = np.linspace(0, 1, 20)
        phi_vals = p.phillips(lam_vals)
        assert all(np.diff(phi_vals) >= -1e-10)  # Non-decreasing


class TestSimulateKeen:
    """Test the main simulation function."""

    def test_default_simulation_succeeds(self):
        """Simulation with default parameters should converge."""
        sol = simulate_keen(t_max=100, t_steps=2000)
        assert sol.success is True
        assert sol.message == "" or "successful" in sol.message.lower()

    def test_default_simulation_bounds(self):
        """State variables should stay within expected ranges."""
        sol = simulate_keen(t_max=100, t_steps=2000)
        # Wage share: should stay between 0 and 1
        assert np.all(sol.omega >= 0)
        assert np.all(sol.omega <= 1)
        # Employment rate: between 0 and 1
        assert np.all(sol.lam >= 0)
        assert np.all(sol.lam <= 1)
        # Debt ratio: non-negative
        assert np.all(sol.d >= 0)

    def test_initial_conditions_match(self):
        """Simulation should start from specified initial conditions."""
        p = KeenParams()
        sol = simulate_keen(p, t_max=50, t_steps=100)
        assert abs(sol.omega[0] - p.omega0) < 1e-6
        assert abs(sol.lam[0] - p.lambda0) < 1e-6
        assert abs(sol.d[0] - p.d0) < 1e-6

    def test_custom_initial_conditions(self):
        """Custom initial conditions should be respected."""
        sol = simulate_keen(omega0=0.5, lambda0=0.90, d0=1.5, t_max=50)
        assert abs(sol.omega[0] - 0.5) < 1e-6
        assert abs(sol.lam[0] - 0.90) < 1e-6
        assert abs(sol.d[0] - 1.5) < 1e-6

    def test_longer_simulation_stable(self):
        """Model should remain stable over longer time horizons."""
        sol = simulate_keen(t_max=200, t_steps=4000)
        assert sol.success is True
        # Final values should be finite
        assert np.isfinite(sol.omega[-1])
        assert np.isfinite(sol.lam[-1])
        assert np.isfinite(sol.d[-1])

    def test_short_simulation(self):
        """Very short simulation should still produce valid output."""
        sol = simulate_keen(t_max=1, t_steps=10)
        assert sol.success is True
        assert len(sol.t) == 10

    def test_computed_properties(self):
        """KeenSolution computed properties should be consistent."""
        sol = simulate_keen(t_max=100)
        # Profit share = 1 - omega - r*d
        expected_pi = 1.0 - sol.omega - sol.params.r * sol.d
        assert np.allclose(sol.profit_share, expected_pi)
        # Debt service = r*d
        assert np.allclose(sol.debt_service_ratio, sol.params.r * sol.d)

    def test_investment_share_consistency(self):
        """Investment share should equal params.investment(profit_share)."""
        sol = simulate_keen(t_max=100)
        expected = sol.params.investment(sol.profit_share)
        assert np.allclose(sol.investment_share, expected)

    def test_real_wage_from_phillips(self):
        """Real wage should come from Phillips curve applied to employment."""
        sol = simulate_keen(t_max=100)
        expected = sol.params.phillips(sol.lam)
        assert np.allclose(sol.real_wage, expected)


class TestKeenODEEdgeCases:
    """Test edge cases in the ODE function."""

    def test_keen_ode_zero_state(self):
        """ODE should handle near-zero state variables without crashing."""
        p = KeenParams()
        # Test with a near-zero state
        result = keen_ode(0.0, np.array([0.001, 0.001, 0.0]), p)
        assert len(result) == 3
        assert all(np.isfinite(r) for r in result)

    def test_keen_ode_high_debt(self):
        """ODE should handle high debt values without crashing."""
        p = KeenParams()
        result = keen_ode(50.0, np.array([0.5, 0.94, 5.0]), p)
        assert len(result) == 3
        assert all(np.isfinite(r) for r in result)

    def test_simulation_with_high_debt_initial(self):
        """Simulation starting from high debt should not crash."""
        sol = simulate_keen(d0=3.0, t_max=50)
        assert sol.success is True
        assert np.all(np.isfinite(sol.d))

    def test_simulation_with_low_wage_initial(self):
        """Simulation starting from very low wage share should not crash."""
        sol = simulate_keen(omega0=0.1, t_max=50)
        assert sol.success is True
        assert np.all(np.isfinite(sol.omega))

    def test_ode_scipy_integration_consistency(self):
        """ODE integration should be deterministic (same seed, same result)."""
        sol1 = simulate_keen(t_max=100, t_steps=2000)
        sol2 = simulate_keen(t_max=100, t_steps=2000)
        assert np.allclose(sol1.omega, sol2.omega)
        assert np.allclose(sol1.lam, sol2.lam)
        assert np.allclose(sol1.d, sol2.d)


class TestKeenSolutionContainer:
    """Test the KeenSolution dataclass."""

    def test_empty_solution_handling(self):
        """KeenSolution should handle edge cases gracefully."""
        p = KeenParams()
        sol = KeenSolution(
            t=np.array([0.0, 1.0]),
            omega=np.array([0.6, 0.6]),
            lam=np.array([0.94, 0.94]),
            d=np.array([0.8, 0.8]),
            params=p,
            success=False,
            message="Test failure",
        )
        assert sol.success is False
        assert "Test failure" in sol.message
        # Computed properties should still work
        assert len(sol.profit_share) == 2
        assert np.isfinite(sol.profit_share[0])

    def test_solution_without_params_fallback(self):
        """KeenSolution should require params to compute properties."""
        p = KeenParams()
        sol = KeenSolution(
            t=np.array([0.0]),
            omega=np.array([0.6]),
            lam=np.array([0.94]),
            d=np.array([0.8]),
            params=p,
            success=True,
        )
        # Properties that rely on params should work
        assert sol.params is not None


class TestKeenHousing:
    """Test the housing extension of the Keen model."""

    def test_housing_simulation_runs(self):
        """Extended model with housing should run without error."""
        core, housing = simulate_keen_housing(t_max=50, t_steps=200)
        assert core.success is True
        assert len(housing) > 0
        assert "price_to_income" in housing

    def test_housing_price_positive(self):
        """Housing prices should remain positive."""
        core, housing = simulate_keen_housing(t_max=50)
        assert np.all(housing["price_to_income"] > 0)

    def test_housing_stock_positive(self):
        """Housing stock should remain positive."""
        core, housing = simulate_keen_housing(t_max=50)
        assert np.all(housing["housing_stock"] > 0)

    def test_housing_affordability_defined(self):
        """Affordability index should be present and finite."""
        core, housing = simulate_keen_housing(t_max=50)
        assert "affordability" in housing
        assert np.all(np.isfinite(housing["affordability"]))

    def test_housing_price_growth_finite(self):
        """Price growth should be finite."""
        core, housing = simulate_keen_housing(t_max=50)
        assert np.all(np.isfinite(housing["price_growth"]))

    def test_custom_housing_params(self):
        """Custom housing parameters should propagate correctly."""
        params = KeenHousingParams(h_initial_price=8.0, h_loan_to_income=6.0)
        core, housing = simulate_keen_housing(params, t_max=50)
        assert abs(housing["price_to_income"][0] - 8.0) < 1e-6
