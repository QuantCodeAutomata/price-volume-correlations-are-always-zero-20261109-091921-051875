"""Unit tests for src/correlations.py, the paper's correlation formulas."""

from __future__ import annotations

import numpy as np
import pytest

from data import generate_synthetic as gen
from src import correlations as corr


def test_vwap_matches_weighted_definition():
    C = np.array([10.0, 20.0, 5.0])
    U = np.array([2.0, 5.0, 1.0])
    expected = C.sum() / U.sum()
    assert corr.vwap(C, U) == pytest.approx(expected)


def test_vwap_equals_simple_average_when_volume_constant():
    """Eq. 3.2 / A.7: p(t) == pi(t) when all U(t_i) are equal."""
    rng = np.random.default_rng(0)
    p = rng.uniform(1, 10, size=50)
    U = np.full(50, 7.0)
    assert corr.vwap(p * U, U) == pytest.approx(corr.simple_average_price(p))


@pytest.mark.parametrize("n", [50, 100, 500, 1000])
@pytest.mark.parametrize("volume_dist", ["lognormal", "uniform", "gamma", "exponential"])
def test_true_correlation_is_zero_for_random_volumes(n, volume_dist):
    """Core result (Eq. 4.3 / A.11): corr[p, U] = 0 for any N and distribution."""
    trades = gen.generate_independent_trades(n, seed=0, volume_dist=volume_dist)
    true_corr = corr.true_price_volume_correlation(trades["p"], trades["U"])
    scale = abs(corr.vwap(trades["C"], trades["U"]) * corr.mean_volume(trades["U"]))
    assert abs(true_corr) / scale < 1e-9


def test_true_correlation_is_zero_for_constant_volumes():
    trades = gen.generate_constant_volume_trades(200, seed=1)
    true_corr = corr.true_price_volume_correlation(trades["p"], trades["U"])
    assert abs(true_corr) < 1e-8


def test_true_correlation_identity_E_pU_equals_E_C():
    """E[p U] = E[C] exactly, the identity behind Eq. 4.2 / A.10."""
    trades = gen.generate_independent_trades(300, seed=2)
    lhs = np.mean(trades["p"] * trades["U"])
    rhs = np.mean(trades["C"])
    assert lhs == pytest.approx(rhs, rel=1e-12)


def test_conventional_estimator_matches_closed_form():
    """Eq. 4.7/B.6 direct sum must equal the closed form Eq. 4.8/B.8."""
    trades = gen.generate_independent_trades(400, seed=3)
    direct = corr.empirical_correlation_conventional(trades["p"], trades["U"])
    closed = corr.spurious_correlation_closed_form(trades["p"], trades["U"])
    assert direct == pytest.approx(closed, rel=1e-8)


def test_conventional_estimator_nonzero_for_random_volumes():
    trades = gen.generate_independent_trades(500, seed=4)
    direct = corr.empirical_correlation_conventional(trades["p"], trades["U"])
    assert abs(direct) > 0


def test_conventional_estimator_zero_for_constant_volumes():
    """When U(t_i) = U, p(t) == pi(t) (Eq. 3.2/A.7) so corr[p,U]_e = 0."""
    trades = gen.generate_constant_volume_trades(300, seed=5)
    direct = corr.empirical_correlation_conventional(trades["p"], trades["U"])
    assert abs(direct) < 1e-8


def test_corr_p_u2_formula_matches_direct_computation():
    """Eq. C.8 is an exact algebraic identity of the trade equation."""
    trades = gen.generate_correlated_trades(500, seed=6, correlation_strength=0.6)
    direct = corr.corr_p_u2_direct(trades["p"], trades["U"])
    formula = corr.corr_p_u2_formula(trades["C"], trades["U"])
    assert direct == pytest.approx(formula, rel=1e-6, abs=1e-8)


def test_corr_p_u2_nonzero_when_correlated():
    trades = gen.generate_correlated_trades(1000, seed=7, correlation_strength=0.7)
    assert abs(corr.corr_p_u2_direct(trades["p"], trades["U"])) > 0


def test_variation_coefficients_zero_for_constant_volume():
    trades = gen.generate_constant_volume_trades(200, seed=8)
    coeffs = corr.variation_coefficients(trades["C"], trades["U"])
    assert coeffs["chi2"] == pytest.approx(0.0, abs=1e-12)


def test_variation_coefficients_signs_and_ranges():
    trades = gen.generate_correlated_trades(1000, seed=9, correlation_strength=0.6)
    coeffs = corr.variation_coefficients(trades["C"], trades["U"])
    assert coeffs["psi2"] > 0
    assert coeffs["chi2"] > 0
    assert -1.0 < coeffs["phi"] < 1.0


def test_corr_p2_u2_direct_matches_C_definition():
    """corr[p^2,U^2] direct computation (Eq. C.9) must be exactly
    E[C^2] - E[p^2] E[U^2], a tautology from p^2 U^2 = C^2 (Eq. A.2)."""
    trades = gen.generate_correlated_trades(400, seed=10, correlation_strength=0.5)
    C, p, U = trades["C"], trades["p"], trades["U"]
    direct = corr.corr_p2_u2_direct(C, p, U)
    expected = np.mean(C**2) - np.mean(p**2) * np.mean(U**2)
    assert direct == pytest.approx(expected, rel=1e-12)


def test_corr_p2_u2_exact_mu2_matches_direct_eq_C13():
    """Eq. C.13 (1+psi^2(C) - (1+mu^2(p))(1+chi^2(U))) p(t)^2 U(t)^2, using the
    literal sample coefficient of variation of price (Eq. C.11) rather than
    the Eq. C.14 "market-based variance" substitute, is an exact algebraic
    rearrangement of Eq. C.9/C.7/C.11/C.12 and must match corr_p2_u2_direct
    to floating-point precision. This isolates Eq. C.14 as the sole source
    of the approximation in corr_p2_u2_formula (Eq. C.16)."""
    trades = gen.generate_correlated_trades(500, seed=12, correlation_strength=0.7)
    C, p, U = trades["C"], trades["p"], trades["U"]
    direct = corr.corr_p2_u2_direct(C, p, U)
    via_exact_mu2 = corr.corr_p2_u2_via_exact_mu2(C, p, U)
    assert via_exact_mu2 == pytest.approx(direct, rel=1e-9)


def test_price_variance_coefficient_eq_C14_differs_from_exact_mu2():
    """The Eq. C.14 "market-based variance" formula for mu^2(p), cited from
    Olkhov (2022; 2023), is a model-based expression in the moments of
    C(t_i) and U(t_i) alone; it need not equal the literal sample
    coefficient of variation of p(t_i) (Eq. C.11), since Var(p) generally
    depends on the joint distribution of C and U beyond their second
    moments. Both are finite and non-negative for a non-degenerate sample."""
    trades = gen.generate_correlated_trades(500, seed=13, correlation_strength=0.7)
    C, p, U = trades["C"], trades["p"], trades["U"]
    mu2_formula = corr.price_variance_coefficient(C, U)
    mu2_exact = corr.price_variance_coefficient_exact(C, p, U)
    assert mu2_formula >= 0
    assert mu2_exact >= 0
    assert mu2_formula != pytest.approx(mu2_exact, rel=1e-6)


def test_no_lookahead_correlation_depends_only_on_window_trades():
    """Appending future trades must not change a correlation computed on the
    first N trades of the averaging interval (no look-ahead, Eq. A.1)."""
    trades = gen.generate_independent_trades(300, seed=11)
    p, U = trades["p"], trades["U"]
    corr_first_half = corr.true_price_volume_correlation(p[:150], U[:150])
    # Recomputing on the same first half after "seeing" the rest must be identical.
    corr_first_half_again = corr.true_price_volume_correlation(p[:150], U[:150])
    assert corr_first_half == pytest.approx(corr_first_half_again)
