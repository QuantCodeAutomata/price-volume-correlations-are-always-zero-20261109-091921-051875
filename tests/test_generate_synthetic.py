"""Unit tests for the synthetic trade-data generators (data/generate_synthetic.py)."""

from __future__ import annotations

import numpy as np
import pytest

from data import generate_synthetic as gen


def test_independent_trades_obey_trade_equation():
    """p(t_i) = C(t_i) / U(t_i) must hold exactly (paper Eq. A.2)."""
    trades = gen.generate_independent_trades(200, seed=0)
    np.testing.assert_allclose(trades["p"] * trades["U"], trades["C"], rtol=1e-12)


def test_independent_trades_strictly_positive():
    trades = gen.generate_independent_trades(100, seed=1, volume_dist="gamma",
                                              value_dist="uniform")
    assert np.all(trades["U"] > 0)
    assert np.all(trades["C"] > 0)
    assert np.all(trades["p"] > 0)


def test_independent_trades_volumes_are_not_constant():
    trades = gen.generate_independent_trades(100, seed=2)
    assert trades["U"].std() > 0


@pytest.mark.parametrize("dist", ["lognormal", "uniform", "gamma", "exponential"])
def test_all_volume_distributions_supported(dist):
    trades = gen.generate_independent_trades(50, seed=3, volume_dist=dist)
    assert np.all(trades["U"] > 0)


@pytest.mark.parametrize("dist", ["normal_positive", "lognormal", "uniform"])
def test_all_value_distributions_supported(dist):
    trades = gen.generate_independent_trades(50, seed=4, value_dist=dist)
    assert np.all(trades["C"] > 0)


def test_constant_volume_case_is_constant():
    """Limiting case of equal consecutive trade volumes (Eq. 3.2 / A.7)."""
    trades = gen.generate_constant_volume_trades(80, seed=5)
    assert trades["U"].std() == 0
    np.testing.assert_allclose(trades["p"] * trades["U"], trades["C"], rtol=1e-12)


def test_correlated_trades_have_nonzero_phi():
    """Shared common factor must induce corr[C, U] != 0 (used by Experiment 3)."""
    trades = gen.generate_correlated_trades(500, seed=6, correlation_strength=0.7)
    corr_cu = np.corrcoef(trades["C"], trades["U"])[0, 1]
    assert corr_cu > 0.3


def test_correlation_strength_zero_reduces_to_independent_like_behaviour():
    trades = gen.generate_correlated_trades(2000, seed=7, correlation_strength=0.0)
    corr_cu = np.corrcoef(trades["C"], trades["U"])[0, 1]
    assert abs(corr_cu) < 0.2


def test_invalid_distribution_raises():
    with pytest.raises(ValueError):
        gen.generate_independent_trades(10, seed=0, volume_dist="not_a_distribution")


def test_invalid_correlation_strength_raises():
    with pytest.raises(ValueError):
        gen.generate_correlated_trades(10, seed=0, correlation_strength=1.0)
    with pytest.raises(ValueError):
        gen.generate_correlated_trades(10, seed=0, correlation_strength=-0.1)


def test_too_few_trades_raises():
    with pytest.raises(ValueError):
        gen.generate_independent_trades(1, seed=0)
    with pytest.raises(ValueError):
        gen.generate_constant_volume_trades(0, seed=0)
    with pytest.raises(ValueError):
        gen.generate_correlated_trades(1, seed=0)


def test_seed_reproducibility():
    t1 = gen.generate_independent_trades(100, seed=42)
    t2 = gen.generate_independent_trades(100, seed=42)
    np.testing.assert_array_equal(t1["U"], t2["U"])
    np.testing.assert_array_equal(t1["C"], t2["C"])
