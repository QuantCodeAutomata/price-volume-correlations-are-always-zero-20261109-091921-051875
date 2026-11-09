"""Integration tests for the three experiments in exp/. Each experiment uses
only synthetic data (data/generate_synthetic.py); no market data or network
access is required, so none of these tests are marked market_data.
"""

from __future__ import annotations

import os

import pytest

from exp import exp1, exp2, exp3


@pytest.fixture(autouse=True)
def _chdir_to_repo_root(tmp_path, monkeypatch):
    """Experiments write to ./results relative to the current working
    directory; run them from the real repository root so the generated
    artifacts land in results/ as the README describes."""
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    monkeypatch.chdir(repo_root)
    yield


def test_exp1_zero_correlation_passes_and_saves_artifacts():
    metrics = exp1.main()
    assert metrics["overall_max_abs_relative_corr"] < 1e-9
    assert metrics["n_trials"] > 0
    assert os.path.exists("results/exp1_results.csv")
    assert os.path.exists("results/exp1_summary.csv")
    assert os.path.exists("results/exp1_zero_correlation.png")


def test_exp2_spurious_correlation_passes_and_saves_artifacts():
    metrics = exp2.main()
    assert metrics["max_formula_vs_direct_error"] < 1e-6
    assert metrics["max_abs_constant_volume_corr_e"] < 1e-8
    assert metrics["median_abs_random_volume_corr_e"] > 0
    assert os.path.exists("results/exp2_results.csv")
    assert os.path.exists("results/exp2_spurious_correlation.png")


def test_exp3_interdependence_passes_and_saves_artifacts():
    metrics = exp3.main()
    assert metrics["max_cpu2_formula_error"] < 1e-6
    assert metrics["median_abs_corr_p_u2_correlated"] > 0
    assert os.path.exists("results/exp3_results.csv")
    assert os.path.exists("results/exp3_interdependence.png")


def test_exp3_direct_vs_formula_ratio_is_reported_not_forced_to_one():
    """The measured corr[p^2,U^2] direct-to-formula ratio (Eq. 4.9/C.16) is a
    diagnostic, not an identity the implementation enforces; it must be a
    finite, strictly positive number for correlated trade values and volumes."""
    metrics = exp3.main()
    ratio = metrics["median_cp2u2_direct_vs_formula_ratio"]
    assert ratio == ratio  # not NaN
    assert ratio > 0
