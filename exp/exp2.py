"""Experiment 2: the conventional empirical correlation formula gives
misleading non-zero results. Applies the textbook estimator
corr[p, U]_e = (1/N) sum (p_i - pi(t))(U_i - U(t)) (paper Eq. 4.4 / B.2),
which uses the simple average price pi(t) instead of VWAP, and shows it
equals [p(t) - pi(t)] * U(t) (Eq. 4.8 / B.8), non-zero whenever trade
volumes are not constant and zero when they are.

Run as: python -m exp.exp2
"""

from __future__ import annotations

import itertools
import logging

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data import generate_synthetic as gen
from src import config
from src.correlations import (
    empirical_correlation_conventional,
    spurious_correlation_closed_form,
    true_price_volume_correlation,
    vwap,
    simple_average_price,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = "results"


def run() -> pd.DataFrame:
    """Compute the conventional empirical correlation corr[p,U]_e, its
    closed form [p(t)-pi(t)]U(t), and the true correlation corr[p,U] for
    every trial, for both random and constant trade volumes."""
    rows = []
    combos = list(
        itertools.product(
            config.EXP2_TRADE_COUNTS,
            config.EXP2_SEEDS,
            config.VOLUME_DISTRIBUTIONS,
            config.VALUE_DISTRIBUTIONS,
        )
    )
    for n, seed, volume_dist, value_dist in combos:
        trades = gen.generate_independent_trades(n, seed, volume_dist, value_dist)
        p, U = trades["p"], trades["U"]
        corr_e_direct = empirical_correlation_conventional(p, U)
        corr_e_formula = spurious_correlation_closed_form(p, U)
        true_corr = true_price_volume_correlation(p, U)
        rows.append(
            {
                "N": n,
                "seed": seed,
                "volume_dist": volume_dist,
                "value_dist": value_dist,
                "case": "random_volume",
                "p_vwap": vwap(p * U, U),
                "pi_simple_average": simple_average_price(p),
                "corr_e_direct": corr_e_direct,
                "corr_e_formula": corr_e_formula,
                "true_corr_p_U": true_corr,
            }
        )

    for n, seed, value_dist in itertools.product(
        config.EXP2_TRADE_COUNTS, config.EXP2_SEEDS, config.VALUE_DISTRIBUTIONS
    ):
        trades = gen.generate_constant_volume_trades(n, seed, value_dist)
        p, U = trades["p"], trades["U"]
        corr_e_direct = empirical_correlation_conventional(p, U)
        corr_e_formula = spurious_correlation_closed_form(p, U)
        true_corr = true_price_volume_correlation(p, U)
        rows.append(
            {
                "N": n,
                "seed": seed,
                "volume_dist": "constant",
                "value_dist": value_dist,
                "case": "constant_volume",
                "p_vwap": vwap(p * U, U),
                "pi_simple_average": simple_average_price(p),
                "corr_e_direct": corr_e_direct,
                "corr_e_formula": corr_e_formula,
                "true_corr_p_U": true_corr,
            }
        )

    return pd.DataFrame(rows)


def summarize_and_save(df: pd.DataFrame) -> dict:
    df.to_csv(f"{RESULTS_DIR}/exp2_results.csv", index=False)

    df["formula_match_error"] = (df["corr_e_direct"] - df["corr_e_formula"]).abs()
    max_formula_error = df["formula_match_error"].max()

    random_case = df[df["case"] == "random_volume"]
    constant_case = df[df["case"] == "constant_volume"]

    max_abs_true_corr = df["true_corr_p_U"].abs().max()
    max_abs_constant_corr_e = constant_case["corr_e_direct"].abs().max()
    median_abs_random_corr_e = random_case["corr_e_direct"].abs().median()

    summary = pd.DataFrame(
        [
            {
                "metric": "max |corr_e_direct - corr_e_formula| (Eq. 4.7 vs 4.8/B.8)",
                "value": max_formula_error,
            },
            {
                "metric": "max |true corr[p,U]| (should be ~0, Eq. 4.3/A.11)",
                "value": max_abs_true_corr,
            },
            {
                "metric": "max |corr[p,U]_e| for constant volumes (should be ~0)",
                "value": max_abs_constant_corr_e,
            },
            {
                "metric": "median |corr[p,U]_e| for random volumes (should be > 0)",
                "value": median_abs_random_corr_e,
            },
        ]
    )
    summary.to_csv(f"{RESULTS_DIR}/exp2_summary.csv", index=False)

    logger.info(
        "exp2: %d trials; max formula-vs-direct error = %.3e; "
        "median |corr_e| (random volumes) = %.4f; "
        "max |corr_e| (constant volumes) = %.3e",
        len(df),
        max_formula_error,
        median_abs_random_corr_e,
        max_abs_constant_corr_e,
    )

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].scatter(random_case["corr_e_direct"], random_case["corr_e_formula"],
                     s=10, alpha=0.6)
    lims = [
        min(random_case["corr_e_direct"].min(), random_case["corr_e_formula"].min()),
        max(random_case["corr_e_direct"].max(), random_case["corr_e_formula"].max()),
    ]
    axes[0].plot(lims, lims, "r--", label="y = x")
    axes[0].set_xlabel(r"$corr[p,U]_e$ direct (Eq. 4.4/B.2)")
    axes[0].set_ylabel(r"$[p(t)-\pi(t)]U(t)$ (Eq. 4.8/B.8)")
    axes[0].set_title("Conventional estimator equals its closed form")
    axes[0].legend()

    axes[1].hist(random_case["corr_e_direct"], bins=25, alpha=0.7,
                 label="random volumes")
    axes[1].hist(constant_case["corr_e_direct"], bins=25, alpha=0.7,
                 label="constant volumes")
    axes[1].axvline(0, color="black", linewidth=1)
    axes[1].set_xlabel(r"$corr[p,U]_e$")
    axes[1].set_ylabel("count")
    axes[1].set_title("Spurious correlation vanishes only for constant volumes")
    axes[1].legend()

    fig.suptitle("Experiment 2: origin of the 'empirical' price-volume correlation "
                 "(Olkhov 2026, Sec. 4.1)")
    fig.tight_layout()
    fig.savefig(f"{RESULTS_DIR}/exp2_spurious_correlation.png", dpi=150)
    plt.close(fig)

    return {
        "n_trials": len(df),
        "max_formula_vs_direct_error": float(max_formula_error),
        "max_abs_true_corr": float(max_abs_true_corr),
        "max_abs_constant_volume_corr_e": float(max_abs_constant_corr_e),
        "median_abs_random_volume_corr_e": float(median_abs_random_corr_e),
    }


def main() -> dict:
    df = run()
    metrics = summarize_and_save(df)
    assert metrics["max_formula_vs_direct_error"] < 1e-6, (
        "corr[p,U]_e computed directly (Eq. 4.4) must equal the closed form "
        "[p(t) - pi(t)] U(t) (Eq. 4.8 / B.8)."
    )
    assert metrics["max_abs_constant_volume_corr_e"] < 1e-8, (
        "corr[p,U]_e must vanish when all trade volumes are constant "
        "(paper Sec. 4.1)."
    )
    assert metrics["median_abs_random_volume_corr_e"] > 0, (
        "corr[p,U]_e must be non-zero for non-constant trade volumes, "
        "demonstrating the spurious 'empirical' correlation."
    )
    logger.info("exp2: PASSED -- conventional estimator matches its closed form and "
                "is non-zero only when volumes vary")
    return metrics


if __name__ == "__main__":
    main()
