"""Experiment 1: verification that the true price-volume correlation
corr[p, U] is always zero (paper Eq. 4.3 / A.11), for random trade volumes
and values drawn from several non-degenerate distributions and for the
limiting case of constant trade volumes (Eq. 3.2 / A.7).

Run as: python -m exp.exp1
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
from src.correlations import true_price_volume_correlation, vwap, mean_volume

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = "results"


def run() -> pd.DataFrame:
    """Compute corr[p, U] for every (N, seed, volume distribution, value
    distribution) combination, plus the constant-volume limiting case, and
    return a tidy DataFrame of the results."""
    rows = []
    combos = list(
        itertools.product(
            config.EXP1_TRADE_COUNTS,
            config.EXP1_SEEDS,
            config.VOLUME_DISTRIBUTIONS,
            config.VALUE_DISTRIBUTIONS,
        )
    )
    for n, seed, volume_dist, value_dist in combos:
        trades = gen.generate_independent_trades(n, seed, volume_dist, value_dist)
        p, U = trades["p"], trades["U"]
        corr = true_price_volume_correlation(p, U)
        scale = abs(vwap(p * U, U) * mean_volume(U))
        rows.append(
            {
                "N": n,
                "seed": seed,
                "volume_dist": volume_dist,
                "value_dist": value_dist,
                "case": "random_volume",
                "corr_p_U": corr,
                "relative_corr": corr / scale if scale else 0.0,
            }
        )

    # Limiting case of constant trade volumes (Eq. 3.2 / A.7).
    for n, seed, value_dist in itertools.product(
        config.EXP1_TRADE_COUNTS, config.EXP1_SEEDS, config.VALUE_DISTRIBUTIONS
    ):
        trades = gen.generate_constant_volume_trades(n, seed, value_dist)
        p, U = trades["p"], trades["U"]
        corr = true_price_volume_correlation(p, U)
        scale = abs(vwap(p * U, U) * mean_volume(U))
        rows.append(
            {
                "N": n,
                "seed": seed,
                "volume_dist": "constant",
                "value_dist": value_dist,
                "case": "constant_volume",
                "corr_p_U": corr,
                "relative_corr": corr / scale if scale else 0.0,
            }
        )

    df = pd.DataFrame(rows)
    return df


def summarize_and_save(df: pd.DataFrame) -> dict:
    """Save the full result table and a summary, log realized scope, and
    return summary metrics for results/RESULTS.md."""
    df.to_csv(f"{RESULTS_DIR}/exp1_results.csv", index=False)

    summary = (
        df.groupby(["case"])["relative_corr"]
        .agg(max_abs_relative_corr=lambda s: s.abs().max())
        .reset_index()
    )
    summary.to_csv(f"{RESULTS_DIR}/exp1_summary.csv", index=False)

    overall_max_abs = df["relative_corr"].abs().max()
    n_trials = len(df)
    n_values = sorted(df["N"].unique().tolist())
    logger.info(
        "exp1: %d trials, N in %s, max |relative corr[p,U]| = %.3e",
        n_trials,
        n_values,
        overall_max_abs,
    )

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for case, group in df.groupby("case"):
        by_n = group.groupby("N")["relative_corr"].apply(lambda s: s.abs().max())
        ax.plot(by_n.index, by_n.values, marker="o", label=case)
    ax.axhline(config.RELATIVE_ZERO_TOLERANCE, color="red", linestyle="--",
               label="zero tolerance")
    ax.set_yscale("log")
    ax.set_xlabel("Number of trades N")
    ax.set_ylabel("max |relative corr[p, U]|")
    ax.set_title("Experiment 1: true price-volume correlation is always zero\n"
                 "(Olkhov 2026, Eq. 4.3 / A.11)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{RESULTS_DIR}/exp1_zero_correlation.png", dpi=150)
    plt.close(fig)

    return {
        "n_trials": n_trials,
        "n_values": n_values,
        "overall_max_abs_relative_corr": float(overall_max_abs),
    }


def main() -> dict:
    df = run()
    metrics = summarize_and_save(df)
    assert metrics["overall_max_abs_relative_corr"] < config.RELATIVE_ZERO_TOLERANCE, (
        "The true price-volume correlation corr[p, U] should be zero up to "
        "floating-point round-off (paper Eq. 4.3 / A.11)."
    )
    logger.info("exp1: PASSED -- corr[p, U] is zero within tolerance for every trial")
    return metrics


if __name__ == "__main__":
    main()
