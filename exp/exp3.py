"""Experiment 3: alternative correlations corr[p, U^2] and corr[p^2, U^2]
that capture the statistical interdependence of prices and volumes which the
zero price-volume correlation corr[p, U] fails to capture (paper App. C).

Verifies:
  - corr[p, U^2] computed directly from trade time series matches the closed
    form [phi(C,U) - chi^2(U)] p(t) U(t)^2 (Eq. C.8);
  - corr[p^2, U^2] computed directly matches 2 corr[p, U^2] p(t) (Eq. C.16 / 4.9);
  - both correlations are generally non-zero when C(t_i) and U(t_i) share a
    common random factor (making phi(C,U) != 0), unlike corr[p, U].

Run as: python -m exp.exp3
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
    corr_p_u2_direct,
    corr_p_u2_formula,
    corr_p2_u2_direct,
    corr_p2_u2_formula,
    variation_coefficients,
    vwap,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = "results"


def _one_trial(trades: dict, n: int, seed: int, volume_dist: str, value_dist: str,
                case: str) -> dict:
    C, U, p = trades["C"], trades["U"], trades["p"]
    coeffs = variation_coefficients(C, U)
    p_vwap = vwap(C, U)

    cpu2_direct = corr_p_u2_direct(p, U)
    cpu2_formula = corr_p_u2_formula(C, U)
    cp2u2_direct = corr_p2_u2_direct(C, p, U)
    cp2u2_formula = corr_p2_u2_formula(C, U)
    cp2u2_from_relation = 2.0 * cpu2_direct * p_vwap

    return {
        "N": n,
        "seed": seed,
        "volume_dist": volume_dist,
        "value_dist": value_dist,
        "case": case,
        "psi2_C": coeffs["psi2"],
        "chi2_U": coeffs["chi2"],
        "phi_CU": coeffs["phi"],
        "p_vwap": p_vwap,
        "corr_p_U2_direct": cpu2_direct,
        "corr_p_U2_formula": cpu2_formula,
        "corr_p2_U2_direct": cp2u2_direct,
        "corr_p2_U2_formula": cp2u2_formula,
        "corr_p2_U2_from_relation": cp2u2_from_relation,
    }


def run() -> pd.DataFrame:
    rows = []
    combos = list(
        itertools.product(
            config.EXP3_TRADE_COUNTS,
            config.EXP3_SEEDS,
            config.VOLUME_DISTRIBUTIONS,
            config.VALUE_DISTRIBUTIONS,
        )
    )
    for n, seed, volume_dist, value_dist in combos:
        # Correlated C, U (phi(C,U) != 0 in general): the case the paper
        # points to for assessing real statistical interdependence.
        trades = gen.generate_correlated_trades(n, seed, volume_dist, value_dist)
        rows.append(_one_trial(trades, n, seed, volume_dist, value_dist,
                                "correlated_C_U"))

        # Independent C, U: phi(C,U) is small but not exactly zero in finite
        # samples, still exercising the same closed-form identities.
        trades_indep = gen.generate_independent_trades(n, seed, volume_dist, value_dist)
        rows.append(_one_trial(trades_indep, n, seed, volume_dist, value_dist,
                                "independent_C_U"))

    for n, seed, value_dist in itertools.product(
        config.EXP3_TRADE_COUNTS, config.EXP3_SEEDS, config.VALUE_DISTRIBUTIONS
    ):
        trades_const = gen.generate_constant_volume_trades(n, seed, value_dist)
        rows.append(_one_trial(trades_const, n, seed, "constant", value_dist,
                                "constant_volume"))

    return pd.DataFrame(rows)


def summarize_and_save(df: pd.DataFrame) -> dict:
    df.to_csv(f"{RESULTS_DIR}/exp3_results.csv", index=False)

    # corr[p, U^2] (Eq. C.8) is an exact algebraic identity: the direct
    # empirical computation and the closed form must agree to floating-point
    # precision for every trial.
    df["cpu2_formula_error"] = (df["corr_p_U2_direct"] - df["corr_p_U2_formula"]).abs()

    # corr[p^2, U^2] (Eq. C.16 / 4.9) is instead obtained in Appendix C by
    # substituting the price variance coefficient mu^2(p) with the
    # "market-based variance" expression of Eq. C.14 (cited from Olkhov,
    # 2022; 2023), which is not re-derived from the trade equation in this
    # appendix. Direct recomputation of corr[p^2, U^2] from the raw trade
    # series via its defining equation (C.9) shows this closed form holds
    # only approximately: the ratio of the direct value to the Eq. C.16
    # value tends to 2, not 1, as the coefficients of variation psi^2(C),
    # chi^2(U) shrink (verified analytically and numerically; see
    # "Deviations from the paper" in results/RESULTS.md). The two
    # quantities are therefore reported and compared, not asserted equal.
    df["cp2u2_direct_vs_formula_ratio"] = (
        df["corr_p2_U2_direct"] / df["corr_p2_U2_formula"]
    )
    df["relation_error"] = (
        df["corr_p2_U2_direct"] - df["corr_p2_U2_from_relation"]
    ).abs()

    correlated = df[df["case"] == "correlated_C_U"]
    constant = df[df["case"] == "constant_volume"]

    summary = pd.DataFrame(
        [
            {"metric": "max |corr[p,U^2] direct - formula| (Eq. C.8, exact)",
             "value": df["cpu2_formula_error"].max()},
            {"metric": "median corr[p^2,U^2] direct / Eq.C.16 formula ratio "
                       "(paper implies 1; measured limit is 2)",
             "value": correlated["cp2u2_direct_vs_formula_ratio"].median()},
            {"metric": "median |corr[p,U^2]| for correlated C,U (should be > 0)",
             "value": correlated["corr_p_U2_direct"].abs().median()},
            {"metric": "max chi^2(U) for constant-volume case (should be ~0)",
             "value": constant["chi2_U"].abs().max()},
        ]
    )
    summary.to_csv(f"{RESULTS_DIR}/exp3_summary.csv", index=False)

    logger.info(
        "exp3: %d trials; max C.8 formula error = %.3e; median corr[p^2,U^2] "
        "direct/formula ratio = %.3f; median |corr[p,U^2]| (correlated) = %.4f",
        len(df),
        df["cpu2_formula_error"].max(),
        correlated["cp2u2_direct_vs_formula_ratio"].median(),
        correlated["corr_p_U2_direct"].abs().median(),
    )

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].scatter(df["corr_p_U2_direct"], df["corr_p_U2_formula"], s=10, alpha=0.5)
    lims = [df["corr_p_U2_direct"].min(), df["corr_p_U2_direct"].max()]
    axes[0].plot(lims, lims, "r--", label="y = x")
    axes[0].set_xlabel(r"$corr[p,U^2]$ direct")
    axes[0].set_ylabel(r"$corr[p,U^2]$ via Eq. C.8")
    axes[0].set_title("corr[p, U^2]: direct vs. closed form")
    axes[0].legend()

    axes[1].scatter(df["corr_p2_U2_direct"], df["corr_p2_U2_from_relation"],
                     s=10, alpha=0.5, label="trials")
    lims2 = [df["corr_p2_U2_direct"].min(), df["corr_p2_U2_direct"].max()]
    axes[1].plot(lims2, lims2, "r--", label="y = x (exact Eq. 4.9/C.16)")
    axes[1].plot(lims2, [2 * v for v in lims2], "g:", label="y = 2x (measured limit)")
    axes[1].set_xlabel(r"$corr[p^2,U^2]$ direct (Eq. C.9)")
    axes[1].set_ylabel(r"$2\,corr[p,U^2]\,p(t)$ (Eq. 4.9/C.16)")
    axes[1].set_title("corr[p^2,U^2] vs. its Eq. 4.9/C.16 closed form")
    axes[1].legend()

    fig.suptitle("Experiment 3: statistical interdependence beyond corr[p,U] "
                 "(Olkhov 2026, App. C)")
    fig.tight_layout()
    fig.savefig(f"{RESULTS_DIR}/exp3_interdependence.png", dpi=150)
    plt.close(fig)

    return {
        "n_trials": len(df),
        "max_cpu2_formula_error": float(df["cpu2_formula_error"].max()),
        "median_cp2u2_direct_vs_formula_ratio": float(
            correlated["cp2u2_direct_vs_formula_ratio"].median()
        ),
        "median_abs_corr_p_u2_correlated": float(
            correlated["corr_p_U2_direct"].abs().median()
        ),
    }


def main() -> dict:
    df = run()
    metrics = summarize_and_save(df)
    assert metrics["max_cpu2_formula_error"] < 1e-6, (
        "corr[p, U^2] computed directly must match the closed form of Eq. C.8, "
        "an exact algebraic identity of the trade equation (Eq. A.2)."
    )
    assert metrics["median_abs_corr_p_u2_correlated"] > 0, (
        "corr[p, U^2] must be non-zero when trade values and volumes are "
        "correlated, capturing statistical interdependence beyond corr[p,U]."
    )
    logger.info(
        "exp3: PASSED -- corr[p,U^2] matches its exact closed form (Eq. C.8) "
        "and is non-zero when C and U are correlated; see results/exp3_summary.csv "
        "for the measured corr[p^2,U^2] vs. Eq. 4.9/C.16 comparison"
    )
    return metrics


if __name__ == "__main__":
    main()
