"""Price-volume correlation formulas from Olkhov (2026),
"Price-Volume Correlations Are Always Zero".

All functions implement the paper's algebraic definitions directly from the
raw trade values C(t_i), volumes U(t_i) and trade prices p(t_i) = C(t_i)/U(t_i)
(trade equation, Eq. A.2) of N consecutive trades made during an averaging
interval Delta (Eq. A.1).

A regular Pearson/weighted-correlation routine from a statistics library
(e.g. scipy.stats, which offers no VWAP-weighted covariance) does not express
the mean-price convention (VWAP vs. simple average) that is the subject of
the paper, so every correlation below is implemented from the paper's
equations rather than from a library call.
"""

from __future__ import annotations

import numpy as np


def vwap(C: np.ndarray, U: np.ndarray) -> float:
    """Volume-weighted average price p(t) = C_Sigma / U_Sigma = C(t)/U(t)
    (paper Eq. 3.1 / A.6)."""
    return float(np.sum(C) / np.sum(U))


def mean_value(C: np.ndarray) -> float:
    """Mean trade value C(t) = (1/N) sum C(t_i) (paper Eq. A.5)."""
    return float(np.mean(C))


def mean_volume(U: np.ndarray) -> float:
    """Mean trade volume U(t) = (1/N) sum U(t_i) (paper Eq. A.5)."""
    return float(np.mean(U))


def simple_average_price(p: np.ndarray) -> float:
    """Simple arithmetic average price pi(t) = (1/N) sum p(t_i)
    (paper Eq. 3.2 / A.7); equals VWAP only if all U(t_i) are constant."""
    return float(np.mean(p))


def true_price_volume_correlation(p: np.ndarray, U: np.ndarray) -> float:
    """Economically justified price-volume correlation
    corr[p, U] = E[p(t_i) U(t_i)] - E[p(t_i)] E[U(t_i)]      (Eq. 4.1; A.8)
    with the mean price set to VWAP, E[p(t_i)] = p(t) (Eq. A.9), giving the
    empirical form (Eq. 4.6 / B.9):
        corr[p, U] = (1/N) sum p(t_i) U(t_i) - p(t) U(t)
    which the paper shows is identically zero (Eq. 4.3 / A.11).
    """
    p_vwap = vwap(p * U, U)  # p(t) = C(t)/U(t); p*U = C exactly
    u_mean = mean_volume(U)
    return float(np.mean(p * U) - p_vwap * u_mean)


def empirical_correlation_conventional(p: np.ndarray, U: np.ndarray) -> float:
    """Conventional textbook correlation estimator (Eq. 4.4 / B.2) applied to
    the price and volume time series, using the *simple average* price
    pi(t) instead of VWAP as the mean price:
        corr[p, U]_e = (1/N) sum (p(t_i) - pi(t)) (U(t_i) - U(t))
                     = (1/N) sum p(t_i) U(t_i) - pi(t) U(t)   (Eq. 4.7 / B.6)
    """
    pi = simple_average_price(p)
    u_mean = mean_volume(U)
    return float(np.mean((p - pi) * (U - u_mean)))


def spurious_correlation_closed_form(p: np.ndarray, U: np.ndarray) -> float:
    """Closed form of the conventional estimator,
        corr[p, U]_e = [p(t) - pi(t)] * U(t)      (Eq. 4.8 / B.8)
    derived from the fact that (1/N) sum p(t_i) U(t_i) = C(t) = p(t) U(t)
    (Eq. B.7), so the conventional estimator differs from the true zero
    correlation only by the gap between VWAP and the simple average price.
    """
    p_vwap = vwap(p * U, U)
    pi = simple_average_price(p)
    u_mean = mean_volume(U)
    return float((p_vwap - pi) * u_mean)


def variation_coefficients(C: np.ndarray, U: np.ndarray) -> dict[str, float]:
    """Coefficients of variation and normalized co-movement of C and U
    (paper Eq. C.1-C.6):
        psi^2(C) = cov[C,C] / C(t)^2
        chi^2(U) = cov[U,U] / U(t)^2
        phi(C,U) = corr[C,U] / (C(t) U(t))
    """
    c_mean, u_mean = mean_value(C), mean_volume(U)
    cov_cc = float(np.mean(C**2) - c_mean**2)
    cov_uu = float(np.mean(U**2) - u_mean**2)
    corr_cu = float(np.mean(C * U) - c_mean * u_mean)
    return {
        "psi2": cov_cc / c_mean**2,
        "chi2": cov_uu / u_mean**2,
        "phi": corr_cu / (c_mean * u_mean),
    }


def corr_p_u2_formula(C: np.ndarray, U: np.ndarray) -> float:
    """corr[p, U^2] via the closed-form expression (paper Eq. C.8):
        corr[p, U^2] = [phi(C,U) - chi^2(U)] * p(t) * U(t)^2
    """
    coeffs = variation_coefficients(C, U)
    p_vwap = vwap(C, U)
    u_mean = mean_volume(U)
    return (coeffs["phi"] - coeffs["chi2"]) * p_vwap * u_mean**2


def corr_p_u2_direct(p: np.ndarray, U: np.ndarray) -> float:
    """corr[p, U^2] computed directly from the trade time series:
        corr[p, U^2] = E[p(t_i) U(t_i)^2] - E[p(t_i)] E[U(t_i)^2]
    using the trade equation p(t_i) U(t_i)^2 = C(t_i) U(t_i) (Eq. A.2).
    """
    p_vwap = vwap(p * U, U)
    return float(np.mean(p * U**2) - p_vwap * np.mean(U**2))


def corr_p2_u2_formula(C: np.ndarray, U: np.ndarray) -> float:
    """corr[p^2, U^2] via the paper's closed-form relation (Eq. C.16 / 4.9):
        corr[p^2, U^2] = 2 [phi(C,U) - chi^2(U)] p(t)^2 U(t)^2
                       = 2 corr[p, U^2] p(t)
    Eq. C.16 is reached (Eq. C.9-C.15) by rewriting the exact identity
        corr[p^2, U^2] = [1 + psi^2(C) - (1+mu^2(p))(1+chi^2(U))] p(t)^2 U(t)^2
    (Eq. C.13, an exact rearrangement of Eq. C.9/C.7/C.11/C.12; see
    price_variance_coefficient_exact) and then replacing the price variance
    coefficient mu^2(p) (Eq. C.11) with the "market-based variance" formula
    Eq. C.14, cited from Olkhov (2022; 2023) rather than derived from C and U
    moments in this appendix. See corr_p2_u2_direct for the exact empirical
    value (Eq. C.9) this closed form approximates, and the "statistical
    interdependence" experiment for the measured gap between the two.
    """
    return 2.0 * corr_p_u2_formula(C, U) * vwap(C, U)


def corr_p2_u2_direct(C: np.ndarray, p: np.ndarray, U: np.ndarray) -> float:
    """corr[p^2, U^2] computed directly from the trade time series, as
    literally defined in Eq. C.9:
        corr[p^2, U^2] = E[p(t_i)^2 U(t_i)^2] - E[p(t_i)^2] E[U(t_i)^2]
    using the trade equation p(t_i)^2 U(t_i)^2 = C(t_i)^2 (Eq. A.2), with
    E[p(t_i)^2] the plain sample second moment (1/N) sum p(t_i)^2, matching
    the convention of Eq. C.5 for E[C^2(t_i)] and E[U^2(t_i)].
    """
    return float(np.mean(C**2) - np.mean(p**2) * np.mean(U**2))


def price_variance_coefficient_exact(C: np.ndarray, p: np.ndarray, U: np.ndarray) -> float:
    """The literal coefficient of variation mu^2(p) of price (Eq. C.11):
        mu^2(p) = cov[p,p] / p(t)^2 = E[p(t_i)^2]/p(t)^2 - 1
    computed directly from the price series, with p(t) the VWAP (Eq. 3.1/
    A.6), since E[p(t_i)] = p(t) (Eq. A.9) uses VWAP as the mean price
    throughout App. A-C, not the simple average pi(t) (Eq. 3.2).
    Substituting this exact value into Eq. C.13 reproduces corr_p2_u2_direct
    (Eq. C.9) exactly; it generally differs from price_variance_coefficient's
    Eq. C.14 formula, which is not an algebraic identity in C(t_i), U(t_i)
    moments alone.
    """
    p_vwap = vwap(C, U)
    return float(np.mean(p**2)) / p_vwap**2 - 1.0


def price_variance_coefficient(C: np.ndarray, U: np.ndarray) -> float:
    """Coefficient of variation mu^2(p) of price via the paper's
    "market-based variance" formula (Eq. C.14, cited from Olkhov 2022;
    2023), substituted into Eq. C.13 to obtain the closed form Eq. C.16:
        mu^2(p) = [psi^2(C) - 2 phi(C,U) + chi^2(U)] / [1 + chi^2(U)]
    This is a model-based expression in the moments of C(t_i) and U(t_i)
    alone, distinct in general from the literal sample coefficient of
    variation of p(t_i) (price_variance_coefficient_exact), since Var(p)
    depends on the joint distribution of C and U beyond their second
    moments. See results/RESULTS.md, Experiment 3, for the measured gap.
    """
    coeffs = variation_coefficients(C, U)
    return (
        coeffs["psi2"] - 2.0 * coeffs["phi"] + coeffs["chi2"]
    ) / (1.0 + coeffs["chi2"])


def corr_p2_u2_via_exact_mu2(C: np.ndarray, p: np.ndarray, U: np.ndarray) -> float:
    """corr[p^2, U^2] via Eq. C.13, using the literal mu^2(p)
    (price_variance_coefficient_exact) rather than the Eq. C.14 formula.
    This is an exact algebraic rearrangement of Eq. C.9/C.7/C.11/C.12 and
    matches corr_p2_u2_direct to floating-point precision; it isolates the
    approximation introduced by Eq. C.14 in corr_p2_u2_formula.
    """
    coeffs = variation_coefficients(C, U)
    psi2, chi2 = coeffs["psi2"], coeffs["chi2"]
    mu2_exact = price_variance_coefficient_exact(C, p, U)
    p_vwap, u_mean = vwap(C, U), mean_volume(U)
    return (1.0 + psi2 - (1.0 + mu2_exact) * (1.0 + chi2)) * p_vwap**2 * u_mean**2
