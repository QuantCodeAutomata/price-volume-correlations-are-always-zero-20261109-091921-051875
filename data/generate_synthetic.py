"""Synthetic trade data for the replication of Olkhov (2026),
"Price-Volume Correlations Are Always Zero".

The paper's results are algebraic identities derived from the trade equation
C(t_i) = p(t_i) * U(t_i) (Eq. A.2) applied to the values C(t_i) and volumes
U(t_i) of N consecutive trades made during an averaging interval Delta
(Eq. A.1). The paper states the identities hold "regardless of the
distribution" of trades and does not prescribe a specific generator; this
module draws random, strictly positive trade volumes and trade values from
a selection of non-degenerate distributions and derives trade prices from
the trade equation (Eq. A.2).
"""

from __future__ import annotations

import numpy as np

from src import config


def _draw_positive(
    rng: np.random.Generator, distribution: str, size: int, scale: float
) -> np.ndarray:
    """Draw ``size`` strictly positive samples from ``distribution`

    Parameters
    ----------
    rng:
        NumPy random generator (controls the seed).
    distribution:
        One of "lognormal", "uniform", "gamma", "exponential",
        "normal_positive".
    size:
        Number of trades N.
    scale:
        Multiplicative scale applied to the unit-mean draw, so that
        different quantities (values vs. volumes) can live on different
        magnitudes while keeping the same relative dispersion.
    """
    if distribution == "lognormal":
        sigma = config.LOGNORMAL_SIGMA
        mu = -0.5 * sigma**2  # mean of exp(normal) equals 1 before scaling
        draw = rng.lognormal(mean=mu, sigma=sigma, size=size)
    elif distribution == "uniform":
        draw = rng.uniform(config.UNIFORM_LOW, config.UNIFORM_HIGH, size=size)
        draw = draw / draw.mean()
    elif distribution == "gamma":
        shape = config.GAMMA_SHAPE
        draw = rng.gamma(shape=shape, scale=1.0 / shape, size=size)
    elif distribution == "exponential":
        draw = rng.exponential(scale=1.0, size=size)
    elif distribution == "normal_positive":
        # Normal draw shifted/clipped to stay strictly positive.
        draw = rng.normal(loc=3.0, scale=0.8, size=size)
        draw = np.clip(draw, 0.05, None)
    else:
        raise ValueError(f"Unknown distribution: {distribution}")
    return scale * draw


def generate_independent_trades(
    n: int,
    seed: int,
    volume_dist: str = "lognormal",
    value_dist: str = "lognormal",
) -> dict[str, np.ndarray]:
    """Generate N consecutive trades with independently drawn C(t_i), U(t_i).

    Returns a dict with keys ``C`` (trade values), ``U`` (trade volumes) and
    ``p`` (trade prices), where ``p = C / U`` obeys the trade equation
    (paper Eq. A.2).
    """
    if n < 2:
        raise ValueError("n must be at least 2 to compute a correlation")
    rng = np.random.default_rng(seed)
    U = _draw_positive(rng, volume_dist, n, config.VOLUME_SCALE)
    C = _draw_positive(rng, value_dist, n, config.VALUE_SCALE)
    p = C / U
    return {"C": C, "U": U, "p": p}


def generate_correlated_trades(
    n: int,
    seed: int,
    volume_dist: str = "lognormal",
    value_dist: str = "lognormal",
    correlation_strength: float | None = None,
) -> dict[str, np.ndarray]:
    """Generate N consecutive trades whose values C(t_i) and volumes U(t_i)
    share a common random factor, so that corr[C, U] (Eq. C.4) is generally
    non-zero. This is needed to exercise the alternative correlations
    corr[p, U^2] (Eq. C.8) and corr[p^2, U^2] (Eq. C.16) of Appendix C, which
    the paper states should "generally" be non-zero, unlike corr[p, U].

    The paper does not specify how to correlate synthetic C and U; here a
    shared positive factor is mixed into both draws with weight
    ``correlation_strength`` in [0, 1).
    """
    if correlation_strength is None:
        correlation_strength = config.CORRELATION_STRENGTH
    if not 0.0 <= correlation_strength < 1.0:
        raise ValueError("correlation_strength must be in [0, 1)")
    if n < 2:
        raise ValueError("n must be at least 2 to compute a correlation")

    rng = np.random.default_rng(seed)
    common = _draw_positive(rng, volume_dist, n, 1.0)
    idio_U = _draw_positive(rng, volume_dist, n, 1.0)
    idio_C = _draw_positive(rng, value_dist, n, 1.0)

    w = correlation_strength
    U = config.VOLUME_SCALE * (w * common + (1 - w) * idio_U)
    C = config.VALUE_SCALE * (w * common + (1 - w) * idio_C)
    p = C / U
    return {"C": C, "U": U, "p": p}


def generate_constant_volume_trades(
    n: int, seed: int, value_dist: str = "lognormal"
) -> dict[str, np.ndarray]:
    """Generate N trades with constant volume U(t_i) = U, the limiting case
    of equal consecutive trade volumes discussed in paper Eq. 3.2 / A.7.
    """
    if n < 2:
        raise ValueError("n must be at least 2 to compute a correlation")
    rng = np.random.default_rng(seed)
    U = np.full(n, config.VOLUME_SCALE)
    C = _draw_positive(rng, value_dist, n, config.VALUE_SCALE)
    p = C / U
    return {"C": C, "U": U, "p": p}
