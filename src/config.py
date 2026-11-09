"""Parameters for the replication of Olkhov (2026), "Price-Volume Correlations
Are Always Zero".

All experiments use synthetic time series of trade values C(t_i) and trade
volumes U(t_i) during an averaging interval Delta (paper Eq. A.1), since the
paper's claims are algebraic identities that hold for any distribution of
trades, not statements about a particular market or sample period. No market
data is required or used by this replication.
"""

# --- Shared trade-generation parameters (paper Sec. 4, App. A) ---

# Numbers of consecutive trades N during the averaging interval Delta (Eq. A.1).
# The paper requires N "large enough to support reliable assessment of at
# least the first two statistical moments"; it gives no exact value, so a
# spread of sizes is used to show the identities hold for every N.
TRADE_COUNTS = [50, 100, 500, 1000]

# Random seeds used to repeat every configuration (paper does not specify a
# seed or a trial count).
SEEDS = [0, 1, 2, 3, 4]

# Distributions used to draw non-degenerate random trade volumes U(t_i).
# The paper does not specify a distribution; lognormal, uniform, gamma and
# exponential cover qualitatively different shapes (skewed, bounded,
# heavy/light tailed), as suggested by the paper's implementation notes.
VOLUME_DISTRIBUTIONS = ["lognormal", "uniform", "gamma", "exponential"]

# Distributions used to draw the random trade values C(t_i).
VALUE_DISTRIBUTIONS = ["normal_positive", "lognormal", "uniform"]

# Distribution parameters (shape only; location/scale fixed so that trade
# values and volumes stay strictly positive, matching the economic meaning of
# a traded value and a traded volume).
LOGNORMAL_SIGMA = 0.5  # shape parameter of the lognormal distribution
GAMMA_SHAPE = 2.0  # shape parameter "k" of the gamma distribution
UNIFORM_LOW = 1.0  # lower bound of the uniform distribution
UNIFORM_HIGH = 5.0  # upper bound of the uniform distribution

# Base scale of trade volumes and values (arbitrary units; only ratios enter
# the correlations computed below).
VOLUME_SCALE = 100.0
VALUE_SCALE = 1000.0

# Relative numerical tolerance used to assert that the true price-volume
# correlation corr[p, U] (Eq. 4.3 / A.11) is zero up to floating-point
# round-off, measured relative to the scale p(t) * U(t) of the two terms
# that cancel in the identity.
RELATIVE_ZERO_TOLERANCE = 1e-9

# --- Experiment 1: verification that corr[p, U] is always zero ---
EXP1_TRADE_COUNTS = TRADE_COUNTS
EXP1_SEEDS = SEEDS

# --- Experiment 2: conventional empirical formula corr[p, U]_e (Eq. 4.4-4.8) ---
EXP2_TRADE_COUNTS = TRADE_COUNTS
EXP2_SEEDS = SEEDS

# --- Experiment 3: corr[p, U^2] and corr[p^2, U^2] (App. C) ---
EXP3_TRADE_COUNTS = TRADE_COUNTS
EXP3_SEEDS = SEEDS
# A correlated (C, U) pair is needed so that corr[p, U^2] (Eq. C.8) is
# generally non-zero; CORRELATION_STRENGTH sets the weight of a common factor
# shared by C(t_i) and U(t_i) (not specified in the paper, which leaves open
# how synthetic C and U should be correlated).
CORRELATION_STRENGTH = 0.6
