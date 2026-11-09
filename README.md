# Price-Volume Correlations Are Always Zero

Replication of: Olkhov, V., "Price-Volume Correlations Are Always Zero."

This repository reproduces the paper's three core claims about the
correlation between trade prices and trade volumes of consecutive market
trades:

1. **The economically justified price-volume correlation `corr[p, U]` is
   identically zero** (Sec. 4, Eq. 4.1-4.3; App. A, Eq. A.8-A.11), once the
   mean price in the definition of correlation is taken to be the
   volume-weighted average price (VWAP), as required by the trade equation
   `C(t_i) = p(t_i) U(t_i)` (Eq. A.2).
2. **The conventional textbook correlation estimator**, which uses the
   simple average price `pi(t)` instead of VWAP, **does not measure this
   zero correlation**: it equals `[p(t) - pi(t)] U(t)` (Sec. 4.1, Eq. 4.7-4.8;
   App. B, Eq. B.6-B.8), which is non-zero whenever trade volumes are not
   constant. This is presented as the origin of the "empirical" positive or
   negative price-volume correlations reported in the literature.
3. **Statistical interdependence beyond `corr[p, U]`** can instead be
   assessed with `corr[p, U^2]` and `corr[p^2, U^2]` (App. C, Eq. C.1-C.9,
   C.16; Sec. 4.1, Eq. 4.9), which are generally non-zero when trade values
   and volumes are statistically dependent.

All three claims are algebraic identities (or, for Eq. C.16, an approximate
relation; see "Deviations from the paper" below) that hold for **any**
distribution of trade values and volumes, not statements about a specific
market, instrument, or sample period. Accordingly, every experiment uses
synthetic time series of trade values `C(t_i)` and trade volumes `U(t_i)`
generated to satisfy the trade equation (Eq. A.2); no market data, API keys,
or network access are required.

## Repository layout

```
src/            correlations.py (paper's formulas), config.py (parameters)
data/           generate_synthetic.py (synthetic trade-data generators)
exp/            exp1.py, exp2.py, exp3.py (one script per experiment)
tests/          pytest unit and integration tests
results/        generated tables, plots and RESULTS.md
```

## Setup

```bash
pip install -r requirements.txt
```

## Running the tests

```bash
python -m pytest -m "not market_data"
```

(No test in this repository is marked `market_data`: all experiments use
synthetic data only.)

## Running the experiments

```bash
python -m exp.exp1   # corr[p, U] is always zero (Eq. 4.3 / A.11)
python -m exp.exp2   # the conventional estimator is misleading (Eq. 4.8 / B.8)
python -m exp.exp3   # corr[p, U^2] and corr[p^2, U^2] (App. C)
```

or run everything (tests, then all three experiments) with:

```bash
./reproduce.sh
```

Each script writes its tables and plots directly to `results/`, overwriting
the files from any previous run. `results/RESULTS.md` compares the numbers
with the paper's stated results.

## Method

Each experiment generates `N` consecutive synthetic trades during an
averaging interval `Delta` (Eq. A.1) by drawing non-degenerate random trade
volumes `U(t_i)` and trade values `C(t_i)` from several distributions
(lognormal, uniform, gamma, exponential for volumes; normal, lognormal,
uniform for values — the paper does not prescribe a distribution), and
setting `p(t_i) = C(t_i) / U(t_i)` so that the trade equation (Eq. A.2) holds
exactly. The number of trades `N in {50, 100, 500, 1000}` and five random
seeds are swept for every distribution combination (`src/config.py`), and a
constant-volume case (`U(t_i) = U`) is included throughout, since the paper
treats it as a special limit (Eq. 3.2 / A.7).

## Deviations and unresolved details

See `results/RESULTS.md` for the "Deviations from the paper" and
"Not reproduced" sections.
