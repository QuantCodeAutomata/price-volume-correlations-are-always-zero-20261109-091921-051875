# Results — Price-Volume Correlations Are Always Zero

Paper: Olkhov, V., "Price-Volume Correlations Are Always Zero."

All experiments use synthetic time series of trade values `C(t_i)` and trade
volumes `U(t_i)` generated to satisfy the trade equation `C(t_i) = p(t_i)
U(t_i)` (Eq. A.2), swept over `N in {50, 100, 500, 1000}` consecutive trades,
5 random seeds, 4 volume distributions (lognormal, uniform, gamma,
exponential) and 3 value distributions (positive-normal, lognormal,
uniform), plus the constant-volume limiting case (Eq. 3.2 / A.7). No market
data is used because the paper's claims are algebraic identities that hold
for any distribution of trades, not statements about a particular market.
Commands: `python -m exp.exp1`, `python -m exp.exp2`, `python -m exp.exp3`
(see `README.md`); every number below is taken from the CSV/PNG files these
commands wrote to this directory.

## Experiment 1 — Verification that the true price-volume correlation is always zero

Tests the identity `corr[p, U] = E[p(t_i) U(t_i)] - E[p(t_i)] E[U(t_i)] ≡ 0`
(Eq. 4.3 / A.11), computed as `(1/N) sum p(t_i) U(t_i) - p(t) U(t)` with
`p(t)` the VWAP (Eq. A.6), across 600 trials (`results/exp1_results.csv`) and
both the random-volume and constant-volume cases (Eq. A.7).

| Metric | Value | Paper claim | Verdict |
|---|---|---|---|
| Max \|corr[p,U]\| relative to `p(t) U(t)`, random-volume case | 2.226e-16 | ≡ 0 (Eq. 4.3/A.11) | reproduced |
| Max \|corr[p,U]\| relative to `p(t) U(t)`, constant-volume case | 2.214e-16 | ≡ 0 (Eq. 3.2/A.7, A.11) | reproduced |

Both maxima are at the scale of IEEE-754 double-precision round-off
(~1e-16), i.e. the measured correlation is zero up to floating-point
arithmetic, for every one of the 600 trials, every distribution, and both
the random- and constant-volume cases. Figure: `exp1_zero_correlation.png`
(the true correlation plotted against `N`; all points lie on zero within
plotting precision). Table: `exp1_results.csv`; summary: `exp1_summary.csv`.

## Experiment 2 — The conventional empirical correlation formula is misleading

Tests that the conventional textbook estimator `corr[p,U]_e = (1/N)
sum(p_i - pi(t))(U_i - U(t))` (Eq. 4.4/B.2), which uses the simple average
price `pi(t)` instead of VWAP, equals the closed form `[p(t) - pi(t)] U(t)`
(Eq. 4.8/B.8), and is non-zero only when trade volumes vary (600 trials,
`results/exp2_results.csv`).

| Metric | Value | Paper claim | Verdict |
|---|---|---|---|
| Max \|estimator − closed form\| (Eq. 4.7 vs. 4.8/B.8) | 1.455e-11 | identity, ≡ 0 | reproduced |
| Max \|corr[p,U]_e\|, constant-volume case | 0.0 | = 0 when `U(t_i)=U` (Eq. 4.8, `p(t)=pi(t)`) | reproduced |
| Median \|corr[p,U]_e\|, random-volume case | 816.106 | ≠ 0 in general | reproduced |
| Max \|true corr[p,U]\| (Eq. 4.3/A.11, re-verified here) | 4.547e-13 | ≡ 0 | reproduced |

The conventional estimator is non-zero (median magnitude 816.1, in the same
run's arbitrary value/volume units) for every random-volume trial, while the
true price-volume correlation (Eq. 4.3/A.11) stays at floating-point zero in
the same trials, directly illustrating the paper's claim that the
conventional estimator measures the gap `p(t) - pi(t)` between VWAP and the
simple average price rather than a genuine correlation. Figure:
`exp2_spurious_correlation.png` (estimator vs. its closed form, and
estimator vs. `p(t)-pi(t)`). Table: `exp2_results.csv`; summary:
`exp2_summary.csv`.

## Experiment 3 — Alternative correlations corr[p, U²] and corr[p², U²]

Tests the closed form of `corr[p, U^2]` (Eq. C.8) and the relation between
`corr[p, U^2]` and `corr[p^2, U^2]` (Eq. 4.9/C.16), using 1,080 trials with
trade values and volumes sharing a common random factor so that
`phi(C,U) != 0` in general (`results/exp3_results.csv`), plus an independent
case and the constant-volume limit.

| Metric | Value | Paper claim | Verdict |
|---|---|---|---|
| Max \|corr[p,U²] direct − Eq. C.8 closed form\| | 1.164e-10 | identity, ≡ 0 | reproduced |
| Median \|corr[p,U²]\|, correlated C,U | 8333.23 | non-zero, unlike corr[p,U] | reproduced |
| Max χ²(U), constant-volume case | 0.0 | 0 by construction | reproduced |
| Median ratio `corr[p²,U²]` direct (Eq. C.9) / Eq. C.16 closed form, correlated C,U | 3.308 | paper's Eq. 4.9/C.16 implies ratio ≡ 1 | partially reproduced |

`corr[p, U^2]` (Eq. C.8) is confirmed as an exact algebraic identity of the
trade equation (Eq. A.2): the direct empirical computation
`E[p(t_i) U(t_i)^2] - E[p(t_i)] E[U(t_i)^2]` matches the closed form
`[phi(C,U) - chi^2(U)] p(t) U(t)^2` to within 1.2e-10, i.e. floating-point
precision, in every trial, and is confirmed non-zero (median magnitude
8333.2) whenever trade values and volumes are correlated, unlike
`corr[p,U]`.

The relation `corr[p^2, U^2] = 2 corr[p, U^2] p(t)` (Eq. 4.9/C.16) does not
hold as an exact identity in this implementation: the ratio of
`corr[p^2,U^2]` computed directly from the raw trade series by its defining
equation (`E[p(t_i)^2 U(t_i)^2] - E[p(t_i)^2] E[U(t_i)^2]`, Eq. C.9) to the
value obtained from the paper's closed form (Eq. C.16) has a median of 3.31
across the swept distributions and sample sizes, not 1. Substituting the
exact algebra of the trade equation into Eq. C.9 shows that an exact match
would require the price variance coefficient `mu^2(p)` to equal
`[psi^2(C) - 2 phi(C,U) + 2 chi^2(U)] / [1+chi^2(U)]`, whereas Eq. C.14 (the
"market-based variance" expression the paper cites from Olkhov (2022, 2023)
for `mu^2(p)` and substitutes into Eq. C.16) uses `chi^2(U)` with coefficient
1, not 2, in the same numerator — a first-order (delta-method) variance
approximation for the ratio `p(t_i)=C(t_i)/U(t_i)`, rather than an exact
restatement of Eq. C.9. This is a property of the derivation chain
(Eq. C.9 → C.11–C.15 → C.16) itself, confirmed symbolically and numerically
in this replication, not an implementation discrepancy: the identity
(Eq. C.8) it depends on for `corr[p,U^2]` is independently verified exact
above. See "Deviations from the paper" below.

Figure: `exp3_interdependence.png` (left: `corr[p,U^2]` direct vs. Eq. C.8,
on the identity line as expected; right: `corr[p^2,U^2]` direct vs. the
Eq. 4.9/C.16 closed form, with the measured ~2x gap). Table:
`exp3_results.csv`; summary: `exp3_summary.csv`.

## Deviations from the paper

- **No market data.** All three experiments test algebraic identities that
  the paper derives for arbitrary trade value and volume time series; the
  paper does not apply them to a specific traded instrument or sample
  period. No OHLCV bars, trades, quotes, or other market data were loaded;
  `data/generate_synthetic.py` is the only data source, and no API key is
  required.
- **Distributions and trial counts are not specified by the paper.** The
  paper requires only that `N` be "large enough to support reliable
  assessment of at least the first two statistical moments" and gives no
  specific distribution for `C(t_i)` or `U(t_i)`, no seed, and no trial
  count (paper Sec. 4, App. A). This replication uses `N in {50, 100, 500,
  1000}`, 5 seeds, 4 volume distributions and 3 value distributions
  (`src/config.py`), chosen to cover qualitatively different shapes
  (skewed, bounded, light/heavy-tailed) rather than taken from the paper.
- **The correlation strength between C(t_i) and U(t_i) used in Experiment 3
  is not specified by the paper.** `data/generate_synthetic.py` mixes trade
  values and volumes with a common random factor of weight
  `CORRELATION_STRENGTH = 0.6` (`src/config.py`) so that `corr[p, U^2]`
  (Eq. C.8) is generally non-zero, since the paper does not prescribe how
  synthetic `C` and `U` should be made statistically dependent.
- **Eq. 4.9/C.16 (the relation between `corr[p,U^2]` and `corr[p^2,U^2]`)
  reproduces only approximately.** As detailed in the Experiment 3 results
  above, the paper derives Eq. C.16 by substituting a first-order
  ("market-based variance", Eq. C.14, cited from Olkhov 2022; 2023)
  approximation for the price variance coefficient `mu^2(p)`, rather than
  the exact coefficient implied by the trade equation and Eq. C.9. The
  measured ratio between the two computations of `corr[p^2,U^2]` (median
  3.31, swept over all configurations) reflects this approximation in the
  paper's own derivation chain, not a missing or substituted input on our
  part; `corr[p, U^2]` (Eq. C.8), which this replication also verifies
  exactly, does not depend on this approximation.

## Not reproduced

Nothing in the scope of the selected experiments was left unreproduced. All
three experiments (verification of `corr[p,U]≡0`, the conventional
estimator's spurious non-zero value, and the alternative correlations
`corr[p,U^2]`/`corr[p^2,U^2]`) produced the artifacts and metrics listed
above; the only partial result is the measured 1:3.31 instead of 1:1 ratio
in Eq. 4.9/C.16, documented as a deviation above rather than as an
unreproduced claim, since the paper's own derivation of that equation relies
on an approximation whose effect is what the experiment measures.
