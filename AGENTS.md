# AGENTS.md

Guide for coding agents and contributors working in this repository.

## What this repository implements

A reproduction of Olkhov, "Price-Volume Correlations Are Always Zero":
the claim that the economically justified price-volume correlation
`corr[p, U]` of consecutive market trades is identically zero (paper
Sec. 4, Eq. 4.1-4.3; App. A, Eq. A.2, A.8-A.11), that the conventional
textbook correlation estimator instead measures the gap between the
volume-weighted average price and the simple average price (Sec. 4.1,
Eq. 4.4-4.8; App. B, Eq. B.1-B.9), and that the alternative correlations
`corr[p, U^2]` and `corr[p^2, U^2]` capture statistical interdependence that
`corr[p, U]` cannot (App. C, Eq. C.1-C.16; Eq. 4.9).

Every claim is an algebraic property of the trade equation
`C(t_i) = p(t_i) U(t_i)` (Eq. A.2) that holds for any distribution of trade
values and volumes, so the repository uses synthetic trade data only; no
market data, data provider, or API key is required.

## Layout

```
src/correlations.py   Paper's correlation formulas (Eq. 3.1-4.9, A.1-C.16)
src/config.py         Module-level experiment parameters (N, seeds, distributions)
data/generate_synthetic.py   Synthetic trade-data generators (not market data)
exp/exp1.py            Experiment 1: corr[p, U] is always zero
exp/exp2.py            Experiment 2: the conventional estimator is misleading
exp/exp3.py            Experiment 3: corr[p, U^2] and corr[p^2, U^2]
tests/                 pytest unit tests (src/, data/) and integration tests (exp/)
results/               Generated tables, plots, and RESULTS.md
```

## Conventions

- All experiment parameters live in `src/config.py` as module-level
  `UPPER_CASE` variables, each with a one-line comment on its meaning, unit,
  and the paper section it comes from (or "not specified in the paper").
  No parameter is hardcoded elsewhere.
- Synthetic data is generated only in `data/generate_synthetic.py`; no
  generator lives in `src/` or `exp/`.
- `exp/*.py` scripts call only the public functions of `src/correlations.py`
  and `data/generate_synthetic.py` with the signatures as written.
- This repository uses no market data and defines no `market_data`-marked
  tests; `.env`/API keys do not apply here.

## Commands

Setup:

```bash
pip install -r requirements.txt
```

Tests:

```bash
python -m pytest -m "not market_data"
```

Experiments (synthetic data, no `--full` flag needed since no universe or
download is involved):

```bash
python -m exp.exp1
python -m exp.exp2
python -m exp.exp3
```

Everything at once:

```bash
./reproduce.sh
```

## Notes for future changes

- Re-running any `exp/*.py` script overwrites the corresponding files in
  `results/`. The numbers and verdicts in `results/RESULTS.md` are not
  regenerated automatically: after a rerun, check them against the new
  files in `results/` and update `results/RESULTS.md` by hand.
- `src/correlations.py::corr_p2_u2_formula` implements the paper's closed
  form of Eq. C.16/4.9 literally. `exp/exp3.py` measures, rather than
  asserts, how closely this closed form matches the value of
  `corr[p^2, U^2]` computed directly from raw trade series (Eq. C.9); see
  `results/RESULTS.md` for the measured relationship.
