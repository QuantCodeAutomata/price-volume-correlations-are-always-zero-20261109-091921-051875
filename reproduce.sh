#!/usr/bin/env bash
# Reproduce the experiments of Olkhov, "Price-Volume Correlations Are Always
# Zero". All experiments use synthetic trade data (data/generate_synthetic.py)
# and require no market-data credentials.
set -e

cd "$(dirname "$0")"

python -m pytest -m "not market_data"

python -m exp.exp1
python -m exp.exp2
python -m exp.exp3

echo "Results written to results/"
