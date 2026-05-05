"""
Layer 5b — Kelly Criterion Position Sizing
Scales blended strategy positions by rolling Kelly fraction, capped at 0.25.
"""

import os
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import SIGNAL_PARAMS, RETURNS_CSV, OUTPUT_DIR


def apply_kelly(blended_signal, raw_returns):
    """
    Scale blended signal by per-pair Kelly fraction computed on a rolling window.
    Returns kelly_returns (portfolio-level) and kelly_fractions (DataFrame).
    """
    window  = SIGNAL_PARAMS["kelly_window"]
    cap     = SIGNAL_PARAMS["kelly_cap"]

    returns = pd.read_csv(RETURNS_CSV, index_col=0, parse_dates=True)
    aligned_returns = returns.reindex(blended_signal.index).ffill()

    # Rolling Kelly fraction per pair: f* = mu / sigma^2
    roll_mu    = aligned_returns.rolling(window).mean()
    roll_var   = aligned_returns.rolling(window).var()
    kelly_frac = (roll_mu / roll_var).clip(0, cap)   # long-only sizing; floor at 0

    # Kelly-scaled positions
    kelly_positions = blended_signal * kelly_frac

    # Portfolio return: sum of (kelly_position × return) per pair, normalised
    pair_returns = kelly_positions * aligned_returns
    kelly_portfolio = pair_returns.mean(axis=1).dropna()

    print(f"\nKelly fraction stats (rolling {window}-day, cap={cap}):")
    print(kelly_frac.describe().round(4))

    return kelly_portfolio, kelly_frac


if __name__ == "__main__":
    print("Run this module via main.py or after blender.py.")
