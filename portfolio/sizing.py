"""
Layer 5b — Volatility-Targeted Position Sizing
Scales blended positions to a target daily volatility, then applies two
additional haircuts:
  1. Vol-regime scalar  — reduces exposure when realised vol spikes above
                          its 6-month average (avoids over-trading in crises)
  2. Correlation scalar — reduces exposure when FX pairs move together
                          (correlated bets don't add diversification)
"""

import os
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import SIGNAL_PARAMS, RETURNS_CSV, OUTPUT_DIR


def apply_kelly(blended_signal, raw_returns):
    """
    Vol-targeted position sizing with regime and correlation adjustments.
    Function name kept for compatibility with main.py.
    Returns (portfolio_returns, scale_series).
    """
    returns = pd.read_csv(RETURNS_CSV, index_col=0, parse_dates=True)
    aligned_returns = returns.reindex(blended_signal.index).ffill()

    target_vol  = SIGNAL_PARAMS["vol_target_daily"]
    scale_cap   = SIGNAL_PARAMS["vol_scale_cap"]
    vol_window  = SIGNAL_PARAMS["vol_regime_window"]
    ref_window  = SIGNAL_PARAMS["vol_regime_ref_window"]
    vol_thresh  = SIGNAL_PARAMS["vol_regime_threshold"]
    corr_window = SIGNAL_PARAMS["corr_window"]

    # ------------------------------------------------------------------
    # 1. Vol targeting: scale so blended portfolio hits target daily vol
    # ------------------------------------------------------------------
    port_vol = raw_returns.rolling(vol_window).std().fillna(raw_returns.std())
    vol_scale = (target_vol / port_vol).clip(0, scale_cap)

    # ------------------------------------------------------------------
    # 2. Vol-regime scalar: reduce when current vol >> long-run vol
    # ------------------------------------------------------------------
    ref_vol = raw_returns.rolling(ref_window).std().fillna(raw_returns.std())
    vol_ratio = (port_vol / ref_vol).fillna(1.0)
    regime_scalar = (vol_thresh / vol_ratio).clip(0, 1.0)

    # ------------------------------------------------------------------
    # 3. Correlation scalar: reduce when pairs are moving in lockstep
    # ------------------------------------------------------------------
    pairs = aligned_returns.columns.tolist()
    n = len(pairs)
    corr_sum = pd.Series(0.0, index=aligned_returns.index)
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            corr_sum += (
                aligned_returns.iloc[:, i]
                .rolling(corr_window)
                .corr(aligned_returns.iloc[:, j])
            )
            count += 1
    avg_corr = (corr_sum / count).clip(0, 1).fillna(0.3)
    corr_scalar = 1.0 - 0.4 * avg_corr   # at avg corr=1 → 60% of normal size

    # ------------------------------------------------------------------
    # 4. Combined scale and scaled portfolio returns
    # ------------------------------------------------------------------
    total_scale = vol_scale * regime_scalar * corr_scalar
    scaled_positions = blended_signal.multiply(total_scale, axis=0)

    pair_returns = scaled_positions * aligned_returns
    portfolio_returns = pair_returns.mean(axis=1).dropna()

    print(f"\nVol-targeting stats (rolling {vol_window}-day vol window):")
    print(f"  Avg vol scale    : {vol_scale.mean():.3f}   (target {target_vol*100:.2f}% daily)")
    print(f"  Avg regime scalar: {regime_scalar.mean():.3f}   (threshold {vol_thresh}×)")
    print(f"  Avg corr scalar  : {corr_scalar.mean():.3f}   (avg pair corr {avg_corr.mean():.3f})")
    print(f"  Avg total scale  : {total_scale.mean():.3f}")

    return portfolio_returns, total_scale.rename("scale").to_frame()


if __name__ == "__main__":
    print("Run this module via main.py or after blender.py.")
