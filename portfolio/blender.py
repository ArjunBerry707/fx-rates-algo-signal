"""
Layer 5a — Signal Blender
Equal-weight combination of carry, momentum, and mean-reversion signals.
Applies transaction costs and slippage on trade days.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import (
    SIGNAL_PARAMS, RETURNS_CSV, OUTPUT_DIR,
    compute_max_drawdown, compute_sharpe,
    compute_annualised_return, compute_annualised_vol,
)


def run(carry_returns, carry_signal,
        momentum_returns, momentum_weights,
        mr_returns, mr_signal):

    returns = pd.read_csv(RETURNS_CSV, index_col=0, parse_dates=True)

    tc  = SIGNAL_PARAMS["transaction_cost_bps"]   # 0.0002
    slip = SIGNAL_PARAMS["slippage_bps"]           # 0.0001

    # ------------------------------------------------------------------
    # 1. Align signals to a common index and blend equally
    # ------------------------------------------------------------------
    # Normalise momentum weights to signal-space [-1, 1] via sign; carry & MR are already {-1, 0, 1}
    # For blending we use the raw weight/signal value from each layer
    common_idx = carry_signal.index \
        .intersection(momentum_weights.index) \
        .intersection(mr_signal.index)

    c_sig  = carry_signal.reindex(common_idx)
    m_sig  = momentum_weights.reindex(common_idx)   # already dollar-neutral weights summing to ±1 per side
    mr_sig = mr_signal.reindex(common_idx)

    # Normalise momentum weights to [-1, 1] range per pair before averaging
    m_sig_norm = m_sig.clip(-1, 1)

    blended = (c_sig + m_sig_norm + mr_sig) / 3.0

    # ------------------------------------------------------------------
    # 2. Transaction costs and slippage
    # ------------------------------------------------------------------
    # Trade occurs when blended signal changes sign or goes to/from zero
    prev_sign = np.sign(blended.shift(1))
    curr_sign = np.sign(blended)
    trade_mask = (prev_sign != curr_sign)   # True on days a trade fires

    aligned_returns = returns.reindex(common_idx).ffill()

    # Raw blended returns (signal × return, averaged across pairs)
    raw_pair_returns = blended * aligned_returns
    portfolio_raw    = raw_pair_returns.mean(axis=1)

    # Subtract costs on trade days (per pair, then averaged)
    cost_per_pair = trade_mask.astype(float) * (tc + slip)
    cost_portfolio = cost_per_pair.mean(axis=1)

    portfolio_returns = (portfolio_raw - cost_portfolio).dropna()

    # ------------------------------------------------------------------
    # 3. Comparison chart: all 4 strategies
    # ------------------------------------------------------------------
    _plot_comparison(carry_returns, momentum_returns, mr_returns, portfolio_returns)

    # ------------------------------------------------------------------
    # 4. Comparison table
    # ------------------------------------------------------------------
    _print_comparison_table(carry_returns, momentum_returns, mr_returns, portfolio_returns)

    return portfolio_returns, blended


def _plot_comparison(carry_r, mom_r, mr_r, blended_r):
    fig, ax = plt.subplots(figsize=(14, 7))

    def _cum(r):
        return (1 + r.dropna()).cumprod()

    _cum(carry_r).plot(ax=ax,   label="Carry",         color="steelblue",    linewidth=1.2)
    _cum(mom_r).plot(ax=ax,     label="Momentum",       color="mediumseagreen", linewidth=1.2)
    _cum(mr_r).plot(ax=ax,      label="Mean-Reversion", color="mediumpurple",  linewidth=1.2)
    _cum(blended_r).plot(ax=ax, label="Blended (after costs)", color="darkorange", linewidth=2)

    ax.set_title("Cumulative Returns — All Strategies", fontsize=13, fontweight="bold")
    ax.set_ylabel("Growth of $1")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fig.savefig(f"{OUTPUT_DIR}/strategy_comparison.png", dpi=150)
    plt.show()
    print(f"Comparison chart saved: {OUTPUT_DIR}/strategy_comparison.png")


def _print_comparison_table(carry_r, mom_r, mr_r, blended_r):
    rows = {}
    for name, r in [("Carry", carry_r), ("Momentum", mom_r),
                    ("Mean-Reversion", mr_r), ("Blended", blended_r)]:
        r = r.dropna()
        rows[name] = {
            "Ann. Return (%)": round(compute_annualised_return(r) * 100, 2),
            "Ann. Vol (%)":    round(compute_annualised_vol(r)    * 100, 2),
            "Sharpe":          round(compute_sharpe(r),                  3),
            "Max Drawdown (%)":round(compute_max_drawdown(r)      * 100, 2),
        }
    df = pd.DataFrame(rows).T
    print("\n=== STRATEGY COMPARISON TABLE ===")
    print(df.to_string())


if __name__ == "__main__":
    print("Run this module via main.py or layer-by-layer.")
