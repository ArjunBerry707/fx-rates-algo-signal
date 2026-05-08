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

    tc  = SIGNAL_PARAMS["transaction_cost_bps"]
    slip = SIGNAL_PARAMS["slippage_bps"]

    # ------------------------------------------------------------------
    # 1. Align signals to a common index
    # ------------------------------------------------------------------
    common_idx = carry_signal.index \
        .intersection(momentum_weights.index) \
        .intersection(mr_signal.index)

    c_sig  = carry_signal.reindex(common_idx)
    m_sig  = momentum_weights.reindex(common_idx)
    mr_sig = mr_signal.reindex(common_idx)

    m_sig_norm = m_sig.clip(-1, 1)

    aligned_returns_for_vol = pd.read_csv(RETURNS_CSV, index_col=0, parse_dates=True) \
        .reindex(common_idx).ffill()

    # ------------------------------------------------------------------
    # 1b. Vol-parity blending: weight each signal inversely by its vol
    #     so every signal contributes equal risk to the portfolio
    # ------------------------------------------------------------------
    vol_window = 63

    c_port  = (c_sig  * aligned_returns_for_vol).mean(axis=1)
    m_port  = (m_sig_norm * aligned_returns_for_vol).mean(axis=1)
    mr_port = (mr_sig * aligned_returns_for_vol).mean(axis=1)

    c_vol  = c_port.rolling(vol_window).std().fillna(c_port.std()).clip(lower=1e-8)
    m_vol  = m_port.rolling(vol_window).std().fillna(m_port.std()).clip(lower=1e-8)
    mr_vol = mr_port.rolling(vol_window).std().fillna(mr_port.std()).clip(lower=1e-8)

    inv_c  = 1.0 / c_vol
    inv_m  = 1.0 / m_vol
    inv_mr = 1.0 / mr_vol
    total_inv = inv_c + inv_m + inv_mr

    w_c  = inv_c  / total_inv
    w_m  = inv_m  / total_inv
    w_mr = inv_mr / total_inv

    print(f"\n  Vol-parity blend weights (averages over full history):")
    print(f"    Carry         : {w_c.mean()*100:.1f}%  (raw vol {c_vol.mean()*100:.3f}%)")
    print(f"    Momentum      : {w_m.mean()*100:.1f}%  (raw vol {m_vol.mean()*100:.3f}%)")
    print(f"    Mean-Reversion: {w_mr.mean()*100:.1f}%  (raw vol {mr_vol.mean()*100:.3f}%)")

    blended = (
        c_sig.multiply(w_c, axis=0)
        + m_sig_norm.multiply(w_m, axis=0)
        + mr_sig.multiply(w_mr, axis=0)
    )

    # ------------------------------------------------------------------
    # 2. Transaction costs and slippage
    # ------------------------------------------------------------------
    # Trade occurs when blended signal changes sign or goes to/from zero
    prev_sign = np.sign(blended.shift(1))
    curr_sign = np.sign(blended)
    trade_mask = (prev_sign != curr_sign)   # True on days a trade fires

    aligned_returns = aligned_returns_for_vol

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
