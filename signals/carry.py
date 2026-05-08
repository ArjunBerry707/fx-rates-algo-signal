"""
Layer 2 — Carry Signal
Generates trading signals from interest rate differentials.
Long the pair when foreign rate > USD rate, short when the reverse.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import (
    SIGNAL_PARAMS, RETURNS_CSV, RATE_DIFF_CSV, VIX_CSV, OUTPUT_DIR,
    compute_max_drawdown, compute_sharpe,
    compute_annualised_return, compute_annualised_vol,
)


def run():
    returns   = pd.read_csv(RETURNS_CSV,  index_col=0, parse_dates=True)
    rate_diff = pd.read_csv(RATE_DIFF_CSV, index_col=0, parse_dates=True)

    threshold = SIGNAL_PARAMS["carry_threshold"]

    # ------------------------------------------------------------------
    # 1. Raw carry signal
    # ------------------------------------------------------------------
    signal = np.sign(rate_diff)
    signal[rate_diff.abs() < threshold] = 0      # flat when differential is tiny

    # 2. Lag by 1 day (no look-ahead bias)
    signal_lagged = signal.shift(1)

    # 2b. VIX regime filter: go flat when market is in risk-off mode
    vix_threshold = SIGNAL_PARAMS["vix_risk_off"]
    try:
        vix = pd.read_csv(VIX_CSV, index_col=0, parse_dates=True).squeeze()
        risk_off = vix.reindex(signal_lagged.index).ffill() > vix_threshold
        signal_lagged[risk_off] = 0
        n_filtered = int(risk_off.sum())
        pct = n_filtered / len(signal_lagged) * 100
        print(f"  VIX regime filter: {n_filtered} days flat ({pct:.1f}% of history, VIX > {vix_threshold})")
    except FileNotFoundError:
        print("  WARNING: VIX data not found — run data ingestion first to enable regime filter")

    # 3. Strategy returns
    aligned_returns = returns.reindex(signal_lagged.index).ffill()
    strat_returns_per_pair = signal_lagged * aligned_returns
    portfolio_returns = strat_returns_per_pair.mean(axis=1).dropna()

    # 4. Rolling 252-day Sharpe
    rolling_sharpe = (
        portfolio_returns.rolling(252).mean()
        / portfolio_returns.rolling(252).std()
    ) * np.sqrt(252)

    # ------------------------------------------------------------------
    # 5. Plots
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    fig.suptitle("Layer 2 — Carry Signal", fontsize=14, fontweight="bold")

    # (a) Cumulative returns
    ax = axes[0]
    cum_strat = (1 + portfolio_returns).cumprod()
    cum_bah   = (1 + aligned_returns.mean(axis=1).dropna()).cumprod()
    cum_strat.plot(ax=ax, label="Carry Strategy", color="steelblue", linewidth=1.5)
    cum_bah.plot(ax=ax, label="Buy & Hold (equal weight)", color="gray", linewidth=1, linestyle="--")
    ax.set_title("Cumulative Returns")
    ax.set_ylabel("Growth of $1")
    ax.legend()
    ax.grid(alpha=0.3)

    # (b) Rolling Sharpe
    ax = axes[1]
    rolling_sharpe.plot(ax=ax, color="darkorange", linewidth=1.2)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Rolling 252-Day Sharpe Ratio")
    ax.set_ylabel("Sharpe")
    ax.grid(alpha=0.3)

    # (c) Signal distribution
    ax = axes[2]
    dist = {}
    for col in signal_lagged.columns:
        counts = signal_lagged[col].value_counts().reindex([-1, 0, 1], fill_value=0)
        dist[col] = counts
    dist_df = pd.DataFrame(dist).T
    dist_df.plot(kind="bar", ax=ax, color=["tomato", "lightgray", "mediumseagreen"],
                 edgecolor="black", linewidth=0.5)
    ax.set_title("Signal Distribution per Pair")
    ax.set_xlabel("Pair")
    ax.set_ylabel("Count of Days")
    ax.set_xticklabels(dist_df.index, rotation=30)
    ax.legend(["-1 (Short)", "0 (Flat)", "+1 (Long)"])
    ax.grid(alpha=0.3, axis="y")

    plt.tight_layout()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fig.savefig(f"{OUTPUT_DIR}/carry_signal.png", dpi=150)
    plt.show()
    print(f"Chart saved: {OUTPUT_DIR}/carry_signal.png")

    # ------------------------------------------------------------------
    # 6. Summary statistics
    # ------------------------------------------------------------------
    ann_ret = compute_annualised_return(portfolio_returns)
    ann_vol = compute_annualised_vol(portfolio_returns)
    sharpe  = compute_sharpe(portfolio_returns)
    mdd     = compute_max_drawdown(portfolio_returns)

    print("\n=== CARRY SIGNAL SUMMARY ===")
    print(f"Annualised Return : {ann_ret*100:.2f}%")
    print(f"Annualised Vol    : {ann_vol*100:.2f}%")
    print(f"Sharpe Ratio      : {sharpe:.3f}")
    print(f"Max Drawdown      : {mdd*100:.2f}%")

    return portfolio_returns, signal_lagged


if __name__ == "__main__":
    run()
