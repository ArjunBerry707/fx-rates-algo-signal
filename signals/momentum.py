"""
Layer 3 — Momentum Signal
12-1 month cross-sectional momentum: long top 2, short bottom 2 pairs.
Dollar-neutral construction.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import (
    SIGNAL_PARAMS, PRICES_CSV, RETURNS_CSV, OUTPUT_DIR,
    compute_max_drawdown, compute_sharpe,
    compute_annualised_return, compute_annualised_vol,
)


def run(carry_returns=None):
    prices  = pd.read_csv(PRICES_CSV,  index_col=0, parse_dates=True)
    returns = pd.read_csv(RETURNS_CSV, index_col=0, parse_dates=True)

    window    = SIGNAL_PARAMS["momentum_window"]  # 231 trading days
    skip      = SIGNAL_PARAMS["momentum_skip"]    # 21 trading days
    long_n    = SIGNAL_PARAMS["momentum_long_n"]
    short_n   = SIGNAL_PARAMS["momentum_short_n"]

    # ------------------------------------------------------------------
    # 1. 12-1 month momentum score for each pair
    # Cumulative return from t-252 to t-21 (skipping last month)
    # ------------------------------------------------------------------
    momentum_score = prices.shift(skip).pct_change(window)

    # 2. Cross-sectional ranks each day (ascending → low score = rank 1)
    ranks = momentum_score.rank(axis=1, method="first")
    n_pairs = prices.shape[1]

    # 3. Dollar-neutral weights
    weights = pd.DataFrame(0.0, index=ranks.index, columns=ranks.columns)
    for idx in range(len(ranks)):
        row = ranks.iloc[idx]
        if row.isna().all():
            continue
        long_mask  = row >= (n_pairs - long_n + 1)    # top 2 ranks
        short_mask = row <= short_n                     # bottom 2 ranks
        n_long  = long_mask.sum()
        n_short = short_mask.sum()
        row_label = ranks.index[idx]
        if n_long > 0:
            weights.loc[row_label, long_mask]  =  1.0 / n_long
        if n_short > 0:
            weights.loc[row_label, short_mask] = -1.0 / n_short

    # 4. Lag weights by 1 day
    weights_lagged = weights.shift(1)

    # 5. Portfolio returns
    aligned_returns = returns.reindex(weights_lagged.index).ffill()
    strat_returns_per_pair = weights_lagged * aligned_returns
    portfolio_returns = strat_returns_per_pair.sum(axis=1).dropna()
    portfolio_returns = portfolio_returns[portfolio_returns.index >= aligned_returns.dropna().index[0]]

    # 6. Rolling 252-day Sharpe
    rolling_sharpe = (
        portfolio_returns.rolling(252).mean()
        / portfolio_returns.rolling(252).std()
    ) * np.sqrt(252)

    # ------------------------------------------------------------------
    # 7. Plots
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    fig.suptitle("Layer 3 — Momentum Signal", fontsize=14, fontweight="bold")

    # (a) Cumulative returns
    ax = axes[0]
    cum_strat = (1 + portfolio_returns).cumprod()
    cum_bah   = (1 + aligned_returns.mean(axis=1).dropna()).cumprod()
    cum_strat.plot(ax=ax, label="Momentum Strategy", color="mediumseagreen", linewidth=1.5)
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

    # (c) Signal distribution (weight sign)
    ax = axes[2]
    signal_sign = np.sign(weights_lagged)
    dist = {}
    for col in signal_sign.columns:
        counts = signal_sign[col].value_counts().reindex([-1, 0, 1], fill_value=0)
        dist[col] = counts
    dist_df = pd.DataFrame(dist).T
    dist_df.plot(kind="bar", ax=ax, color=["tomato", "lightgray", "mediumseagreen"],
                 edgecolor="black", linewidth=0.5)
    ax.set_title("Position Direction Distribution per Pair")
    ax.set_xlabel("Pair")
    ax.set_ylabel("Count of Days")
    ax.set_xticklabels(dist_df.index, rotation=30)
    ax.legend(["-1 (Short)", "0 (Flat/Neutral)", "+1 (Long)"])
    ax.grid(alpha=0.3, axis="y")

    plt.tight_layout()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fig.savefig(f"{OUTPUT_DIR}/momentum_signal.png", dpi=150)
    plt.show()
    print(f"Chart saved: {OUTPUT_DIR}/momentum_signal.png")

    # ------------------------------------------------------------------
    # 8. Correlation matrix with carry
    # ------------------------------------------------------------------
    if carry_returns is not None:
        _plot_correlation(carry_returns, portfolio_returns)

    # ------------------------------------------------------------------
    # 9. Summary
    # ------------------------------------------------------------------
    ann_ret = compute_annualised_return(portfolio_returns)
    ann_vol = compute_annualised_vol(portfolio_returns)
    sharpe  = compute_sharpe(portfolio_returns)
    mdd     = compute_max_drawdown(portfolio_returns)

    print("\n=== MOMENTUM SIGNAL SUMMARY ===")
    print(f"Annualised Return : {ann_ret*100:.2f}%")
    print(f"Annualised Vol    : {ann_vol*100:.2f}%")
    print(f"Sharpe Ratio      : {sharpe:.3f}")
    print(f"Max Drawdown      : {mdd*100:.2f}%")

    return portfolio_returns, weights_lagged


def _plot_correlation(carry_returns, momentum_returns):
    common = carry_returns.align(momentum_returns, join="inner")
    corr_df = pd.DataFrame({
        "Carry":    common[0],
        "Momentum": common[1],
    }).corr()

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(corr_df, annot=True, fmt=".3f", cmap="coolwarm",
                center=0, ax=ax, vmin=-1, vmax=1, linewidths=0.5)
    ax.set_title("Strategy Return Correlations")
    plt.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/correlation_carry_momentum.png", dpi=150)
    plt.show()
    print(f"Correlation chart saved: {OUTPUT_DIR}/correlation_carry_momentum.png")


if __name__ == "__main__":
    run()
