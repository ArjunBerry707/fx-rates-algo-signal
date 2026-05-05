"""
Layer 4 — Mean-Reversion Signal
Z-score of price relative to 60-day rolling mean.
Stop-loss exits after 10 days of no reversion.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import (
    SIGNAL_PARAMS, PRICES_CSV, RETURNS_CSV, OUTPUT_DIR,
    compute_max_drawdown, compute_sharpe,
    compute_annualised_return, compute_annualised_vol,
)


def run(carry_returns=None, momentum_returns=None):
    prices  = pd.read_csv(PRICES_CSV,  index_col=0, parse_dates=True)
    returns = pd.read_csv(RETURNS_CSV, index_col=0, parse_dates=True)

    z_window    = SIGNAL_PARAMS["z_window"]
    z_entry     = SIGNAL_PARAMS["z_entry"]
    z_exit      = SIGNAL_PARAMS["z_exit"]
    stop_days   = SIGNAL_PARAMS["stop_loss_days"]

    # ------------------------------------------------------------------
    # 1. Z-scores
    # ------------------------------------------------------------------
    roll_mean = prices.rolling(z_window).mean()
    roll_std  = prices.rolling(z_window).std()
    z_scores  = (prices - roll_mean) / roll_std

    # ------------------------------------------------------------------
    # 2. Signals with stop-loss
    # ------------------------------------------------------------------
    signal = _build_signal_with_stoploss(z_scores, z_entry, z_exit, stop_days)

    # 3. Lag
    signal_lagged = signal.shift(1)

    # 4. Portfolio returns
    aligned_returns = returns.reindex(signal_lagged.index).ffill()
    strat_returns_per_pair = signal_lagged * aligned_returns
    portfolio_returns = strat_returns_per_pair.mean(axis=1).dropna()

    # 5. Rolling Sharpe
    rolling_sharpe = (
        portfolio_returns.rolling(252).mean()
        / portfolio_returns.rolling(252).std()
    ) * np.sqrt(252)

    # ------------------------------------------------------------------
    # 6. Plots
    # ------------------------------------------------------------------
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Main 3-panel chart
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    fig.suptitle("Layer 4 — Mean-Reversion Signal", fontsize=14, fontweight="bold")

    ax = axes[0]
    cum_strat = (1 + portfolio_returns).cumprod()
    cum_bah   = (1 + aligned_returns.mean(axis=1).dropna()).cumprod()
    cum_strat.plot(ax=ax, label="Mean-Reversion Strategy", color="mediumpurple", linewidth=1.5)
    cum_bah.plot(ax=ax, label="Buy & Hold", color="gray", linewidth=1, linestyle="--")
    ax.set_title("Cumulative Returns")
    ax.set_ylabel("Growth of $1")
    ax.legend(); ax.grid(alpha=0.3)

    ax = axes[1]
    rolling_sharpe.plot(ax=ax, color="darkorange", linewidth=1.2)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Rolling 252-Day Sharpe Ratio")
    ax.set_ylabel("Sharpe"); ax.grid(alpha=0.3)

    ax = axes[2]
    dist = {}
    for col in signal_lagged.columns:
        counts = signal_lagged[col].value_counts().reindex([-1, 0, 1], fill_value=0)
        dist[col] = counts
    dist_df = pd.DataFrame(dist).T
    dist_df.plot(kind="bar", ax=ax, color=["tomato", "lightgray", "mediumseagreen"],
                 edgecolor="black", linewidth=0.5)
    ax.set_title("Signal Distribution per Pair")
    ax.set_xticklabels(dist_df.index, rotation=30)
    ax.legend(["-1 (Short)", "0 (Flat)", "+1 (Long)"]); ax.grid(alpha=0.3, axis="y")

    plt.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/mean_reversion_signal.png", dpi=150)
    plt.show()

    # EURUSD z-score + price chart
    _plot_eurusd_zscore(prices, z_scores, signal_lagged, z_entry)

    # 3×3 correlation matrix
    if carry_returns is not None and momentum_returns is not None:
        _plot_correlation(carry_returns, momentum_returns, portfolio_returns)

    # ------------------------------------------------------------------
    # 7. Summary
    # ------------------------------------------------------------------
    ann_ret = compute_annualised_return(portfolio_returns)
    ann_vol = compute_annualised_vol(portfolio_returns)
    sharpe  = compute_sharpe(portfolio_returns)
    mdd     = compute_max_drawdown(portfolio_returns)

    print("\n=== MEAN-REVERSION SIGNAL SUMMARY ===")
    print(f"Annualised Return : {ann_ret*100:.2f}%")
    print(f"Annualised Vol    : {ann_vol*100:.2f}%")
    print(f"Sharpe Ratio      : {sharpe:.3f}")
    print(f"Max Drawdown      : {mdd*100:.2f}%")

    return portfolio_returns, signal_lagged, z_scores


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def _build_signal_with_stoploss(z_scores, z_entry, z_exit, stop_days):
    signal = pd.DataFrame(0.0, index=z_scores.index, columns=z_scores.columns)
    for col in z_scores.columns:
        z = z_scores[col].values
        sig = np.zeros(len(z))
        holding = 0   # days currently in a position

        for i in range(1, len(z)):
            if np.isnan(z[i]):
                sig[i] = 0
                holding = 0
                continue

            prev_sig = sig[i - 1]

            if prev_sig != 0:
                holding += 1
                # Stop-loss: exit if held too long without reversion toward zero
                if holding >= stop_days and abs(z[i]) > z_exit:
                    sig[i] = 0
                    holding = 0
                # Normal exit: z-score has reverted near zero
                elif abs(z[i]) < z_exit:
                    sig[i] = 0
                    holding = 0
                else:
                    sig[i] = prev_sig  # maintain position
            else:
                # Entry conditions
                if z[i] < -z_entry:
                    sig[i] = 1
                    holding = 1
                elif z[i] > z_entry:
                    sig[i] = -1
                    holding = 1
                else:
                    sig[i] = 0
                    holding = 0

        signal[col] = sig
    return signal


def _plot_eurusd_zscore(prices, z_scores, signal_lagged, z_entry):
    pair = "EURUSD"
    if pair not in prices.columns:
        return

    fig = plt.figure(figsize=(14, 8))
    gs  = gridspec.GridSpec(2, 1, height_ratios=[1, 1.5])

    # Top: z-score
    ax1 = fig.add_subplot(gs[0])
    z_scores[pair].plot(ax=ax1, color="steelblue", linewidth=1)
    ax1.axhline( z_entry, color="tomato",       linestyle="--", linewidth=1, label=f"+{z_entry} (Short entry)")
    ax1.axhline(-z_entry, color="mediumseagreen", linestyle="--", linewidth=1, label=f"-{z_entry} (Long entry)")
    ax1.axhline(0,        color="black",         linestyle=":",  linewidth=0.8)
    ax1.set_title(f"EURUSD — 60-Day Z-Score")
    ax1.set_ylabel("Z-Score"); ax1.legend(loc="upper right"); ax1.grid(alpha=0.3)

    # Bottom: price with entry/exit markers
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    prices[pair].plot(ax=ax2, color="black", linewidth=0.8, alpha=0.7, label="EURUSD Price")

    sig = signal_lagged[pair]
    long_entries  = prices[pair][sig.diff() > 0]
    short_entries = prices[pair][sig.diff() < 0]
    exits         = prices[pair][(sig == 0) & (sig.shift(1) != 0)]

    ax2.scatter(long_entries.index,  long_entries.values,  marker="^", color="mediumseagreen", s=40, zorder=5, label="Long entry")
    ax2.scatter(short_entries.index, short_entries.values, marker="v", color="tomato",          s=40, zorder=5, label="Short entry")
    ax2.scatter(exits.index,         exits.values,         marker="x", color="gray",            s=30, zorder=5, label="Exit")
    ax2.set_title("EURUSD Price with Entry/Exit Signals")
    ax2.set_ylabel("Price"); ax2.legend(loc="upper right"); ax2.grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/eurusd_zscore.png", dpi=150)
    plt.show()
    print(f"EURUSD z-score chart saved: {OUTPUT_DIR}/eurusd_zscore.png")


def _plot_correlation(carry_returns, momentum_returns, mr_returns):
    # Align all three series to common dates
    df = pd.concat([carry_returns, momentum_returns, mr_returns], axis=1, join="inner")
    df.columns = ["Carry", "Momentum", "Mean-Rev"]
    corr = df.corr()

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(corr, annot=True, fmt=".3f", cmap="coolwarm",
                center=0, ax=ax, vmin=-1, vmax=1, linewidths=0.5)
    ax.set_title("Strategy Return Correlations (3×3)")
    plt.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/correlation_3x3.png", dpi=150)
    plt.show()
    print(f"3×3 correlation chart saved: {OUTPUT_DIR}/correlation_3x3.png")


if __name__ == "__main__":
    run()
