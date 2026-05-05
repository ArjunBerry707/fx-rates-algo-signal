"""
Layer 6 — Performance Tearsheet
Full professional tearsheet for the blended strategy, saved as PDF.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.backends.backend_pdf as pdf_backend
import seaborn as sns

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import (
    RETURNS_CSV, TEARSHEET_PDF, OUTPUT_DIR,
    compute_max_drawdown, compute_sharpe,
    compute_annualised_return, compute_annualised_vol,
)


def run(blended_returns):
    returns = blended_returns.dropna()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Benchmark: equally-weighted buy-and-hold
    raw = pd.read_csv(RETURNS_CSV, index_col=0, parse_dates=True)
    bah_returns = raw.reindex(returns.index).mean(axis=1).dropna()

    metrics = _compute_metrics(returns)
    _print_metrics(metrics)

    with pdf_backend.PdfPages(TEARSHEET_PDF) as pdf:
        _page_summary_and_drawdown(returns, bah_returns, metrics, pdf)
        _page_rolling_metrics(returns, pdf)
        _page_monthly_heatmap(returns, pdf)
        _page_trade_stats(returns, metrics, pdf)

    print(f"\nTearsheet saved: {TEARSHEET_PDF}")


# ------------------------------------------------------------------
# Metric computation
# ------------------------------------------------------------------
def _compute_metrics(r):
    cum = (1 + r).cumprod()
    roll_max = cum.cummax()
    drawdown = (cum - roll_max) / roll_max

    ann_ret = compute_annualised_return(r)
    ann_vol = compute_annualised_vol(r)
    sharpe  = compute_sharpe(r)
    mdd     = drawdown.min()

    # Sortino
    downside = r[r < 0].std() * np.sqrt(252)
    sortino  = ann_ret / downside if downside > 0 else np.nan

    # Calmar
    calmar = ann_ret / abs(mdd) if mdd != 0 else np.nan

    # Drawdown duration
    in_dd = drawdown < 0
    dd_starts = in_dd & ~in_dd.shift(1).fillna(False)
    dd_ends   = ~in_dd & in_dd.shift(1).fillna(False)
    durations = []
    start = None
    for i, idx in enumerate(drawdown.index):
        if dd_starts.loc[idx]:
            start = i
        if dd_ends.loc[idx] and start is not None:
            durations.append(i - start)
            start = None
    avg_dd_dur = np.mean(durations) if durations else 0

    # Win rate
    win_rate    = (r > 0).mean()
    avg_win     = r[r > 0].mean()
    avg_loss    = r[r < 0].mean()
    gross_profit = r[r > 0].sum()
    gross_loss   = abs(r[r < 0].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.nan

    return {
        "ann_ret":      ann_ret,
        "ann_vol":      ann_vol,
        "sharpe":       sharpe,
        "sortino":      sortino,
        "calmar":       calmar,
        "mdd":          mdd,
        "avg_dd_dur":   avg_dd_dur,
        "drawdown":     drawdown,
        "win_rate":     win_rate,
        "avg_win":      avg_win,
        "avg_loss":     avg_loss,
        "profit_factor": profit_factor,
        "n_trades":     int((r != 0).sum()),
    }


def _print_metrics(m):
    print("\n=== TEARSHEET METRICS ===")
    print(f"Annualised Return  : {m['ann_ret']*100:.2f}%")
    print(f"Annualised Vol     : {m['ann_vol']*100:.2f}%")
    print(f"Sharpe Ratio       : {m['sharpe']:.3f}")
    print(f"Sortino Ratio      : {m['sortino']:.3f}")
    print(f"Calmar Ratio       : {m['calmar']:.3f}")
    print(f"Max Drawdown       : {m['mdd']*100:.2f}%")
    print(f"Avg Drawdown Dur.  : {m['avg_dd_dur']:.0f} days")
    print(f"Win Rate           : {m['win_rate']*100:.1f}%")
    print(f"Avg Win            : {m['avg_win']*100:.4f}%")
    print(f"Avg Loss           : {m['avg_loss']*100:.4f}%")
    print(f"Profit Factor      : {m['profit_factor']:.3f}")


# ------------------------------------------------------------------
# PDF pages
# ------------------------------------------------------------------
def _page_summary_and_drawdown(returns, bah, metrics, pdf):
    fig = plt.figure(figsize=(14, 10))
    gs  = gridspec.GridSpec(3, 1, figure=fig, hspace=0.4)

    # Cumulative returns vs benchmark
    ax1 = fig.add_subplot(gs[0])
    (1 + returns).cumprod().plot(ax=ax1, label="Blended Strategy", color="darkorange", linewidth=1.5)
    (1 + bah).cumprod().plot(ax=ax1, label="Buy & Hold (equal weight)", color="gray",
                              linewidth=1, linestyle="--")
    ax1.set_title("Cumulative Returns vs Benchmark")
    ax1.set_ylabel("Growth of $1"); ax1.legend(); ax1.grid(alpha=0.3)

    # Underwater equity curve
    ax2 = fig.add_subplot(gs[1])
    metrics["drawdown"].plot(ax=ax2, color="tomato", linewidth=1)
    ax2.fill_between(metrics["drawdown"].index, metrics["drawdown"], 0, alpha=0.3, color="tomato")
    ax2.set_title("Underwater Equity Curve (Drawdown)")
    ax2.set_ylabel("Drawdown"); ax2.grid(alpha=0.3)

    # Key metrics text box
    ax3 = fig.add_subplot(gs[2])
    ax3.axis("off")
    txt = (
        f"Annualised Return: {metrics['ann_ret']*100:.2f}%     "
        f"Annualised Vol: {metrics['ann_vol']*100:.2f}%\n"
        f"Sharpe: {metrics['sharpe']:.3f}     "
        f"Sortino: {metrics['sortino']:.3f}     "
        f"Calmar: {metrics['calmar']:.3f}\n"
        f"Max Drawdown: {metrics['mdd']*100:.2f}%     "
        f"Avg DD Duration: {metrics['avg_dd_dur']:.0f} days\n"
        f"Win Rate: {metrics['win_rate']*100:.1f}%     "
        f"Profit Factor: {metrics['profit_factor']:.3f}"
    )
    ax3.text(0.05, 0.5, txt, transform=ax3.transAxes, fontsize=11,
             verticalalignment="center", fontfamily="monospace",
             bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))
    ax3.set_title("Key Performance Metrics", fontsize=11)
    pdf.savefig(fig); plt.close(fig)


def _page_rolling_metrics(returns, pdf):
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    rolling_sharpe = (returns.rolling(252).mean() / returns.rolling(252).std()) * np.sqrt(252)
    rolling_vol    = returns.rolling(63).std() * np.sqrt(252)

    rolling_sharpe.plot(ax=axes[0], color="steelblue", linewidth=1.2)
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].set_title("Rolling 252-Day Sharpe Ratio"); axes[0].grid(alpha=0.3)

    rolling_vol.plot(ax=axes[1], color="darkorange", linewidth=1.2)
    axes[1].set_title("Rolling 63-Day Annualised Volatility"); axes[1].grid(alpha=0.3)

    plt.tight_layout()
    pdf.savefig(fig); plt.close(fig)


def _page_monthly_heatmap(returns, pdf):
    monthly = returns.copy()
    monthly.index = pd.to_datetime(monthly.index)
    monthly_r = monthly.groupby([monthly.index.year, monthly.index.month]).sum()
    monthly_r.index.names = ["Year", "Month"]
    pivot = monthly_r.unstack(level="Month")
    pivot.columns = ["Jan","Feb","Mar","Apr","May","Jun",
                     "Jul","Aug","Sep","Oct","Nov","Dec"]

    fig, ax = plt.subplots(figsize=(14, max(4, len(pivot) * 0.5 + 2)))
    sns.heatmap(
        pivot * 100, annot=True, fmt=".1f", cmap="RdYlGn", center=0,
        ax=ax, linewidths=0.3, linecolor="gray", cbar_kws={"label": "Monthly Return (%)"}
    )
    ax.set_title("Monthly Returns Heatmap (%)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    pdf.savefig(fig); plt.close(fig)


def _page_trade_stats(returns, metrics, pdf):
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.axis("off")
    stats = {
        "Total Observations":  len(returns),
        "Win Rate":            f"{metrics['win_rate']*100:.1f}%",
        "Avg Daily Win":       f"{metrics['avg_win']*100:.4f}%",
        "Avg Daily Loss":      f"{metrics['avg_loss']*100:.4f}%",
        "Profit Factor":       f"{metrics['profit_factor']:.3f}",
        "Annualised Return":   f"{metrics['ann_ret']*100:.2f}%",
        "Annualised Vol":      f"{metrics['ann_vol']*100:.2f}%",
        "Sharpe Ratio":        f"{metrics['sharpe']:.3f}",
        "Sortino Ratio":       f"{metrics['sortino']:.3f}",
        "Calmar Ratio":        f"{metrics['calmar']:.3f}",
        "Max Drawdown":        f"{metrics['mdd']*100:.2f}%",
        "Avg DD Duration":     f"{metrics['avg_dd_dur']:.0f} days",
    }
    rows = [[k, v] for k, v in stats.items()]
    table = ax.table(cellText=rows, colLabels=["Metric", "Value"],
                     loc="center", cellLoc="left")
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)
    ax.set_title("Full Performance Statistics", fontsize=13, fontweight="bold", pad=20)
    pdf.savefig(fig); plt.close(fig)


if __name__ == "__main__":
    print("Run this module via main.py — it requires blended_returns as input.")
