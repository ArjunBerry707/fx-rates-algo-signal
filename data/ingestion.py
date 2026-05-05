"""
Layer 1 — Data Pipeline
Pulls FX prices from yfinance and interest rates from FRED,
aligns them to a common daily index, and saves cleaned CSVs.
"""

import os
import numpy as np
import pandas as pd
import yfinance as yf

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import (
    PAIRS, PAIR_LABELS, PAIR_FOREIGN_CCY,
    FRED_USD, FRED_FOREIGN,
    DATA_START, DATA_END,
    OUTPUT_DIR, PRICES_CSV, RETURNS_CSV, RATE_DIFF_CSV,
)


def run():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Pull FX prices from yfinance
    # ------------------------------------------------------------------
    print("Fetching FX prices from yfinance...")
    raw = yf.download(PAIRS, start=DATA_START, end=DATA_END, auto_adjust=True, progress=False)

    # yfinance returns a MultiIndex when multiple tickers are requested.
    # We want the 'Close' column (auto_adjust=True replaces Adj Close with Close).
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"].copy()
    else:
        prices = raw[["Close"]].copy()

    prices.columns = [PAIR_LABELS[c] for c in prices.columns]
    prices.index = pd.to_datetime(prices.index)
    prices = prices.sort_index()

    # ------------------------------------------------------------------
    # 2. Pull interest rates from FRED
    # ------------------------------------------------------------------
    print("Fetching interest rates from FRED...")
    rates = _fetch_rates(prices.index)

    # ------------------------------------------------------------------
    # 3. Align indexes: forward-fill rates, drop rows where FX is missing
    # ------------------------------------------------------------------
    rates = rates.reindex(prices.index).ffill()
    prices = prices.dropna(how="all")
    rates   = rates.reindex(prices.index).ffill()

    # Drop any remaining NaN rows in prices (e.g. market holidays)
    prices = prices.dropna()
    rates   = rates.reindex(prices.index).ffill()

    # ------------------------------------------------------------------
    # 4. Compute log returns
    # ------------------------------------------------------------------
    log_returns = np.log(prices / prices.shift(1)).dropna()

    # ------------------------------------------------------------------
    # 5. Compute interest rate differentials
    # ------------------------------------------------------------------
    usd_rate = rates["USD"]
    rate_diffs = pd.DataFrame(index=rates.index)
    for pair, label in PAIR_LABELS.items():
        ccy = PAIR_FOREIGN_CCY[pair]
        # Convert annual % to decimal
        rate_diffs[label] = (rates[ccy] - usd_rate) / 100.0

    rate_diffs = rate_diffs.reindex(log_returns.index).ffill()

    # ------------------------------------------------------------------
    # 6. Save CSVs
    # ------------------------------------------------------------------
    prices.to_csv(PRICES_CSV)
    log_returns.to_csv(RETURNS_CSV)
    rate_diffs.to_csv(RATE_DIFF_CSV)
    print(f"Saved: {PRICES_CSV}, {RETURNS_CSV}, {RATE_DIFF_CSV}")

    # ------------------------------------------------------------------
    # 7. Print summary
    # ------------------------------------------------------------------
    print("\n=== DATA SUMMARY ===")
    print(f"Date range : {log_returns.index[0].date()} → {log_returns.index[-1].date()}")
    print(f"Observations: {len(log_returns)}")
    print("\nLog return descriptive statistics:")
    print(log_returns.describe().round(6))
    print("\nRate differentials (first 5 rows, annualised %):")
    print((rate_diffs * 100).head().round(4))

    return prices, log_returns, rate_diffs


# ------------------------------------------------------------------
# Internal helper
# ------------------------------------------------------------------
def _fetch_rates(date_index):
    """
    Pull short-term rates from FRED. Falls back to hardcoded proxy
    values if a series is unavailable.
    """
    try:
        from fredapi import Fred
        fred = Fred()          # reads FRED_API_KEY from environment
        use_fred = True
    except Exception:
        use_fred = False
        print("  WARNING: fredapi not configured (missing FRED_API_KEY). Using hardcoded proxies.")

    rates = pd.DataFrame(index=date_index)

    # USD rate
    rates["USD"] = _get_series("USD", FRED_USD, None, use_fred, date_index)

    # Foreign rates
    for ccy, info in FRED_FOREIGN.items():
        rates[ccy] = _get_series(ccy, info["series"], info["fallback"], use_fred, date_index)

    return rates


def _get_series(ccy, series_id, fallback, use_fred, date_index):
    if use_fred:
        try:
            from fredapi import Fred
            fred = Fred()
            s = fred.get_series(series_id, observation_start=DATA_START)
            s = s.reindex(date_index).ffill()
            print(f"  FRED OK : {ccy} ({series_id})")
            return s
        except Exception as e:
            print(f"  FRED FAIL: {ccy} ({series_id}) — {e}. Using fallback={fallback}.")

    # Fallback: constant proxy rate
    fb = fallback if fallback is not None else 0.0
    return pd.Series(fb, index=date_index)


if __name__ == "__main__":
    run()
