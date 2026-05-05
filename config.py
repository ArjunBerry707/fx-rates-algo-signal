import numpy as np

# --- FX Pairs ---
PAIRS = ["EURUSD=X", "GBPUSD=X", "AUDUSD=X", "USDJPY=X", "USDCAD=X"]

PAIR_LABELS = {
    "EURUSD=X": "EURUSD",
    "GBPUSD=X": "GBPUSD",
    "AUDUSD=X": "AUDUSD",
    "USDJPY=X": "USDJPY",
    "USDCAD=X": "USDCAD",
}

# Foreign currency for each pair (the non-USD leg)
PAIR_FOREIGN_CCY = {
    "EURUSD=X": "EUR",
    "GBPUSD=X": "GBP",
    "AUDUSD=X": "AUD",
    "USDJPY=X": "JPY",
    "USDCAD=X": "CAD",
}

# --- FRED Series ---
# USD base rate
FRED_USD = "FEDFUNDS"

# Foreign currency short-term rates
# Fallback proxies (hardcoded annual %) used if FRED series unavailable
FRED_FOREIGN = {
    "EUR": {"series": "ECBDFR",             "fallback": 0.0},
    "GBP": {"series": "IUDSOIA",            "fallback": 0.5},
    "AUD": {"series": "AUCBCR",             "fallback": 1.5},
    "JPY": {"series": "IRSTCI01JPM156N",    "fallback": -0.1},
    "CAD": {"series": "IRSTCI01CAM156N",    "fallback": 1.0},
}

# --- Data Parameters ---
DATA_START = "2014-01-01"   # ~10 years of history
DATA_END   = None           # None = today

# --- Signal Parameters ---
SIGNAL_PARAMS = {
    "carry_threshold":      0.001,   # 10 bps dead-band for carry signal
    "momentum_window":      231,     # 12 months - 1 month = 252 - 21 trading days
    "momentum_skip":        21,      # skip most recent month to avoid reversal
    "momentum_long_n":      2,       # number of pairs to go long
    "momentum_short_n":     2,       # number of pairs to go short
    "z_window":             60,      # days for rolling mean/std in mean-reversion
    "z_entry":              1.5,     # z-score threshold to enter a trade
    "z_exit":               0.5,     # z-score threshold considered "near zero"
    "stop_loss_days":       10,      # max holding period without reversion
    "kelly_window":         63,      # days for rolling Kelly estimation
    "kelly_cap":            0.25,    # maximum Kelly fraction
    "transaction_cost_bps": 0.0002,  # 2 bps round-trip cost per trade
    "slippage_bps":         0.0001,  # 1 bp adverse slippage per trade
}

# --- Output Paths ---
OUTPUT_DIR        = "outputs"
PRICES_CSV        = f"{OUTPUT_DIR}/prices.csv"
RETURNS_CSV       = f"{OUTPUT_DIR}/returns.csv"
RATE_DIFF_CSV     = f"{OUTPUT_DIR}/rate_differentials.csv"
TEARSHEET_PDF     = f"{OUTPUT_DIR}/tearsheet.pdf"


def compute_max_drawdown(returns):
    """Compute maximum drawdown from a series of daily returns."""
    cum = (1 + returns).cumprod()
    rolling_max = cum.cummax()
    drawdown = (cum - rolling_max) / rolling_max
    return drawdown.min()


def compute_sharpe(returns, periods=252):
    """Annualised Sharpe ratio (assumes zero risk-free rate)."""
    if returns.std() == 0:
        return np.nan
    return (returns.mean() / returns.std()) * np.sqrt(periods)


def compute_annualised_return(returns, periods=252):
    """Annualised arithmetic return."""
    return returns.mean() * periods


def compute_annualised_vol(returns, periods=252):
    """Annualised volatility."""
    return returns.std() * np.sqrt(periods)
