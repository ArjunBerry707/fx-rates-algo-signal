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
    "AUD": {"series": "IRSTCI01AUM156N",     "fallback": 1.5},
    "JPY": {"series": "IRSTCI01JPM156N",    "fallback": -0.1},
    "CAD": {"series": "IRSTCI01CAM156N",    "fallback": 1.0},
}

# --- Data Parameters ---
DATA_START = "2014-01-01"   # ~10 years of history
DATA_END   = None           # None = today

# --- Signal Parameters ---
SIGNAL_PARAMS = {
    # Carry
    "carry_threshold":       0.005,   # 50 bps dead-band (was 10 bps — widens filter)
    "vix_risk_off":          20.0,    # go flat on carry when VIX exceeds this level

    # Momentum
    "momentum_window":       231,
    "momentum_skip":         21,
    "momentum_long_n":       2,
    "momentum_short_n":      2,
    "momentum_min_spread":   0.03,    # min cross-pair score spread to trade (3%)

    # Mean-reversion
    "z_window":              60,
    "z_entry":               1.5,
    "z_exit":                0.5,
    "stop_loss_days":        10,

    # Position sizing — volatility targeting
    "vol_target_daily":      0.005,   # 0.5% daily vol target (~8% annualised)
    "vol_scale_cap":         2.0,     # maximum scale factor
    "vol_regime_window":     21,      # short window for current vol
    "vol_regime_ref_window": 126,     # reference window (~6 months)
    "vol_regime_threshold":  1.5,     # scale down when current/ref vol ratio > this
    "corr_window":           63,      # window for pairwise correlation

    # Legacy (kept so old code that references these doesn't break)
    "kelly_window":          63,
    "kelly_cap":             0.25,

    # Costs
    "transaction_cost_bps":  0.0002,
    "slippage_bps":          0.0001,
}

# --- Output Paths ---
OUTPUT_DIR        = "outputs"
PRICES_CSV        = f"{OUTPUT_DIR}/prices.csv"
RETURNS_CSV       = f"{OUTPUT_DIR}/returns.csv"
RATE_DIFF_CSV     = f"{OUTPUT_DIR}/rate_differentials.csv"
VIX_CSV           = f"{OUTPUT_DIR}/vix.csv"
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
