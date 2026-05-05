# FX & Rates Algorithmic Trading Signal Engine

A complete end-to-end quantitative FX signal engine built in Python. Pulls live data from yfinance and the FRED API, constructs three trading signals (carry, momentum, mean-reversion), backtests them individually and as a blended strategy with realistic transaction costs and slippage, applies Kelly Criterion position sizing, generates a professional performance tearsheet, and runs a live signal monitoring dashboard in Plotly Dash.

---

## Project Architecture

```
FX & Rates Algo Signal Engine/
├── config.py                   # Central config: pairs, parameters, shared helpers
├── data/ingestion.py           # Layer 1: Data pipeline
├── signals/carry.py            # Layer 2: Carry signal
├── signals/momentum.py         # Layer 3: Momentum signal
├── signals/mean_reversion.py   # Layer 4: Mean-reversion signal
├── portfolio/blender.py        # Layer 5a: Signal blending + transaction costs
├── portfolio/sizing.py         # Layer 5b: Kelly Criterion position sizing
├── analytics/performance.py   # Layer 6: Performance tearsheet
├── dashboard/app.py            # Layer 7: Plotly Dash live dashboard
├── outputs/                   # Generated CSVs, charts, tearsheet.pdf
└── main.py                    # Runner: sequences all layers, launches dashboard
```

Separating signals from the backtester is a core principle of professional system design. Signals define *when* to trade; the backtester defines *how* trades are simulated (costs, slippage, sizing). Mixing them makes it impossible to swap one without breaking the other.

---

## The Three Signals

### Carry
**Economic rationale:** Borrow in a low-rate currency, invest in a high-rate currency, and pocket the interest rate differential. The carry trade is essentially a bet that Uncovered Interest Rate Parity (UIP) fails — i.e., that exchange rates do not fully offset the rate differential. Empirically, UIP fails in the short run, making carry a persistent source of alpha.

**Implementation:** Signal = sign of the interest rate differential. Positive differential → long the pair; negative → short; near-zero → flat (10bps dead-band). Lagged by 1 day.

**Risk:** Carry crashes. During risk-off episodes (2008 JPY unwind, 2020 COVID), high-carry currencies collapse simultaneously as positions are unwound, producing severe left-tail losses.

### Momentum
**Economic rationale:** Currencies that have outperformed over the past 12 months tend to continue outperforming over the next month. Behaviourally explained by underreaction to news and herding; risk-based explanations point to compensation for crash risk.

**Implementation:** 12-1 month formation period (cumulative return from t-252 to t-21, skipping the most recent month to avoid short-term reversal). Cross-sectional ranking: long top 2 pairs, short bottom 2. Dollar-neutral construction (long book sums to 1.0, short book sums to -1.0). Lagged by 1 day.

### Mean-Reversion
**Economic rationale:** FX prices revert toward purchasing power parity (PPP) over medium-term horizons. When a pair trades significantly above or below its recent average, there is a statistical tendency to revert.

**Implementation:** Z-score of current price vs 60-day rolling mean and standard deviation. Enter long below -1.5σ, short above +1.5σ. Exit when |z| < 0.5. Stop-loss: exit after 10 days if z-score has not reverted, regardless of current level. Lagged by 1 day.

---

## Key Performance Metrics

| Metric | Definition | What it tells you |
|--------|-----------|-------------------|
| **Sharpe Ratio** | Annualised return / annualised vol | Risk-adjusted return per unit of total volatility |
| **Sortino Ratio** | Annualised return / downside deviation | Like Sharpe but only penalises downside volatility |
| **Calmar Ratio** | Annualised return / max drawdown | How much return you earn per unit of worst-case loss |
| **Max Drawdown** | Largest peak-to-trough decline | The worst loss a buy-and-hold investor would have experienced |
| **Profit Factor** | Gross profit / gross loss | >1.0 means the strategy makes more than it loses in aggregate |
| **Win Rate** | % of days with positive return | Does not account for magnitude; high win rate ≠ profitable |

---

## How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Set your FRED API key (free at fred.stlouisfed.org)
```bash
export FRED_API_KEY=your_key_here
```
Without a FRED key, the engine falls back to hardcoded proxy rates for carry signal computation.

### 3. Run the full pipeline
```bash
python main.py
```
This sequences all 7 layers, saves `outputs/tearsheet.pdf`, and launches the Dash dashboard at `http://localhost:8050`.

### 4. Run individual layers
```bash
python data/ingestion.py          # Layer 1: pull and save data
python signals/carry.py           # Layer 2: carry signal + charts
python signals/momentum.py        # Layer 3: momentum signal + charts
python signals/mean_reversion.py  # Layer 4: mean-reversion signal + charts
```

---

## Data Sources

| Data | Source | Tickers / Series |
|------|--------|-----------------|
| FX prices (10Y daily OHLCV) | yfinance | EURUSD=X, GBPUSD=X, AUDUSD=X, USDJPY=X, USDCAD=X |
| Fed Funds Rate | FRED | FEDFUNDS |
| ECB Deposit Rate | FRED | ECBDFR |
| Bank of England Rate | FRED | IUDSOIA |
| RBA Cash Rate | FRED | AUCBCR |
| Bank of Japan Rate | FRED | IRSTCI01JPM156N |
| Bank of Canada Rate | FRED | IRSTCI01CAM156N |
| VIX (regime indicator) | yfinance | ^VIX |

---

## What Would Be Built Next

1. **Machine learning signal layer** — train an LSTM or gradient-boosted model on the signal features to produce a fourth blended signal with non-linear regime awareness
2. **Regime-conditional weighting** — dynamically upweight mean-reversion in low-VIX regimes and downweight carry in high-VIX regimes using the existing regime indicator
3. **Live execution via broker API** — wire the dashboard signals to an Interactive Brokers or OANDA API for automated order generation
4. **Cross-asset expansion** — extend the signal framework to rates (government bond futures) and commodities to test whether the same signals generalise
5. **Walk-forward optimisation** — replace fixed signal parameters with rolling optimisation to reduce overfitting and test out-of-sample robustness
