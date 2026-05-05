"""
Layer 7 — Live Signal Dashboard
Plotly Dash application with 4 panels: signal monitor, live P&L,
signal history (z-scores), and regime indicator.
"""

import os
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, dash_table, Input, Output, State

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import (
    PAIRS, PAIR_LABELS, SIGNAL_PARAMS,
    PRICES_CSV, RETURNS_CSV, RATE_DIFF_CSV,
    compute_max_drawdown,
)

# ------------------------------------------------------------------
# Colour palette for dark theme
# ------------------------------------------------------------------
DARK_BG   = "#1a1a2e"
CARD_BG   = "#16213e"
ACCENT    = "#e94560"
POSITIVE  = "#4ecca3"
NEGATIVE  = "#e94560"
NEUTRAL   = "#888888"
TEXT      = "#eaeaea"

# ------------------------------------------------------------------
# App layout
# ------------------------------------------------------------------
app = dash.Dash(__name__, title="FX Signal Engine")
app.layout = html.Div(style={"backgroundColor": DARK_BG, "color": TEXT,
                               "fontFamily": "monospace", "padding": "20px"}, children=[

    html.H1("FX & Rates Algorithmic Signal Engine",
            style={"textAlign": "center", "color": POSITIVE, "marginBottom": "30px"}),

    # ---- Panel 1: Signal Monitor ----
    html.Div(style={"backgroundColor": CARD_BG, "padding": "20px",
                     "borderRadius": "8px", "marginBottom": "20px"}, children=[
        html.H3("Panel 1 — Current Signal Monitor", style={"color": POSITIVE}),
        html.Button("Refresh Signals", id="refresh-btn", n_clicks=0,
                    style={"backgroundColor": ACCENT, "color": TEXT, "border": "none",
                           "padding": "8px 18px", "borderRadius": "4px",
                           "cursor": "pointer", "marginBottom": "15px"}),
        html.Div(id="signal-table"),
    ]),

    # ---- Panel 2: Live P&L ----
    html.Div(style={"backgroundColor": CARD_BG, "padding": "20px",
                     "borderRadius": "8px", "marginBottom": "20px"}, children=[
        html.H3("Panel 2 — Live Cumulative P&L", style={"color": POSITIVE}),
        dcc.Dropdown(
            id="lookback-dropdown",
            options=[{"label": l, "value": v} for l, v in
                     [("1 Month", "1M"), ("3 Months", "3M"),
                      ("6 Months", "6M"), ("1 Year", "1Y"), ("Full History", "ALL")]],
            value="1Y",
            style={"width": "200px", "marginBottom": "10px",
                   "backgroundColor": CARD_BG, "color": "#000"},
        ),
        dcc.Graph(id="pnl-chart", style={"height": "450px"}),
    ]),

    # ---- Panel 3: Signal History (Z-scores) ----
    html.Div(style={"backgroundColor": CARD_BG, "padding": "20px",
                     "borderRadius": "8px", "marginBottom": "20px"}, children=[
        html.H3("Panel 3 — Signal History (Z-Scores, last 60 days)", style={"color": POSITIVE}),
        dcc.Dropdown(
            id="pair-dropdown",
            options=[{"label": PAIR_LABELS[p], "value": PAIR_LABELS[p]} for p in PAIRS],
            value="EURUSD",
            style={"width": "200px", "marginBottom": "10px",
                   "backgroundColor": CARD_BG, "color": "#000"},
        ),
        dcc.Graph(id="zscore-chart", style={"height": "350px"}),
    ]),

    # ---- Panel 4: Regime Indicator ----
    html.Div(style={"backgroundColor": CARD_BG, "padding": "20px",
                     "borderRadius": "8px", "marginBottom": "20px"}, children=[
        html.H3("Panel 4 — Market Regime Indicator", style={"color": POSITIVE}),
        html.Div(id="regime-badge",
                 style={"fontSize": "28px", "fontWeight": "bold",
                        "marginBottom": "15px", "textAlign": "center"}),
        dcc.Graph(id="regime-chart", style={"height": "350px"}),
    ]),

    # Hidden store for computed signals
    dcc.Store(id="signal-store"),
])


# ------------------------------------------------------------------
# Callbacks
# ------------------------------------------------------------------
@app.callback(
    Output("signal-store", "data"),
    Output("signal-table", "children"),
    Input("refresh-btn", "n_clicks"),
)
def refresh_signals(n_clicks):
    prices, returns, rate_diff, z_scores, blended, kelly_frac = _compute_live_signals()

    rows = []
    for pair_label in blended.columns:
        c_sig  = _last_valid(rate_diff[pair_label]) if pair_label in rate_diff.columns else 0
        c_sig  = 1 if c_sig > 0.001 else (-1 if c_sig < -0.001 else 0)
        mr_sig = _last_valid(z_scores[pair_label]) if pair_label in z_scores.columns else 0
        mr_sig = -1 if mr_sig > 1.5 else (1 if mr_sig < -1.5 else 0)
        bl_val = _last_valid(blended[pair_label])
        kf_val = _last_valid(kelly_frac[pair_label]) if pair_label in kelly_frac.columns else 0

        row_color = POSITIVE if bl_val > 0.1 else (NEGATIVE if bl_val < -0.1 else NEUTRAL)
        rows.append({
            "Pair": pair_label,
            "Carry": f"{c_sig:+d}",
            "Mean-Rev": f"{mr_sig:+d}",
            "Blended": f"{bl_val:.3f}",
            "Kelly Size": f"{kf_val:.3f}",
            "_color": row_color,
        })

    table = dash_table.DataTable(
        data=[{k: v for k, v in r.items() if k != "_color"} for r in rows],
        columns=[{"name": c, "id": c} for c in ["Pair", "Carry", "Mean-Rev", "Blended", "Kelly Size"]],
        style_table={"overflowX": "auto"},
        style_header={"backgroundColor": DARK_BG, "color": POSITIVE,
                      "fontWeight": "bold", "border": f"1px solid {NEUTRAL}"},
        style_cell={"backgroundColor": CARD_BG, "color": TEXT,
                    "textAlign": "center", "border": f"1px solid {NEUTRAL}",
                    "fontFamily": "monospace"},
        style_data_conditional=[
            {"if": {"filter_query": f'{{Blended}} > 0.1'},
             "backgroundColor": "#1a3a2a", "color": POSITIVE},
            {"if": {"filter_query": f'{{Blended}} < -0.1'},
             "backgroundColor": "#3a1a1a", "color": NEGATIVE},
        ],
    )

    store = {
        "blended_returns": blended.mean(axis=1).to_json(),
        "z_scores": z_scores.to_json(),
        "rate_diff": rate_diff.to_json(),
    }
    return store, table


@app.callback(
    Output("pnl-chart", "figure"),
    Input("lookback-dropdown", "value"),
    Input("signal-store", "data"),
)
def update_pnl(lookback, store):
    if store is None:
        return go.Figure()

    strat_returns = pd.read_json(store["blended_returns"], typ="series").sort_index()
    strat_returns.index = pd.to_datetime(strat_returns.index)

    cutoff = _lookback_cutoff(strat_returns.index[-1], lookback)
    r = strat_returns[strat_returns.index >= cutoff]

    cum = (1 + r).cumprod()
    roll_max = cum.cummax()
    dd = (cum - roll_max) / roll_max

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.65, 0.35],
                        subplot_titles=["Cumulative Return", "Drawdown"])

    fig.add_trace(go.Scatter(x=cum.index, y=cum.values, name="Strategy",
                              line=dict(color=POSITIVE, width=1.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=dd.index, y=dd.values, name="Drawdown",
                              fill="tozeroy", line=dict(color=NEGATIVE, width=1),
                              fillcolor="rgba(233,69,96,0.2)"), row=2, col=1)

    fig.update_layout(paper_bgcolor=CARD_BG, plot_bgcolor=CARD_BG,
                      font_color=TEXT, showlegend=True,
                      margin=dict(l=40, r=20, t=40, b=20))
    fig.update_xaxes(gridcolor="#333"); fig.update_yaxes(gridcolor="#333")
    return fig


@app.callback(
    Output("zscore-chart", "figure"),
    Input("pair-dropdown", "value"),
    Input("signal-store", "data"),
)
def update_zscore(pair, store):
    if store is None:
        return go.Figure()

    z = pd.read_json(store["z_scores"]).sort_index()
    z.index = pd.to_datetime(z.index)
    z60 = z.tail(60)

    fig = go.Figure()
    if pair in z60.columns:
        fig.add_trace(go.Scatter(x=z60.index, y=z60[pair], name=f"{pair} Z-Score",
                                  line=dict(color=POSITIVE, width=1.5)))
    fig.add_hline(y=1.5,  line=dict(color=NEGATIVE, dash="dash", width=1), annotation_text="+1.5 Short")
    fig.add_hline(y=-1.5, line=dict(color=POSITIVE, dash="dash", width=1), annotation_text="-1.5 Long")
    fig.add_hline(y=0,    line=dict(color=TEXT,     dash="dot",  width=0.8))
    fig.update_layout(title=f"{pair} — 60-Day Z-Score History",
                      paper_bgcolor=CARD_BG, plot_bgcolor=CARD_BG,
                      font_color=TEXT, margin=dict(l=40, r=20, t=40, b=20))
    fig.update_xaxes(gridcolor="#333"); fig.update_yaxes(gridcolor="#333")
    return fig


@app.callback(
    Output("regime-badge", "children"),
    Output("regime-badge", "style"),
    Output("regime-chart", "figure"),
    Input("signal-store", "data"),
)
def update_regime(store):
    vix = yf.download("^VIX", period="2y", progress=False, auto_adjust=True)
    if isinstance(vix.columns, pd.MultiIndex):
        vix_close = vix["Close"].iloc[:, 0]
    else:
        vix_close = vix["Close"]

    vix_ma = vix_close.rolling(20).mean()
    regime = pd.Series("Risk-On", index=vix_ma.index)
    regime[vix_ma > 20] = "Risk-Off"

    current_regime = regime.iloc[-1]
    badge_color = POSITIVE if current_regime == "Risk-On" else "#f5a623"
    badge_style = {"fontSize": "28px", "fontWeight": "bold",
                   "color": badge_color, "textAlign": "center",
                   "padding": "10px", "borderRadius": "8px",
                   "border": f"2px solid {badge_color}",
                   "display": "inline-block", "marginBottom": "15px"}

    fig = go.Figure()
    risk_on  = regime == "Risk-On"
    risk_off = regime == "Risk-Off"

    fig.add_trace(go.Scatter(x=vix_ma.index, y=vix_ma.values, name="VIX 20-Day MA",
                              line=dict(color="#aaaaaa", width=1.2)))
    fig.add_hline(y=20, line=dict(color=NEGATIVE, dash="dash", width=1),
                  annotation_text="VIX 20 threshold")

    if store is not None:
        strat_returns = pd.read_json(store["blended_returns"], typ="series").sort_index()
        strat_returns.index = pd.to_datetime(strat_returns.index)
        cum = (1 + strat_returns).cumprod()
        common = cum.index.intersection(regime.index)
        fig.add_trace(go.Scatter(x=common, y=cum.reindex(common).values,
                                  name="Strategy (cum.)",
                                  line=dict(color=POSITIVE, width=1.5),
                                  yaxis="y2"))

    fig.update_layout(
        title="VIX Regime vs Blended Strategy",
        paper_bgcolor=CARD_BG, plot_bgcolor=CARD_BG,
        font_color=TEXT, margin=dict(l=40, r=60, t=40, b=20),
        yaxis=dict(title="VIX MA", gridcolor="#333"),
        yaxis2=dict(title="Cumulative Return", overlaying="y",
                    side="right", gridcolor="#333", showgrid=False),
        legend=dict(bgcolor=CARD_BG),
    )
    return f"Current Regime: {current_regime}", badge_style, fig


# ------------------------------------------------------------------
# Signal computation helpers
# ------------------------------------------------------------------
def _compute_live_signals():
    prices   = pd.read_csv(PRICES_CSV,   index_col=0, parse_dates=True)
    returns  = pd.read_csv(RETURNS_CSV,  index_col=0, parse_dates=True)
    rate_diff = pd.read_csv(RATE_DIFF_CSV, index_col=0, parse_dates=True)

    z_window = SIGNAL_PARAMS["z_window"]
    z_entry  = SIGNAL_PARAMS["z_entry"]
    kelly_window = SIGNAL_PARAMS["kelly_window"]
    kelly_cap    = SIGNAL_PARAMS["kelly_cap"]

    roll_mean = prices.rolling(z_window).mean()
    roll_std  = prices.rolling(z_window).std()
    z_scores  = (prices - roll_mean) / roll_std

    carry_sig = np.sign(rate_diff)
    carry_sig[rate_diff.abs() < SIGNAL_PARAMS["carry_threshold"]] = 0

    mr_sig = pd.DataFrame(0.0, index=z_scores.index, columns=z_scores.columns)
    mr_sig[z_scores < -z_entry] = 1
    mr_sig[z_scores >  z_entry] = -1

    blended = (carry_sig + mr_sig) / 2.0

    roll_mu  = returns.rolling(kelly_window).mean()
    roll_var = returns.rolling(kelly_window).var()
    kelly_frac = (roll_mu / roll_var).clip(0, kelly_cap)

    return prices, returns, rate_diff, z_scores, blended, kelly_frac


def _last_valid(series):
    s = series.dropna()
    return float(s.iloc[-1]) if len(s) > 0 else 0.0


def _lookback_cutoff(last_date, lookback):
    if lookback == "1M":
        return last_date - pd.DateOffset(months=1)
    elif lookback == "3M":
        return last_date - pd.DateOffset(months=3)
    elif lookback == "6M":
        return last_date - pd.DateOffset(months=6)
    elif lookback == "1Y":
        return last_date - pd.DateOffset(years=1)
    return pd.Timestamp("2000-01-01")


def run_server(debug=False):
    app.run(debug=debug, host="0.0.0.0", port=8050)


if __name__ == "__main__":
    run_server(debug=True)
