"""
Main Runner — FX & Rates Algorithmic Signal Engine
Executes all 7 layers in sequence with timing output,
saves the tearsheet PDF, then launches the Dash dashboard.
"""

import time
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)


def _header(n, name):
    bar = "=" * 60
    print(f"\n{bar}")
    print(f"  LAYER {n}: {name}")
    print(f"{bar}")


def main():
    total_start = time.perf_counter()

    # ------------------------------------------------------------------
    # Setup: ensure outputs directory exists
    # ------------------------------------------------------------------
    os.makedirs("outputs", exist_ok=True)

    # ------------------------------------------------------------------
    # Layer 1 — Data Ingestion
    # ------------------------------------------------------------------
    _header(1, "DATA PIPELINE")
    t0 = time.perf_counter()
    from data.ingestion import run as ingest
    prices, log_returns, rate_diffs = ingest()
    print(f"  Done in {time.perf_counter() - t0:.1f}s")

    # ------------------------------------------------------------------
    # Layer 2 — Carry Signal
    # ------------------------------------------------------------------
    _header(2, "CARRY SIGNAL")
    t0 = time.perf_counter()
    from signals.carry import run as carry_run
    carry_returns, carry_signal = carry_run()
    print(f"  Done in {time.perf_counter() - t0:.1f}s")

    # ------------------------------------------------------------------
    # Layer 3 — Momentum Signal
    # ------------------------------------------------------------------
    _header(3, "MOMENTUM SIGNAL")
    t0 = time.perf_counter()
    from signals.momentum import run as momentum_run
    momentum_returns, momentum_weights = momentum_run(carry_returns=carry_returns)
    print(f"  Done in {time.perf_counter() - t0:.1f}s")

    # ------------------------------------------------------------------
    # Layer 4 — Mean-Reversion Signal
    # ------------------------------------------------------------------
    _header(4, "MEAN-REVERSION SIGNAL")
    t0 = time.perf_counter()
    from signals.mean_reversion import run as mr_run
    mr_returns, mr_signal, z_scores = mr_run(
        carry_returns=carry_returns,
        momentum_returns=momentum_returns,
    )
    print(f"  Done in {time.perf_counter() - t0:.1f}s")

    # ------------------------------------------------------------------
    # Layer 5 — Signal Blending + Kelly Position Sizing
    # ------------------------------------------------------------------
    _header(5, "SIGNAL BLENDING + KELLY POSITION SIZING")
    t0 = time.perf_counter()
    from portfolio.blender import run as blend_run
    blended_returns, blended_signal = blend_run(
        carry_returns, carry_signal,
        momentum_returns, momentum_weights,
        mr_returns, mr_signal,
    )
    from portfolio.sizing import apply_kelly
    kelly_returns, kelly_fractions = apply_kelly(blended_signal, blended_returns)
    print(f"  Done in {time.perf_counter() - t0:.1f}s")

    # ------------------------------------------------------------------
    # Layer 6 — Performance Tearsheet
    # ------------------------------------------------------------------
    _header(6, "PERFORMANCE TEARSHEET")
    t0 = time.perf_counter()
    from analytics.performance import run as tearsheet_run
    tearsheet_run(blended_returns)
    print(f"  Done in {time.perf_counter() - t0:.1f}s")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    elapsed = time.perf_counter() - total_start
    print(f"\n{'='*60}")
    print(f"  All layers complete in {elapsed:.1f}s")
    print(f"  Tearsheet PDF: outputs/tearsheet.pdf")
    print(f"{'='*60}")

    # ------------------------------------------------------------------
    # Layer 7 — Live Dashboard (blocking)
    # ------------------------------------------------------------------
    _header(7, "LAUNCHING LIVE DASHBOARD (http://localhost:8050)")
    from dashboard.app import run_server
    run_server(debug=False)


if __name__ == "__main__":
    main()
