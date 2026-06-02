#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate the public JSON snapshot used by Cloudflare Pages."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dashboard_core import RiskSettings, build_snapshot_payload, compute_dashboard_metrics, write_snapshot
from data_sources import add_implied_volatility, fetch_50etf_options_sina, fetch_50etf_options_yfinance
from strategy.indicators import StrategyIndicators


def _load_etf_cache(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if "Date" not in df.columns:
        df = pd.read_csv(path, index_col=0, parse_dates=True).reset_index().rename(columns={"index": "Date"})
    if df.empty:
        return None
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    return df.dropna(subset=["Date", "Close"])


def _load_options_cache(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    df = pd.read_csv(path)
    unnamed = [c for c in df.columns if str(c).startswith("Unnamed:")]
    if unnamed:
        df = df.drop(columns=unnamed)
    return None if df.empty else df


def _fetch_etf_online() -> pd.DataFrame | None:
    import yfinance as yf

    df = yf.download("510050.SS", period="5y", progress=False, auto_adjust=False)
    if df is None or df.empty:
        return None
    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    df = df.reset_index()
    return df.rename(columns={"index": "Date"})


def _fetch_options_online() -> tuple[pd.DataFrame | None, str]:
    df, msg = fetch_50etf_options_sina()
    if df is not None and not df.empty:
        return df, msg
    df, msg = fetch_50etf_options_yfinance()
    if df is not None and not df.empty:
        return df, msg
    return None, msg


def generate_snapshot(
    *,
    output: Path,
    refresh: bool,
    target_otm_pct: float,
    stop_loss_pct: float,
    rv_threshold_pct: float,
) -> Path:
    data_dir = ROOT / "data"
    etf_cache = data_dir / "etf_510050.csv"
    options_cache = data_dir / "options_50.csv"

    etf_df = _fetch_etf_online() if refresh else None
    etf_source = "online:yfinance" if etf_df is not None and not etf_df.empty else "cache"
    if etf_df is None or etf_df.empty:
        etf_df = _load_etf_cache(etf_cache)
        etf_source = "cache"

    options_df = None
    options_source = "cache"
    if refresh:
        options_df, options_source = _fetch_options_online()
    if options_df is None or options_df.empty:
        options_df = _load_options_cache(options_cache)
        options_source = "cache"

    if etf_df is None or etf_df.empty:
        raise RuntimeError("ETF data is unavailable; cannot generate snapshot")

    prices = pd.to_numeric(etf_df["Close"], errors="coerce").dropna()
    spot = float(prices.iloc[-1])
    if options_df is not None and not options_df.empty:
        options_df = add_implied_volatility(options_df, spot)

    indicators = StrategyIndicators()
    garch_result = indicators.calculate_garch_var(prices, confidence_levels=[0.95, 0.975, 0.99])
    bsadf_result = indicators.calculate_bsadf(prices, window=100)
    metrics = compute_dashboard_metrics(
        etf_df=etf_df,
        options_df=options_df,
        garch_result=garch_result,
        bsadf_result=bsadf_result,
        risk_settings=RiskSettings(target_otm_pct, stop_loss_pct, rv_threshold_pct),
        indicators=indicators,
        etf_source=etf_source,
        options_source=options_source,
    )
    payload = build_snapshot_payload(metrics, etf_df, options_df, limit_options=40)
    return write_snapshot(payload, output)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate public/data/latest.json for Cloudflare Pages.")
    parser.add_argument("--output", default=str(ROOT / "public" / "data" / "latest.json"))
    parser.add_argument("--refresh", action="store_true", help="Fetch online data before falling back to local cache.")
    parser.add_argument("--target-otm-pct", type=float, default=11.0)
    parser.add_argument("--stop-loss-pct", type=float, default=6.0)
    parser.add_argument("--rv-threshold-pct", type=float, default=30.0)
    args = parser.parse_args(argv)

    path = generate_snapshot(
        output=Path(args.output),
        refresh=args.refresh,
        target_otm_pct=args.target_otm_pct,
        stop_loss_pct=args.stop_loss_pct,
        rv_threshold_pct=args.rv_threshold_pct,
    )
    print(f"snapshot written: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
