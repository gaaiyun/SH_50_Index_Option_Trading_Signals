# -*- coding: utf-8 -*-
"""Shared business logic for the live Streamlit dashboard and public snapshots."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from strategy.indicators import StrategyIndicators


@dataclass(frozen=True)
class RiskSettings:
    target_otm_pct: float
    stop_loss_pct: float
    rv_threshold_pct: float


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        value = float(value)
        if np.isnan(value) or np.isinf(value):
            return default
        return value
    except Exception:
        return default


def _iso(value: Any) -> str:
    try:
        ts = pd.Timestamp(value)
        if ts.tzinfo is None:
            return ts.isoformat()
        return ts.isoformat()
    except Exception:
        return str(value)


def _last_data_time(df: pd.DataFrame) -> str | None:
    if df is None or df.empty:
        return None
    if "Date" in df.columns:
        return _iso(df["Date"].iloc[-1])
    return None


def infer_option_type(row: Mapping[str, Any]) -> str:
    """Infer call/put type from Chinese labels, explicit type columns, or C/P codes."""

    values = []
    for col in ["名称", "类型", "option_type", "期权类型", "代码"]:
        if col in row:
            values.append(str(row[col]))
    text = " ".join(values).upper()
    if "购" in text or "认购" in text or "CALL" in text:
        return "认购"
    if "沽" in text or "认沽" in text or "PUT" in text:
        return "认沽"
    if "CON_OP" in text:
        return str(row.get("类型") or row.get("option_type") or "未知")
    if "C" in text and "P" not in text:
        return "认购"
    if "P" in text and "C" not in text:
        return "认沽"
    return "未知"


def compute_dashboard_metrics(
    *,
    etf_df: pd.DataFrame,
    options_df: pd.DataFrame | None,
    garch_result: Mapping[str, Any],
    bsadf_result: Mapping[str, Any],
    risk_settings: RiskSettings,
    indicators: Any | None = None,
    etf_source: str = "",
    options_source: str = "",
    now: Any | None = None,
) -> dict[str, Any]:
    """Compute the dashboard state once, without rendering Streamlit UI."""

    if etf_df is None or etf_df.empty or "Close" not in etf_df.columns:
        raise ValueError("ETF dataframe must include non-empty Close prices")

    prices = pd.to_numeric(etf_df["Close"], errors="coerce").dropna()
    if len(prices) < 2:
        raise ValueError("ETF dataframe must include at least two valid Close prices")

    now_ts = pd.Timestamp.now(tz="Asia/Shanghai") if now is None else pd.Timestamp(now)
    spot = float(prices.iloc[-1])
    prev = float(prices.iloc[-2])
    returns = np.log(prices / prices.shift(1)).dropna()

    change_pct = ((spot / prev) - 1.0) * 100.0 if prev else 0.0
    hv30 = float(returns.iloc[-30:].std() * np.sqrt(252) * 100) if len(returns) >= 30 else 0.0

    var_95 = _as_float(garch_result.get("var_95"), 0.02) * 100
    var_99 = _as_float(garch_result.get("var_99"), 0.03) * 100
    var_95_call = _as_float(garch_result.get("var_95_call"), 0.02) * 100
    var_95_put = _as_float(garch_result.get("var_95_put"), 0.02) * 100
    sigma_ann = _as_float(garch_result.get("sigma_norm"), 0.01) * np.sqrt(252) * 100
    robust_vol = _as_float(garch_result.get("robust_vol"), 0.01) * np.sqrt(252) * 100
    jump_lambda = _as_float(garch_result.get("jump_lambda_60"), 0.0) * 100

    bsadf_stat = _as_float(bsadf_result.get("adf_stat"), 0.0)
    bsadf_cv = _as_float(bsadf_result.get("cv"), 2.0)
    triggered = bool(bsadf_result.get("is_significant", False))

    iv_avg = 0.0
    exposure = {"gex_net": 0.0, "dex_net": 0.0, "max_pain_strike": spot}
    if options_df is not None and not options_df.empty:
        if "隐含波动率" in options_df.columns:
            iv_avg = _as_float(pd.to_numeric(options_df["隐含波动率"], errors="coerce").mean(), 0.0)
        engine = indicators or StrategyIndicators()
        exposure = engine.calculate_market_exposure(options_df, spot)

    if triggered:
        signal = "执行: 建立空仓"
        action = f"卖出偏离 {var_99:.1f}%–{risk_settings.target_otm_pct:.0f}% 虚值合约"
        signal_tone = "warning"
    else:
        signal = "状态: 观望戒备"
        action = f"BSADF({bsadf_stat:.2f}) < CV({bsadf_cv:.2f})"
        signal_tone = "neutral"

    return {
        "generated_at": _iso(now_ts),
        "spot": spot,
        "change_pct": float(change_pct),
        "hv30": hv30,
        "iv_avg": iv_avg,
        "var_95": var_95,
        "var_99": var_99,
        "var_95_call": var_95_call,
        "var_95_put": var_95_put,
        "sigma_ann": float(sigma_ann),
        "robust_vol": float(robust_vol),
        "jump_lambda_60": jump_lambda,
        "bsadf_stat": bsadf_stat,
        "bsadf_cv": bsadf_cv,
        "bsadf_triggered": triggered,
        "signal": signal,
        "action": action,
        "signal_tone": signal_tone,
        "gex_call": _as_float(exposure.get("gex_call"), 0.0),
        "gex_put": _as_float(exposure.get("gex_put"), 0.0),
        "gex_net": _as_float(exposure.get("gex_net"), 0.0),
        "dex_call": _as_float(exposure.get("dex_call"), 0.0),
        "dex_put": _as_float(exposure.get("dex_put"), 0.0),
        "dex_net": _as_float(exposure.get("dex_net"), 0.0),
        "max_pain": _as_float(exposure.get("max_pain_strike"), spot),
        "data_status": {"etf": etf_source, "options": options_source},
        "data_asof": {"etf": _last_data_time(etf_df), "options": None},
        "risk_settings": {
            "target_otm_pct": risk_settings.target_otm_pct,
            "stop_loss_pct": risk_settings.stop_loss_pct,
            "rv_threshold_pct": risk_settings.rv_threshold_pct,
        },
    }


def _json_value(value: Any) -> Any:
    if isinstance(value, (pd.Timestamp,)):
        return _iso(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return _as_float(value)
    if isinstance(value, float):
        return _as_float(value)
    if pd.isna(value):
        return None
    return value


def build_snapshot_payload(
    metrics: Mapping[str, Any],
    etf_df: pd.DataFrame,
    options_df: pd.DataFrame | None,
    *,
    limit_options: int = 25,
) -> dict[str, Any]:
    """Build a compact JSON payload for Cloudflare Pages."""

    options: list[dict[str, Any]] = []
    if options_df is not None and not options_df.empty:
        present_cols = [
            col
            for col in ["代码", "名称", "最新价", "行权价", "隐含波动率", "成交量", "持仓量", "买入价", "卖出价"]
            if col in options_df.columns
        ]
        trimmed = options_df[present_cols].head(limit_options)
        options = [
            {str(k): _json_value(v) for k, v in row.items()}
            for row in trimmed.to_dict(orient="records")
        ]

    closes = pd.to_numeric(etf_df["Close"], errors="coerce").dropna().tail(80)
    dates = etf_df.loc[closes.index, "Date"] if "Date" in etf_df.columns else closes.index
    price_series = [
        {"date": _iso(date), "close": _as_float(close)}
        for date, close in zip(dates, closes)
    ]

    return {
        "schema_version": 1,
        "generated_at": metrics["generated_at"],
        "market": {
            "symbol": "510050.SS",
            "spot": _as_float(metrics.get("spot")),
            "change_pct": _as_float(metrics.get("change_pct")),
            "data_asof": metrics.get("data_asof", {}).get("etf"),
            "data_status": metrics.get("data_status", {}),
        },
        "risk": {
            "signal": metrics.get("signal", ""),
            "action": metrics.get("action", ""),
            "bsadf_stat": _as_float(metrics.get("bsadf_stat")),
            "bsadf_cv": _as_float(metrics.get("bsadf_cv")),
            "bsadf_triggered": bool(metrics.get("bsadf_triggered")),
            "var_95": _as_float(metrics.get("var_95")),
            "var_95_call": _as_float(metrics.get("var_95_call")),
            "var_95_put": _as_float(metrics.get("var_95_put")),
            "hv30": _as_float(metrics.get("hv30")),
            "iv_avg": _as_float(metrics.get("iv_avg")),
        },
        "exposure": {
            "gex_net": _as_float(metrics.get("gex_net")),
            "dex_net": _as_float(metrics.get("dex_net")),
            "max_pain": _as_float(metrics.get("max_pain")),
        },
        "price_series": price_series,
        "options": options,
    }


def write_snapshot(payload: Mapping[str, Any], output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
