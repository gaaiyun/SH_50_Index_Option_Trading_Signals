# -*- coding: utf-8 -*-
"""Dashboard business-layer tests for live metrics and public snapshots."""

import json

import pandas as pd


class RecordingIndicators:
    def __init__(self):
        self.exposure_calls = []

    def calculate_market_exposure(self, options_df, spot):
        self.exposure_calls.append((options_df.copy(), spot))
        return {
            "gex_call": 1.25,
            "gex_put": -0.75,
            "gex_net": 0.50,
            "dex_call": 0.12,
            "dex_put": -0.18,
            "dex_net": -0.06,
            "max_pain_strike": 2.55,
        }


def _sample_etf():
    return pd.DataFrame(
        {
            "Date": pd.date_range("2026-05-25", periods=40, freq="D"),
            "Close": [2.50 + i * 0.002 for i in range(40)],
            "Open": [2.49 + i * 0.002 for i in range(40)],
            "High": [2.52 + i * 0.002 for i in range(40)],
            "Low": [2.48 + i * 0.002 for i in range(40)],
            "Volume": [1000000 + i for i in range(40)],
        }
    )


def _sample_options():
    return pd.DataFrame(
        {
            "代码": ["CON_OP_1", "CON_OP_2"],
            "名称": ["50ETF购26062550", "50ETF沽26062500"],
            "最新价": [0.08, 0.07],
            "行权价": [2.55, 2.50],
            "隐含波动率": [18.5, 20.0],
            "成交量": [1200, 900],
            "持仓量": [22000, 18000],
            "买入价": [0.079, 0.069],
            "卖出价": [0.081, 0.071],
        }
    )


def test_dashboard_metrics_uses_market_exposure_engine():
    from dashboard_core import RiskSettings, compute_dashboard_metrics

    indicator = RecordingIndicators()
    metrics = compute_dashboard_metrics(
        etf_df=_sample_etf(),
        options_df=_sample_options(),
        garch_result={
            "var_95": 0.021,
            "var_99": 0.034,
            "var_95_call": 0.023,
            "var_95_put": 0.026,
            "sigma_norm": 0.012,
            "robust_vol": 0.011,
            "jump_lambda_60": 0.08,
        },
        bsadf_result={"adf_stat": 1.4, "cv": 2.3, "is_significant": False},
        risk_settings=RiskSettings(target_otm_pct=11, stop_loss_pct=6, rv_threshold_pct=30),
        indicators=indicator,
        etf_source="loaded",
        options_source="loaded",
        now=pd.Timestamp("2026-06-02T10:00:00+08:00"),
    )

    assert len(indicator.exposure_calls) == 1
    assert indicator.exposure_calls[0][1] == metrics["spot"]
    assert metrics["gex_net"] == 0.50
    assert metrics["dex_net"] == -0.06
    assert metrics["max_pain"] == 2.55
    assert metrics["signal"] == "状态: 观望戒备"
    assert metrics["data_status"]["etf"] == "loaded"
    assert metrics["data_status"]["options"] == "loaded"


def test_snapshot_payload_and_file_are_pages_ready(tmp_path):
    from dashboard_core import RiskSettings, build_snapshot_payload, compute_dashboard_metrics, write_snapshot

    metrics = compute_dashboard_metrics(
        etf_df=_sample_etf(),
        options_df=_sample_options(),
        garch_result={"var_95": 0.021, "var_95_call": 0.023, "var_95_put": 0.026},
        bsadf_result={"adf_stat": 3.1, "cv": 2.3, "is_significant": True},
        risk_settings=RiskSettings(target_otm_pct=11, stop_loss_pct=6, rv_threshold_pct=30),
        indicators=RecordingIndicators(),
        etf_source="loaded",
        options_source="loaded",
        now=pd.Timestamp("2026-06-02T10:00:00+08:00"),
    )
    payload = build_snapshot_payload(metrics, _sample_etf(), _sample_options(), limit_options=1)

    assert payload["schema_version"] == 1
    assert payload["generated_at"].startswith("2026-06-02T10:00:00")
    assert payload["market"]["symbol"] == "510050.SS"
    assert payload["risk"]["signal"] == "执行: 建立空仓"
    assert payload["exposure"]["gex_net"] == 0.50
    assert len(payload["options"]) == 1

    output = tmp_path / "latest.json"
    write_snapshot(payload, output)
    loaded = json.loads(output.read_text(encoding="utf-8"))
    assert loaded["market"]["spot"] == payload["market"]["spot"]


def test_infer_option_type_prefers_explicit_type_columns():
    from dashboard_core import infer_option_type

    assert infer_option_type({"名称": "510050", "类型": "认购"}) == "认购"
    assert infer_option_type({"名称": "510050", "option_type": "认沽"}) == "认沽"
    assert infer_option_type({"名称": "510050C2602M02500"}) == "认购"
    assert infer_option_type({"名称": "510050P2602M02400"}) == "认沽"
