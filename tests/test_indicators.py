# -*- coding: utf-8 -*-
"""VolGuard Pro — 策略指标内核单元测试"""

import pytest
import pandas as pd
import numpy as np


class TestCalculateBsadf:
    """BSADF 泡沫测试"""

    def test_returns_structure_with_sample_prices(self, indicators, sample_prices):
        result = indicators.calculate_bsadf(sample_prices)
        assert "adf_stat" in result
        assert "cv" in result
        assert "is_significant" in result
        assert "series" in result
        assert result["cv"] > 0
        assert isinstance(result["series"], pd.Series)
        # 窗口回到最近 200 根，series 长度约在 200 以内
        assert len(result["series"]) <= 200
        assert len(result["series"]) >= 1
        # series index 与价格最后 200 根对齐（最后若干日有值）
        last_dates = sample_prices.index[-200:]
        for idx in result["series"].index:
            assert idx in last_dates.values or idx in sample_prices.index.values

    def test_short_data_returns_error(self, indicators, sample_prices_short):
        result = indicators.calculate_bsadf(sample_prices_short)
        assert result.get("error") == "Not enough data"
        assert result.get("is_significant") is False


class TestCalculateGarchVar:
    """GARCH VaR 三重分布防线"""

    def test_keys_present_with_sample_prices(self, indicators, sample_prices):
        result = indicators.calculate_garch_var(sample_prices)
        assert "var_95_put" in result
        assert "var_95_call" in result
        assert "var_99_put" in result
        assert "var_99_call" in result
        assert "robust_vol" in result
        assert "jump_lambda_60" in result

    def test_var_positive_and_ordering(self, indicators, sample_prices):
        result = indicators.calculate_garch_var(sample_prices)
        assert result["var_95_put"] > 0
        assert result["var_95_call"] > 0
        assert result["var_99_put"] >= result["var_95_put"]
        assert result["var_99_call"] >= result["var_95_call"]

    def test_custom_confidence_levels(self, indicators, sample_prices):
        result = indicators.calculate_garch_var(
            sample_prices, confidence_levels=[0.95]
        )
        assert "var_95_put" in result
        assert "var_95_call" in result
        assert "error" not in result or result.get("var_95_put", 0) > 0


class TestCalculateDailyRv:
    """已实现波动率 RV"""

    def test_returns_two_days_positive(self, indicators, sample_hf_df):
        rv = indicators.calculate_daily_rv(sample_hf_df)
        assert len(rv) == 2
        assert (rv > 0).all()


class TestCalculateOtmLevel:
    """虚值程度 OTM"""

    def test_call_otm_positive_when_strike_above_spot(self, indicators):
        # spot=2.5, strike=2.6 -> call 虚值 (2.6-2.5)/2.5*100 = 4%
        otm = indicators.calculate_otm_level(2.5, 2.6, "call")
        assert otm == pytest.approx(4.0)

    def test_put_otm_positive_when_spot_above_strike(self, indicators):
        # spot=2.5, strike=2.3 -> put 虚值 (2.5-2.3)/2.5*100 = 8%
        otm = indicators.calculate_otm_level(2.5, 2.3, "put")
        assert otm == pytest.approx(8.0)

    def test_call_otm_negative_when_itr(self, indicators):
        # 认购实值: strike < spot -> (strike - spot) / spot * 100
        otm = indicators.calculate_otm_level(2.6, 2.5, "call")
        assert otm == pytest.approx((2.5 - 2.6) / 2.6 * 100, rel=0.01)

    def test_put_otm_negative_when_itr(self, indicators):
        # 认沽实值: spot < strike -> (spot - strike) / spot * 100
        otm = indicators.calculate_otm_level(2.3, 2.5, "put")
        assert otm == pytest.approx((2.3 - 2.5) / 2.3 * 100, rel=0.01)

    def test_auto_absolute_distance(self, indicators):
        otm = indicators.calculate_otm_level(2.5, 2.6, "auto")
        assert otm == pytest.approx(4.0)
        otm2 = indicators.calculate_otm_level(2.5, 2.4, "auto")
        assert otm2 == pytest.approx(4.0)

    def test_zero_spot_or_strike_returns_zero(self, indicators):
        assert indicators.calculate_otm_level(0, 2.5, "call") == 0.0
        assert indicators.calculate_otm_level(2.5, 0, "call") == 0.0


class TestCheckStopLoss:
    """止损触发逻辑"""

    def test_put_stop_triggered_when_otm_below_threshold(self, indicators):
        # spot=2.5, strike=2.4 -> put otm = 4% < 6.4 -> 应触发
        assert indicators.check_stop_loss(2.5, 2.4, "put", stop_otm=6.4) is True

    def test_put_stop_not_triggered_when_otm_above_threshold(self, indicators):
        # spot=2.5, strike=2.2 -> put otm = 12% > 6.4
        assert indicators.check_stop_loss(2.5, 2.2, "put", stop_otm=6.4) is False

    def test_call_stop_triggered_when_otm_below_threshold(self, indicators):
        # spot=2.5, strike=2.6 -> call otm = 4% < 6.4
        assert indicators.check_stop_loss(2.5, 2.6, "call", stop_otm=6.4) is True

    def test_call_stop_not_triggered_when_otm_above_threshold(self, indicators):
        # spot=2.5, strike=2.8 -> call otm = 12%
        assert indicators.check_stop_loss(2.5, 2.8, "call", stop_otm=6.4) is False


class TestCalculateGreeks:
    """Black-Scholes Greeks 计算"""

    def test_call_greeks_structure(self, indicators):
        result = indicators.calculate_greeks(
            spot=2.5, strike=2.6, time_to_expiry=30/365, volatility=0.25, option_type='call'
        )
        assert "delta" in result
        assert "gamma" in result
        assert "vega" in result
        assert "theta" in result
        assert "price" in result

    def test_call_delta_range(self, indicators):
        result = indicators.calculate_greeks(
            spot=2.5, strike=2.6, time_to_expiry=30/365, volatility=0.25, option_type='call'
        )
        assert 0 <= result["delta"] <= 1

    def test_put_delta_range(self, indicators):
        result = indicators.calculate_greeks(
            spot=2.5, strike=2.4, time_to_expiry=30/365, volatility=0.25, option_type='put'
        )
        assert -1 <= result["delta"] <= 0

    def test_gamma_positive(self, indicators):
        result = indicators.calculate_greeks(
            spot=2.5, strike=2.5, time_to_expiry=30/365, volatility=0.25, option_type='call'
        )
        assert result["gamma"] > 0

    def test_vega_positive(self, indicators):
        result = indicators.calculate_greeks(
            spot=2.5, strike=2.5, time_to_expiry=30/365, volatility=0.25, option_type='call'
        )
        assert result["vega"] > 0

    def test_invalid_inputs_return_zeros(self, indicators):
        result = indicators.calculate_greeks(
            spot=0, strike=2.5, time_to_expiry=30/365, volatility=0.25, option_type='call'
        )
        assert result["delta"] == 0.0
        assert result["gamma"] == 0.0


class TestCalculateMarketExposure:
    """GEX/DEX 市场暴露计算"""

    def test_returns_structure(self, indicators):
        df = pd.DataFrame({
            '名称': ['510050C2602M02500', '510050P2602M02400'],
            '行权价': [2.5, 2.4],
            '隐含波动率': [25.0, 28.0],
            '持仓量': [10000, 8000]
        })
        result = indicators.calculate_market_exposure(df, spot=2.45)
        assert "gex_call" in result
        assert "gex_put" in result
        assert "gex_net" in result
        assert "dex_call" in result
        assert "dex_put" in result
        assert "dex_net" in result
        assert "max_pain_strike" in result

    def test_empty_dataframe_returns_zeros(self, indicators):
        df = pd.DataFrame()
        result = indicators.calculate_market_exposure(df, spot=2.5)
        assert result["gex_net"] == 0.0
        assert result["dex_net"] == 0.0

    def test_max_pain_calculated(self, indicators):
        df = pd.DataFrame({
            '名称': ['510050C2602M02500', '510050P2602M02400'],
            '行权价': [2.5, 2.4],
            '隐含波动率': [25.0, 28.0],
            '持仓量': [10000, 8000]
        })
        result = indicators.calculate_market_exposure(df, spot=2.45)
        assert result["max_pain_strike"] > 0

    def test_uses_type_column_when_name_lacks_call_put_marker(self, indicators):
        """真实新浪缓存里名称可能不含方向，应回退到 类型/option_type 列判断认购认沽"""
        df = pd.DataFrame({
            '名称': ['510050', '510050'],
            '类型': ['认购', '认沽'],
            'option_type': ['认购', '认沽'],
            '行权价': [2.55, 2.45],
            '隐含波动率': [18.0, 20.0],
            '持仓量': [10000, 12000]
        })

        result = indicators.calculate_market_exposure(df, spot=2.50)

        assert "error" not in result
        assert result["total_oi_call"] == 10000
        assert result["total_oi_put"] == 12000
        assert result["gex_net"] != 0
