#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VolGuard Pro — 核心量化指标计算模块 (v6.0)

严格金融数学实现:
- BSADF 泡沫测试 (Phillips, Shi & Yu 2015) — 正统渐进临界值
- Multi-Dist GARCH VaR — Skew-t 真实分位数, 双侧 (Call/Put) 防线
- 已实现波动率 RV — 频率修正年化
"""

import os
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional

logger = logging.getLogger(__name__)


class StrategyIndicators:
    """策略指标计算类 — 纯无状态算法内核"""

    def calculate_bsadf(
        self,
        prices: pd.Series,
        window: int = 100,
        max_eval_points: int = 40,
        start_step: int = 8
    ) -> Dict:
        """
        计算 BSADF 泡沫测试 (Backward Supremum ADF)

        使用 Phillips, Shi & Yu (2015) 渐进临界值近似:
            cv ≈ 1.0 + 0.26 * log(T)   (5% 显著水平, 基于 PSY 2015 Table 1)

        参数:
            prices: 价格序列
            window: 最小滑窗 (r0)，默认 100
            max_eval_points: 最近最多评估多少个右端点；最终时点始终完整评估
            start_step: 历史曲线采样时的左端点步长，用于控制看板响应时间

        返回:
            dict: {adf_stat, is_significant, cv, series}
        """
        try:
            from statsmodels.tsa.stattools import adfuller

            log_prices = np.log(prices.dropna())
            n = len(log_prices)

            if n < window + 10:
                return {'error': 'Not enough data', 'is_significant': False, 'adf_stat': 0.0}

            # ===== 正统 PSY (2015) 渐进 5% 临界值 ========================
            # 基于 Monte Carlo 拟合: cv ≈ 1.0 + 0.26 * ln(sample_size)
            # 在 T=250 时 ≈ 2.43, T=500 时 ≈ 2.61 — 均为正的右尾极值
            critical_value = 1.0 + 0.26 * np.log(n)

            bsadf_series = pd.Series(index=log_prices.index, dtype=float)
            sup_adf = -np.inf

            start_search_idx = max(window, n - max_eval_points)
            eval_points = list(range(start_search_idx, n))

            for t_idx in eval_points:
                current_sup_adf = -np.inf
                left_candidates = range(max(0, t_idx - 250), t_idx - window + 1)
                if t_idx != n - 1 and start_step > 1:
                    left_candidates = range(max(0, t_idx - 250), t_idx - window + 1, start_step)
                # 右尾滑动起始点: 从 (t - 250) 到 (t - window) 保证计算充分
                for s_idx in left_candidates:
                    window_data = log_prices.iloc[s_idx: t_idx + 1]
                    try:
                        # regression='ct' 含常数+趋势项, 与 PSY 2015 标准一致
                        adf_stat = adfuller(window_data, regression='ct', autolag='AIC')[0]
                        if adf_stat > current_sup_adf:
                            current_sup_adf = adf_stat
                    except Exception:
                        continue

                bsadf_series.iloc[t_idx] = current_sup_adf
                if t_idx == n - 1:
                    sup_adf = current_sup_adf

            is_significant = sup_adf > critical_value

            return {
                'adf_stat': float(sup_adf),
                'is_significant': is_significant,
                'cv': float(critical_value),
                'series': bsadf_series.dropna()
            }

        except Exception as e:
            logger.error(f"BSADF calculation failed: {e}", exc_info=True)
            return {'error': str(e), 'is_significant': False, 'adf_stat': 0.0, 'cv': 2.0}

    def calculate_garch_var(
        self,
        prices: pd.Series,
        confidence_levels: Optional[List[float]] = None,
        window: int = 250
    ) -> Dict:
        """
        Multi-Distribution GARCH VaR — 三重分布防线

        参数:
            prices: 日线收盘价序列
            confidence_levels: 置信水平列表, 默认 [0.90, 0.95, 0.99]
            window: GARCH 拟合窗口 (250 个交易日 = 1 年)

        返回:
            dict — 包含 var_95_call, var_95_put 双向防线
        """
        try:
            from arch import arch_model
            import scipy.stats as stats

            if confidence_levels is None:
                confidence_levels = [0.90, 0.95, 0.99]

            returns = np.log(prices / prices.shift(1)).dropna()
            recent_returns = returns.iloc[-window:]
            data_scaled = recent_returns * 100  # arch 需要百分制
            results = {}

            # ════════════════════════════════════════════════════════
            # 全新优化: 引入 R 代码中的鲁棒波动率与跳跃频率 (λ)
            # ════════════════════════════════════════════════════════
            # 1. 稳健波动率 (剔除 1%和99% 异常值后的 SD), 防止 GARCH 预估被极值过度干扰
            lower_q = float(np.percentile(recent_returns, 1))
            upper_q = float(np.percentile(recent_returns, 99))
            filtered_rets = recent_returns[(recent_returns >= lower_q) & (recent_returns <= upper_q)]
            robust_sd = float(np.std(filtered_rets))
            results['robust_vol'] = robust_sd
            
            # 2. 60天跳跃频率 λ (过去60天内, 绝对涨跌幅超过 99% 分位数的频率)
            # 使用类似于 R 代码中的 rolling(60).mean()
            vol_threshold = abs(stats.norm.ppf(0.01)) * robust_sd  # 约 2.33 * robust_sd
            jumps = (np.abs(recent_returns) > vol_threshold).astype(int)
            lambda_60 = float(jumps.rolling(window=min(60, len(jumps))).mean().iloc[-1])
            results['jump_lambda_60'] = lambda_60
            
            logger.info(f"R-Style Metrics: Robust Vol={robust_sd*250**0.5:.2%}, Jump Lambda(60d)={lambda_60:.2%}")

            # ════════════════════════════════════════════════════════
            # 模型 1: 标准正态 GARCH(1,1)
            # ════════════════════════════════════════════════════════
            try:
                am_norm = arch_model(data_scaled, vol='Garch', p=1, q=1, dist='Normal')
                res_norm = am_norm.fit(disp='off')
                fc_norm = res_norm.forecast(horizon=1)
                sigma_norm = np.sqrt(fc_norm.variance.iloc[-1, 0]) / 100  # 还原到小数

                results['sigma_norm'] = float(sigma_norm)
                results['alpha_norm'] = float(res_norm.params.get('alpha[1]', 0.1))
                results['beta_norm'] = float(res_norm.params.get('beta[1]', 0.8))

                for cl in confidence_levels:
                    cl_str = "975" if cl == 0.975 else str(int(cl*100))
                    z = stats.norm.ppf(cl)          # 双侧: put 用左尾, call 用右尾
                    results[f'norm_{cl_str}_call'] = float(z * sigma_norm)   # 上行
                    results[f'norm_{cl_str}_put'] = float(z * sigma_norm)    # 正态对称

            except Exception as e:
                logger.warning(f"Normal GARCH failed: {e}")
                results['norm_error'] = str(e)

            # ════════════════════════════════════════════════════════
            # 模型 2: 偏态 t 分布 GARCH(1,1) — 使用 arch 内置 SkewStudent ppf
            # ════════════════════════════════════════════════════════
            try:
                am_skew = arch_model(data_scaled, vol='Garch', p=1, q=1, dist='skewstudent')
                res_skew = am_skew.fit(disp='off')
                fc_skew = res_skew.forecast(horizon=1)
                sigma_skew = np.sqrt(fc_skew.variance.iloc[-1, 0]) / 100

                results['sigma_skew'] = float(sigma_skew)

                nu = float(res_skew.params.get('nu', 10.0))
                nu = max(nu, 2.5)
                lam = float(res_skew.params.get('lambda', 0.0))

                skew_dist = am_skew.distribution
                dist_params = np.array([nu, lam])

                for cl in confidence_levels:
                    cl_str = "975" if cl == 0.975 else str(int(cl*100))
                    q_put_raw = skew_dist.ppf(1.0 - cl, dist_params)
                    q_call_raw = skew_dist.ppf(cl, dist_params)
                    results[f'skew_{cl_str}_put'] = float(abs(q_put_raw) * sigma_skew)
                    results[f'skew_{cl_str}_call'] = float(q_call_raw * sigma_skew)

            except Exception as e:
                logger.warning(f"Skew-t GARCH failed: {e}")
                results['skew_error'] = str(e)

            # ════════════════════════════════════════════════════════
            # 模型 3: 极端事件补偿缓冲 (Extreme Event Buffer)
            # 不再用 1.35 魔法数，改用历史最大单日损失的 99% 分位数补偿
            # ════════════════════════════════════════════════════════
            historical_99_quantile = float(np.percentile(np.abs(recent_returns), 99))
            sigma_base = results.get('sigma_skew', results.get('sigma_norm', 0.02))
            # 极端事件缓冲 = max(GARCH预测的2σ, 历史99%分位数)
            extreme_buf = max(2.0 * sigma_base, historical_99_quantile)
            results['extreme_buffer'] = extreme_buf
            results['sigma_jump'] = extreme_buf  # 向后兼容

            for cl in confidence_levels:
                cl_str = "975" if cl == 0.975 else str(int(cl*100))
                z = stats.norm.ppf(cl)
                results[f'jump_{cl_str}_put'] = float(z * extreme_buf)
                results[f'jump_{cl_str}_call'] = float(z * extreme_buf)

            # ════════════════════════════════════════════════════════
            # 合并输出: 取三重模型的最宽防线 (最保守)
            # 输出 1%, 2.5%, 5% 风险分位数 (对应 99%, 97.5%, 95% 置信度)
            # ════════════════════════════════════════════════════════
            def _widest(prefix_list: List[str]) -> float:
                vals = [results[k] for k in prefix_list if k in results]
                return max(vals) if vals else 0.02

            results['var_95_put'] = _widest(['skew_95_put', 'jump_95_put', 'norm_95_put'])
            results['var_95_call'] = _widest(['skew_95_call', 'jump_95_call', 'norm_95_call'])
            
            # 97.5% 置信度 (2.5% tail)
            results['var_975_put'] = _widest(['skew_975_put', 'jump_975_put', 'norm_975_put']) 
            results['var_975_call'] = _widest(['skew_975_call', 'jump_975_call', 'norm_975_call'])
            
            results['var_99_put'] = _widest(['skew_99_put', 'jump_99_put', 'norm_99_put'])
            results['var_99_call'] = _widest(['skew_99_call', 'jump_99_call', 'norm_99_call'])

            # 向后兼容键
            results['var_95'] = results['var_95_put']
            results['var_99'] = results['var_99_put']

            return results

        except Exception as e:
            logger.error(f"GARCH VaR calculation failed: {e}", exc_info=True)
            return {'error': str(e), 'var_95': 0.02, 'var_99': 0.03,
                    'var_95_put': 0.02, 'var_95_call': 0.02, 
                    'var_975_put': 0.025, 'var_975_call': 0.025,
                    'robust_vol': 0.02, 'jump_lambda_60': 0.0}

    def calculate_daily_rv(
        self,
        high_freq_df: pd.DataFrame,
        time_col: str = 'time',
        price_col: str = 'close',
        freq_minutes: int = 5
    ) -> pd.Series:
        """
        计算日内已实现波动率 (Realized Volatility, RV)

        修正: 年化需乘 sqrt(每日K线段数 × 250交易日)
            n_per_day = 240 / freq_minutes  (A股交易 240 分钟/日)
            annual_rv = intraday_rv × sqrt(n_per_day × 250)

        参数:
            high_freq_df: 高频 K 线 DataFrame
            time_col: 时间列名
            price_col: 价格列名
            freq_minutes: K 线频率(分钟), 默认 5
        """
        try:
            df = high_freq_df.copy()
            if time_col in df.columns:
                df[time_col] = pd.to_datetime(df[time_col])
                df.set_index(time_col, inplace=True)

            df['log_ret'] = np.log(df[price_col] / df[price_col].shift(1))

            # 按日分组, 计算日内 RV = sqrt(sum(r^2))
            daily_rv = df.groupby(df.index.date)['log_ret'].apply(
                lambda x: np.sqrt(np.nansum(x ** 2))
            )

            # 正确年化: n_per_day × 250 个交易日
            n_per_day = 240 // freq_minutes  # A股正常交易时间 240 分钟
            annual_rv = daily_rv * np.sqrt(n_per_day * 250)
            return annual_rv

        except Exception as e:
            logger.warning(f"RV calculation failed: {e}")
            return pd.Series(dtype=float)

    def calculate_otm_level(
        self,
        spot_price: float,
        strike_price: float,
        option_type: str = 'auto'
    ) -> float:
        """
        计算虚值程度 (Out-The-Money Distance)

        修正: 区分认购 (Call) 和认沽 (Put) 方向 —
            Call 虚值 = (strike - spot) / spot × 100  (仅当 strike > spot)
            Put  虚值 = (spot - strike) / spot × 100  (仅当 spot > strike)
            当期权实值时返回负值 (已击穿)

        参数:
            spot_price: 现货价格
            strike_price: 行权价
            option_type: 'call' | 'put' | 'auto' (auto 取绝对距离)
        """
        if spot_price <= 0 or strike_price <= 0:
            return 0.0

        if option_type == 'call':
            return float((strike_price - spot_price) / spot_price * 100)
        elif option_type == 'put':
            return float((spot_price - strike_price) / spot_price * 100)
        else:
            return float(abs(spot_price - strike_price) / spot_price * 100)

    def check_stop_loss(
        self,
        spot_price: float,
        strike_price: float,
        option_type: str = 'put',
        stop_otm: float = 6.4
    ) -> bool:
        """
        触发止损检查

        参数:
            spot_price: 现货价格
            strike_price: 行权价
            option_type: 期权类型 ('call' or 'put')
            stop_otm: 止损虚值阈值 (%)
        """
        current_otm = self.calculate_otm_level(spot_price, strike_price, option_type)
        return current_otm < stop_otm

    def calculate_greeks(
        self,
        spot: float,
        strike: float,
        time_to_expiry: float,
        volatility: float,
        risk_free_rate: float = 0.015,
        option_type: str = 'call'
    ) -> Dict:
        """
        Black-Scholes Greeks 计算 (Delta/Gamma/Vega/Theta)

        参数:
            spot: 标的现价
            strike: 行权价
            time_to_expiry: 到期时间(年), 如 30天 = 30/365
            volatility: 隐含波动率(小数), 如 0.25 = 25%
            risk_free_rate: 无风险利率, 默认 1.5%
            option_type: 'call' | 'put'

        返回:
            dict: {delta, gamma, vega, theta, price}
        """
        from scipy import stats
        
        if spot <= 0 or strike <= 0 or time_to_expiry <= 0 or volatility <= 0:
            return {'delta': 0.0, 'gamma': 0.0, 'vega': 0.0, 'theta': 0.0, 'price': 0.0}

        try:
            sqrt_t = np.sqrt(time_to_expiry)
            d1 = (np.log(spot / strike) + (risk_free_rate + 0.5 * volatility ** 2) * time_to_expiry) / (volatility * sqrt_t)
            d2 = d1 - volatility * sqrt_t

            nd1 = stats.norm.cdf(d1)
            nd2 = stats.norm.cdf(d2)
            nprime_d1 = stats.norm.pdf(d1)

            if option_type.lower() == 'call':
                delta = nd1
                price = spot * nd1 - strike * np.exp(-risk_free_rate * time_to_expiry) * nd2
                theta = (-spot * nprime_d1 * volatility / (2 * sqrt_t) 
                        - risk_free_rate * strike * np.exp(-risk_free_rate * time_to_expiry) * nd2) / 365
            else:
                delta = nd1 - 1
                price = strike * np.exp(-risk_free_rate * time_to_expiry) * stats.norm.cdf(-d2) - spot * stats.norm.cdf(-d1)
                theta = (-spot * nprime_d1 * volatility / (2 * sqrt_t) 
                        + risk_free_rate * strike * np.exp(-risk_free_rate * time_to_expiry) * stats.norm.cdf(-d2)) / 365

            gamma = nprime_d1 / (spot * volatility * sqrt_t)
            vega = spot * nprime_d1 * sqrt_t / 100

            return {
                'delta': float(delta),
                'gamma': float(gamma),
                'vega': float(vega),
                'theta': float(theta),
                'price': float(price)
            }

        except Exception as e:
            logger.warning(f"Greeks calculation failed: {e}")
            return {'delta': 0.0, 'gamma': 0.0, 'vega': 0.0, 'theta': 0.0, 'price': 0.0}

    def calculate_market_exposure(
        self,
        options_df: pd.DataFrame,
        spot: float,
        risk_free_rate: float = 0.015
    ) -> Dict:
        """
        计算市场 Gamma/Delta 暴露 (GEX/DEX)

        GEX = Σ(Gamma × OI × 100 × S²) / 10^9  (单位: 十亿)
        DEX = Σ(Delta × OI × 100 × S) / 10^9

        参数:
            options_df: 期权链, 需包含: 名称/行权价/隐含波动率/持仓量
            spot: 标的现价
            risk_free_rate: 无风险利率

        返回:
            dict: {gex_call, gex_put, gex_net, dex_call, dex_put, dex_net, 
                   total_oi_call, total_oi_put, max_pain_strike}
        """
        try:
            df = options_df.copy()
            
            for col in ['行权价', '隐含波动率', '持仓量']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            df = df.dropna(subset=['行权价', '隐含波动率', '持仓量'])
            
            if df.empty:
                return {'gex_net': 0.0, 'dex_net': 0.0, 'error': 'No valid data'}

            def _type(row):
                values = []
                for col in ['名称', '类型', 'option_type', '期权类型']:
                    if col in row.index:
                        values.append(str(row[col]))
                text = " ".join(values).upper()
                if '购' in text or '认购' in text or 'CALL' in text:
                    return 'call'
                if '沽' in text or '认沽' in text or 'PUT' in text:
                    return 'put'
                if 'C' in text and 'P' not in text:
                    return 'call'
                if 'P' in text and 'C' not in text:
                    return 'put'
                return 'unknown'

            df['type'] = df.apply(_type, axis=1)
            df = df[df['type'].isin(['call', 'put'])]

            from datetime import datetime, timedelta
            today = datetime.now()
            
            def _tte(name):
                try:
                    if isinstance(name, str) and len(name) >= 11:
                        year_month = name[7:11]
                        year = 2000 + int(year_month[:2])
                        month = int(year_month[2:])
                        first_day = datetime(year, month, 1)
                        first_wed = first_day + timedelta(days=(2 - first_day.weekday() + 7) % 7)
                        expiry = first_wed + timedelta(weeks=3)
                        days = (expiry - today).days
                        return max(1, days) / 365.0
                except Exception:
                    pass
                return 30 / 365.0
            
            df['tte'] = df['名称'].apply(_tte)

            gex_call, gex_put = 0.0, 0.0
            dex_call, dex_put = 0.0, 0.0
            oi_call, oi_put = 0.0, 0.0

            for _, row in df.iterrows():
                strike = row['行权价']
                iv = row['隐含波动率'] / 100.0 if row['隐含波动率'] > 1 else row['隐含波动率']
                oi = row['持仓量']
                tte = row['tte']
                opt_type = row['type']

                greeks = self.calculate_greeks(spot, strike, tte, iv, risk_free_rate, opt_type)
                
                gamma_exp = greeks['gamma'] * oi * 100 * spot * spot / 1e9
                delta_exp = greeks['delta'] * oi * 100 * spot / 1e9

                if opt_type == 'call':
                    gex_call += gamma_exp
                    dex_call += delta_exp
                    oi_call += oi
                else:
                    gex_put += gamma_exp
                    dex_put += delta_exp
                    oi_put += oi

            gex_net = gex_call - gex_put
            dex_net = dex_call + dex_put

            strikes = df['行权价'].unique()
            max_pain = spot
            if len(strikes) > 0:
                pain_scores = []
                for k in strikes:
                    call_pain = df[df['type'] == 'call']['持仓量'].sum() * max(0, spot - k)
                    put_pain = df[df['type'] == 'put']['持仓量'].sum() * max(0, k - spot)
                    pain_scores.append((k, call_pain + put_pain))
                max_pain = min(pain_scores, key=lambda x: x[1])[0] if pain_scores else spot

            return {
                'gex_call': float(gex_call),
                'gex_put': float(gex_put),
                'gex_net': float(gex_net),
                'dex_call': float(dex_call),
                'dex_put': float(dex_put),
                'dex_net': float(dex_net),
                'total_oi_call': float(oi_call),
                'total_oi_put': float(oi_put),
                'max_pain_strike': float(max_pain)
            }

        except Exception as e:
            logger.error(f"Market exposure calculation failed: {e}", exc_info=True)
            return {'gex_net': 0.0, 'dex_net': 0.0, 'error': str(e)}


if __name__ == "__main__":
    print("StrategyIndicators import OK")
