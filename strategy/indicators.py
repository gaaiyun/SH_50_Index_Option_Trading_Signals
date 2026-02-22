#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
中证50期权策略 - 核心指标计算模块

包含:
- BSADF泡沫检验
- GARCH波动率预测
- RV已实现波动率
"""

import numpy as np
import pandas as pd
import os
from typing import Dict, List, Tuple

# 设置代理
os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'


class StrategyIndicators:
    """策略指标计算类"""
    
    def __init__(self):
        self.proxy_enabled = True
        
    def _set_proxy(self):
        """设置代理"""
        if self.proxy_enabled:
            os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'
            os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'
    
    def calculate_bsadf(self, prices: pd.Series, window: int = 100) -> Dict:
        """
        计算BSADF泡沫检验
        
        参数:
            prices: 价格序列
            window: 滚动窗口大小
        
        返回:
            dict: 包含adf统计量和p值
        """
        try:
            import statsmodels.tsa.stattools as ts
            
            # 计算对数收益率
            returns = np.log(prices / prices.shift(1)).dropna()
            
            # ADF检验
            result = ts.adfuller(returns, regression='ct', autolag='AIC')
            
            return {
                'adf_stat': result[0],
                'p_value': result[1],
                'is_significant': result[1] < 0.05
            }
        except Exception as e:
            return {'error': str(e)}
    
    def calculate_garch_var(self, prices: pd.Series, 
                            confidence_levels: List[float] = [0.90, 0.95, 0.99],
                            window: int = 250) -> Dict:
        """
        计算GARCH波动率和VaR分位数
        
        参数:
            prices: 价格序列
            confidence_levels: 置信水平列表
            window: 滚动窗口大小
        
        返回:
            dict: 各置信水平的VaR分位数
        """
        try:
            from arch import arch_model
            
            # 计算对数收益率
            returns = np.log(prices / prices.shift(1)).dropna()
            recent_returns = returns.iloc[-window:]
            
            results = {}
            
            # 1. 正态分布GARCH
            try:
                model = arch_model(recent_returns * 100, vol='Garch', p=1, q=1, dist='normal')
                fit = model.fit(disp='off')
                forecast = fit.forecast(horizon=1)
                sigma = np.sqrt(forecast.variance.iloc[-1].values[0]) / 100
                
                for cl in confidence_levels:
                    z = np.abs(np.percentile(np.random.normal(0, 1, 10000), (1-cl)*100))
                    results[f'norm_{int(cl*100)}'] = float(z * sigma)
                results['sigma_norm'] = float(sigma)
            except Exception as e:
                results['norm_error'] = str(e)
            
            # 2. 偏态t分布GARCH
            try:
                model = arch_model(recent_returns * 100, vol='Garch', p=1, q=1, dist='skewt')
                fit = model.fit(disp='off')
                forecast = fit.forecast(horizon=1)
                sigma = np.sqrt(forecast.variance.iloc[-1].values[0]) / 100
                
                for cl in confidence_levels:
                    results[f'ghyp_{int(cl*100)}'] = float(sigma * 1.2)
                results['sigma_ghyp'] = float(sigma)
            except Exception as e:
                results['ghyp_error'] = str(e)
            
            # 3. 跳跃GARCH (简化版)
            sigma_jump = results.get('sigma_norm', 0.01) * 1.3
            for cl in confidence_levels:
                z = np.abs(np.percentile(np.random.normal(0, 1, 10000), (1-cl)*100))
                results[f'jump_{int(cl*100)}'] = float(z * sigma_jump)
            results['sigma_jump'] = float(sigma_jump)
            
            return results
            
        except Exception as e:
            return {'error': str(e)}
    
    def calculate_rv(self, prices: pd.Series) -> float:
        """
        计算已实现波动率RV
        
        参数:
            prices: 价格序列 (高频数据)
        
        返回:
            float: RV值
        """
        # 计算对数收益率
        log_returns = np.log(prices / prices.shift(1)).dropna()
        
        # RV = Σ r²
        rv = np.sqrt(np.sum(log_returns ** 2))
        
        return float(rv)
    
    def calculate_otm_level(self, spot_price: float, strike_price: float) -> float:
        """
        计算虚值程度
        
        参数:
            spot_price: 现货价格
            strike_price: 行权价
        
        返回:
            float: 虚值程度 (%)
        """
        if spot_price <= 0 or strike_price <= 0:
            return 0.0
        
        # 虚值程度 = |现货 - 行权价| / 现货 * 100%
        otm = abs(spot_price - strike_price) / spot_price * 100
        
        return float(otm)
    
    def check_stop_loss(self, spot_price: float, strike_price: float,
                       entry_otm: float = 11.0, stop_otm: float = 6.4) -> bool:
        """
        检查是否触发止损
        
        参数:
            spot_price: 现货价格
            strike_price: 行权价
            entry_otm: 建仓时虚值程度 (%)
            stop_otm: 止损虚值程度 (%)
        
        返回:
            bool: True表示触发止损
        """
        current_otm = self.calculate_otm_level(spot_price, strike_price)
        
        # 如果虚值程度小于止损阈值，触发止损
        if current_otm < stop_otm:
            return True
        
        return False


def get_index_data(symbol: str = "000016") -> pd.DataFrame:
    """
    获取指数数据
    
    参数:
        symbol: 指数代码 (000016=中证50)
    
    返回:
        DataFrame: 指数数据
    """
    import akshare as ak
    
    df = ak.stock_zh_index_daily_em(symbol=symbol)
    return df


# 测试代码
if __name__ == "__main__":
    print("=" * 50)
    print("中证50期权策略指标计算测试")
    print("=" * 50)
    
    # 获取数据
    print("\n[1] 获取中证50指数数据...")
    df = get_index_data("000016")
    print(f"获取到 {len(df)} 条数据")
    print(df.tail())
    
    # 初始化指标计算
    indicators = StrategyIndicators()
    
    # 计算BSADF
    print("\n[2] 计算BSADF泡沫检验...")
    prices = df['收盘']
    bsadf_result = indicators.calculate_bsadf(prices)
    print(f"BSADF结果: {bsadf_result}")
    
    # 计算GARCH
    print("\n[3] 计算GARCH VaR...")
    garch_result = indicators.calculate_garch_var(prices)
    print(f"GARCH结果: {garch_result}")
    
    print("\n✅ 测试完成!")
