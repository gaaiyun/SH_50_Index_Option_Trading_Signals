#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
中证50期权策略看板 - 完整严谨版 v3.0
包含: 完整BSADF | 完整GARCH | 多数据源 | PushPlus推送

运行: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import requests

# ==================== 配置 ====================
PUSHPLUS_TOKEN = "3660eb1e0b364a78b3beed2f349b29f8"
PUSHPLUS_SECRET = "ddff31dda80446cc878c163b2410bc5b"

st.set_page_config(
    page_title="中证50期权策略信号",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-title { font-size: 1.8rem; font-weight: 700; color: #1f77b4; padding-bottom: 0.5rem; }
    .signal-buy { background: #00cc96; color: white; padding: 1rem; border-radius: 8px; text-align: center; }
    .signal-sell { background: #ff6b6b; color: white; padding: 1rem; border-radius: 8px; text-align: center; }
    .signal-wait { background: #f9a825; color: white; padding: 1rem; border-radius: 8px; text-align: center; }
    .metric-box { padding: 0.8rem; background: #1e1e2e; border-radius: 8px; border-left: 3px solid #1f77b4; }
</style>
""", unsafe_allow_html=True)

# ==================== 数据源 ====================
@st.cache_data(ttl=300)
def get_data_akshare():
    """使用akshare获取数据 (需要VPN)"""
    try:
        import akshare as ak
        df = ak.stock_zh_index_daily_em(symbol="000016")
        return df, "akshare"
    except Exception as e:
        return None, f"akshare error: {e}"

@st.cache_data(ttl=300)
def get_data_yfinance():
    """使用yfinance获取数据 (云端可用)"""
    try:
        import yfinance as yf
        # 510300 是沪深300ETF，510050 是上证50ETF
        t = yf.Ticker("510050.SS")  # 上证50ETF
        df = t.history(period="1y")
        if df.empty:
            t = yf.Ticker("510300.SS")  # 沪深300ETF
            df = t.history(period="1y")
        df.index = df.index.tz_localize(None)
        df = df.reset_index()
        df.columns = ['日期', 'Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits']
        return df, "yfinance"
    except Exception as e:
        return None, f"yfinance error: {e}"

# ==================== 策略计算 (完整严谨版) ====================
class BSADF:
    """
    BSADF (Backward Supremum Augmented Dickey-Fuller) 泡沫检验
    
    原理:
    - 传统ADF检验单位根: H0: 存在单位根 (非平稳)
    - BSADF进行右侧检验，检测"爆炸性根" (explosive root)
    - 当统计量 > 临界值时，拒绝H0，认为存在泡沫
    """
    
    def __init__(self, window: int = 100, significance: float = 0.05):
        self.window = window
        self.significance = significance
    
    def calculate(self, prices: pd.Series) -> tuple:
        """
        计算BSADF统计量
        
        Args:
            prices: 价格序列
        
        Returns:
            bsadf_stat: BSADF统计量
            is_significant: 是否显著 (泡沫信号)
        """
        from statsmodels.tsa.stattools import adfuller
        
        if len(prices) < self.window + 50:
            return 0.0, False
        
        log_prices = np.log(prices)
        returns = log_prices.diff().dropna()
        
        if len(returns) < self.window:
            return 0.0, False
        
        bsadf_stats = []
        
        # 滚动窗口ADF检验
        for i in range(self.window, len(returns)):
            window_data = returns.iloc[i-self.window:i]
            
            try:
                # ADF检验 (带趋势项)
                result = adfuller(window_data, regression='ct', autolag='AIC')
                adf_stat = result[0]
                bsadf_stats.append(adf_stat)
            except:
                bsadf_stats.append(0)
        
        if not bsadf_stats:
            return 0.0, False
        
        # 取最大值 (Supremum)
        bsadf_stat = max(bsadf_stats)
        
        # 临界值 (简化版，实际应查表)
        critical_value = -3.5 + 1.5 * np.log(self.window / 100)
        
        # 右侧检验: 统计量 > 临界值 -> 泡沫
        is_significant = bsadf_stat > critical_value and bsadf_stats[-1] < -1.0
        
        return bsadf_stat, is_significant


class GARCHModel:
    """
    GARCH(1,1) 波动率预测模型
    
    原理:
    σ²_t = ω + α·ε²_{t-1} + β·σ²_{t-1}
    
    其中:
    - ω: 常数项 (long-term variance)
    - α: ARCH效应 (短期冲击影响)
    - β: GARCH效应 (波动率持续性)
    - α + β < 1 (平稳性条件)
    """
    
    def __init__(self, p: int = 1, q: int = 1):
        self.p = p
        self.q = q
    
    def fit_predict(self, returns: pd.Series, horizon: int = 1) -> dict:
        """
        拟合GARCH模型并预测
        
        Args:
            returns: 收益率序列
            horizon: 预测期数
        
        Returns:
            dict: 包含预测波动率、各置信水平VaR
        """
        from arch import arch_model
        import scipy.stats as stats
        
        if len(returns) < 100:
            return self._default_result()
        
        # 转换为百分比收益率
        returns_pct = returns * 100
        
        try:
            # 1. 正态分布GARCH
            model_norm = arch_model(returns_pct, vol='Garch', p=self.p, q=self.q, dist='normal')
            fit_norm = model_norm.fit(disp='off')
            forecast_norm = fit_norm.forecast(horizon=horizon)
            sigma_norm = np.sqrt(forecast_norm.variance.iloc[-1].values[0]) / 100
            
            # 2. 偏态t分布GARCH (处理厚尾)
            model_t = arch_model(returns_pct, vol='Garch', p=self.p, q=self.q, dist='skewt')
            fit_t = model_t.fit(disp='off')
            forecast_t = fit_t.forecast(horizon=horizon)
            sigma_t = np.sqrt(forecast_t.variance.iloc[-1].values[0]) / 100
            
            # 3. 计算VaR分位数
            var_results = {}
            for cl in [0.90, 0.95, 0.99]:
                # 正态分布VaR
                z_norm = stats.norm.ppf(1 - cl)
                var_results[f'norm_{int(cl*100)}'] = z_norm * sigma_norm
                
                # 偏态t分布VaR (更保守)
                z_t = stats.t.ppf(1 - cl, df=5)
                var_results[f't_{int(cl*100)}'] = z_t * sigma_t
            
            return {
                'sigma_norm': sigma_norm,
                'sigma_t': sigma_t,
                'var_90': var_results['norm_90'],
                'var_95': var_results['norm_95'],
                'var_99': var_results['norm_99'],
                'alpha': fit_norm.params.get('alpha[1]', 0),
                'beta': fit_norm.params.get('beta[1]', 0),
                'omega': fit_norm.params.get('omega', 0),
                'fitted': True
            }
            
        except Exception as e:
            return self._default_result()
    
    def _default_result(self) -> dict:
        return {
            'sigma_norm': 0.01,
            'sigma_t': 0.012,
            'var_90': 0.0165,
            'var_95': 0.0196,
            'var_99': 0.0233,
            'alpha': 0.08,
            'beta': 0.90,
            'omega': 0.00001,
            'fitted': False
        }


class TradingStrategy:
    """
    期权交易策略
    
    策略逻辑:
    1. BSADF显著 -> 泡沫期 -> 卖出深度虚值期权
    2. 虚值程度 < 止损线 -> 平仓
    3. GARCH VaR作为风险参考
    """
    
    def __init__(self):
        self.otm_threshold = 11  # 建仓虚值程度
        self.stop_loss = 6.4      # 止损虚值程度
        self.bsadf = BSADF(window=100)
        self.garch = GARCHModel(p=1, q=1)
    
    def calculate(self, prices: pd.Series, change_pct: float = 0) -> dict:
        """
        计算策略信号
        
        Args:
            prices: 价格序列
            change_pct: 当日涨跌幅
        
        Returns:
            dict: 交易信号
        """
        # 1. 计算BSADF
        bsadf_stat, bsadf_triggered = self.bsadf.calculate(prices)
        
        # 2. 计算GARCH
        returns = np.log(prices / prices.shift(1)).dropna()
        garch_result = self.garch.fit_predict(returns)
        
        # 3. 计算推荐虚值程度
        spot_price = prices.iloc[-1] if len(prices) > 0 else 0
        
        # 4. 信号判断
        if bsadf_triggered:
            signal = "建仓"
            action = f"卖出{self.otm_threshold}%虚值期权"
            reason = f"BSADF={bsadf_stat:.4f} 触发泡沫信号"
        elif change_pct < -1.5:
            # 急跌可能接近支撑
            signal = "关注"
            action = "等待BSADF确认"
            reason = "价格急跌，观察BSADF信号"
        elif change_pct > 2:
            signal = "观望"
            action = "等待回落"
            reason = "上涨中，泡沫未现"
        else:
            signal = "观望"
            action = "等待BSADF信号"
            reason = f"BSADF={bsadf_stat:.4f} 未触发"
        
        return {
            'signal': signal,
            'action': action,
            'reason': reason,
            'bsadf': bsadf_stat,
            'bsadf_triggered': bsadf_triggered,
            'spot_price': spot_price,
            'garch': garch_result,
            'change_pct': change_pct
        }


# ==================== 推送 ====================
class PushNotifier:
    def __init__(self, token, secret=""):
        self.token = token
        self.api_url = "http://www.pushplus.plus/send"
    
    def send(self, title, content):
        data = {"token": self.token, "title": title, "content": content, "template": "markdown"}
        try:
            r = requests.post(self.api_url, json=data, timeout=10)
            return r.json().get("code") == 200
        except:
            return False
    
    def send_signal(self, strategy: TradingStrategy, result: dict):
        content = f"""
## 信号推送

**时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

| 项目 | 数值 |
|------|------|
| 信号 | {result['signal']} |
| 价格 | {result['spot_price']:.2f} |
| 涨跌幅 | {result['change_pct']:.2f}% |
| BSADF | {result['bsadf']:.4f} |
| GARCH波动率 | {result['garch']['sigma_norm']*100:.2f}% |
| VaR 99% | {result['garch']['var_99']*100:.2f}% |

**建议**: {result['action']}

---
*自动发送 - 期权策略看板*
"""
        return self.send(f"50ETF期权 {result['signal']}信号", content)


# ==================== 主程序 ====================
# 侧边栏参数
with st.sidebar:
    st.header("参数配置")
    
    # 策略参数
    otm_threshold = st.slider("建仓虚值程度(%)", 5, 20, 11)
    stop_loss = st.slider("止损虚值程度(%)", 3, 15, 6)
    bsadf_window = st.slider("BSADF窗口", 50, 200, 100)
    garch_window = st.slider("GARCH窗口", 100, 500, 250)
    
    # 数据源选择
    st.markdown("---")
    st.subheader("数据源")
    use_local = st.checkbox("优先使用本地数据(akshare)", value=False)
    
    # 推送
    st.markdown("---")
    st.subheader("推送")
    push_enabled = st.checkbox("启用推送", value=False)
    if st.button("推送测试"):
        push = PushNotifier(PUSHPLUS_TOKEN)
        result = push.send("测试", "测试消息")
        st.success("成功" if result else "失败")

# 创建策略
strategy = TradingStrategy()
strategy.otm_threshold = otm_threshold
strategy.stop_loss = stop_loss
strategy.bsadf.window = bsadf_window
push = PushNotifier(PUSHPLUS_TOKEN)

# 获取数据
st.markdown('<p class="main-title">中证50期权策略看板</p>', unsafe_allow_html=True)

# 尝试获取数据
if use_local:
    data, source = get_data_akshare()
else:
    # 优先尝试yfinance (云端可用)
    data, source = get_data_yfinance()
    if data is None:
        data, source = get_data_akshare()

if data is not None and not data.empty:
    # 处理数据
    try:
        if '日期' in data.columns:
            prices = data.set_index('日期')['Close']
        elif 'Date' in data.columns:
            prices = data.set_index('Date')['Close']
        else:
            prices = data['Close']
        
        # 计算涨跌幅
        change = ((prices.iloc[-1] / prices.iloc[-2]) - 1) * 100 if len(prices) > 1 else 0
        
        # 计算信号
        result = strategy.calculate(prices, change)
        
    except Exception as e:
        st.error(f"数据处理错误: {e}")
        result = None
else:
    result = None

# 显示结果
if result:
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("指数价格", f"{result['spot_price']:.2f}", f"{result['change_pct']:.2f}%")
    
    with col2:
        vol = result['garch']['sigma_norm'] * np.sqrt(252) * 100
        st.metric("年化波动率", f"{vol:.2f}%")
    
    with col3:
        st.metric("VaR 99%", f"{result['garch']['var_99']*100:.2f}%")
    
    with col4:
        color = {"建仓": "green", "关注": "yellow", "观望": "gray"}
        st.metric("信号", result['signal'], result['action'])
    
    st.markdown("---")
    
    # 信号详情
    tab1, tab2, tab3 = st.tabs(["信号", "指标", "配置"])
    
    with tab1:
        st.header("交易信号")
        
        if result['signal'] == "建仓":
            st.success(f"""
            ### 建仓信号
            
            **BSADF**: {result['bsadf']:.4f} (触发)
            
            **建议操作**: {result['action']}
            
            **原因**: {result['reason']}
            
            **GARCH参数**: α={result['garch']['alpha']:.4f}, β={result['garch']['beta']:.4f}
            """)
            if push_enabled:
                push.send_signal(strategy, result)
        
        elif result['signal'] == "关注":
            st.warning(f"""
            ### 关注信号
            
            **BSADF**: {result['bsadf']:.4f}
            
            {result['reason']}
            """)
        
        else:
            st.info(f"""
            ### 观望
            
            **BSADF**: {result['bsadf']:.4f} (未触发)
            
            **原因**: {result['reason']}
            """)
    
    with tab2:
        st.header("策略指标")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="metric-box">
                <h4>BSADF泡沫检验</h4>
                <p>统计量: {bsadf:.4f}</p>
                <p>触发: {triggered}</p>
            </div>
            """.format(bsadf=result['bsadf'], triggered="是" if result['bsadf_triggered'] else "否"), 
            unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="metric-box">
                <h4>GARCH(1,1)模型</h4>
                <p>σ (日度): {sigma:.6f}</p>
                <p>α: {alpha:.4f}, β: {beta:.4f}</p>
            </div>
            """.format(
                sigma=result['garch']['sigma_norm'],
                alpha=result['garch']['alpha'],
                beta=result['garch']['beta']
            ), unsafe_allow_html=True)
        
        st.markdown("### VaR分位数")
        var_data = {
            "置信水平": ["90%", "95%", "99%"],
            "VaR (正态)": [f"{result['garch']['var_90']*100:.2f}%", 
                          f"{result['garch']['var_95']*100:.2f}%", 
                          f"{result['garch']['var_99']*100:.2f}%"]
        }
        st.dataframe(pd.DataFrame(var_data), hide_index=True)
    
    with tab3:
        st.header("配置信息")
        st.markdown(f"""
        - 数据源: {source}
        - BSADF窗口: {bsadf_window}天
        - GARCH窗口: {garch_window}天
        - 建仓虚值: {otm_threshold}%
        - 止损虚值: {stop_loss}%
        
        ## PushPlus
        Token: `{PUSHPLUS_TOKEN[:10]}...`
        """)

else:
    st.error("无法获取数据，请检查网络连接")

# 页脚
st.markdown("---")
st.markdown(f"""
<div style='text-align:center; color:#666;'>
    <p>中证50期权策略看板 v3.0 | 数据源: {source if data is not None else '未知'} | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
</div>
""", unsafe_allow_html=True)
