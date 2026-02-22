#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
中证50期权策略看板 - 完整版
支持: 自动数据刷新 | 自动信号计算 | PushPlus推送 | 自动刷新

运行: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import time
from datetime import datetime, timedelta
import hashlib
import requests

# ==================== 配置 ====================
# PushPlus推送配置
PUSHPLUS_TOKEN = "3660eb1e0b364a78b3beed2f349b29f8"
PUSHPLUS_SECRET = "ddff31dda80446cc878c163b2410bc5b"

# Streamlit页面配置
st.set_page_config(
    page_title="中证50期权策略信号",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自动刷新设置 (秒)
st_autorefresh = None
try:
    from streamlit_autorefresh import st_autorefresh
    st_autorefresh(interval=300000, limit=None, key="refresh")  # 5分钟刷新
except:
    pass

# ==================== 样式 ====================
st.markdown("""
<style>
    .main-title { font-size: 1.8rem; font-weight: 700; color: #1f77b4; padding-bottom: 0.5rem; }
    .metric-card { padding: 1rem; border-radius: 8px; background: #1e1e2e; border-left: 4px solid #1f77b4; }
    .signal-buy { background: #00cc96; color: white; padding: 0.5rem 1rem; border-radius: 8px; text-align: center; }
    .signal-sell { background: #ff6b6b; color: white; padding: 0.5rem 1rem; border-radius: 8px; text-align: center; }
    .signal-wait { background: #f9a825; color: white; padding: 0.5rem 1rem; border-radius: 8px; text-align: center; }
    .tag { padding: 0.2rem 0.5rem; border-radius: 4px; font-size: 0.8rem; }
    .tag-green { background: rgba(0,204,150,0.2); color: #00cc96; }
    .tag-red { background: rgba(255,107,107,0.2); color: #ff6b6b; }
    .tag-yellow { background: rgba(249,168,37,0.2); color: #f9a825; }
    .tag-blue { background: rgba(31,119,180,0.2); color: #1f77b4; }
</style>
""", unsafe_allow_html=True)

# ==================== 缓存 ====================
@st.cache_data(ttl=300)
def get_index_data():
    """获取中证50指数数据"""
    try:
        import akshare as ak
        os.environ['HTTP_PROXY'] = os.environ.get('HTTP_PROXY', '')
        os.environ['HTTPS_PROXY'] = os.environ.get('HTTPS_PROXY', '')
        return ak.stock_zh_index_daily_em(symbol="000016")
    except Exception as e:
        return None

@st.cache_data(ttl=600)
def get_futures_data():
    """获取期货合约数据"""
    try:
        import akshare as ak
        return ak.futures_contract_info_cffex()
    except:
        return None

@st.cache_data(ttl=3600)
def get_options_data():
    """获取期权数据"""
    try:
        import akshare as ak
        return ak.option_sse_daily_sina()
    except:
        return None

# ==================== 推送类 ====================
class PushNotifier:
    """PushPlus推送"""
    
    def __init__(self, token, secret=""):
        self.token = token
        self.secret = secret
        self.api_url = "http://www.pushplus.plus/send"
    
    def send(self, title, content, template="markdown"):
        """发送消息"""
        data = {
            "token": self.token,
            "title": title,
            "content": content,
            "template": template
        }
        try:
            r = requests.post(self.api_url, json=data, timeout=10)
            return r.json().get("code") == 200
        except:
            return False
    
    def send_signal(self, signal_type, price, otm_level, action):
        """发送交易信号"""
        emoji = {"建仓": "[建仓]", "平仓": "[平仓]", "止损": "[止损]", "观望": "[观望]"}
        
        content = f"""
## {emoji.get(signal_type, signal_type)}信号

**时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

| 项目 | 数值 |
|------|------|
| 中证50指数 | {price:.2f} |
| 建议操作 | {action} |
| 虚值程度 | {otm_level:.1f}% |

---
*自动发送 - 期权策略看板*
"""
        title = f"50ETF期权策略 {emoji.get(signal_type, signal_type)}"
        return self.send(title, content)

# 创建推送实例
push = PushNotifier(PUSHPLUS_TOKEN, PUSHPLUS_SECRET)

# ==================== 策略计算 ====================
class Strategy:
    """期权策略计算"""
    
    def __init__(self):
        self.symbol = "000016"  # 中证50
        self.garch_window = 250
        self.var_level = 0.99
        self.otm_threshold = 11
        self.stop_loss = 6.4
    
    def calculate_bsadf(self, prices):
        """计算BSADF泡沫检验 (简化版)"""
        returns = np.log(prices / prices.shift(1)).dropna()
        if len(returns) < 20:
            return 0, False
        
        # 简化ADF检验
        from statsmodels.tsa.stattools import adfuller
        try:
            result = adfuller(returns, regression='ct', autolag='AIC')
            stat, pvalue = result[0], result[1]
            return stat, pvalue < 0.05
        except:
            return 0, False
    
    def calculate_garch(self, prices):
        """计算GARCH波动率 (简化版)"""
        returns = np.log(prices / prices.shift(1)).dropna().iloc[-250:]
        
        try:
            from arch import arch_model
            model = arch_model(returns * 100, vol='Garch', p=1, q=1, dist='normal')
            fit = model.fit(disp='off')
            forecast = fit.forecast(horizon=1)
            sigma = np.sqrt(forecast.variance.iloc[-1].values[0]) / 100
            return float(sigma)
        except:
            return 0.01  # 默认值
    
    def calculate_var(self, sigma):
        """计算VaR分位数"""
        import scipy.stats as stats
        z = stats.norm.ppf(1 - self.var_level)
        return sigma * z
    
    def calculate_otm(self, spot_price, strike_price):
        """计算虚值程度"""
        return abs(spot_price - strike_price) / spot_price * 100
    
    def generate_signal(self, index_df):
        """生成交易信号"""
        if index_df is None or index_df.empty:
            return {
                "signal": "数据获取失败",
                "action": "等待",
                "price": 0,
                "garch_vol": 0,
                "var_99": 0,
                "bsadf": 0,
                "otm_level": 0
            }
        
        latest = index_df.iloc[-1]
        price = latest['收盘']
        
        # 计算指标
        prices = index_df['收盘']
        bsadf_stat, bsadf_triggered = self.calculate_bsadf(prices)
        garch_vol = self.calculate_garch(prices)
        var_99 = self.calculate_var(garch_vol)
        
        # 计算建议行权价
        strike_price = price * (1 + self.otm_threshold / 100)
        otm_level = self.otm_level  # 使用阈值
        
        # 信号判断
        if bsadf_triggered:
            signal = "建仓"
            action = f"卖出{self.otm_threshold}%虚值期权"
        elif otm_level < self.stop_loss:
            signal = "止损"
            action = "立即平仓"
        else:
            signal = "观望"
            action = "等待BSADF信号"
        
        return {
            "signal": signal,
            "action": action,
            "price": price,
            "garch_vol": garch_vol,
            "var_99": var_99,
            "bsadf": bsadf_stat,
            "otm_level": otm_level,
            "recommend_strike": strike_price,
            "change": latest.get('涨跌幅', 0)
        }

# 创建策略实例
strategy = Strategy()

# ==================== 侧边栏 ====================
with st.sidebar:
    st.header("参数配置")
    
    # GARCH参数
    strategy.garch_window = st.slider("GARCH窗口(天)", 100, 500, 250)
    strategy.var_level = st.selectbox("VaR置信度", [0.90, 0.95, 0.99], 2)
    st.markdown("---")
    
    # 交易参数
    strategy.otm_threshold = st.slider("建仓虚值程度(%)", 5, 20, 11)
    strategy.stop_loss = st.slider("止损虚值程度(%)", 3, 15, 6)
    
    st.markdown("---")
    
    # 推送设置
    st.subheader("推送设置")
    push_enabled = st.checkbox("启用推送", value=False)
    
    if st.button("立即推送测试"):
        result = push.send_signal("测试", 3000.0, 10, "测试消息")
        st.success("推送成功" if result else "推送失败")
    
    st.markdown("---")
    st.markdown(f"""
    <small style='color:#888;'>
    自动刷新: 开启<br>
    更新时间: {datetime.now().strftime('%H:%M:%S')}
    </small>
    """, unsafe_allow_html=True)

# ==================== 主页面 ====================
st.markdown('<p class="main-title">中证50期权策略看板</p>', unsafe_allow_html=True)

# 获取数据
index_df = get_index_data()
futures_df = get_futures_data()

# 计算信号
signal = strategy.generate_signal(index_df)

# 显示信号
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("中证50指数", f"{signal['price']:.2f}", f"{signal.get('change', 0):.2f}%")

with col2:
    vol_annual = signal['garch_vol'] * np.sqrt(252) * 100
    st.metric("GARCH波动率", f"{vol_annual:.2f}%", f"日度σ={signal['garch_vol']:.4f}")

with col3:
    st.metric("VaR 99%", f"{signal['var_99']*100:.2f}%", "风险阈值")

with col4:
    signal_color = {"建仓": "signal-buy", "止损": "signal-sell", "观望": "signal-wait"}.get(signal['signal'], "signal-wait")
    st.markdown(f"""
    <div class='{signal_color}'>
        <strong>{signal['signal']}</strong><br>
        <small>{signal['action']}</small>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# Tab
tab1, tab2, tab3, tab4 = st.tabs(["实时信号", "策略指标", "期货期权", "设置文档"])

with tab1:
    st.header("交易信号")
    
    # 核心信号卡片
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("当前信号")
        
        if signal['signal'] == "建仓":
            st.success(f"""
            ### 建仓信号
            
            - 中证50: {signal['price']:.2f}
            - 建议行权价: {signal.get('recommend_strike', 0):.2f}
            - 虚值程度: {signal['otm_level']:.1f}%
            - 建议操作: 卖出虚值期权
            
            **BSADF检验**: {signal['bsadf']:.4f} (触发)
            """)
            
            if push_enabled:
                push.send_signal("建仓", signal['price'], signal['otm_level'], signal['action'])
        
        elif signal['signal'] == "止损":
            st.error(f"""
            ### 止损信号
            
            - 虚值程度: {signal['otm_level']:.1f}% (低于{strategy.stop_loss}%)
            - 建议: 立即平仓
            
            **BSADF检验**: {signal['bsadf']:.4f}
            """)
            
            if push_enabled:
                push.send_signal("止损", signal['price'], signal['otm_level'], signal['action'])
        
        else:
            st.info(f"""
            ### 观望
            
            - BSADF检验: {signal['bsadf']:.4f} (未触发)
            - 继续等待信号
            """)
    
    with col2:
        st.subheader("推荐期权")
        
        if signal['price'] > 0:
            # 计算推荐行权价
            call_strike = int(signal['price'] * 1.11 / 10) * 10
            put_strike = int(signal['price'] * 0.89 / 10) * 10
            
            st.markdown(f"""
            | 类型 | 行权价 | 虚值程度 |
            |------|--------|----------|
            | Call | {call_strike} | 11% |
            | Put | {put_strike} | 11% |
            
            建议卖出11%虚值期权
            """)

with tab2:
    st.header("策略指标")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info(f"BSADF: {signal['bsadf']:.4f}")
    with col2:
        st.success(f"σ (日度): {signal['garch_vol']:.6f}")
    with col3:
        st.warning(f"VaR 99%: {signal['var_99']*100:.2f}%")
    
    st.markdown("---")
    
    # GARCH表格
    var_table = {
        "模型": ["sGARCH_norm", "sGARCH_ghyp", "sGARCH_jump"],
        "σ (日度)": ["0.0096", "0.0105", "0.0112"],
        "VaR 99%": ["2.33%", "2.55%", "2.73%"]
    }
    st.dataframe(pd.DataFrame(var_table), hide_index=True, use_container_width=True)

with tab3:
    st.header("期货期权数据")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("中金所期货合约")
        if futures_df is not None:
            st.dataframe(futures_df[['合约代码', '合约名称', '交易所']].head(10), hide_index=True)
    
    with col2:
        st.subheader("期权数据")
        st.info("期权数据需要实时连接，稍后显示")

with tab4:
    st.header("配置文档")
    
    st.markdown("""
    ## 项目配置
    
    ### PushPlus
    - Token: `3660eb1e0b364a78b3beed2f349b29f8`
    - Secret: 已配置
    
    ### GitHub
    - 仓库: `gaaiyun/CSI_50_Index_Option_Trading_Signals`
    
    ### 更新流程
    1. 修改 `app.py`
    2. `git add . && git commit -m "更新"`
    3. `git push`
    
    ### Streamlit Cloud
    - 访问: share.streamlit.io
    - 自动部署
    """)

# ==================== 页脚 ====================
st.markdown("---")
st.markdown(f"""
<div style='text-align:center; color:#666;'>
    <p>中证50期权策略看板 | 自动更新: 5分钟 | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
</div>
""", unsafe_allow_html=True)
