#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
中证50期权策略看板 - 完整版 v2.3
支持: 自动数据刷新 | 自动信号计算 | PushPlus推送

运行: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
from datetime import datetime
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

# ==================== 样式 ====================
st.markdown("""
<style>
    .main-title { font-size: 1.8rem; font-weight: 700; color: #1f77b4; padding-bottom: 0.5rem; }
    .signal-buy { background: #00cc96; color: white; padding: 1rem; border-radius: 8px; text-align: center; }
    .signal-sell { background: #ff6b6b; color: white; padding: 1rem; border-radius: 8px; text-align: center; }
    .signal-wait { background: #f9a825; color: white; padding: 1rem; border-radius: 8px; text-align: center; }
</style>
""", unsafe_allow_html=True)

# ==================== 缓存函数 ====================
@st.cache_data(ttl=600)
def get_index_data():
    """获取中证50指数数据"""
    try:
        import akshare as ak
        df = ak.stock_zh_index_daily_em(symbol="000016")
        return df
    except Exception as e:
        st.error(f"获取指数数据失败: {e}")
        return None

@st.cache_data(ttl=3600)
def get_futures_data():
    """获取期货合约数据"""
    try:
        import akshare as ak
        df = ak.futures_contract_info_cffex()
        return df
    except Exception as e:
        return None

# ==================== 推送类 ====================
class PushNotifier:
    def __init__(self, token, secret=""):
        self.token = token
        self.api_url = "http://www.pushplus.plus/send"
    
    def send(self, title, content, template="markdown"):
        data = {"token": self.token, "title": title, "content": content, "template": template}
        try:
            r = requests.post(self.api_url, json=data, timeout=10)
            return r.json().get("code") == 200
        except:
            return False
    
    def send_signal(self, signal_type, price, action):
        content = f"""
## {signal_type}信号

**时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

| 项目 | 数值 |
|------|------|
| 中证50指数 | {price:.2f} |
| 建议操作 | {action} |

---
*自动发送 - 期权策略看板*
"""
        return self.send(f"50ETF期权策略 {signal_type}", content)

push = PushNotifier(PUSHPLUS_TOKEN, PUSHPLUS_SECRET)

# ==================== 策略类 ====================
class Strategy:
    def __init__(self):
        self.symbol = "000016"
        self.garch_window = 250
        self.var_level = 0.99
        self.otm_threshold = 11
        self.stop_loss = 6.4
    
    def calculate_garch(self, prices):
        """简化GARCH计算"""
        returns = np.log(prices / prices.shift(1)).dropna().iloc[-self.garch_window:]
        if len(returns) < 50:
            return 0.01
        # 简化计算：使用滚动标准差
        vol = returns.rolling(20).std().iloc[-1]
        return vol if not np.isnan(vol) else 0.01
    
    def calculate_var(self, sigma):
        """计算VaR"""
        z = 2.326  # 99%置信度
        return sigma * z
    
    def generate_signal(self, index_df):
        """生成交易信号"""
        if index_df is None or index_df.empty:
            return {"signal": "数据获取失败", "action": "等待", "price": 0, "garch_vol": 0, "var_99": 0}
        
        try:
            latest = index_df.iloc[-1]
            price = float(latest['收盘'])
        except:
            return {"signal": "数据解析失败", "action": "等待", "price": 0, "garch_vol": 0, "var_99": 0}
        
        # 计算波动率
        prices = index_df['收盘']
        garch_vol = self.calculate_garch(prices)
        var_99 = self.calculate_var(garch_vol)
        
        # 简化信号判断 (随机模拟，实际需要BSADF)
        change = latest.get('涨跌幅', 0) if '涨跌幅' in latest.index else 0
        
        # 简单判断
        if change < -1:
            signal = "建仓"
            action = f"卖出{self.otm_threshold}%虚值Put"
        elif change > 1:
            signal = "止损"
            action = "考虑平仓"
        else:
            signal = "观望"
            action = "等待信号"
        
        return {
            "signal": signal,
            "action": action,
            "price": price,
            "garch_vol": garch_vol,
            "var_99": var_99,
            "change": change
        }

strategy = Strategy()

# ==================== 侧边栏 ====================
with st.sidebar:
    st.header("参数配置")
    
    strategy.garch_window = st.slider("GARCH窗口(天)", 100, 500, 250)
    strategy.var_level = st.selectbox("VaR置信度", [0.90, 0.95, 0.99], 2)
    st.markdown("---")
    strategy.otm_threshold = st.slider("建仓虚值程度(%)", 5, 20, 11)
    strategy.stop_loss = st.slider("止损虚值程度(%)", 3, 15, 6)
    st.markdown("---")
    
    st.subheader("推送设置")
    push_enabled = st.checkbox("启用推送", value=False)
    
    if st.button("推送测试"):
        result = push.send_signal("测试", 3000.0, "测试消息")
        st.success("成功" if result else "失败")

# ==================== 主页面 ====================
st.markdown('<p class="main-title">中证50期权策略看板</p>', unsafe_allow_html=True)

# 获取数据
index_df = get_index_data()
futures_df = get_futures_data()

# 计算信号
signal = strategy.generate_signal(index_df)

# 显示指标
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("中证50指数", f"{signal['price']:.2f}", f"{signal.get('change', 0):.2f}%")

with col2:
    vol_annual = signal['garch_vol'] * np.sqrt(252) * 100
    st.metric("波动率(年化)", f"{vol_annual:.2f}%")

with col3:
    st.metric("VaR 99%", f"{signal['var_99']*100:.2f}%")

with col4:
    signal_map = {"建仓": "signal-buy", "止损": "signal-sell", "观望": "signal-wait"}
    cls = signal_map.get(signal['signal'], "signal-wait")
    st.markdown(f'<div class="{cls}"><strong>{signal["signal"]}</strong><br><small>{signal["action"]}</small></div>', unsafe_allow_html=True)

st.markdown("---")

# Tab
tab1, tab2, tab3, tab4 = st.tabs(["信号", "指标", "期货", "配置"])

with tab1:
    st.header("交易信号")
    
    if signal['signal'] == "建仓":
        st.success(f"""
        ### 建仓信号
        
        - 价格: {signal['price']:.2f}
        - 建议: 卖出{strategy.otm_threshold}%虚值Put
        - 波动率: {signal['garch_vol']*100:.2f}%
        
        如果启用推送，信号已自动推送
        """)
        if push_enabled:
            push.send_signal("建仓", signal['price'], signal['action'])
    
    elif signal['signal'] == "止损":
        st.error(f"### 止损信号\n\n价格: {signal['price']:.2f}\n\n建议: 立即平仓")
        if push_enabled:
            push.send_signal("止损", signal['price'], signal['action'])
    
    else:
        st.info(f"""
        ### 观望
        
        - 价格: {signal['price']:.2f}
        - 涨跌幅: {signal.get('change', 0):.2f}%
        - 继续等待建仓信号
        """)

with tab2:
    st.header("策略指标")
    st.info(f"BSADF: 需实时计算")
    st.success(f"波动率: {signal['garch_vol']*100:.4f}%")
    st.warning(f"VaR 99%: {signal['var_99']*100:.2f}%")

with tab3:
    st.header("期货数据")
    if futures_df is not None and not futures_df.empty:
        # 处理列名
        cols = futures_df.columns.tolist()
        st.write("可用列:", cols[:5])
        
        # 显示前几列
        st.dataframe(futures_df.head(10))
    else:
        st.warning("期货数据获取失败 (需要VPN)")

with tab4:
    st.header("配置文档")
    st.markdown("""
    ## PushPlus
    - Token: `3660eb1e0b364a78b3beed2f349b29f8`
    
    ## GitHub
    - 仓库: `gaaiyun/CSI_50_Index_Option_Trading_Signals`
    
    ## 更新流程
    1. 修改代码
    2. `git add . && git commit -m "更新" && git push`
    """)

# 页脚
st.markdown("---")
st.markdown(f"""
<div style='text-align:center; color:#666;'>
    <p>中证50期权策略看板 | 更新时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
</div>
""", unsafe_allow_html=True)
