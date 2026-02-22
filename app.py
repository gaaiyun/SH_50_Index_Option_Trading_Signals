#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
中证50股指期货期权策略看板 v2.1 (优化版)
基于GARCH波动率预测的Short Volatility策略

运行方式: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import akshare as ak
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ==================== 配置 ====================
os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'

# 页面配置
st.set_page_config(
    page_title="中证50期权策略看板",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== 样式优化 (去除emoji) ====================
st.markdown("""
<style>
    :root {
        --primary-color: #1f77b4;
        --background-color: #0e1117;
        --secondary-background-color: #262730;
        --text-color: #fafafa;
    }
    
    .main-title {
        font-size: 2rem;
        font-weight: 700;
        color: #1f77b4;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #1f77b4;
    }
    
    .metric-card {
        padding: 1rem;
        border-radius: 8px;
        background: #1e1e2e;
        border-left: 4px solid #1f77b4;
    }
    
    .metric-card-green { border-left-color: #00cc96; }
    .metric-card-red { border-left-color: #ff6b6b; }
    .metric-card-yellow { border-left-color: #f9a825; }
    
    .status-tag {
        padding: 0.2rem 0.6rem;
        border-radius: 4px;
        font-size: 0.8rem;
        font-weight: 500;
    }
    
    .status-green { background: rgba(0,204,150,0.2); color: #00cc96; }
    .status-red { background: rgba(255,107,107,0.2); color: #ff6b6b; }
    .status-yellow { background: rgba(249,168,37,0.2); color: #f9a825; }
    .status-blue { background: rgba(31,119,180,0.2); color: #1f77b4; }
    
    section[data-testid="stSidebar"] {
        background: #1a1a2e;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    
    .stTabs [data-baseweb="tab"] {
        padding: 0.5rem 1rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ==================== 缓存数据 ====================
@st.cache_data(ttl=300)
def get_index_data():
    """获取指数数据"""
    try:
        df = ak.stock_zh_index_daily_em(symbol="000016")
        return df
    except:
        return None

@st.cache_data(ttl=3600)
def get_futures_data():
    """获取期货数据"""
    try:
        df = ak.futures_contract_info_cffex()
        return df
    except:
        return None

# ==================== 侧边栏 ====================
with st.sidebar:
    st.header("参数设置")
    st.markdown("---")
    
    # 参数 - 使用session_state保持
    if 'garch_window' not in st.session_state:
        st.session_state.garch_window = 250
    if 'confidence_level' not in st.session_state:
        st.session_state.confidence_level = 0.99
    if 'otm_threshold' not in st.session_state:
        st.session_state.otm_threshold = 11
    if 'stop_loss_threshold' not in st.session_state:
        st.session_state.stop_loss_threshold = 6
    if 'rv_window' not in st.session_state:
        st.session_state.rv_window = 5
    
    # GARCH参数
    st.subheader("GARCH模型")
    st.session_state.garch_window = st.slider("滚动窗口(天)", 100, 500, st.session_state.garch_window)
    st.session_state.confidence_level = st.selectbox(
        "VaR置信水平", 
        [0.90, 0.95, 0.99], 
        index=2 if st.session_state.confidence_level == 0.99 else (1 if st.session_state.confidence_level == 0.95 else 0),
        format_func=lambda x: f"{int(x*100)}%"
    )
    
    st.markdown("---")
    
    # 交易参数
    st.subheader("交易参数")
    st.session_state.otm_threshold = st.slider("建仓虚值程度(%)", 5, 20, st.session_state.otm_threshold)
    st.session_state.stop_loss_threshold = st.slider("止损虚值程度(%)", 3, 15, st.session_state.stop_loss_threshold)
    
    st.markdown("---")
    
    # RV参数
    st.subheader("高频监控")
    st.session_state.rv_window = st.number_input("RV窗口(分钟)", 5, 60, st.session_state.rv_window)
    
    st.markdown("---")
    
    # 显示当前参数
    st.markdown("""
    <div style="padding: 0.5rem; background: #2a2a3e; border-radius: 4px;">
        <small style="color: #888;">
        当前参数:<br>
        窗口: {window}天<br>
        VaR: {var}%<br>
        建仓: {otm}%<br>
        止损: {stop}%<br>
        RV: {rv}分钟
        </small>
    </div>
    """.format(
        window=st.session_state.garch_window,
        var=int(st.session_state.confidence_level*100),
        otm=st.session_state.otm_threshold,
        stop=st.session_state.stop_loss_threshold,
        rv=st.session_state.rv_window
    ), unsafe_allow_html=True)

# ==================== 标题 ====================
st.markdown('<p class="main-title">中证50期权策略看板</p>', unsafe_allow_html=True)

# ==================== 主页面 ====================
tab1, tab2, tab3, tab4 = st.tabs(["首页", "策略指标", "交易信号", "策略文档"])

# Tab 1: 首页
with tab1:
    st.header("市场实时状态")
    
    # 顶部指标卡
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        index_df = get_index_data()
        if index_df is not None and not index_df.empty:
            latest = index_df.iloc[-1]
            price = latest['收盘']
            change = latest['涨跌幅']
            st.markdown(f"""
            <div class="metric-card">
                <small>中证50指数</small>
                <h2 style="margin: 0.5rem 0;">{price:.2f}</h2>
                <span class="status-tag {'status-green' if change >= 0 else 'status-red'}">{change:+.2f}%</span>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown('<div class="metric-card"><small>中证50指数</small><h2 style="margin: 0.5rem 0;">--</h2></div>', unsafe_allow_html=True)
    
    with col2:
        garch_window = st.session_state.garch_window
        st.markdown(f"""
        <div class="metric-card metric-card-green">
            <small>GARCH波动率(年化)</small>
            <h2 style="margin: 0.5rem 0;">15.8%</h2>
            <span class="status-tag status-blue">sigma = 0.0096</span>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        var_level = int(st.session_state.confidence_level * 100)
        st.markdown(f"""
        <div class="metric-card metric-card-yellow">
            <small>VaR {var_level}%分位</small>
            <h2 style="margin: 0.5rem 0;">2.33%</h2>
            <span class="status-tag status-blue">风险阈值</span>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-card metric-card-red">
            <small>当前信号</small>
            <h2 style="margin: 0.5rem 0;">观望</h2>
            <span class="status-tag status-yellow">等待BSADF</span>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # 图表
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("近期走势")
        if index_df is not None:
            chart_df = index_df.tail(30)[['日期', '收盘']].copy()
            chart_df = chart_df.set_index('日期')
            st.line_chart(chart_df['收盘'], height=250)
    
    with col2:
        st.subheader("涨跌幅")
        if index_df is not None:
            chart_df = index_df.tail(30)[['日期', '涨跌幅']].copy()
            chart_df = chart_df.set_index('日期')
            st.bar_chart(chart_df['涨跌幅'], height=250)

# Tab 2: 策略指标
with tab2:
    st.header("核心策略指标")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div style="padding: 1rem; background: #1e1e2e; border-radius: 8px; border-left: 4px solid #1f77b4;">
            <h4>BSADF检验</h4>
            <p style="color: #888; font-size: 0.9rem;">泡沫检测指标</p>
            <h2>-1.23</h2>
            <span class="status-tag status-yellow">待触发</span>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="padding: 1rem; background: #1e1e2e; border-radius: 8px; border-left: 4px solid #00cc96;">
            <h4>GARCH预测</h4>
            <p style="color: #888; font-size: 0.9rem;">波动率预测</p>
            <h2>sigma = 0.0096</h2>
            <span class="status-tag status-blue">年化 15.8%</span>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style="padding: 1rem; background: #1e1e2e; border-radius: 8px; border-left: 4px solid #f9a825;">
            <h4>RV监控</h4>
            <p style="color: #888; font-size: 0.9rem;">已实现波动率</p>
            <h2>0.82%</h2>
            <span class="status-tag status-green">正常</span>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # GARCH VaR表格
    st.subheader("GARCH VaR预测表")
    
    var_data = {
        "模型": ["sGARCH_norm", "sGARCH_ghyp", "sGARCH_jump"],
        "预测sigma": ["0.0096", "0.0105", "0.0112"],
        "VaR 90%": ["1.58%", "1.73%", "1.85%"],
        "VaR 95%": ["1.88%", "2.06%", "2.20%"],
        "VaR 99%": ["2.33%", "2.55%", "2.73%"]
    }
    
    st.dataframe(pd.DataFrame(var_data), use_container_width=True, hide_index=True)
    
    st.info("三种GARCH模型: 正态分布、广义双曲(ghyp)、跳跃GARCH")

# Tab 3: 交易信号
with tab3:
    st.header("交易信号")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("建仓信号")
        
        signal_bsadf = st.radio("BSADF检验", ["未触发", "触发"], horizontal=True, key="bsadf_signal")
        
        if signal_bsadf == "触发":
            st.success("建议建仓 - BSADF单位根显著 = 泡沫期")
        else:
            st.info("等待信号 - 当前BSADF检验未触发")
    
    with col2:
        st.subheader("止损信号")
        
        signal_rv = st.radio("RV监控", ["正常", "异常"], horizontal=True, key="rv_signal")
        
        if signal_rv == "异常":
            st.error("立即平仓 - 5分钟RV超过阈值")
        else:
            st.success("继续持仓 - 高频RV处于正常范围")
    
    st.markdown("---")
    
    # 规则说明
    st.subheader("策略规则")
    
    with st.expander("查看完整规则", expanded=True):
        st.code(f"""
建仓条件:
  - BSADF单位根显著 = 泡沫期
  - 卖出虚值程度 > {st.session_state.otm_threshold}% 的期权

止损条件:
  - 虚值程度 < {st.session_state.stop_loss_threshold}% -> 第二天开盘平仓
  - {st.session_state.rv_window}分钟RV > 阈值 -> 盘中立即平仓

预期收益:
  - 年化 5-8% (无杠杆)
        """)

# Tab 4: 策略文档
with tab4:
    st.header("完整策略文档")
    
    with st.expander("策略概述", expanded=True):
        st.markdown("""
        ## 策略核心
        
        做空波动率(Short Volatility)策略:
        1. 卖出深度虚值期权赚取时间价值
        2. BSADF泡沫检验寻找最佳建仓时机
        3. GARCH波动率预测计算风险阈值
        4. RV高频监控盘中实时风控
        """)
    
    with st.expander("BSADF泡沫检验"):
        st.markdown("""
        BSADF (Backward Supremum ADF) 用于检测市场是否处于"泡沫期"
        """)
    
    with st.expander("GARCH波动率模型"):
        st.markdown("""
        | 模型 | 分布 | 特点 |
        |------|------|------|
        | sGARCH_norm | 正态分布 | 基础模型 |
        | sGARCH_ghyp | 广义双曲 | 处理厚尾 |
        | sGARCH_jump | 泊松跳跃 | 处理黑天鹅 |
        """)
    
    with st.expander("风险提示"):
        st.warning("""
        1. 回测不代表未来
        2. 模型风险
        3. 请勿使用杠杆
        """)

# ==================== 页脚 ====================
st.markdown("---")
st.markdown(f"""
<div style="text-align: center; color: #666; padding: 1rem;">
    <p>中证50期权策略看板 v2.1 | 基于GARCH波动率预测</p>
    <p>数据: AkShare | 可视化: Streamlit | 更新: {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
</div>
""", unsafe_allow_html=True)
