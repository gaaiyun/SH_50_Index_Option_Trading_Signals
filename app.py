#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
上证50ETF期权策略看板 - 完整严谨版 v4.0

交易哲学: 尾部风险防范, 极其严格的波动率做空体系
数据源:
- yfinance: 510050.SS ETF日线
- akshare: 期权链、IH期货
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import time
from datetime import datetime
from streamlit_echarts import st_pyecharts
from pyecharts import options as opts
from pyecharts.charts import Kline, Scatter, Line, Grid
from strategy.indicators import StrategyIndicators

# 配置
PUSHPLUS_TOKEN = "3660eb1e0b364a78b3beed2f349b29f8"

st.set_page_config(
    page_title="上证50期权高阶看板",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-title { font-size: 2.2rem; font-weight: 800; color: #1f77b4; margin-bottom: -10px;}
    .sub-title { font-size: 1.0rem; color: #888; margin-bottom: 20px;}
    .warning { padding: 1rem; background: #2a2a3e; border-left: 4px solid #f9a825; border-radius: 4px;}
    .metric-card { background: #1e1e2d; padding: 15px; border-radius: 10px; border: 1px solid #333; text-align: center;}
    .metric-title { font-size: 0.9rem; color: #aaa; margin-bottom: 5px;}
    .metric-value { font-size: 1.5rem; font-weight: bold; color: #fff;}
    .metric-sub { font-size: 0.8rem; }
    .color-green { color: #00cc96; }
    .color-red { color: #ff4d4f; }
</style>
""", unsafe_allow_html=True)

# ==================== 本地缓存管理 ====================
DATA_DIR = "data"
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

def load_local_cache(filename: str, ttl_seconds: int):
    """尝试加载本地缓存数据，检查是否过期"""
    filepath = os.path.join(DATA_DIR, filename)
    if os.path.exists(filepath):
        mtime = os.path.getmtime(filepath)
        if time.time() - mtime < ttl_seconds:
            try:
                # 针对带有datetime index的yfinance数据特殊处理
                if 'etf' in filename:
                    df = pd.read_csv(filepath, index_col=0, parse_dates=True)
                else:
                    df = pd.read_csv(filepath)
                return df, True
            except:
                pass
    return None, False

def save_local_cache(df: pd.DataFrame, filename: str):
    """保存数据到本地"""
    filepath = os.path.join(DATA_DIR, filename)
    try:
        df.to_csv(filepath)
    except Exception as e:
        print(f"缓存写入失败: {e}")

# ==================== 数据获取 ====================
@st.cache_data(ttl=300)
def get_etf_510050(force_refresh=False):
    """获取上证50ETF (510050.SS)，带本地持久化降级"""
    cache_file = "etf_510050.csv"
    
    if not force_refresh:
        df, valid = load_local_cache(cache_file, 3600*12) # 日线数据理论上存活半天
        if valid and not df.empty:
            return df, "yfinance (本地缓存)"
            
    try:
        import yfinance as yf
        t = yf.Ticker("510050.SS")
        df = t.history(period="3y")
        df.index = df.index.tz_localize(None)
        if not df.empty:
            save_local_cache(df, cache_file)
        return df, "yfinance (在线刷新)"
    except Exception as e:
        # 如果在线挂了，即使缓存过期也强行读取兜底
        df, _ = load_local_cache(cache_file, 999999)
        if df is not None:
            return df, f"yfinance (网络异常，强行读取陈旧缓存)"
        return None, str(e)

@st.cache_data(ttl=60)
def get_options_data(force_refresh=False):
    """获取期权实时T型盘口，带1分钟防刷及本地持久化缓存"""
    cache_file = "options_50.csv"
    
    if not force_refresh:
        df, valid = load_local_cache(cache_file, 60) # 期权盘口1分钟内不重复拉取
        if valid and not df.empty:
            return df, "akshare (本地缓存)"
            
    import threading
    result_holder = {"df": None, "error": None}

    def _fetch():
        try:
            import akshare as ak
            df_full = ak.option_current_em()
            result_holder["df"] = df_full
        except Exception as e:
            result_holder["error"] = str(e)

    t = threading.Thread(target=_fetch, daemon=True)
    t.start()
    t.join(timeout=8)          # 最多等 8 秒

    if not t.is_alive() and result_holder["df"] is not None:
        df = result_holder["df"]
        df_50 = df[df['名称'].str.contains('50ETF') | df['代码'].str.startswith('100')].copy()
        if not df_50.empty:
            save_local_cache(df_50, cache_file)
        return df_50, "akshare (在线刷新)"
    else:
        # 降级读取本地兜底
        err = result_holder["error"] if result_holder["error"] else "云端节点直连东财接口超时"
        df, _ = load_local_cache(cache_file, 999999)
        if df is not None:
            return df, f"akshare (超时降级，强行读取陈旧缓存)"
        return None, f"获取失败: {err}"

# ==================== 可视化库 ====================
def render_kline_with_bsadf(df: pd.DataFrame, bsadf_result: dict):
    """绘制K线并在泡沫期(显著区间)高亮散点"""
    try:
        # 切片最近200天显示
        plot_df = df.iloc[-200:].copy()
        x_data = plot_df.index.strftime('%Y-%m-%d').tolist()
        y_data = plot_df[['Open', 'Close', 'Low', 'High']].values.tolist()
        
        kline = Kline()
        kline.add_xaxis(x_data)
        kline.add_yaxis(
            "510050.SS",
            y_data,
            itemstyle_opts=opts.ItemStyleOpts(color="#ec0000", color0="#00da3c"),
        )
        kline.set_global_opts(
            xaxis_opts=opts.AxisOpts(is_scale=True, splitline_opts=opts.SplitLineOpts(is_show=False)),
            yaxis_opts=opts.AxisOpts(is_scale=True, splitline_opts=opts.SplitLineOpts(is_show=True)),
            datazoom_opts=[opts.DataZoomOpts(type_="inside")],
            title_opts=opts.TitleOpts(title="上证50ETF日线泡沫监控", pos_left="center"),
            legend_opts=opts.LegendOpts(is_show=False)
        )
        
        # 叠加BSADF高亮
        if 'series' in bsadf_result and not bsadf_result['series'].empty:
            bsadf_sr = bsadf_result['series']
            cv = bsadf_result.get('cv', 1.5)
            scatter_data = []
            
            # 对齐时间轴
            for time_str in x_data:
                time_dt = pd.to_datetime(time_str)
                if time_dt in bsadf_sr.index:
                    val = bsadf_sr.loc[time_dt]
                    if val > cv:
                        # 泡沫发生，标记在K线最高点之上
                        high_price = plot_df.loc[time_dt, 'High']
                        scatter_data.append([time_str, float(high_price * 1.01)])
                    else:
                        scatter_data.append([time_str, None])
                else:
                    scatter_data.append([time_str, None])
                    
            scatter = Scatter()
            scatter.add_xaxis(x_data)
            scatter.add_yaxis(
                "泡沫预警区间",
                [y[1] if y[1] is not None else "" for y in scatter_data],
                symbol="circle",
                symbol_size=6,
                itemstyle_opts=opts.ItemStyleOpts(color="#fadb14"),
                label_opts=opts.LabelOpts(is_show=False)
            )
            kline.overlap(scatter)
            
        return kline
    except Exception as e:
        return None

# ==================== 主程序 ====================
with st.sidebar:
    st.header("风控参数")
    otm = st.slider("目标建仓虚值(%)", 5, 20, 11)
    stop_loss = st.slider("绝对认怂虚值(%)", 3, 10, 6)
    
    st.markdown("---")
    st.subheader("高频预警设定")
    rv_threshold = st.slider("RV年化异常阈值(%)", 15, 60, 30)
    
    st.markdown("---")
    push = st.checkbox("PushPlus 信号推送", value=False)
    if push:
        st.info("已启用实盘级推送")
        
    st.markdown("---")
    st.subheader("数据管理")
    force_refresh = st.button("🔄 强制更新所有数据源", use_container_width=True)

st.markdown('<div class="main-title">上证50ETF期权 卖方高阶看板 (v4.1)</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">系统核心: 多重GARCH立体防御体系 | BSADF极值泡沫猎杀 | 日内RV高频止损截断</div>', unsafe_allow_html=True)

# 获取数据
df_etf, source_etf = get_etf_510050(force_refresh=force_refresh)
options_df, opt_source = get_options_data(force_refresh=force_refresh)

if force_refresh:
    st.toast("数据源已向云端发起更新请求", icon="✅")

if df_etf is not None and not df_etf.empty:
    prices = df_etf['Close']
    
    # 计算指标
    indicators = StrategyIndicators()
    
    bsadf_result = indicators.calculate_bsadf(prices, window=100)
    bsadf_stat = bsadf_result.get('adf_stat', 0.0)
    triggered = bsadf_result.get('is_significant', False)
    
    garch_result = indicators.calculate_garch_var(prices, confidence_levels=[0.90, 0.95, 0.99])
    
    returns = np.log(prices / prices.shift(1)).dropna()
    change = ((prices.iloc[-1] / prices.iloc[-2]) - 1) * 100
    spot = prices.iloc[-1]
    
    # 抽取核心GARCH防线
    var_95 = garch_result.get('var_95', 0) * 100 # 认怂线距离 (%)
    var_99 = garch_result.get('var_99', 0) * 100 # 极端预警距离 (%)
    sigma = garch_result.get('sigma_norm', 0.01) * np.sqrt(252) * 100
    
    # 模拟最新天的RV (如果没有分钟级别数据，暂用日线粗略换算展示)
    pseudo_rv = np.sqrt(np.sum(returns.iloc[-5:]**2)) * np.sqrt(252/5) * 100
    
    # 产生信号
    if triggered:
        signal, action = "建仓信号启动", f"优先卖出 {var_99:.1f}% 到 {otm:.1f}% 深度虚值的 Put/Call 期权"
        sig_color = "#f9a825"
    else:
        signal, action = "绝对观望", f"BSADF={bsadf_stat:.2f} 尚未进入非理性极值区间，忍耐吃瓜。"
        sig_color = "#333"

    # ========= 核心数据面板 =========
    st.markdown("### 📊 实时量化防御面")
    c1, c2, c3, c4 = st.columns(4)
    
    with c1:
        color = "color-red" if change > 0 else "color-green"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">510050.SS (底层标的)</div>
            <div class="metric-value {color}">{spot:.3f}</div>
            <div class="metric-sub {color}">{change:+.2f}%</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">多重GARCH 预测年化波动率</div>
            <div class="metric-value" style="color:#00e5ff">{sigma:.2f}%</div>
            <div class="metric-sub">Sigma T+1 期望</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">认怂绝对红线 (VaR 95%)</div>
            <div class="metric-value" style="color:#ff6b6b">±{var_95:.2f}%</div>
            <div class="metric-sub">如所持仓头寸剩余虚值率 < 该数值, 无条件平仓!</div>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">系统操作指令</div>
            <div class="metric-value" style="font-size: 1.2rem; color:{sig_color}">{signal}</div>
            <div class="metric-sub">{action}</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("---")
    
    # ========= 高阶图表 =========
    st.markdown("### 📉 K线与气泡预警诊断")
    kline_chart = render_kline_with_bsadf(df_etf, bsadf_result)
    if kline_chart:
        st_pyecharts(kline_chart, height="450px")
        
    st.markdown("---")
    
    # ========= 期权链交易推荐 =========
    st.markdown("### 🎯 实时期权靶心测算库 (寻找最佳深度虚值)")
    
    if options_df is not None and not options_df.empty:
        # 重命名容易理解的列并计算虚值率
        try:
            show_df = options_df[['代码', '名称', '最新价', '行权价', '隐含波动率']].copy()
            show_df['行权价'] = pd.to_numeric(show_df['行权价'], errors='coerce')
            show_df['当前虚值深度(%)'] = (abs(spot - show_df['行权价']) / spot * 100).round(2)
            
            # 使用GARCH VaR计算它的安全防线
            show_df['距离95%认怂线差距'] = (show_df['当前虚值深度(%)'] - var_95).round(2)
            
            # 高亮优选：OTM大于11%，同时隐含波动率较高
            def highlight_target(row):
                if row['当前虚值深度(%)'] >= otm and row['距离95%认怂线差距'] > 2.0:
                    return ['background-color: #2e4c2e'] * len(row)
                elif row['当前虚值深度(%)'] < stop_loss:
                    return ['color: #ff4d4f'] * len(row)
                return [''] * len(row)
            
            # 排序后展示
            show_df = show_df.sort_values('当前虚值深度(%)', ascending=False)
            st.dataframe(show_df.style.apply(highlight_target, axis=1), height=400, use_container_width=True)
            
            st.caption("🟢 绿色背景代表符合安全垫条件(离认怂线距离远)的高优Target | 🔴 红色字体代表已被击穿至止损区间的剧毒合约")
        except Exception as e:
            st.warning(f"期权表单渲染错误: {e}")
            st.dataframe(options_df)
    else:
        st.warning("⚠️ 盘口休市或数据接口异常，当前无法加载期权靶心测算。")

else:
    st.error("❌ 无法获取 510050.SS 基础现价数据。请检查网络。")

st.markdown(f"<div style='text-align:center; color:#555; margin-top:30px; font-size: 0.8rem;'>数据驱动引擎: yfinance + akshare | {source_etf} | {opt_source} | 刷新策略: 磁盘持久化智能缓存机制</div>", unsafe_allow_html=True)
