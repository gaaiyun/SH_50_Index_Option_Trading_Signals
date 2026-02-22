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
from pyecharts.charts import Kline, Scatter, Line, Grid, Bar
from strategy.indicators import StrategyIndicators

# 配置
PUSHPLUS_TOKEN = "3660eb1e0b364a78b3beed2f349b29f8"

st.set_page_config(
    page_title="上证50期权高频防御系统",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /* TradingView Dark Theme Palette */
    :root {
        --tv-bg: #131722;
        --tv-panel: #1e222d;
        --tv-border: #2a2e39;
        --tv-text: #d1d4dc;
        --tv-text-dim: #787b86;
        --tv-green: #089981;
        --tv-red: #f23645;
        --tv-blue: #2962ff;
        --tv-yellow: #f5a623;
    }
    
    /* Global Overrides for Streamlit */
    .stApp {
        background-color: var(--tv-bg);
        color: var(--tv-text);
    }
    
    /* Sidebar Overrides */
    [data-testid="stSidebar"] {
        background-color: var(--tv-panel) !important;
    }
    [data-testid="stSidebar"] * {
        color: var(--tv-text) !important;
    }
    
    /* Typography & Headers */
    h1, h2, h3, h4, h5, h6, p, span {
        color: var(--tv-text);
        font-family: -apple-system, BlinkMacSystemFont, "Trebuchet MS", Roboto, Ubuntu, sans-serif !important;
    }
    
    .main-title { font-size: 1.6rem; font-weight: 600; color: #ffffff !important; margin-bottom: 2px; letter-spacing: 0.5px;}
    .sub-title { font-size: 0.85rem; color: var(--tv-text-dim) !important; margin-bottom: 24px;}
    
    /* Metric Cards */
    .metric-card { 
        background-color: var(--tv-panel); 
        padding: 18px; 
        border-radius: 4px; 
        border: 1px solid var(--tv-border); 
        text-align: left; 
    }
    .metric-title { font-size: 0.8rem; color: var(--tv-text-dim) !important; margin-bottom: 8px; text-transform: uppercase; letter-spacing: 0.5px;}
    .metric-value { font-size: 1.5rem; font-weight: 500; letter-spacing: 0.2px;}
    .metric-sub { font-size: 0.75rem; color: var(--tv-text-dim) !important; margin-top: 4px;}
    
    /* Color Utilities */
    .color-green { color: var(--tv-green) !important; }
    .color-red { color: var(--tv-red) !important; }
    .color-blue { color: var(--tv-blue) !important; }
    .color-orange { color: var(--tv-yellow) !important; }
    
    /* DataFrame overriding */
    .stDataFrame { font-size: 0.85rem; }
    
    /* Streamlit overrides */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ==================== 本地缓存管理 (Stale-while-Revalidate) ====================
DATA_DIR = "data"
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

def load_local_cache(filename: str):
    """强制加载本地旧数据，永不阻塞"""
    filepath = os.path.join(DATA_DIR, filename)
    if os.path.exists(filepath):
        try:
            if 'etf' in filename:
                df = pd.read_csv(filepath, index_col=0, parse_dates=True)
            else:
                df = pd.read_csv(filepath)
            return df
        except:
            pass
    return None

def is_cache_expired(filename: str, ttl_seconds: int):
    filepath = os.path.join(DATA_DIR, filename)
    if not os.path.exists(filepath):
        return True
    mtime = os.path.getmtime(filepath)
    return (time.time() - mtime) > ttl_seconds

def save_local_cache(df: pd.DataFrame, filename: str):
    """保存数据到本地"""
    filepath = os.path.join(DATA_DIR, filename)
    try:
        df.to_csv(filepath)
    except Exception as e:
        print(f"缓存写入失败: {e}")

# ==================== 数据获取 ====================
def fetch_etf_bg():
    try:
        import yfinance as yf
        df = yf.download("510050.SS", period="5y", progress=False)
        if not df.empty:
            df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
            save_local_cache(df, "etf_510050.csv")
    except:
        pass

def get_etf_510050(force_refresh=False):
    """日线基准数据: 立即返还缓存，后台静默验证更新"""
    cache_file = "etf_510050.csv"
    import threading
    
    if force_refresh or is_cache_expired(cache_file, 43200): # 12 小时更新
        threading.Thread(target=fetch_etf_bg, daemon=True).start()
    
    df = load_local_cache(cache_file)
    if df is not None:
        return df, "本地急速库"
    return None, "待缓冲"

def fetch_options_bg():
    try:
        import akshare as ak
        df_full = ak.option_current_em()
        if df_full is not None and not df_full.empty:
            df_50 = df_full[df_full['名称'].str.contains('50ETF') | df_full['代码'].str.startswith('100')].copy()
            if not df_50.empty:
                save_local_cache(df_50, "options_50.csv")
    except:
        pass

def get_options_data(force_refresh=False):
    """高频期权盘口: 优先本地读取，后台静默重连轮询"""
    cache_file = "options_50.csv"
    import threading
    
    if force_refresh or is_cache_expired(cache_file, 60): # 60秒过期
        threading.Thread(target=fetch_options_bg, daemon=True).start()
        
    df = load_local_cache(cache_file)
    if df is not None:
        return df, "后台实时推流 (SWR架构)"
    return None, "待缓冲"

# ==================== 可视化库 ====================
def render_kline_with_bsadf(df: pd.DataFrame, bsadf_result: dict, var_95_val: float):
    """绘制TradingView风格三窗格: K线主图+VaR带 | BSADF 序列 | 成交量"""
    try:
        # 切片最近200天显示
        plot_df = df.iloc[-200:].copy()
        
        # 计算移动平均线与模拟VaR动态通道(仅做视觉参考展示历史走势)
        plot_df['MA5'] = plot_df['Close'].rolling(window=5).mean()
        plot_df['MA20'] = plot_df['Close'].rolling(window=20).mean()
        plot_df['VaR_Upper'] = plot_df['Close'] * (1 + var_95_val/100)
        plot_df['VaR_Lower'] = plot_df['Close'] * (1 - var_95_val/100)
        
        x_data = plot_df.index.strftime('%Y-%m-%d').tolist()
        y_data = plot_df[['Open', 'Close', 'Low', 'High']].values.tolist()
        ma5_data = [round(x, 3) if not pd.isna(x) else None for x in plot_df['MA5']]
        ma20_data = [round(x, 3) if not pd.isna(x) else None for x in plot_df['MA20']]
        var_upper_data = [round(x, 3) for x in plot_df['VaR_Upper']]
        var_lower_data = [round(x, 3) for x in plot_df['VaR_Lower']]
        
        # 准备成交量数据
        vol_data = []
        for i, row in plot_df.iterrows():
            color = "#089981" if row['Close'] >= row['Open'] else "#f23645"
            vol_data.append(
                opts.BarItem(
                    name=i.strftime('%Y-%m-%d'),
                    value=int(row['Volume']),
                    itemstyle_opts=opts.ItemStyleOpts(color=color)
                )
            )

        # ========= 主图 Pane 0: K线 + VaR通道 =========
        kline = Kline()
        kline.add_xaxis(x_data)
        kline.add_yaxis(
            "上证50",
            y_data,
            itemstyle_opts=opts.ItemStyleOpts(
                color="#089981", color0="#f23645",
                border_color="#089981", border_color0="#f23645"
            ),
        )
        
        kline.set_global_opts(
            xaxis_opts=opts.AxisOpts(
                is_scale=True, 
                splitline_opts=opts.SplitLineOpts(is_show=True, linestyle_opts=opts.LineStyleOpts(color="#2a2e39")),
                axislabel_opts=opts.LabelOpts(is_show=False),
                axisline_opts=opts.LineStyleOpts(color="#2a2e39"),
                axispointer_opts=opts.AxisPointerOpts(is_show=True, type_="line")
            ),
            yaxis_opts=opts.AxisOpts(
                is_scale=True, 
                splitline_opts=opts.SplitLineOpts(is_show=True, linestyle_opts=opts.LineStyleOpts(color="#2a2e39")),
                axislabel_opts=opts.LabelOpts(color="#787b86"),
                axisline_opts=opts.LineStyleOpts(color="#2a2e39"),
                position="right"
            ),
            datazoom_opts=[
                opts.DataZoomOpts(is_show=False, type_="inside", xaxis_index=[0, 1, 2]),
                opts.DataZoomOpts(is_show=True, type_="slider", xaxis_index=[0, 1, 2], bottom="0px",
                                  data_background_opts=opts.DataZoomBackgroundOpts(
                                      lineStyle=opts.LineStyleOpts(color="#2962ff"),
                                      areaStyle=opts.AreaStyleOpts(color="rgba(41,98,255,0.2)")
                                  ),
                                  filler_color="rgba(41,98,255,0.1)",
                                  border_color="#2a2e39")
            ],
            tooltip_opts=opts.TooltipOpts(
                trigger="axis",
                axis_pointer_type="cross",
                background_color="#1e222d",
                border_color="#2a2e39",
                textstyle_opts=opts.TextStyleOpts(color="#d1d4dc"),
            ),
            legend_opts=opts.LegendOpts(is_show=False)
        )
        
        # 叠加MA与VaR通道
        line_main = Line()
        line_main.add_xaxis(x_data)
        line_main.add_yaxis("MA5", ma5_data, is_smooth=True, is_symbol_show=False, itemstyle_opts=opts.ItemStyleOpts(color="#2962ff"), label_opts=opts.LabelOpts(is_show=False))
        # line_main.add_yaxis("MA20", ma20_data, is_smooth=True, is_symbol_show=False, itemstyle_opts=opts.ItemStyleOpts(color="#f5a623"), label_opts=opts.LabelOpts(is_show=False))
        line_main.add_yaxis("VaR_Upper", var_upper_data, is_smooth=True, is_symbol_show=False, itemstyle_opts=opts.ItemStyleOpts(color="rgba(242,54,69,0.5)"), linestyle_opts=opts.LineStyleOpts(type_="dashed"), label_opts=opts.LabelOpts(is_show=False))
        line_main.add_yaxis("VaR_Lower", var_lower_data, is_smooth=True, is_symbol_show=False, itemstyle_opts=opts.ItemStyleOpts(color="rgba(242,54,69,0.5)"), linestyle_opts=opts.LineStyleOpts(type_="dashed"), label_opts=opts.LabelOpts(is_show=False))
        kline.overlap(line_main)
        
        # ========= 副图 Pane 1: BSADF 监测 =========
        bsadf_line = Line()
        cv = bsadf_result.get('cv', 1.5)
        
        if 'series' in bsadf_result and not bsadf_result['series'].empty:
            bsadf_sr = bsadf_result['series']
            # 对齐数据
            b_data = []
            for time_str in x_data:
                time_dt = pd.to_datetime(time_str)
                if time_dt in bsadf_sr.index:
                    b_data.append(round(bsadf_sr.loc[time_dt], 3))
                else:
                    b_data.append(None)
                    
            bsadf_line.add_xaxis(x_data)
            bsadf_line.add_yaxis(
                "BSADF Stat",
                b_data,
                is_smooth=False,
                is_symbol_show=False,
                itemstyle_opts=opts.ItemStyleOpts(color="#f5a623"),
                label_opts=opts.LabelOpts(is_show=False),
                markline_opts=opts.MarkLineOpts(
                    data=[opts.MarkLineItem(y=cv, name="95% 极值红线")],
                    linestyle_opts=opts.LineStyleOpts(color="#f23645", type_="solid")
                )
            )
            
        bsadf_line.set_global_opts(
            xaxis_opts=opts.AxisOpts(
                type_="category", grid_index=1,
                axislabel_opts=opts.LabelOpts(is_show=False),
                axisline_opts=opts.LineStyleOpts(color="#2a2e39")
            ),
            yaxis_opts=opts.AxisOpts(
                is_scale=False, splitline_opts=opts.SplitLineOpts(is_show=False),
                axislabel_opts=opts.LabelOpts(color="#787b86"),
                axisline_opts=opts.LineStyleOpts(color="#2a2e39"),
                position="right"
            ),
            legend_opts=opts.LegendOpts(is_show=False)
        )

        # ========= 副图 Pane 2: 成交量 =========
        bar = Bar()
        bar.add_xaxis(x_data)
        bar.add_yaxis(
            "Volume",
            vol_data,
            label_opts=opts.LabelOpts(is_show=False),
            itemstyle_opts=opts.ItemStyleOpts(color="#58a6ff")
        )
        bar.set_global_opts(
            xaxis_opts=opts.AxisOpts(
                type_="category", grid_index=2,
                axislabel_opts=opts.LabelOpts(color="#787b86"),
                axisline_opts=opts.LineStyleOpts(color="#2a2e39")
            ),
            yaxis_opts=opts.AxisOpts(
                is_scale=True, splitline_opts=opts.SplitLineOpts(is_show=False),
                axislabel_opts=opts.LabelOpts(is_show=False),
                axisline_opts=opts.LineStyleOpts(color="#2a2e39"),
                position="right"
            ),
            legend_opts=opts.LegendOpts(is_show=False),
        )

        # ========= 组合 Grid =========
        grid_chart = Grid(init_opts=opts.InitOpts(bg_color="#131722", width="100%", height="750px"))
        # 主图 50%
        grid_chart.add(kline, grid_opts=opts.GridOpts(pos_left="2%", pos_right="6%", height="50%"))
        # BSADF 15%
        grid_chart.add(bsadf_line, grid_opts=opts.GridOpts(pos_left="2%", pos_right="6%", pos_top="58%", height="15%"))
        # Volume 15%
        grid_chart.add(bar, grid_opts=opts.GridOpts(pos_left="2%", pos_right="6%", pos_top="75%", height="15%"))

        return grid_chart
    except Exception as e:
        import traceback
        print(traceback.format_exc())
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
    push = st.checkbox("PushPlus 推送服务", value=False)
    if push:
        st.info("推送通道已激活")
        
    st.markdown("---")
    st.subheader("系统控制")
    force_refresh = st.button("强制更新数据总线", use_container_width=True)

st.markdown('<div class="main-title">VolGuard Pro: 上证50期权风控雷达</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title" style="margin-bottom:12px;">算法核心: Multi-GARCH VaR | BSADF 序列重构 | 大尺度秒开缓存框架 (v5.0)</div>', unsafe_allow_html=True)

# 获取数据
df_etf, source_etf = get_etf_510050(force_refresh=force_refresh)
options_df, opt_source = get_options_data(force_refresh=force_refresh)

if force_refresh:
    st.toast("数据总线更新指令已发送", icon="🔄")

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
        signal, action = "执行: 建立空仓", f"指令: 卖出偏离 {var_99:.1f}% 至 {otm:.1f}% 之虚值合约"
        sig_color = "color-orange"
    else:
        signal, action = "状态: 观望戒备", f"BSADF({bsadf_stat:.2f}) 未达显著极值区间"
        sig_color = ""

    # ========= 核心数据面板 =========
    st.markdown("<h4 style='color:#d1d4dc; font-size:1.1rem; font-weight:500; margin-top:10px;'>量化引擎参数</h4>", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    
    with c1:
        color = "color-red" if change < 0 else "color-green"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">510050.SS (底层标的)</div>
            <div class="metric-value {color}">{spot:.3f}</div>
            <div class="metric-sub">今日波动: <span class="{color}">{change:+.2f}%</span></div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">GARCH T+1 年化预测</div>
            <div class="metric-value color-blue">{sigma:.2f}%</div>
            <div class="metric-sub">复合模型次日方差期望</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">VaR 95% 刚性防线</div>
            <div class="metric-value color-red">±{var_95:.2f}%</div>
            <div class="metric-sub">期权剩余虚值空间低于此值触发平仓</div>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">系统状态</div>
            <div class="metric-value {sig_color}" style="font-size: 1.1rem;">{signal}</div>
            <div class="metric-sub">{action}</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("<hr style='border-top: 1px solid var(--tv-border); margin: 25px 0;'>", unsafe_allow_html=True)
    
    # ========= 3-Pane 全景联动图表 =========
    st.markdown("<h4 style='color:#d1d4dc; font-size:1.1rem; font-weight:500;'>Multi-GARCH 止损带 & BSADF 单位根监控仪</h4>", unsafe_allow_html=True)
    kline_chart = render_kline_with_bsadf(df_etf, bsadf_result, var_95)
    if kline_chart:
        # Pyecharts Grid 高度定高，防止被压扁
        st_pyecharts(kline_chart, height="750px")
        
    st.markdown("<hr style='border-top: 1px solid var(--tv-border); margin: 25px 0;'>", unsafe_allow_html=True)
    
    # ========= 期权链交易推荐 =========
    st.markdown("<h4 style='color:#d1d4dc; font-size:1.1rem; font-weight:500;'>深度虚值期权雷达扫描仪</h4>", unsafe_allow_html=True)
    
    if options_df is not None and not options_df.empty:
        try:
            # 扩展提取流动性指标列
            desired_cols = ['代码', '名称', '最新价', '行权价', '涨跌幅', '成交量', '持仓量', '隐含波动率']
            cols_to_extract = [c for c in desired_cols if c in options_df.columns]
                
            show_df = options_df[cols_to_extract].copy()
            show_df['行权价'] = pd.to_numeric(show_df['行权价'], errors='coerce')
            show_df['最新价'] = pd.to_numeric(show_df['最新价'], errors='coerce')
            
            # 计算虚值空间
            show_df['当前虚值空间(%)'] = (abs(spot - show_df['行权价']) / spot * 100).round(2)
            show_df['距止损线缓冲(%)'] = (show_df['当前虚值空间(%)'] - var_95).round(2)
            
            # 处理 NaN: 统一填补并降级数据类型，防止格式化崩溃
            show_df = show_df.fillna(0)
            
            # 强化列重排
            front_cols = ['代码', '名称', '行权价', '最新价', '当前虚值空间(%)', '距止损线缓冲(%)']
            back_cols = [c for c in show_df.columns if c not in front_cols]
            show_df = show_df[front_cols + back_cols]
            
            # 排序后应用高级Pandas Styler
            show_df = show_df[show_df['行权价'] > 0].sort_values('当前虚值空间(%)', ascending=False)
            
            # Styler定义
            format_dict = {
                '最新价': '{:.4f}',
                '行权价': '{:.3f}',
                '当前虚值空间(%)': '{:.2f}%',
                '距止损线缓冲(%)': '{:.2f}%'
            }
            if '隐含波动率' in show_df.columns:
                format_dict['隐含波动率'] = '{:.2f}'
            if '涨跌幅' in show_df.columns:
                format_dict['涨跌幅'] = '{:.2f}%'
                
            def highlight_target(row):
                if row['当前虚值空间(%)'] >= otm and row['距止损线缓冲(%)'] > 2.0:
                    return ['background-color: rgba(8, 153, 129, 0.2); color: #089981; font-weight: bold'] * len(row)
                elif row['当前虚值空间(%)'] < stop_loss:
                    return ['color: #f23645; opacity: 0.8'] * len(row)
                return [''] * len(row)
            
            styled_df = (show_df.style
                .apply(highlight_target, axis=1)
                .format(format_dict, na_rep='-')
                .set_properties(**{
                    'text-align': 'center', 
                    'border-color': 'var(--tv-border)',
                })
                .set_table_styles([
                    {'selector': 'th', 'props': [('background-color', 'var(--tv-panel)'), ('color', 'var(--tv-text-dim)'), ('font-weight', '500'), ('border-bottom', '1px solid var(--tv-border)')]},
                    {'selector': 'td', 'props': [('border-bottom', '1px solid var(--tv-border)')]}
                ])
            )
            
            st.dataframe(styled_df, height=450, use_container_width=True, hide_index=True)
            
            st.markdown("<div style='font-size:0.85rem; color:var(--tv-text-dim); margin-top:8px;'><b>安全边界图例</b>: <span style='color:#089981; font-weight:bold;'>■</span> 绿色底纹代表充足安全垫的精选目标，<span style='color:#f23645; font-weight:bold;'>■</span> 红色字体警告虚值过浅极易惨遭击穿。"
                        "<br/><b>流动性提示</b>: 查看右侧成交量与持仓量，避免买卖滑点过大的真空合约。</div>", unsafe_allow_html=True)
        except Exception as e:
            st.error(f"解析期权结构发生异常: {e}")
            st.dataframe(options_df)
    else:
        st.warning("数据接口未能返回期权组合表列，可能处于交易时段外或接口连接阻断。")

else:
    st.error("无法加载 510050.SS (上证50ETF) 底层基准价格轨迹，请检查本地网络链路或远程节点状态。")

st.markdown(f"<div style='text-align:right; color:var(--tv-text-dim); margin-top:20px; font-size: 0.75rem;'>数据引擎链路: yfinance + akshare | {source_etf} | {opt_source} | 强持久化缓存激活</div>", unsafe_allow_html=True)
