#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VolGuard Pro — 上证50ETF期权全景风控系统 (v6.0)

架构:
  - 数据层: yfinance (ETF) + akshare (期权链), Stale-While-Revalidate 异步缓存
  - 算法层: BSADF 泡沫测试 (PSY 2015) + Multi-Dist GARCH VaR (双向防线) + RV
  - 视图层: ECharts 4-Pane Grid (K线+布林带 / BSADF / Volume+MA / HV vs IV)
  - 安全层: Token 通过 st.secrets / 环境变量注入, 不得硬编码
"""

import os
import time
import logging
import threading
import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime
from streamlit_echarts import st_pyecharts
from pyecharts import options as opts
from pyecharts.charts import Kline, Line, Grid, Bar

from strategy.indicators import StrategyIndicators

# ══════════════════════════════════════════════════════
# 日志配置
# ══════════════════════════════════════════════════════
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════
# PushPlus Token — 从 st.secrets 或环境变量读取
# ══════════════════════════════════════════════════════
def _get_secret(key: str) -> str:
    try:
        return st.secrets.get(key, os.environ.get(key.upper(), ""))
    except Exception:
        return os.environ.get(key.upper(), "")

PUSHPLUS_TOKEN = _get_secret("pushplus_token")

# ══════════════════════════════════════════════════════
# Streamlit 页面配置
# ══════════════════════════════════════════════════════
st.set_page_config(
    page_title="VolGuard Pro | 上证50期权风控雷达",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
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
        --tv-purple: #9c27b0;
    }

    .stApp { background-color: var(--tv-bg); color: var(--tv-text); }

    /* Sidebar */
    [data-testid="stSidebar"] { background-color: var(--tv-panel) !important; }
    [data-testid="stSidebar"] * { color: var(--tv-text) !important; }
    [data-testid="stSidebar"] .stSlider > div > div > div { background: var(--tv-blue) !important; }

    /* Typography */
    h1, h2, h3, h4, h5, h6 {
        color: var(--tv-text) !important;
        font-family: -apple-system, BlinkMacSystemFont, "Inter", "Trebuchet MS", Roboto, Ubuntu, sans-serif !important;
    }

    /* Page title */
    .main-title { font-size: 1.6rem; font-weight: 700; color: #ffffff !important; margin-bottom: 2px; letter-spacing: 0.3px;}
    .sub-title { font-size: 0.82rem; color: var(--tv-text-dim) !important; margin-bottom: 20px; }

    /* Metric cards */
    .metric-card {
        background-color: var(--tv-panel);
        padding: 16px 20px;
        border-radius: 4px;
        border: 1px solid var(--tv-border);
        border-left: 3px solid var(--tv-border);
        text-align: left;
    }
    .metric-card.card-green { border-left-color: var(--tv-green); }
    .metric-card.card-red   { border-left-color: var(--tv-red); }
    .metric-card.card-blue  { border-left-color: var(--tv-blue); }
    .metric-card.card-orange{ border-left-color: var(--tv-yellow); }

    .metric-title { font-size: 0.75rem; color: var(--tv-text-dim) !important; margin-bottom: 8px; text-transform: uppercase; letter-spacing: 0.8px; }
    .metric-value { font-size: 1.45rem; font-weight: 600; letter-spacing: 0.1px; }
    .metric-sub   { font-size: 0.72rem; color: var(--tv-text-dim) !important; margin-top: 5px; }

    /* Color utilities */
    .color-green  { color: var(--tv-green) !important; }
    .color-red    { color: var(--tv-red) !important; }
    .color-blue   { color: var(--tv-blue) !important; }
    .color-orange { color: var(--tv-yellow) !important; }
    .color-purple { color: var(--tv-purple) !important; }

    /* VaR bar */
    .var-bar {
        margin-top: 8px;
        height: 4px;
        border-radius: 2px;
        background: linear-gradient(90deg, var(--tv-green) 0%, var(--tv-yellow) 60%, var(--tv-red) 100%);
    }

    /* DataFrame */
    .stDataFrame { font-size: 0.83rem; }

    /* Section divider */
    .section-divider { border-top: 1px solid var(--tv-border); margin: 22px 0; }

    /* Streamlit chrome — hide default header bar and footer */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    [data-testid="stHeader"] { background-color: transparent !important; border-bottom: none !important; }
    [data-testid="stToolbar"] { visibility: hidden !important; }
    section[data-testid="stSidebar"] > div:first-child { padding-top: 0; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════
# 缓存目录 — 使用脚本绝对路径防止 CWD 漂移
# ══════════════════════════════════════════════════════
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(_SCRIPT_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

# ══════════════════════════════════════════════════════
# 并发控制 — Semaphore 防止线程爆炸
# ══════════════════════════════════════════════════════
_ETF_LOCK = threading.Semaphore(1)
_OPT_LOCK = threading.Semaphore(1)
_BSADF_LOCK = threading.Semaphore(1)

# ══════════════════════════════════════════════════════
# SWR 缓存工具函数
# ══════════════════════════════════════════════════════
def load_local_cache(filename: str, is_timeseries: bool = False):
    filepath = os.path.join(DATA_DIR, filename)
    if not os.path.exists(filepath):
        return None
    try:
        if is_timeseries:
            df = pd.read_csv(filepath, index_col=0, parse_dates=True)
            # ── 修正 yfinance MultiIndex 列名 ─────────────────
            # yfinance >= 0.2 有时以 ('Close', '510050.SS') 形式保存列名
            new_cols = {}
            for c in df.columns:
                name = c[0] if isinstance(c, tuple) else str(c)
                # 去掉括号等冗余字符
                name = str(c).strip("()'\"").split(",")[0].strip().strip("'")
                new_cols[c] = name
            df.rename(columns=new_cols, inplace=True)
            return df
        return pd.read_csv(filepath)
    except Exception as e:
        logger.warning(f"Cache read failed [{filename}]: {e}")
        return None

def save_local_cache(df: pd.DataFrame, filename: str):
    filepath = os.path.join(DATA_DIR, filename)
    try:
        df.to_csv(filepath)
    except Exception as e:
        logger.warning(f"Cache write failed [{filename}]: {e}")

def is_cache_expired(filename: str, ttl_seconds: int) -> bool:
    filepath = os.path.join(DATA_DIR, filename)
    if not os.path.exists(filepath):
        return True
    return (time.time() - os.path.getmtime(filepath)) > ttl_seconds

# ══════════════════════════════════════════════════════
# 数据获取 — ETF 日线 (SWR, 12h TTL)
# ══════════════════════════════════════════════════════
def _fetch_etf_bg():
    if not _ETF_LOCK.acquire(blocking=False):
        return
    try:
        import yfinance as yf
        df = yf.download("510050.SS", period="5y", progress=False)
        if not df.empty:
            df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
            save_local_cache(df, "etf_510050.csv")
            logger.info("ETF cache refreshed")
    except Exception as e:
        logger.warning(f"ETF fetch failed: {e}")
    finally:
        _ETF_LOCK.release()

def get_etf_510050(force_refresh: bool = False):
    if force_refresh or is_cache_expired("etf_510050.csv", 43200):
        threading.Thread(target=_fetch_etf_bg, daemon=True).start()
    df = load_local_cache("etf_510050.csv", is_timeseries=True)
    return (df, "本地急速库") if df is not None else (None, "待缓冲")

# ══════════════════════════════════════════════════════
# 数据获取 — 期权盘口 (SWR, 60s TTL)
# ══════════════════════════════════════════════════════
def _fetch_options_bg():
    if not _OPT_LOCK.acquire(blocking=False):
        return
    try:
        import akshare as ak
        df_full = ak.option_current_em()
        if df_full is not None and not df_full.empty:
            mask = df_full['名称'].str.contains('50ETF', na=False) | \
                   df_full['代码'].str.startswith('100', na=False)
            df_50 = df_full[mask].copy()
            if not df_50.empty:
                save_local_cache(df_50, "options_50.csv")
                logger.info(f"Options cache refreshed: {len(df_50)} contracts")
    except Exception as e:
        logger.warning(f"Options fetch failed: {e}")
    finally:
        _OPT_LOCK.release()

def get_options_data(force_refresh: bool = False):
    if force_refresh or is_cache_expired("options_50.csv", 60):
        threading.Thread(target=_fetch_options_bg, daemon=True).start()
    df = load_local_cache("options_50.csv")
    return (df, "后台SWR推流") if df is not None else (None, "待缓冲")

# ══════════════════════════════════════════════════════
# 缓存的重计算函数 — GARCH (TTL 1h)
# ══════════════════════════════════════════════════════
@st.cache_data(ttl=3600, show_spinner=False)
def _cached_garch(prices_tuple: tuple) -> dict:
    prices = pd.Series(prices_tuple)
    ind = StrategyIndicators()
    return ind.calculate_garch_var(prices, confidence_levels=[0.90, 0.95, 0.99])

@st.cache_data(ttl=900, show_spinner=False)
def _cached_bsadf(prices_tuple: tuple) -> dict:
    prices = pd.Series(prices_tuple)
    ind = StrategyIndicators()
    return ind.calculate_bsadf(prices, window=100)

# ══════════════════════════════════════════════════════
# ECharts 4-Pane 全景图
# ══════════════════════════════════════════════════════
def _safe(x, decimals=3):
    """将浮点值或 NaN 安全转为 ECharts 可序列化对象"""
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return None
    return round(float(x), decimals)

def render_4pane_chart(
    df: pd.DataFrame,
    bsadf_result: dict,
    var_95_val: float,
    options_df: pd.DataFrame | None = None
) -> Grid | None:
    """
    TradingView 风格 4 窗格图表:
      Pane 0 (50%): K线 + 布林带(20,2) + EMA5/EMA20 + VaR 通道
      Pane 1 (15%): BSADF 统计量序列 + 95% 临界值红线
      Pane 2 (15%): Volume 成交量 + 5日量能 MA
      Pane 3 (14%): HV30 (历史已实现波动率) vs IV (平均隐含波动率)
    """
    try:
        plot_df = df.iloc[-200:].copy()

        # ── 技术指标计算 ─────────────────────────────
        ema5  = plot_df['Close'].ewm(span=5, adjust=False).mean()
        ema20 = plot_df['Close'].ewm(span=20, adjust=False).mean()

        # 布林带 (20, 2)
        bb_mid   = plot_df['Close'].rolling(20).mean()
        bb_std   = plot_df['Close'].rolling(20).std()
        bb_upper = bb_mid + 2 * bb_std
        bb_lower = bb_mid - 2 * bb_std

        # VaR 通道 (基于最新 GARCH 预测展示历史距离)
        var_upper = plot_df['Close'] * (1 + var_95_val / 100)
        var_lower = plot_df['Close'] * (1 - var_95_val / 100)

        # HV30 (历史已实现波动率, 年化)
        log_ret = np.log(plot_df['Close'] / plot_df['Close'].shift(1))
        hv30 = log_ret.rolling(30).std() * np.sqrt(252) * 100

        x_data = plot_df.index.strftime('%Y-%m-%d').tolist()

        # OHLC 转 list, 确保无 NaN
        y_ohlc = []
        for _, r in plot_df.iterrows():
            o, c, l, h = r['Open'], r['Close'], r['Low'], r['High']
            if any(pd.isna(v) for v in [o, c, l, h]):
                y_ohlc.append([None, None, None, None])
            else:
                y_ohlc.append([_safe(o), _safe(c), _safe(l), _safe(h)])

        # 成交量 MA5
        vol_series = plot_df['Volume'].fillna(0)
        vol_ma5 = vol_series.rolling(5).mean()

        # ── Pane 0: K 线 ─────────────────────────────
        kline = Kline()
        kline.add_xaxis(x_data)
        kline.add_yaxis(
            "上证50ETF",
            y_ohlc,
            itemstyle_opts=opts.ItemStyleOpts(
                color="#089981", color0="#f23645",
                border_color="#089981", border_color0="#f23645"
            ),
        )
        kline.set_global_opts(
            xaxis_opts=opts.AxisOpts(
                grid_index=0,
                is_scale=True,
                axislabel_opts=opts.LabelOpts(is_show=False),
                splitline_opts=opts.SplitLineOpts(is_show=True, linestyle_opts=opts.LineStyleOpts(color="#2a2e39", width=0.5)),
                axisline_opts=opts.AxisLineOpts(linestyle_opts=opts.LineStyleOpts(color="#2a2e39")),
                axispointer_opts=opts.AxisPointerOpts(is_show=True, link=[{"xAxisIndex": "all"}])
            ),
            yaxis_opts=opts.AxisOpts(
                grid_index=0, is_scale=True, position="right",
                splitline_opts=opts.SplitLineOpts(is_show=True, linestyle_opts=opts.LineStyleOpts(color="#2a2e39", width=0.5)),
                axislabel_opts=opts.LabelOpts(color="#787b86", font_size=10),
                axisline_opts=opts.AxisLineOpts(linestyle_opts=opts.LineStyleOpts(color="#2a2e39")),
            ),
            datazoom_opts=[
                opts.DataZoomOpts(is_show=False, type_="inside", xaxis_index=[0, 1, 2, 3], range_start=60, range_end=100),
                opts.DataZoomOpts(
                    is_show=True, type_="slider",
                    xaxis_index=[0, 1, 2, 3],
                    pos_bottom="1%", height="4%",
                    filler_color="rgba(41,98,255,0.08)",
                    border_color="#2a2e39",
                    handle_style=opts.HandleStyleOpts(color="#2962ff")
                )
            ],
            tooltip_opts=opts.TooltipOpts(
                trigger="axis", axis_pointer_type="cross",
                background_color="#1e222d", border_color="#2a2e39",
                textstyle_opts=opts.TextStyleOpts(color="#d1d4dc", font_size=12),
            ),
            legend_opts=opts.LegendOpts(is_show=False),
            background_color="#131722",
        )

        # ── 叠加线图 (EMA, BB, VaR) ──────────────────
        line_overlay = Line()
        line_overlay.add_xaxis(x_data)

        def _add_line(name, series, color, dash=False, width=1, opacity=1.0):
            style = opts.LineStyleOpts(
                type_="dashed" if dash else "solid",
                width=width, opacity=opacity
            )
            line_overlay.add_yaxis(
                name, [_safe(v) for v in series],
                is_smooth=False, is_symbol_show=False,
                itemstyle_opts=opts.ItemStyleOpts(color=color, opacity=opacity),
                linestyle_opts=style,
                label_opts=opts.LabelOpts(is_show=False),
            )

        _add_line("EMA5",  ema5,  "#2962ff",  width=1.2)
        _add_line("EMA20", ema20, "#ff9800",  width=1.2)
        _add_line("BB Upper", bb_upper, "#9c27b0", dash=True, width=0.8, opacity=0.6)
        _add_line("BB Mid",   bb_mid,   "#9c27b0", width=0.8, opacity=0.4)
        _add_line("BB Lower", bb_lower, "#9c27b0", dash=True, width=0.8, opacity=0.6)
        _add_line("VaR↑",  var_upper, "rgba(242,54,69,0.35)", dash=True, width=1)
        _add_line("VaR↓",  var_lower, "rgba(242,54,69,0.35)", dash=True, width=1)

        kline.overlap(line_overlay)

        # ── Pane 1: BSADF ─────────────────────────────
        bsadf_line = Line()
        cv = bsadf_result.get('cv', 2.0)

        bsadf_sr = bsadf_result.get('series', pd.Series(dtype=float))
        b_data = []
        sr_idx_set = set(bsadf_sr.index) if not bsadf_sr.empty else set()
        for ds in x_data:
            dt = pd.to_datetime(ds)
            if sr_idx_set and dt in sr_idx_set:
                b_data.append(_safe(bsadf_sr.loc[dt]))
            else:
                b_data.append(None)

        bsadf_line.add_xaxis(x_data)
        bsadf_line.add_yaxis(
            "BSADF Stat", b_data,
            is_smooth=False, is_symbol_show=False,
            itemstyle_opts=opts.ItemStyleOpts(color="#f5a623"),
            linestyle_opts=opts.LineStyleOpts(width=1.5),
            label_opts=opts.LabelOpts(is_show=False),
            markline_opts=opts.MarkLineOpts(
                data=[opts.MarkLineItem(y=cv, name=f"5% CV={cv:.2f}")],
                linestyle_opts=opts.LineStyleOpts(color="#f23645", type_="solid", width=1.5),
                label_opts=opts.LabelOpts(color="#f23645", font_size=10)
            )
        )
        bsadf_line.set_global_opts(
            xaxis_opts=opts.AxisOpts(
                grid_index=1, type_="category",
                axislabel_opts=opts.LabelOpts(is_show=False),
                axisline_opts=opts.AxisLineOpts(linestyle_opts=opts.LineStyleOpts(color="#2a2e39")),
                splitline_opts=opts.SplitLineOpts(is_show=False),
            ),
            yaxis_opts=opts.AxisOpts(
                grid_index=1, is_scale=True, position="right",
                splitline_opts=opts.SplitLineOpts(is_show=True, linestyle_opts=opts.LineStyleOpts(color="#2a2e39", width=0.5)),
                axislabel_opts=opts.LabelOpts(color="#787b86", font_size=9),
            ),
            legend_opts=opts.LegendOpts(is_show=False),
            tooltip_opts=opts.TooltipOpts(trigger="axis", axis_pointer_type="cross",
                background_color="#1e222d", border_color="#2a2e39",
                textstyle_opts=opts.TextStyleOpts(color="#d1d4dc", font_size=11)),
        )

        # ── Pane 2: Volume + MA5 ──────────────────────
        vol_data = []
        for i, row in plot_df.iterrows():
            color = "#089981" if row['Close'] >= row['Open'] else "#f23645"
            vol_data.append(
                opts.BarItem(
                    name=i.strftime('%Y-%m-%d'),
                    value=int(row['Volume']) if not pd.isna(row['Volume']) else 0,
                    itemstyle_opts=opts.ItemStyleOpts(color=color, opacity=0.8)
                )
            )

        vol_bar = Bar()
        vol_bar.add_xaxis(x_data)
        vol_bar.add_yaxis("Volume", vol_data, label_opts=opts.LabelOpts(is_show=False),
                          bar_max_width=6)
        vol_bar.set_global_opts(
            xaxis_opts=opts.AxisOpts(
                grid_index=2, type_="category",
                axislabel_opts=opts.LabelOpts(color="#787b86", font_size=9),
                axisline_opts=opts.AxisLineOpts(linestyle_opts=opts.LineStyleOpts(color="#2a2e39")),
                splitline_opts=opts.SplitLineOpts(is_show=False),
            ),
            yaxis_opts=opts.AxisOpts(
                grid_index=2, is_scale=False, position="right",
                axislabel_opts=opts.LabelOpts(is_show=False),
            ),
            legend_opts=opts.LegendOpts(is_show=False),
            tooltip_opts=opts.TooltipOpts(trigger="axis", axis_pointer_type="shadow",
                background_color="#1e222d", border_color="#2a2e39",
                textstyle_opts=opts.TextStyleOpts(color="#d1d4dc", font_size=11)),
        )

        vol_ma5_line = Line()
        vol_ma5_line.add_xaxis(x_data)
        vol_ma5_line.add_yaxis(
            "Vol MA5", [_safe(v, 0) for v in vol_ma5],
            is_symbol_show=False,
            itemstyle_opts=opts.ItemStyleOpts(color="#2962ff"),
            linestyle_opts=opts.LineStyleOpts(width=1.2),
            label_opts=opts.LabelOpts(is_show=False),
        )
        vol_bar.overlap(vol_ma5_line)

        # ── Pane 3: HV30 vs IV ────────────────────────
        # IV — 从期权表提取平均隐含波动率
        iv_series = [None] * len(x_data)  # 默认全 None
        if options_df is not None and '隐含波动率' in options_df.columns:
            try:
                avg_iv = pd.to_numeric(options_df['隐含波动率'], errors='coerce').mean()
                iv_series = [_safe(avg_iv)] * len(x_data) if not np.isnan(avg_iv) else iv_series
            except Exception:
                pass

        hv_iv_line = Line()
        hv_iv_line.add_xaxis(x_data)
        hv_iv_line.add_yaxis(
            "HV30(%)", [_safe(v, 2) for v in hv30],
            is_smooth=True, is_symbol_show=False,
            itemstyle_opts=opts.ItemStyleOpts(color="#089981"),
            linestyle_opts=opts.LineStyleOpts(width=1.5),
            label_opts=opts.LabelOpts(is_show=False),
        )
        hv_iv_line.add_yaxis(
            "Avg IV(%)", iv_series,
            is_smooth=False, is_symbol_show=False,
            itemstyle_opts=opts.ItemStyleOpts(color="#f23645"),
            linestyle_opts=opts.LineStyleOpts(width=1.5, type_="dashed"),
            label_opts=opts.LabelOpts(is_show=False),
        )
        hv_iv_line.set_global_opts(
            xaxis_opts=opts.AxisOpts(
                grid_index=3, type_="category",
                axislabel_opts=opts.LabelOpts(color="#787b86", font_size=9),
                axisline_opts=opts.AxisLineOpts(linestyle_opts=opts.LineStyleOpts(color="#2a2e39")),
                splitline_opts=opts.SplitLineOpts(is_show=False),
            ),
            yaxis_opts=opts.AxisOpts(
                grid_index=3, is_scale=True, position="right",
                axislabel_opts=opts.LabelOpts(color="#787b86", font_size=9),
                splitline_opts=opts.SplitLineOpts(is_show=True, linestyle_opts=opts.LineStyleOpts(color="#2a2e39", width=0.5)),
            ),
            legend_opts=opts.LegendOpts(
                is_show=True, pos_top="76%", pos_right="8%",
                textstyle_opts=opts.TextStyleOpts(color="#787b86", font_size=10)
            ),
            tooltip_opts=opts.TooltipOpts(trigger="axis", axis_pointer_type="cross",
                background_color="#1e222d", border_color="#2a2e39",
                textstyle_opts=opts.TextStyleOpts(color="#d1d4dc", font_size=11)),
        )

        # ── 组合 Grid ─────────────────────────────────
        grid = Grid(init_opts=opts.InitOpts(
            bg_color="#131722", width="100%", height="900px"
        ))

        # Pane 0: K线 50%
        grid.add(kline,       grid_opts=opts.GridOpts(pos_left="1%", pos_right="7%", pos_top="4%",   height="46%"))
        # Pane 1: BSADF 14%
        grid.add(bsadf_line,  grid_opts=opts.GridOpts(pos_left="1%", pos_right="7%", pos_top="53%",  height="12%"))
        # Pane 2: Volume 14%
        grid.add(vol_bar,     grid_opts=opts.GridOpts(pos_left="1%", pos_right="7%", pos_top="68%",  height="12%"))
        # Pane 3: HV/IV 13%
        grid.add(hv_iv_line,  grid_opts=opts.GridOpts(pos_left="1%", pos_right="7%", pos_top="83%",  height="11%"))

        return grid

    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        logger.error(f"Chart render failed: {tb}")
        # 将错误存入 session_state 供 UI 展示
        st.session_state['chart_error'] = f"{type(e).__name__}: {e}"

# ══════════════════════════════════════════════════════
# 主界面
# ══════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("<div style='font-size:1.1rem; font-weight:600; color:#d1d4dc; padding:8px 0 12px;'>风控参数</div>", unsafe_allow_html=True)
    otm        = st.slider("目标建仓虚值 (%)", 5, 25, 11, help="BSADF 触发后, 卖出偏离现价至少此值的虚值合约")
    stop_loss  = st.slider("强制止损虚值 (%)", 2, 12, 6,  help="期权虚值空间低于此值立刻买回平仓")
    st.markdown("<hr style='border-color:#2a2e39; margin:10px 0;'>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:0.85rem; font-weight:500; color:#d1d4dc;'>高频预警参数</div>", unsafe_allow_html=True)
    rv_threshold = st.slider("RV 年化异常阈值 (%)", 15, 60, 30, help="盘中日化 RV 超出此值触发即时平仓预警")
    st.markdown("<hr style='border-color:#2a2e39; margin:10px 0;'>", unsafe_allow_html=True)
    push_enabled = st.checkbox("启用 PushPlus 推送", value=bool(PUSHPLUS_TOKEN))
    if push_enabled and not PUSHPLUS_TOKEN:
        st.warning("Token 未配置。请在 `.streamlit/secrets.toml` 中设置 `pushplus_token`。")
    elif push_enabled:
        st.success("推送通道激活", icon="📲")
    st.markdown("<hr style='border-color:#2a2e39; margin:10px 0;'>", unsafe_allow_html=True)
    force_refresh = st.button("强制更新数据总线", use_container_width=True)

# ── 标题 ─────────────────────────────────────────────
st.markdown('<div class="main-title">VolGuard Pro: 上证50期权风控雷达</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Multi-GARCH VaR (PSY临界值) | BSADF 泡沫测试 | HV/IV 对比 | SWR 实时缓存框架 (v6.0)</div>', unsafe_allow_html=True)

# ── 数据加载 ──────────────────────────────────────────
df_etf, source_etf = get_etf_510050(force_refresh=force_refresh)
options_df, opt_source = get_options_data(force_refresh=force_refresh)

if force_refresh:
    st.toast("数据总线更新指令已发送", icon="🔄")

if df_etf is None or df_etf.empty:
    st.error("无法加载上证50ETF基准数据，请检查网络或点击'强制更新数据总线'。")
    st.stop()

prices = df_etf['Close'].dropna()

# ── 计算指标 (缓存) ────────────────────────────────────
with st.spinner("GARCH VaR 分析中…"):
    garch_result = _cached_garch(tuple(prices.round(6).tolist()))

with st.spinner("BSADF 泡沫测试中…"):
    bsadf_result = _cached_bsadf(tuple(prices.round(6).tolist()))

returns      = np.log(prices / prices.shift(1)).dropna()
spot         = float(prices.iloc[-1])
change_pct   = float((prices.iloc[-1] / prices.iloc[-2] - 1) * 100)
bsadf_stat   = bsadf_result.get('adf_stat', 0.0)
bsadf_cv     = bsadf_result.get('cv', 2.0)
triggered    = bsadf_result.get('is_significant', False)

var_95    = garch_result.get('var_95', 0.02) * 100
var_99    = garch_result.get('var_99', 0.03) * 100
var_95_c  = garch_result.get('var_95_call', 0.02) * 100
var_95_p  = garch_result.get('var_95_put', 0.02) * 100
sigma_ann = garch_result.get('sigma_norm', 0.01) * np.sqrt(252) * 100

# HV30
hv30_val = float(returns.iloc[-30:].std() * np.sqrt(252) * 100) if len(returns) >= 30 else 0.0

# 信号生成
if triggered:
    signal    = "执行: 建立空仓"
    action    = f"卖出偏离 {var_99:.1f}%–{otm:.0f}% 虚值合约"
    sig_color = "color-orange"
    card_cls  = "card-orange"
else:
    signal    = "状态: 观望戒备"
    action    = f"BSADF({bsadf_stat:.2f}) < CV({bsadf_cv:.2f})"
    sig_color = ""
    card_cls  = ""

# ── 4 核心数据面板 ─────────────────────────────────────
st.markdown("<div style='font-size:0.9rem; font-weight:500; color:#787b86; text-transform:uppercase; letter-spacing:1px; margin:12px 0 10px;'>量化引擎参数</div>", unsafe_allow_html=True)
c1, c2, c3, c4 = st.columns(4)

with c1:
    cc = "color-red" if change_pct < 0 else "color-green"
    bc = "card-red" if change_pct < 0 else "card-green"
    st.markdown(f"""
    <div class="metric-card {bc}">
        <div class="metric-title">510050.SS (底层标的)</div>
        <div class="metric-value {cc}">{spot:.3f}</div>
        <div class="metric-sub">今日波动: <span class="{cc}">{change_pct:+.2f}%</span></div>
    </div>""", unsafe_allow_html=True)

with c2:
    iv_val = 0.0
    if options_df is not None and '隐含波动率' in options_df.columns:
        try:
            iv_val = float(pd.to_numeric(options_df['隐含波动率'], errors='coerce').mean())
        except Exception:
            pass
    hv_gt_iv = "↑ HV>IV 溢价" if hv30_val > iv_val > 0 else ("↓ IV>HV 吃权" if iv_val > hv30_val else "─")
    st.markdown(f"""
    <div class="metric-card card-blue">
        <div class="metric-title">HV30 / Avg IV</div>
        <div class="metric-value color-blue">{hv30_val:.1f}% / {iv_val:.1f}%</div>
        <div class="metric-sub">{hv_gt_iv} | GARCH σ≈{sigma_ann:.1f}%</div>
    </div>""", unsafe_allow_html=True)

with c3:
    st.markdown(f"""
    <div class="metric-card card-red">
        <div class="metric-title">VaR 95% 双向刚性防线</div>
        <div class="metric-value color-red">Put ↓{var_95_p:.2f}% / Call ↑{var_95_c:.2f}%</div>
        <div class="metric-sub">虚值空间低于任一边触发强制止损</div>
        <div class="var-bar"></div>
    </div>""", unsafe_allow_html=True)

with c4:
    st.markdown(f"""
    <div class="metric-card {card_cls}">
        <div class="metric-title">系统状态</div>
        <div class="metric-value {sig_color}" style="font-size:1.0rem;">{signal}</div>
        <div class="metric-sub">{action}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)

# ── 4-Pane 图表 ───────────────────────────────────────
col_label, col_legend = st.columns([3, 2])
with col_label:
    st.markdown("<div style='font-size:0.9rem; font-weight:500; color:#d1d4dc;'>全景联动图: K线·布林带·BSADF·量能·HV/IV</div>", unsafe_allow_html=True)
with col_legend:
    st.markdown(
        "<div style='font-size:0.75rem; color:#787b86; text-align:right; padding-top:4px;'>"
        "<span style='color:#2962ff;'>━</span> EMA5 &nbsp;"
        "<span style='color:#ff9800;'>━</span> EMA20 &nbsp;"
        "<span style='color:#9c27b0;'>⋯</span> BB(20,2) &nbsp;"
        "<span style='color:#f23645;'>⋯</span> VaR±95%</div>",
        unsafe_allow_html=True
    )

chart = render_4pane_chart(df_etf, bsadf_result, var_95, options_df)
if chart:
    st.session_state.pop('chart_error', None)
    st_pyecharts(chart, height="900px")
else:
    err = st.session_state.get('chart_error', '未知错误，请查看 Streamlit 日志')
    with st.expander("⚠️ 图表渲染失败 — 点击查看错误详情", expanded=True):
        st.code(err, language="python")

st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)

# ── 期权链雷达表格 ────────────────────────────────────
st.markdown("<div style='font-size:0.9rem; font-weight:500; color:#d1d4dc; margin-bottom:8px;'>深度虚值期权雷达扫描仪</div>", unsafe_allow_html=True)

if options_df is not None and not options_df.empty:
    try:
        # ── 字段提取 ─────────────────────────────────
        want_cols = ['代码', '名称', '最新价', '行权价', '涨跌幅', '成交量', '持仓量', '隐含波动率', '买入价', '卖出价']
        present   = [c for c in want_cols if c in options_df.columns]
        show_df   = options_df[present].copy()

        # 数值化
        for col in ['行权价', '最新价', '隐含波动率', '买入价', '卖出价', '成交量', '持仓量']:
            if col in show_df.columns:
                show_df[col] = pd.to_numeric(show_df[col], errors='coerce')

        # ── 区分认购/认沽方向计算 OTM ────────────────
        def _option_type(name: str) -> str:
            if isinstance(name, str):
                if '购' in name: return 'call'
                if '沽' in name: return 'put'
            return 'auto'

        def _otm(row) -> float:
            ot = _option_type(str(row.get('名称', '')))
            s, k = spot, row.get('行权价', np.nan)
            if pd.isna(k) or k <= 0: return 0.0
            if ot == 'call': return (k - s) / s * 100
            if ot == 'put':  return (s - k) / s * 100
            return abs(s - k) / s * 100

        show_df['类型'] = show_df['名称'].apply(_option_type).map({'call': '认购', 'put': '认沽', 'auto': '-'})
        show_df['虚值空间(%)'] = show_df.apply(_otm, axis=1).round(2)
        show_df['VaR缓冲(%)']  = (show_df['虚值空间(%)'] - var_95).round(2)

        # 买卖价差
        if '买入价' in show_df.columns and '卖出价' in show_df.columns:
            show_df['买卖价差'] = (show_df['卖出价'] - show_df['买入价']).round(4)

        show_df = show_df[show_df['行权价'] > 0]
        show_df = show_df.fillna(0).sort_values('虚值空间(%)', ascending=False)

        # 列排序
        front = ['代码', '名称', '类型', '行权价', '最新价', '虚值空间(%)', 'VaR缓冲(%)']
        back  = [c for c in show_df.columns if c not in front]
        show_df = show_df[front + back]

        # 格式化字典
        fmt = {'最新价': '{:.4f}', '行权价': '{:.3f}',
               '虚值空间(%)': '{:.2f}%', 'VaR缓冲(%)': '{:.2f}%'}
        if '隐含波动率' in show_df.columns: fmt['隐含波动率'] = '{:.1f}%'
        if '涨跌幅' in show_df.columns:    fmt['涨跌幅']    = '{:.2f}%'
        if '买卖价差' in show_df.columns:  fmt['买卖价差']  = '{:.4f}'

        def _highlight(row):
            otm_v = row.get('虚值空间(%)', 0)
            buf_v = row.get('VaR缓冲(%)', 0)
            option_type = row.get('类型', '-')
            stop = (var_95_p if option_type == '认沽' else var_95_c)

            # 绿: 深度虚值 + 安全垫充足
            if otm_v >= otm and buf_v > 2.0:
                return ['background-color: rgba(8,153,129,0.15); color:#089981; font-weight:600'] * len(row)
            # 红: 虚值已低于止损线
            elif otm_v < stop_loss:
                return ['color:#f23645; font-weight:600'] * len(row)
            # 黄: 接近警戒
            elif buf_v < 1.0:
                return ['color:#f5a623'] * len(row)
            return [''] * len(row)

        styled = (show_df.style
            .apply(_highlight, axis=1)
            .format(fmt, na_rep='—')
            .set_properties(**{'text-align': 'right', 'border-color': 'var(--tv-border)'})
            .set_table_styles([
                {'selector': 'th', 'props': [
                    ('background-color', '#1e222d'), ('color', '#787b86'),
                    ('font-weight', '500'), ('border-bottom', '1px solid #2a2e39'),
                    ('text-align', 'center'), ('font-size', '0.78rem')
                ]},
                {'selector': 'td', 'props': [('border-bottom', '1px solid #2a2e39'), ('font-size', '0.82rem')]},
            ])
        )

        st.dataframe(styled, height=480, use_container_width=True, hide_index=True)
        st.markdown(
            f"<div style='font-size:0.78rem; color:#787b86; margin-top:6px;'>"
            f"<b>图例</b>: <span style='color:#089981;'>■</span> 绿 = 深度虚值·安全垫充足 (&ge;{otm}% 且 VaR缓冲&gt;2%)&emsp;"
            f"<span style='color:#f5a623;'>■</span> 黄 = VaR 缓冲不足 1%&emsp;"
            f"<span style='color:#f23645;'>■</span> 红 = 已穿越止损线 (&lt;{stop_loss}%)&emsp;"
            f"类型列区分认购/认沽，虚值方向已修正。</div>",
            unsafe_allow_html=True
        )

    except Exception as e:
        st.error(f"期权表格解析异常: {e}")
        st.dataframe(options_df.head(50))
else:
    st.warning("期权盘口数据暂未就绪，系统后台正在异步拉取 akshare 接口…")

# ── 底部状态栏 ────────────────────────────────────────
st.markdown(
    f"<div style='text-align:right; color:#787b86; margin-top:16px; font-size:0.72rem; border-top:1px solid #2a2e39; padding-top:8px;'>"
    f"VolGuard Pro v6.0 &nbsp;|&nbsp; 数据源: yfinance + akshare &nbsp;|&nbsp; {source_etf} / {opt_source} "
    f"&nbsp;|&nbsp; {datetime.now().strftime('%H:%M:%S')}</div>",
    unsafe_allow_html=True
)
