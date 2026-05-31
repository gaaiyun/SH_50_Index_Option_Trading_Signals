#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VolGuard Pro — 期权数据源模块

主要解决问题：
- akshare.option_current_em 在海外环境 (如 Streamlit Cloud) 经常失效
- 本模块提供新浪财经数据源作为主数据源，akshare 作为可选备用

公开函数：
- fetch_50etf_options_sina() -> (df, source_msg)
"""

import logging
import re
import time
import random
from typing import List, Tuple, Callable, Any
from functools import wraps

import pandas as pd
import requests
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq

logger = logging.getLogger(__name__)


# 反爬虫策略：多个 User-Agent 轮换
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:123.0) Gecko/20100101 Firefox/123.0",
]

SINA_HEADERS = {
    "User-Agent": USER_AGENTS[0],
    "Referer": "https://stock.finance.sina.com.cn/",
}


def retry_with_backoff(max_retries: int = 3, base_delay: float = 0.5) -> Callable:
    """重试装饰器，带指数退避"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise
                    delay = base_delay * (2 ** attempt) + random.uniform(0, 0.5)
                    logger.warning(f"{func.__name__} failed (attempt {attempt+1}/{max_retries}): {e}, retrying in {delay:.2f}s")
                    time.sleep(delay)
            return None
        return wrapper
    return decorator


def get_random_headers(referer: str = None) -> dict:
    """获取随机 User-Agent 的请求头"""
    headers = {"User-Agent": random.choice(USER_AGENTS)}
    if referer:
        headers["Referer"] = referer
    return headers


@retry_with_backoff(max_retries=3, base_delay=0.5)
def _get_option_months_sina_raw(underlying: str = "510050") -> List[str]:
    """
    从新浪获取可用期权月份列表（底层实现）。

    返回类似 ["2503", "2504", "2506", ...] 的字符串列表；
    网络持续失败时由 retry_with_backoff 在重试耗尽后抛出异常，
    交由 _get_option_months_sina 包装层兜底。
    """
    url = (
        "http://stock.finance.sina.com.cn/futures/api/openapi.php/"
        "StockOptionService.getStockName"
    )
    headers = get_random_headers("https://stock.finance.sina.com.cn/")
    resp = requests.get(url, headers=headers, timeout=8)
    resp.raise_for_status()
    data = resp.json()
    result = data.get("result", {}).get("data", {})
    months = result.get("contractMonth", []) or []
    parsed = []
    for m in months:
        if isinstance(m, str):
            if m.isdigit():
                parsed.append(m)
            elif len(m) == 7 and m[4] == "-":  # "YYYY-MM" -> "YYMM"
                parsed.append(m[2:4] + m[5:7])
    months = parsed
    if not months:
        logger.warning(f"Sina getStockName returned empty months, raw data keys: {list(result.keys())}")
    else:
        logger.info(f"Sina months OK: {months}")
    return months


def _get_option_months_sina(underlying: str = "510050") -> List[str]:
    """
    新浪期权月份列表（对外稳定接口）。

    在 _get_option_months_sina_raw 之上加一层兜底：重试耗尽（网络始终失败）时返回 []，
    而非向上抛异常。这与同模块 _get_option_codes_sina / _get_option_detail_sina
    "失败即返回空" 的约定一致；调用方 fetch_50etf_options_sina 依赖此约定，
    在月份为空时给出 "Sina 合约月份获取失败" 提示而非崩溃。
    """
    try:
        return _get_option_months_sina_raw(underlying)
    except Exception as e:
        logger.warning(f"Sina getStockName failed after retries: {e}")
        return []


def _get_option_codes_sina(underlying: str, month: str) -> Tuple[List[str], List[str]]:
    """
    获取某个月份的认购 / 认沽合约代码列表。

    返回: (call_codes, put_codes)
    """
    call_url = f"http://hq.sinajs.cn/list=OP_UP_{underlying}{month}"
    put_url = f"http://hq.sinajs.cn/list=OP_DOWN_{underlying}{month}"

    call_codes: List[str] = []
    put_codes: List[str] = []

    try:
        call_resp = requests.get(call_url, headers=SINA_HEADERS, timeout=8)
        call_resp.encoding = "gbk"
        call_codes = re.findall(r"CON_OP_\d+", call_resp.text)
        if not call_codes:
            logger.warning(f"Sina {month} call codes empty, resp length: {len(call_resp.text)}")
    except Exception as e:
        logger.warning(f"Sina call code list failed for {month}: {e}")

    try:
        put_resp = requests.get(put_url, headers=SINA_HEADERS, timeout=8)
        put_resp.encoding = "gbk"
        put_codes = re.findall(r"CON_OP_\d+", put_resp.text)
        if not put_codes:
            logger.warning(f"Sina {month} put codes empty, resp length: {len(put_resp.text)}")
    except Exception as e:
        logger.warning(f"Sina put code list failed for {month}: {e}")

    return call_codes, put_codes


def _get_option_detail_sina(codes: List[str]) -> pd.DataFrame:
    """
    批量获取新浪期权合约行情。

    输出字段会在上层统一重命名为：
      - 代码 / 名称 / 最新价 / 行权价 / 涨跌幅 / 成交量 / 持仓量 / 买入价 / 卖出价
    """
    if not codes:
        return pd.DataFrame()

    code_str = ",".join(codes)
    url = f"http://hq.sinajs.cn/list={code_str}"

    try:
        resp = requests.get(url, headers=SINA_HEADERS, timeout=8)
        resp.encoding = "gbk"
    except Exception as e:
        logger.warning(f"Sina option detail request failed: {e}")
        return pd.DataFrame()

    rows = []
    lines = resp.text.strip().split("\n")
    for line in lines:
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        code = key.replace("var hq_str_", "").strip()
        value = value.strip().strip('";')
        fields = value.split(",")

        # 期权返回通常有 42+ 字段，这里做长度保护
        if len(fields) < 40:
            continue

        try:
            rows.append(
                {
                    "合约代码": code,
                    "买价": float(fields[1]) if fields[1] else 0.0,
                    "最新价": float(fields[2]) if fields[2] else 0.0,
                    "卖价": float(fields[3]) if fields[3] else 0.0,
                    "持仓量": float(fields[5]) if fields[5] else 0.0,
                    "涨幅": float(fields[6]) if fields[6] else 0.0,
                    "行权价": float(fields[7]) if fields[7] else 0.0,
                    "昨收价": float(fields[8]) if fields[8] else 0.0,
                    "开盘价": float(fields[9]) if fields[9] else 0.0,
                    "合约简称": fields[36] if len(fields) > 36 else code,
                    "最高价": float(fields[38]) if fields[38] else 0.0,
                    "最低价": float(fields[39]) if fields[39] else 0.0,
                    "成交量": float(fields[40]) if len(fields) > 40 and fields[40] else 0.0,
                    "成交额": float(fields[41]) if len(fields) > 41 and fields[41] else 0.0,
                }
            )
        except Exception:
            # 单行解析失败直接跳过
            continue

    if not rows:
        logger.warning(f"Sina detail parsed 0 rows from {len(lines)} lines, {len(codes)} codes requested")
    return pd.DataFrame(rows)


def fetch_50etf_options_sina() -> tuple[pd.DataFrame, str]:
    """
    从新浪财经获取 50ETF 全期权链数据。

    返回: (DataFrame, source_msg)
    DataFrame 至少包含 app.py 使用到的字段：
      - 代码 / 名称 / 最新价 / 行权价 / 涨跌幅 / 成交量 / 持仓量 / 买入价 / 卖出价
    """
    months = _get_option_months_sina("510050")
    if not months:
        return pd.DataFrame(), "Sina 合约月份获取失败"

    all_df: list[pd.DataFrame] = []

    for month in months:
        call_codes, put_codes = _get_option_codes_sina("510050", month)

        if call_codes:
            call_df = _get_option_detail_sina(call_codes)
            if not call_df.empty:
                call_df["类型"] = "认购"
                call_df["月份"] = month
                all_df.append(call_df)

        if put_codes:
            put_df = _get_option_detail_sina(put_codes)
            if not put_df.empty:
                put_df["类型"] = "认沽"
                put_df["月份"] = month
                all_df.append(put_df)

    if not all_df:
        return pd.DataFrame(), "Sina 期权行情为空"

    df = pd.concat(all_df, ignore_index=True)

    # 统一重命名为主程序习惯的中文列名
    rename_map = {
        "合约代码": "代码",
        "合约简称": "名称",
        "最新价": "最新价",
        "行权价": "行权价",
        "涨幅": "涨跌幅",
        "成交量": "成交量",
        "持仓量": "持仓量",
        "买价": "买入价",
        "卖价": "卖出价",
    }
    df = df.rename(columns=rename_map)
    df["option_type"] = df["类型"]  # 添加英文列名以便测试和标准化

    return df, "Sina options loaded"


def fetch_50etf_options_yfinance() -> tuple[pd.DataFrame, str]:
    """
    从 yfinance 获取 50ETF 期权链（仅作 fallback，国内标的在 Yahoo 上可能无数据或不全）。

    返回: (DataFrame, source_msg)，列名与 Sina 一致：代码/名称/最新价/行权价/涨跌幅/成交量/持仓量/买入价/卖出价
    """
    try:
        import yfinance as yf
    except Exception as e:
        logger.warning(f"yfinance import failed: {e}")
        return pd.DataFrame(), "yfinance 未安装或不可用"

    for symbol in ["510050.SS", "510050"]:
        try:
            ticker = yf.Ticker(symbol)
            dates = getattr(ticker, "options", None)
            if not dates:
                continue
            all_calls, all_puts = [], []
            for exp in dates[:6]:
                try:
                    chain = ticker.option_chain(exp)
                    if chain.calls is not None and not chain.calls.empty:
                        c = chain.calls.copy()
                        c["类型"], c["到期"] = "认购", exp
                        all_calls.append(c)
                    if chain.puts is not None and not chain.puts.empty:
                        p = chain.puts.copy()
                        p["类型"], p["到期"] = "认沽", exp
                        all_puts.append(p)
                except Exception:
                    continue
            if not all_calls and not all_puts:
                continue

            frames = []
            for df_list in [all_calls, all_puts]:
                if not df_list:
                    continue
                raw = pd.concat(df_list, ignore_index=True)
                raw = raw.rename(columns={
                    "contractSymbol": "代码", "strike": "行权价", "lastPrice": "最新价",
                    "bid": "买入价", "ask": "卖出价", "volume": "成交量", "openInterest": "持仓量"
                })
                if "代码" not in raw.columns and "contractSymbol" in raw.columns:
                    raw["代码"] = raw["contractSymbol"]
                raw["名称"] = raw["代码"] if "代码" in raw.columns else ""
                raw["涨跌幅"] = raw["percentChange"] if "percentChange" in raw.columns else 0.0
                if "impliedVolatility" in raw.columns:
                    raw["隐含波动率"] = raw["impliedVolatility"] * 100
                frames.append(raw)
            
            if not frames:
                continue
            out = pd.concat(frames, ignore_index=True)
            out["option_type"] = out["类型"]  # 添加英文列名以便测试和标准化
            return out, "yfinance options loaded"
        except Exception as e:
            logger.warning(f"yfinance options for {symbol} failed: {e}")
    return pd.DataFrame(), "yfinance 无 50ETF 期权数据"


def fetch_50etf_options_eastmoney() -> tuple[pd.DataFrame, str]:
    """
    从东方财富获取 50ETF 期权数据（作为 fallback）
    
    返回: (DataFrame, source_msg)
    """
    try:
        url = "http://push2.eastmoney.com/api/qt/clist/get"
        params = {
            "pn": "1",
            "pz": "500",
            "fs": "m:10+t:3",  # 期权市场
            "fields": "f12,f13,f14,f2,f3,f15,f16,f17,f18,f152,f5,f6",
        }
        headers = get_random_headers("http://quote.eastmoney.com/")
        
        resp = requests.get(url, params=params, headers=headers, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        
        if not data.get("data") or not data["data"].get("diff"):
            return pd.DataFrame(), "东方财富无期权数据"
        
        items = data["data"]["diff"]
        rows = []
        for item in items:
            code = item.get("f12", "")
            if "510050" not in code:  # 只保留 50ETF 相关
                continue
            
            rows.append({
                "代码": code,
                "名称": item.get("f14", ""),
                "最新价": item.get("f2", 0) / 1000 if item.get("f2") else 0,
                "涨跌幅": item.get("f3", 0) / 100 if item.get("f3") else 0,
                "成交量": item.get("f5", 0),
                "持仓量": item.get("f152", 0),
                "买入价": item.get("f17", 0) / 1000 if item.get("f17") else 0,
                "卖出价": item.get("f18", 0) / 1000 if item.get("f18") else 0,
            })
        
        if not rows:
            return pd.DataFrame(), "东方财富未找到 50ETF 期权"
        
        df = pd.DataFrame(rows)
        
        # 从名称判断类型和行权价
        df["类型"] = df["名称"].apply(lambda x: "认购" if "购" in str(x) else "认沽")
        df["option_type"] = df["类型"]
        
        # 提取行权价（从名称中解析，格式如 "50ETF购3月2800"）
        def extract_strike(name):
            import re
            match = re.search(r'(\d+)$', str(name))
            if match:
                return float(match.group(1)) / 1000  # 2800 -> 2.800
            return 0.0
        
        df["行权价"] = df["名称"].apply(extract_strike)
        
        logger.info(f"Eastmoney loaded {len(df)} options")
        return df, "Eastmoney options loaded"
        
    except Exception as e:
        logger.warning(f"Eastmoney options failed: {e}")
        return pd.DataFrame(), f"东方财富获取失败: {e}"


def clean_option_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    数据清洗：参考 iVIX 项目的数据处理流程
    
    - 移除无效数据（价格为0、行权价为0）
    - 移除异常波动数据（涨跌幅超过±50%）
    - 填充缺失值
    """
    if df is None or df.empty:
        return df
    
    df = df.copy()
    
    # 移除无效数据
    if "最新价" in df.columns:
        df = df[df["最新价"] > 0]
    if "行权价" in df.columns:
        df = df[df["行权价"] > 0]
    
    # 移除异常波动
    if "涨跌幅" in df.columns:
        df = df[(df["涨跌幅"] >= -50) & (df["涨跌幅"] <= 50)]
    
    # 填充缺失的数值列
    numeric_cols = ["买入价", "卖出价", "成交量", "持仓量", "涨跌幅"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)
    
    # 填充缺失的字符串列
    string_cols = ["代码", "名称"]
    for col in string_cols:
        if col in df.columns:
            df[col] = df[col].fillna("")
    
    logger.info(f"Data cleaned: {len(df)} rows remaining")
    return df


def _bs_call_price(S, K, T, r, sigma):
    """Black-Scholes 认购期权定价"""
    if T <= 0 or sigma <= 0:
        return max(S - K, 0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def _bs_put_price(S, K, T, r, sigma):
    """Black-Scholes 认沽期权定价"""
    if T <= 0 or sigma <= 0:
        return max(K - S, 0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def _calculate_tte(name: str) -> float:
    """从合约名称提取到期时间（年）"""
    from datetime import datetime, timedelta
    try:
        if isinstance(name, str) and len(name) >= 11:
            year_month = name[7:11]
            year = 2000 + int(year_month[:2])
            month = int(year_month[2:])
            first_day = datetime(year, month, 1)
            first_wed = first_day + timedelta(days=(2 - first_day.weekday() + 7) % 7)
            expiry = first_wed + timedelta(weeks=3)
            days = (expiry - datetime.now()).days
            return max(1, days) / 365.0
    except Exception:
        pass
    return 30 / 365.0


def _implied_volatility(price, S, K, T, r, opt_type='call'):
    """从期权价格反推隐含波动率（Brent 方法）"""
    if price <= 0 or T <= 0:
        return 0.0
    
    intrinsic = max(S - K, 0) if opt_type == 'call' else max(K - S, 0)
    if price <= intrinsic:
        return 0.0
    
    bs_func = _bs_call_price if opt_type == 'call' else _bs_put_price
    
    try:
        def objective(sigma):
            return bs_func(S, K, T, r, sigma) - price
        
        iv = brentq(objective, 0.001, 5.0, xtol=1e-6, maxiter=100)
        return iv * 100  # 转换为百分比
    except Exception:
        return 20.0  # 默认 20%


def add_implied_volatility(df: pd.DataFrame, spot: float, risk_free_rate: float = 0.015) -> pd.DataFrame:
    """
    为期权数据添加隐含波动率列
    
    参数:
        df: 期权数据，需包含：名称、最新价、行权价
        spot: 标的现价
        risk_free_rate: 无风险利率
    
    返回:
        添加了"隐含波动率"列的 DataFrame
    """
    if df is None or df.empty:
        return df
    
    df = df.copy()
    
    # 如果已有隐含波动率且非空，直接返回
    if '隐含波动率' in df.columns and df['隐含波动率'].notna().any():
        return df
    
    ivs = []
    for _, row in df.iterrows():
        try:
            price = float(row.get('最新价', 0))
            strike = float(row.get('行权价', 0))
            name = str(row.get('名称', ''))
            
            if price <= 0 or strike <= 0:
                ivs.append(0.0)
                continue
            
            tte = _calculate_tte(name)
            opt_type = 'call' if '购' in name else 'put'
            
            iv = _implied_volatility(price, spot, strike, tte, risk_free_rate, opt_type)
            ivs.append(iv)
        except Exception:
            ivs.append(0.0)
    
    df['隐含波动率'] = ivs
    return df


__all__ = ["fetch_50etf_options_sina", "fetch_50etf_options_yfinance", "fetch_50etf_options_eastmoney", "add_implied_volatility", "clean_option_data"]
