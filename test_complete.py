#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""完整测试脚本：验证数据源修复和性能"""

import sys
import os
import time
sys.path.insert(0, os.path.dirname(__file__))

from data_sources import (
    _get_option_months_sina,
    fetch_50etf_options_sina,
    fetch_50etf_options_yfinance
)

def test_sina_months():
    """测试 Sina 月份解析"""
    print("\n=== 测试 Sina 月份解析 ===")
    start = time.time()
    months = _get_option_months_sina("510050")
    elapsed = time.time() - start
    
    if months:
        print(f"SUCCESS: 获取 {len(months)} 个月份")
        print(f"月份列表: {months}")
        print(f"格式验证: ", end="")
        all_valid = all(len(m) == 4 and m.isdigit() for m in months)
        print("PASS" if all_valid else "FAIL")
        print(f"耗时: {elapsed:.2f}s")
        return True
    else:
        print(f"FAIL: 未获取到月份数据")
        print(f"耗时: {elapsed:.2f}s")
        return False

def test_sina_options():
    """测试 Sina 期权数据获取"""
    print("\n=== 测试 Sina 期权数据获取 ===")
    start = time.time()
    df, msg = fetch_50etf_options_sina()
    elapsed = time.time() - start
    
    if df is not None and not df.empty:
        print(f"SUCCESS: 获取 {len(df)} 条期权数据")
        print(f"数据源: {msg}")
        print(f"列名: {list(df.columns)}")
        print(f"认购/认沽分布: {df['option_type'].value_counts().to_dict()}")
        print(f"耗时: {elapsed:.2f}s")
        return True
    else:
        print(f"FAIL: {msg}")
        print(f"耗时: {elapsed:.2f}s")
        return False

def test_yfinance_options():
    """测试 yfinance 期权数据获取"""
    print("\n=== 测试 yfinance 期权数据获取 ===")
    start = time.time()
    df, msg = fetch_50etf_options_yfinance()
    elapsed = time.time() - start
    
    if df is not None and not df.empty:
        print(f"SUCCESS: 获取 {len(df)} 条期权数据")
        print(f"数据源: {msg}")
        print(f"认购/认沽分布: {df['option_type'].value_counts().to_dict()}")
        print(f"耗时: {elapsed:.2f}s")
        return True
    else:
        print(f"INFO: {msg}")
        print(f"耗时: {elapsed:.2f}s")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("VolGuard Pro - 数据源完整测试")
    print("=" * 60)
    
    results = []
    
    # 测试 Sina 月份解析
    results.append(("Sina 月份解析", test_sina_months()))
    
    # 测试 Sina 期权数据
    results.append(("Sina 期权数据", test_sina_options()))
    
    # 测试 yfinance 期权数据
    results.append(("yfinance 期权数据", test_yfinance_options()))
    
    # 汇总结果
    print("\n" + "=" * 60)
    print("测试结果汇总")
    print("=" * 60)
    for name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{name}: {status}")
    
    passed = sum(1 for _, r in results if r)
    print(f"\n通过: {passed}/{len(results)}")
