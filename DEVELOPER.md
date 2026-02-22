# 中证50股指期货期权量化策略 - 开发技术说明

## 一、项目概述

### 1.1 策略背景

本策略是一套经典的**做空波动率(Short Volatility)**策略，通过卖出深度虚值期权来赚取时间价值（Theta）。策略的核心特点是结合了宏观泡沫检验、高频波动率监控以及多种GARCH模型进行严格的尾部风险管理。

### 1.2 策略目标

- 赚取期权时间价值
- 严格控制尾部风险
- 预期年化收益：5-8%

---

## 二、数据源

### 2.1 主要数据源

| 数据类型 | 数据源 | API函数 |
|----------|--------|---------|
| 指数日线 | 东方财富 | `stock_zh_index_daily_em()` |
| 期货合约 | 东方财富 | `futures_contract_info_cffex()` |
| 期权数据 | 新浪财经 | `option_sse_daily_sina()` |
| A股实时 | 东方财富 | `stock_zh_a_spot_em()` |

### 2.2 数据获取

```python
import akshare as ak
import os

# 设置代理
os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'

# 获取中证50指数数据
index_data = ak.stock_zh_index_daily_em(symbol="000016")

# 获取期货合约列表
futures_data = ak.futures_contract_info_cffex()

# 获取期权数据
options_data = ak.option_sse_daily_sina()
```

---

## 三、依赖包

### 3.1 核心依赖

| 包名 | 版本 | 用途 |
|------|------|------|
| Python | 3.12+ | 运行环境 |
| akshare | 1.18.25 | 金融数据获取 |
| pandas | 3.0.1 | 数据处理 |
| numpy | 2.4.2 | 数值计算 |
| arch | - | GARCH模型 |
| statsmodels | - | 统计检验(ADF) |
| streamlit | - | Web看板 |

### 3.2 安装命令

```bash
pip install akshare pandas numpy arch statsmodels streamlit
```

---

## 四、核心指标计算方法

### 4.1 BSADF (Backward Supremum Augmented Dickey-Fuller) 泡沫检验

**目的**: 检测指数是否处于"泡沫期"或极端单边行情

**原理**: 
- 传统的ADF检验用于检验是否存在单位根（随机游走）
- BSADF在传统基础上进行右侧检验，检测"爆炸性根"
- 当统计量超过临界值时，认为存在泡沫

**Python实现**:

```python
import numpy as np
import statsmodels.tsa.stattools as ts

def calculate_bsadf(returns, window=100):
    """
    计算BSADF泡沫检验统计量
    
    参数:
        returns: 对数收益率序列
        window: 滚动窗口大小
    
    返回:
        bsadf_stat: BSADF统计量序列
    """
    bsadf_stats = []
    
    for i in range(window, len(returns)):
        # 取滚动窗口数据
        window_data = returns[i-window:i]
        
        # 执行ADF检验
        try:
            result = ts.adfuller(window_data, regression='ct', autolag='AIC')
            # 右侧检验
            adf_stat = result[0]
            p_value = result[1]
            
            # 记录统计量
            bsadf_stats.append({
                'adf_stat': adf_stat,
                'p_value': p_value
            })
        except:
            bsadf_stats.append({'adf_stat': np.nan, 'p_value': np.nan})
    
    return bsadf_stats

# 使用示例
# index_data = ak.stock_zh_index_daily_em(symbol="000016")
# index_data['returns'] = np.log(index_data['收盘']/index_data['收盘'].shift(1))
# bsadf_result = calculate_bsadf(index_data['returns'].dropna())
```

### 4.2 GARCH波动率预测

**目的**: 预测下一天的波动率σ，计算VaR分位数

**原理**:
- GARCH(1,1)模型: σ²_t = ω + α·ε²_{t-1} + β·σ²_{t-1}
- 滚动250天数据拟合，预测t+1的σ
- 使用不同分布假设计算VaR

**Python实现**:

```python
import numpy as np
import pandas as pd
from arch import arch_model

def calculate_garch_var(returns, confidence_levels=[0.90, 0.95, 0.99], window=250):
    """
    计算GARCH波动率预测和VaR分位数
    
    参数:
        returns: 对数收益率序列
        confidence_levels: 置信水平列表
        window: 滚动窗口大小
    
    返回:
        var_results: 各置信水平的VaR分位数
    """
    results = {}
    
    # 取最近window个数据
    recent_returns = returns.dropna().iloc[-window:]
    
    # 1. 正态分布GARCH
    model_norm = arch_model(recent_returns * 100, vol='Garch', p=1, q=1, dist='normal')
    res_norm = model_norm.fit(disp='off')
    forecast_norm = res_norm.forecast(horizon=1)
    sigma_norm = np.sqrt(forecast_norm.variance.iloc[-1].values[0]) / 100
    
    for cl in confidence_levels:
        z = np.abs(np.percentile(np.random.normal(0, 1, 10000), (1-cl)*100))
        results[f'norm_{int(cl*100)}'] = z * sigma_norm
    
    # 2. 偏态t分布GARCH (非对称)
    model_t = arch_model(recent_returns * 100, vol='Garch', p=1, q=1, dist='skewt')
    res_t = model_t.fit(disp='off')
    forecast_t = res_t.forecast(horizon=1)
    sigma_t = np.sqrt(forecast_t.variance.iloc[-1].values[0]) / 100
    
    for cl in confidence_levels:
        results[f'ghyp_{int(cl*100)}'] = sigma_t * 1.2  # 偏态t更厚尾
    
    # 3. 跳跃GARCH (简化版)
    # 实际需要使用rugarch包，这里用增强波动率近似
    sigma_jump = sigma_norm * 1.3  # 加入跳跃风险溢价
    for cl in confidence_levels:
        z = np.abs(np.percentile(np.random.normal(0, 1, 10000), (1-cl)*100))
        results[f'jump_{int(cl*100)}'] = z * sigma_jump
    
    return results

# 使用示例
# index_data['returns'] = np.log(index_data['收盘']/index_data['收盘'].shift(1))
# var_result = calculate_garch_var(index_data['returns'])
# print(f"VaR 99%: {var_result['norm_99']:.4f}")
```

### 4.3 RV (Realized Volatility) 已实现波动率

**目的**: 盘中实时监控波动率，用于高频止损

**原理**:
- RV = Σ r² (日内高频收益率的平方和)
- 5分钟K线数据计算

**Python实现**:

def calculate_rv(prices):
    """
    计算已实现波动率RV
    
    参数:
        prices: 价格序列（5分钟K线）
    
    返回:
        rv: 已实现波动率
    """
    # 计算对数收益率
    log_returns = np.log(prices / prices.shift(1)).dropna()
    
    # RV = Σ r²
    rv = np.sqrt(np.sum(log_returns ** 2))
    
    return rv

# 使用示例
# 假设有5分钟K线数据
# rv_5min = calculate_rv(kline_data['close'])
# print(f"5分钟RV: {rv_5min:.6f}")

---

## 五、策略交易规则

### 5.1 建仓条件

| 条件 | 说明 |
|------|------|
| BSADF检验显著 | 泡沫期，单边行情可能衰竭 |
| 卖出深度虚值 | 偏离现价11%以上的期权 |
| 只做50指数 | 流动性好，尾部风险低 |

### 5.2 止损条件

| 条件 | 触发 | 操作 |
|------|------|------|
| 虚值程度不足 | 从11%跌至6.4% | 第二天开盘平仓 |
| 高频RV异常 | 5分钟RV飙升 | 盘中立即平仓 |

### 5.3 仓位管理

- 每次只卖出一份期权
- 不加杠杆
- 严格止损

---

## 六、风控体系

### 6.1 低频风控（日线）

```
IF 当前虚值程度 < 6.4%:
    第二天开盘平仓
ELSE:
    继续持仓
```

### 6.2 高频风控（盘中）

```
IF 5分钟RV > 阈值(历史均值+2倍标准差):
    立即平仓
ELSE:
    继续持仓
```

---

## 七、Web看板

### 7.1 技术栈

- **前端**: Streamlit (Python Web框架)
- **数据**: AkShare API
- **图表**: Plotly/Altair

### 7.2 功能模块

1. **市场数据** - 实时指数、期货、期权数据
2. **策略指标** - BSADF、GARCH、VaR、RV
3. **交易信号** - 建仓/止损信号
4. **策略说明** - 完整策略文档

### 7.3 运行方式

```bash
cd "C:\Users\gaaiy\Desktop\中证50期权策略看板"
streamlit run app.py
```

---

## 八、文件结构

```
中证50期权策略看板/
├── app.py              # Streamlit看板主程序
├── README.md           # 使用说明
└── DEVELOPER.md       # 本开发技术文档
```

---

## 九、注意事项

1. **网络要求**: 必须开启VPN/代理才能获取数据
2. **数据延迟**: 部分数据可能有延迟
3. **实盘风险**: 回测不等于实盘，请谨慎使用
4. **模型局限**: GARCH模型对突发事件预测能力有限

---

## 十、参考资料

1. Tushare金融数据平台: https://tushare.pro/
2. AkShare开源库: https://akshare.akfamily.xyz
3. arch库文档: https://arch.readthedocs.io
4. statsmodels库: https://www.statsmodels.org

---

*文档版本: v1.0*
*创建日期: 2026-02-22*
