# VolGuard Pro — 算法原理与技术架构 (v6.0)

本文档面向开发者和量化研究员，详细阐述系统三大核心引擎的计算方法、判断逻辑和实现细节。

---

## 1. BSADF 泡沫检测引擎

### 1.1 理论背景

BSADF (Backward Supremum Augmented Dickey-Fuller) 是 Phillips, Shi & Yu (2015) 提出的右尾单位根泡沫检测方法。其核心思想：

- 标准 ADF 检验检测左尾（平稳性），但泡沫是**右尾爆炸过程**
- 泡沫期间价格序列表现为**爆炸性自回归**：`ρ > 1`，即 ADF 统计量变成大的正值
- 通过滑动窗口取 Supremum（极大值），可以捕捉到最强的爆炸性信号

### 1.2 算法步骤

```
输入: log_prices = ln(收盘价序列), 最小窗口 r0 = 100

对于 t = max(r0, n-200) ... n-1:
    对于每个起始点 s = max(0, t-250) ... t-r0:
        计算 ADF(log_prices[s:t+1], regression='ct', autolag='AIC')
        记录 adf_stat
    BSADF[t] = sup(adf_stat)  // 所有子窗口的最大值

最终信号 = BSADF[n-1]
```

### 1.3 PSY 临界值

系统使用 Phillips, Shi & Yu (2015) Table 1 的 5% 渐进临界值近似公式：

```
cv = 1.0 + 0.26 × ln(n)
```

其中 `n` 为样本总长度。例如：
| 样本量 n | 临界值 cv |
|----------|-----------|
| 250      | 2.44      |
| 500      | 2.61      |
| 1000     | 2.79      |

### 1.4 判断规则

| 条件 | 判断结果 | 操作建议 |
|------|----------|----------|
| `BSADF[latest] > cv` | 泡沫显著 | 系统发出"执行: 建立空仓"信号，寻找符合虚值目标的期权卖出 |
| `BSADF[latest] <= cv` | 无泡沫 | 系统显示"状态: 观望戒备"，不入场 |

### 1.5 实现要点

- 仅计算最近 200 根 K 线的 BSADF，保证副图与 K 线主图时间对齐
- 每个时间点的向后滑窗最多回溯 250 天，保证计算不溢出
- 通过 `@st.cache_data(ttl=900)` 缓存 15 分钟，避免每次刷新重算

---

## 2. Multi-Distribution GARCH VaR 引擎

### 2.1 理论背景

GARCH(1,1) (Generalized Autoregressive Conditional Heteroskedasticity) 用于预测次日收益的条件方差：

```
σ²[t] = ω + α × r²[t-1] + β × σ²[t-1]
```

其中：
- `ω` — 长期方差基底
- `α` — 滞后残差的冲击系数（短期波动响应）
- `β` — 持久性系数（长期记忆）
- `α + β < 1` — 保证方差平稳性

VaR (Value at Risk) 95% 的含义：**有 95% 的概率，次日涨跌幅不超过此值**。

### 2.2 三重分布防线

系统同时拟合三种假设，取最宽（最保守）的防线：

| 模型 | 分布假设 | 分位数计算 | 特点 |
|------|----------|------------|------|
| sGARCH-Normal | 正态分布 | `z × σ`，z = Φ⁻¹(cl) | 基线模型，低估厚尾 |
| sGARCH-SkewT | Hansen's Skewed t | `t.ppf(1-cl, df=ν) × σ × adj` | 捕捉偏态和厚尾 |
| Extreme Buffer | 历史 99% 分位数 | `max(2σ, P99(|r|))` | 应对黑天鹅跳空 |

### 2.3 Skew-t 分位数计算（v6.0 修正）

旧版使用 `z_norm × σ × 1.2` 的无理论依据近似，v6.0 改为正确计算：

```python
nu = max(res_skew.params['nu'], 2.5)     # 自由度
lam = res_skew.params['lambda']           # 偏度参数

# 左尾 (Put 认沽防线)
q_put = scipy.stats.t.ppf(1 - cl, df=nu)   # 负值
var_put = |q_put| × σ_skew × (1 + max(-lam, 0) × 0.3)

# 右尾 (Call 认购防线)  
q_call = scipy.stats.t.ppf(cl, df=nu)      # 正值
var_call = q_call × σ_skew × (1 + max(lam, 0) × 0.3)
```

### 2.4 双向 VaR 输出

v6.0 区分认购 (Call) 和认沽 (Put) 方向：

| 输出键 | 含义 | 用途 |
|--------|------|------|
| `var_95_put` | Put 下行 95% VaR | 认沽期权止损参考 |
| `var_95_call` | Call 上行 95% VaR | 认购期权止损参考 |
| `var_99_put` | Put 下行 99% VaR | 极端事件警报 |
| `var_99_call` | Call 上行 99% VaR | 极端事件警报 |

**判断规则**：若持仓认沽期权的虚值空间 < `var_95_put`，触发强制止损。认购方向同理。

### 2.5 Extreme Event Buffer（v6.0 修正）

旧版使用 `σ × 1.35` 的无模型依据乘数，v6.0 改为数据驱动：

```python
# 取最近 250 日收益绝对值的 99% 分位数
historical_99 = np.percentile(np.abs(returns[-250:]), 99)
# 极端缓冲 = max(2 × GARCH预测σ, 历史99分位)
extreme_buffer = max(2 * sigma_skew, historical_99)
```

---

## 3. HV vs IV 波动率对比

### 3.1 理论背景

期权卖方的核心 Edge（正期望来源）在于：**隐含波动率 (IV) 普遍高估实际发生的波动率 (HV)**。这被称为 Variance Risk Premium（方差风险溢价）。

| 指标 | 定义 | 来源 |
|------|------|------|
| HV30 | 过去 30 个交易日对数收益率标准差的年化值 | `std(ln(Ct/Ct-1)) × sqrt(252) × 100` |
| Avg IV | 期权链中所有合约隐含波动率的平均值 | akshare 期权盘口数据 |

### 3.2 判断逻辑

| 条件 | 含义 | 操作 |
|------|------|------|
| IV > HV | 波动率溢价存在 | 卖方有正 Edge，可以考虑建仓 |
| HV > IV | 波动率折价 | 卖方处于逆风，谨慎或不入场 |
| IV >> HV (差值 > 10%) | 极高溢价 | 卖方黄金窗口，但需关注是否因事件驱动 |

---

## 4. OTM 虚值空间计算（v6.0 修正）

### 4.1 方向区分

v6.0 通过合约名称中的"购"/"沽"字符判断期权类型：

```python
认购 (Call) 虚值 = (行权价 - 现价) / 现价 × 100%     # 仅 strike > spot 时为正虚值
认沽 (Put)  虚值 = (现价 - 行权价) / 现价 × 100%     # 仅 spot > strike 时为正虚值
```

旧版使用 `abs(spot - strike)` 不区分方向，导致实值期权被误判为安全。

### 4.2 期权表三色告警

| 颜色 | 条件 | 含义 |
|------|------|------|
| 绿色底纹 | 虚值 >= 目标建仓值 且 VaR缓冲 > 2% | 安全垫充沛，可建仓目标 |
| 黄色字体 | VaR 缓冲 < 1% | 接近防线，需关注 |
| 红色字体 | 虚值 < 止损线 | 已击穿，禁止开仓 / 必须平仓 |

---

## 5. 已实现波动率 RV 计算（v6.0 修正）

### 5.1 正确年化公式

```python
# 日内 RV = sqrt(Σ(r_i²))，其中 r_i 为每根 K 线的对数收益
# 年化 RV = daily_rv × sqrt(n_per_day × 250)
#   n_per_day = 240 / freq_minutes = 48 (5分钟K线)
```

旧版使用 `daily_rv × sqrt(250)` 漏乘了日内区间数，低估了约 7 倍。

---

## 6. 系统架构

### 6.1 SWR 缓存时序图

```
用户刷新 → 读 data/etf_510050.csv → 立即渲染 UI
                                    ↓ (后台线程)
                        TTL 过期? → yfinance.download() → 更新 CSV
                                    ↓
                        threading.Semaphore(1) 保证最多 1 个线程
```

### 6.2 并发控制

| 信号量 | 保护资源 | TTL |
|--------|----------|-----|
| `_ETF_LOCK` | yfinance ETF 日线拉取 | 12h |
| `_OPT_LOCK` | akshare 期权盘口拉取 | 60s |

### 6.3 计算缓存

| 函数 | 缓存方式 | TTL |
|------|----------|-----|
| `_cached_garch()` | `@st.cache_data` | 1h |
| `_cached_bsadf()` | `@st.cache_data` | 15min |

---

## 7. 文件结构

```
├── app.py                    # Streamlit 主程序 + 4-Pane ECharts
├── strategy/
│   ├── __init__.py
│   └── indicators.py         # BSADF / GARCH / RV / OTM 算法内核
├── push_client.py            # PushPlus 微信推送
├── .streamlit/
│   └── secrets.toml          # Token 密钥 (不纳入 Git)
├── data/                     # SWR 本地缓存 (不纳入 Git)
├── requirements.txt
├── 启动看板.bat
├── CONFIG.md                 # 参数配置说明
├── DEVELOPER.md              # 本文档
├── README.md                 # 项目概览
└── LICENSE                   # MIT License
```
