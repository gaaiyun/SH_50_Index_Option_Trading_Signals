# VolGuard Pro — 系统参数与数据流配置 (v6.0)

---

## 1. 风控参数 (Sidebar Controls)

以下参数可在看板左侧面板实时调节：

| 参数 | 范围 | 默认值 | 说明 |
|------|------|--------|------|
| 目标建仓虚值 (%) | 5–25 | 11 | BSADF 触发后，仅卖出偏离现价 >= 此值的虚值合约 |
| 强制止损虚值 (%) | 2–12 | 6 | 持仓期权虚值空间缩窄至此值以下，立即买回平仓 |
| RV 年化异常阈值 (%) | 15–60 | 30 | 盘中已实现波动率超出此值，触发高频平仓预警 |

---

## 2. BSADF 引擎参数

| 参数 | 值 | 说明 |
|------|-----|------|
| `window` | 100 | 最小 ADF 检验子窗口长度 (r0) |
| 最大回溯 | 250 | 每个时间点向后扩展窗口的最大长度 |
| 计算范围 | 最后 200 天 | 保证 BSADF 副图与 K 线主图时间对齐 |
| 临界值公式 | `1.0 + 0.26 × ln(n)` | Phillips, Shi & Yu (2015) 5% 渐进近似 |
| ADF 参数 | `regression='ct', autolag='AIC'` | 含常数项和时间趋势，自动选择滞后阶数 |
| 缓存 TTL | 900s (15 min) | `@st.cache_data` 避免重复计算 |

---

## 3. GARCH VaR 引擎参数

| 参数 | 值 | 说明 |
|------|-----|------|
| `window` | 250 | 拟合窗口 (约 1 年交易日) |
| 置信水平 | 90%, 95%, 99% | 三档分位数 |
| 正态模型 | `vol='Garch', p=1, q=1, dist='Normal'` | 基线 |
| 偏态 t 模型 | `vol='Garch', p=1, q=1, dist='skewstudent'` | Hansen's Skewed Student-t |
| 极端缓冲 | `max(2σ, P99(abs(r)))` | 历史极端值驱动，不再使用魔法系数 |
| 缓存 TTL | 3600s (1h) | `@st.cache_data` 避免重复计算 |
| 输出 | `var_95_put`, `var_95_call` 等 | 双向 VaR (认沽/认购) |

---

## 4. 数据源与缓存策略

### 4.1 数据源

| 数据 | 来源 | 函数 | 说明 |
|------|------|------|------|
| 510050.SS ETF 日线 | yfinance | `get_etf_510050()` | 近 5 年 OHLCV |
| 50ETF 期权盘口 | akshare `option_current_em()` | `get_options_data()` | 全量期权链实时报价 |

### 4.2 SWR 缓存 TTL

| 缓存文件 | TTL | 说明 |
|----------|-----|------|
| `data/etf_510050.csv` | 43200s (12h) | 日线数据日内不变 |
| `data/options_50.csv` | 60s | 盘口秒级变化 |

### 4.3 并发控制

每种数据源有独立的 `threading.Semaphore(1)`，保证最多一个后台线程在拉取。网络缓慢时不会累积线程。

---

## 5. 密钥配置

| 密钥 | 来源 | 说明 |
|------|------|------|
| `pushplus_token` | `.streamlit/secrets.toml` 或环境变量 `PUSHPLUS_TOKEN` | PushPlus 推送 Token |
| `pushplus_secret` | `.streamlit/secrets.toml` 或环境变量 `PUSHPLUS_SECRET` | PushPlus 签名密钥 |

secrets.toml 示例：
```toml
pushplus_token  = "your_token_here"
pushplus_secret = "your_secret_here"
```

密钥文件已在 `.gitignore` 中排除，绝不提交至版本控制。

---

## 6. ECharts 4-Pane 图表配置

| 窗格 | 图表类型 | Grid 位置 | 技术指标 |
|------|----------|-----------|----------|
| Pane 0 | Kline + Line overlay | top=4%, height=46% | EMA(5), EMA(20), BB(20,2), VaR 95% |
| Pane 1 | Line | top=53%, height=12% | BSADF 序列 + CV 红线 |
| Pane 2 | Bar + Line overlay | top=68%, height=12% | Volume (阴阳色) + MA(5) |
| Pane 3 | Line (双系列) | top=83%, height=11% | HV30 vs Avg IV |

DataZoom 配置：
- Inside (鼠标滚轮缩放): `xaxis_index=[0,1,2,3]`
- Slider: `pos_bottom=1%, height=4%`
- 默认显示范围: 60%–100% (最近约 80 个交易日)

---

## 7. 年化常数对照

| 场景 | 公式 | 值 |
|------|------|-----|
| 日线收益标准差年化 | `σ_daily × sqrt(252)` | 252 个交易日/年 |
| 5 分钟线 RV 年化 | `rv_intraday × sqrt(48 × 250)` | 48 段/日 × 250 日 |
| GARCH σ 年化 | `σ_garch × sqrt(252)` | 与日线一致 |
