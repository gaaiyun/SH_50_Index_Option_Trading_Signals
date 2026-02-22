# VolGuard Pro: 上证 50 期权全景风控系统 (v5.0)

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/) 
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg)](https://streamlit.io/)
[![Quant](https://img.shields.io/badge/Quant-ShortVol-yellow.svg)]()

> 本系统为期权卖方（Short Volatility）构建的防御级量化监控终端。策略核心逻辑：**寻找局部泡沫吃取时间价值，严守尾部黑天鹅极值防线。**系统以滚动 BSADF 单位根测试捕捉单边极值，以三重分布假定的 GARCH 预测构建隔夜 VaR 绝对警戒线。

---

## 核心特性 (v5.0)

1. **毫秒级闪电开盘 (SWR 架构)**：彻底重构数据流，采用 Stale-While-Revalidate (SWR) 守护线程机制。UI 渲染与网络拉取完全解耦。看板始终瞬间（<100ms）加载本地急速缓存，同时后台静默向 `akshare`/`yfinance` 轮询最新数据并落盘。无阻塞，零假死。
2. **TradingView 级全景三联窗格 (3-Pane Grid)**：
    *   **Pane 1 (主量价与防线)**：K 线主图叠加 GARCH VaR 95% 动态压力/支撑带，直观标识价格生死线。
    *   **Pane 2 (泡沫警示系统)**：独立的 BSADF 评估统计量时间序列与 95% 显著水位红线。
    *   **Pane 3 (动能刻度)**：严格根据日线收盘价阴阳染色的 Volume 当期成交量。
3. **极简金融电报词族**：去除一切主观模糊表达与过度包装（“AI 味”），文本指令全盘转换为诸如：“执行: 建立空仓”、“状态: 观望戒备” 等专业交易终端的极简指令集。
4. **期权深度雷达矩阵**：结合高阶 Pandas Styler 的深度资金盘口表。实时解算当前合约的虚值空间及距离 VaR 95% 止损线的具体缓冲厚度，并通过红黄绿极差色带引导视觉决策，规避非理性抗单。

---

## 极速启动手册

### 步骤一：环境对齐
请确保使用 Python 3.9+ 并在纯净虚拟环境中安装专属武器库：
```bash
pip install streamlit pandas numpy akshare arch statsmodels scipy yfinance pyecharts streamlit-echarts
```

### 步骤二：指令点火
打开终端进入本项目的根目录，运行看板渲染引擎：
```bash
cd C:\Users\gaaiy\Desktop\CSI_50_Index_Option_Trading_Signals
streamlit run app.py
```
终端将自动弹出浏览器直接进入 `http://localhost:8501/`。

---

## 驾驶舱仪表盘速查

### 1. 【系统状态指令】
- 位于顶部核心数值卡片。当行情极值触发 BSADF 时，宣告“执行: 建立空仓”。若未达显著极值区间，将显示“状态: 观望戒备”，此时坚决不予入场，防止做空波动率被趋势粉碎。

### 2. 【VaR 95% 刚性防线】
- GARCH 引擎预测的复合边界。如果系统输出 `±2.50%`，意味着您的持仓期权若距现价虚值空间小于 2.50%，即面临极高行权风险，**必须无条件清仓止损**。

### 3. 【三窗格高阶图表】
- 聚焦副图的 BSADF Stats 橙色折线，当其向上穿透血红色的 95% 极值红线时，意味着市场单边泡沫进入极限发酵期，胜率窗口开启。

### 4. 【深度虚值扫描雷达 (表格)】
- 🟢 **绿底标识**：安全垫充沛，距离被击穿拥有巨大缓冲空间。
- 🔴 **红字/红底标识**：已进入 GARCH 计算的危险射程内，禁止开新仓；若持有务必即发止损。

---

## 关于底层数学架构开发
欲深入打磨底层量化逻辑、调整三重 GARCH 分布模型极值边界或探索 SWR 异步机制的开发者，请详参本项目配套白皮书：
*   算法内核与 UI 渲染技术解密：[`DEVELOPER.md`](DEVELOPER.md)
*   系统风控参数与数据流约束：[`CONFIG.md`](CONFIG.md)
