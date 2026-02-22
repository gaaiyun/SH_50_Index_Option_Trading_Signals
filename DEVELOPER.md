# VolGuard Pro 量化大屏 - 开发者技术与算法原理白皮书 (v5.0)

本文档面向对该衍生品量化大屏进行二次开发、算法重构的工程师与量化研究员。详细阐述了底层的纯数学逻辑与最新 v5.0 的渲染黑科技。

---

## 一、核心引擎与目录架构

项目极度内聚，数学计算核心与 UI 完全隔离：
*   `app.py`: Streamlit 应用主干，内嵌原生级高频交易专用的 Stale-While-Revalidate 并发引擎，以及承载 Pyecharts Grid 布局的图表控制栈。
*   `strategy/indicators.py`: 纯量化算法内核大类 `StrategyIndicators` 所在地。

### 数值体系依赖拓扑:
*   **arch**: 实现多分布族 GARCH 波动预测引擎。
*   **statsmodels**: 执行时间序列的高级分析测算，底层调用 `adfuller` 实现严谨单位根检验。
*   **scipy**: 负责极值分布模型的逆累积分布函数推算 (PPF)。

---

## 二、数学防雷区：三大核心指标的底层抽象

本策略为典型的极右偏“赌大概率回归/卖波动”算法，因此核心代码集中在过滤尾部悬崖崩盘事件。

### 2.1 泡沫发生器：向后极大右尾 ADF 检验 (BSADF)
**金融学动机**：寻找是否存在一个导致资产价格指数膨胀的局部巨型几何根 (Explosive Unit Root)。
**实现 (`calculate_bsadf`)**：
1. 取对数价格 $p_t = \ln(P_t)$。
2. 设定最小测算滑窗为 100 根线。
3. 从每一个子样本区间执行 Augmented Dickey-Fuller 测试，提取 $\gamma > 0$ （右尾爆炸过程）的结果。
4. 取出测试结果之中的上确界（Supremum）形塑成时序结构。
5. **v5.0 UI级改动**：抛弃简陋黄点，将其整体投射至独立 Pane，与实证界限 $cv$（例如 1.5）直接绘制时序追踪对比。

### 2.2 防护穹顶：多重分布假定的 GARCH VaR 评估
**金融学动机**：预测明天到底多大波动率区间算是安全降落伞，借此作为实盘斩仓死线（认怂线）。
**实现 (`calculate_garch_var`)**：
提纯基于 `arch` 工具库的波动率方差模型：
$$ \sigma_t^2 = \omega + \alpha_1 \varepsilon_{t-1}^2 + \beta_1 \sigma_{t-1}^2 $$
1. **纯正态sGARCH**：`dist='Normal'`，给出基础 95% 与 99% VaR 幅度。
2. **偏态斯图登特t (Skew-t)**：`dist='skewstudent'`，拟合金融重尾分布自由度 $\nu$。通过分布极限求取比正态模型严密得多的护城河缓冲带厚度。
*   **系统调用**：`app.py` 中会直接提取预测出的 `var_95` 将其具象化。在主 K 线上方直接生成一条基于收盘价向外拓展的伪布林带，作为直观的高危预警线。

### 2.3 极致高频盘中 RV 熔断器 (Realized Volatility)
**技术实现 (`calculate_daily_rv`)**：
提取 5分钟 级别的盘中连续对数收益率重构盘中日化波动率。若 RV 断层式暴涨超出设定的 2σ 等极限，强行切断等待日开盘隔夜结果的思维定势，发送平单提示。

---

## 三、UI 前端底层原理及黑科技 (v5.0 突破)

### 3.1 毫秒级 SWR 异步推流架构 (`Stale-While-Revalidate`)
在 v4.0 及以前，只要数据源阻塞 3 秒，UI 就会报出网络迟延警报乃至假死。
v5.0 中重写了 `load_local_cache` 极速调度模型：
*   所有读取行为直接使用 Pandas 开凿本地 CSV 文件系统。
*   对于带有时间戳失效认证 (`is_cache_expired`) 的文件，使用 Python 内置库 `threading` 启动一个原生 Daemon Background Thread 执行网络 I/O 等费操作。
*   由于主线程无死锁继续流通，UI 会维持令人发指的毫秒级渲染响应速度。

### 3.2 纯血 TradingView 三联 Grid 视窗 (`Pyecharts`)
在 `render_kline_with_bsadf` 函数中，摒弃了原有单图叠加 (`Overlap`) 的局限性，动用了最高级版图排期 `Grid` 引擎。
*   主视图以 50% 的核心屏宽占据 `grid_index=0`，嵌入 K线 + 双面虚线描摹的 GARCH 收口通道。
*   次位窗采用 15% 屏宽分流至 `grid_index=1`，展示带红线的 BSADF。
*   底位窗采用 15% 屏宽汇合至 `grid_index=2`，绘制红绿放榜成色量 (Volume)。
三幅子组件皆由一组高度精巧设置的内置 `DataZoom` 组件强锁定，共享着唯一的 X 轴联动尺度，彻底达成了复刻交易员终端神韵的壮举！
