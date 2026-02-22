# VolGuard Pro: 上证50期权全景风控系统

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> 期权卖方 (Short Volatility) 量化防御终端。策略逻辑：以 BSADF 捕捉单边泡沫窗口卖出虚值期权吃取时间价值，以三重分布 GARCH VaR 构建双向止损防线，严守尾部风险。

---

## v6.0 核心特性

### 数据层：SWR 异步缓存架构
- UI 渲染与网络拉取完全解耦，看板始终 <100ms 加载本地缓存
- 后台守护线程向 yfinance / akshare 静默轮询并落盘
- `threading.Semaphore(1)` 限制并发，防止线程爆炸
- ETF 日线 12h TTL，期权盘口 60s TTL

### 算法层：三大核心引擎
| 引擎 | 方法 | 用途 |
|------|------|------|
| BSADF | Phillips, Shi & Yu (2015) 滚动极值 ADF | 检测价格序列是否进入单边泡沫区 |
| Multi-GARCH VaR | Normal + Skew-t + 历史极端缓冲 | 预测次日价格极值边界，构建止损防线 |
| HV vs IV | 30日历史已实现波动率 vs 期权链平均隐含波动率 | 判断波动率溢价，确认卖方 Edge |

### 视图层：4-Pane ECharts 全景联动
| 窗格 | 内容 | 高度 |
|------|------|------|
| Pane 0 | K线 + EMA(5,20) + Bollinger Bands(20,2) + VaR 95% 通道 | 46% |
| Pane 1 | BSADF 统计量序列 + PSY 5% 临界值红线 | 12% |
| Pane 2 | Volume 成交量 (阴阳染色) + 5日量能 MA | 12% |
| Pane 3 | HV30 (绿) vs Avg IV (红) 波动率对比 | 11% |

四窗格通过 DataZoom `xaxis_index=[0,1,2,3]` 实现同步联动缩放。

### 安全层
- Token/Secret 通过 `.streamlit/secrets.toml` 或环境变量注入，代码零明文
- `.gitignore` 排除所有密钥文件

---

## 快速启动

### 方式一：批处理一键启动
双击 `启动看板.bat`，脚本会自动检查全量依赖并启动。

### 方式二：命令行启动
```bash
pip install -r requirements.txt
streamlit run app.py
```
浏览器自动打开 `http://localhost:8501/`。

### PushPlus 推送配置 (可选)
在项目根目录创建 `.streamlit/secrets.toml`：
```toml
pushplus_token  = "your_token"
pushplus_secret = "your_secret"
```

---

## 看板操作速查

### 系统状态指令
- **执行: 建立空仓** — BSADF 统计量突破 PSY 临界值，泡沫窗口开启
- **状态: 观望戒备** — BSADF 未达显著区间，不入场

### VaR 双向防线
- 面板显示 `Put` 和 `Call` 两个方向的 95% VaR 距离
- 持仓期权虚值空间小于任一方向 VaR 值时，必须无条件止损

### 期权扫描表三色标注
- **绿色底纹**：深度虚值，VaR 缓冲 > 2%，安全垫充沛
- **黄色字体**：VaR 缓冲 < 1%，接近警戒线
- **红色字体**：虚值已低于止损线，禁止开仓 / 立即平仓

---

## 技术文档
- 算法原理与判断逻辑：[DEVELOPER.md](DEVELOPER.md)
- 系统参数与数据流配置：[CONFIG.md](CONFIG.md)

## 许可证
[MIT License](LICENSE)
