# VolGuard Pro: 上证50期权风控终端

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Live_App-FF4B4B.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

VolGuard Pro 是一个面向上证50ETF期权卖方风控的双站系统：

- **Streamlit 主站**：实时点击更新行情，计算 GARCH VaR、BSADF、GEX/DEX、HV/IV 和期权雷达表。
- **Cloudflare Pages 展示站**：读取 `public/data/latest.json`，展示公开快照、项目门面和历史快照入口。

主站负责真实计算，展示站负责轻量访问。Cloudflare Pages 的刷新按钮只重新读取最新 JSON 快照，不直接运行 Python 量化计算。

## 核心能力

### 数据层

- ETF 日线：`yfinance`
- 期权链：Sina 为主源，yfinance / akshare 作为 fallback
- Streamlit 主站使用 SWR 缓存：本地缓存优先，后台刷新，避免页面被网络波动拖死
- GitHub Actions 可定时生成 `public/data/latest.json`

### 算法层

| 引擎 | 用途 |
|---|---|
| Multi-GARCH VaR | 估计 Put / Call 双向 95% 风险边界 |
| BSADF | 检测价格序列是否进入单边泡沫窗口 |
| GEX / DEX | 计算期权链 Gamma / Delta 市场暴露 |
| HV / IV | 比较已实现波动与期权隐含波动 |

BSADF 在看板中采用“当前时点完整扫描 + 历史曲线采样”的方式，保留当前信号判断的完整性，同时避免首屏长期卡在 ADF 回归。

### 视图层

- Streamlit 主站使用专业交易终端风格，顶部展示实时数据状态、生成时间和快照可用性。
- 4-pane ECharts 联动图包含 K 线、BSADF、成交量、HV/IV。
- OTM 期权雷达表按认购/认沽方向计算虚值空间和 VaR 缓冲。
- Cloudflare Pages 展示站提供静态快照、价格序列、风险卡片和期权链样例。

## 快速启动

```bash
pip install -r requirements.txt
streamlit run app.py
```

浏览器打开 `http://localhost:8501/`。

也可以双击 `启动看板.bat` 启动本地 Streamlit 主站。

## 生成 Cloudflare Pages 快照

使用本地缓存生成：

```bash
python scripts/export_snapshot.py --output public/data/latest.json
```

尝试在线刷新后生成：

```bash
python scripts/export_snapshot.py --refresh --output public/data/latest.json
```

本地预览展示站：

```bash
python -m http.server 8787 --bind 127.0.0.1 --directory public
```

浏览器打开 `http://127.0.0.1:8787/`。

## 测试

```bash
python -m pytest tests/ -q -m "not online"
```

在线数据源测试带 `online` 标记，默认 CI 不运行。

## 可选推送配置

在项目根目录创建 `.streamlit/secrets.toml`：

```toml
pushplus_token = "your_token"
pushplus_secret = "your_secret"
```

也可以通过环境变量 `PUSHPLUS_TOKEN` 注入。

## 部署

- Streamlit 主站部署见 [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md)。
- Cloudflare Pages 展示站部署目录为 `public/`。
- 定时快照 workflow 为 `.github/workflows/pages-snapshot.yml`。

## 技术文档

- 算法原理与判断逻辑：[DEVELOPER.md](DEVELOPER.md)
- 系统参数与数据流配置：[CONFIG.md](CONFIG.md)

## 许可证

[MIT License](LICENSE)
