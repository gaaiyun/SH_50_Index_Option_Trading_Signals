# 部署指南：Streamlit 主站 + Cloudflare Pages 展示站

VolGuard Pro 推荐双站部署：

- **Streamlit 主站**：运行 `app.py`，负责实时刷新、Python 指标计算和交互式风控终端。
- **Cloudflare Pages 展示站**：部署 `public/`，读取 `public/data/latest.json`，负责公开展示和快照访问。

## 1. Streamlit 主站

### Streamlit Cloud

创建应用时填写：

- Repository: `gaaiyun/SH_50_Index_Option_Trading_Signals`
- Branch: `main`
- Main file path: `app.py`
- Python version: `3.11` 或 `3.12`

可选 secrets：

```toml
pushplus_token = "your_pushplus_token_here"
pushplus_secret = ""  # 可选；普通 /send 消息接口不需要
```

部署后如需验证 PushPlus：

1. 打开 Streamlit 主站。
2. 侧栏勾选“启用 PushPlus 推送”。
3. 点击“发送测试推送”。
4. 页面显示“测试推送已发送”且 PushPlus 收到消息，才说明真实通道通畅。

不要把 `pushplus_token` 或 `pushplus_secret` 写入 README、Issue、PR 或提交历史。

### 本地验证

```bash
pip install -r requirements.txt
python -m pytest tests/ -q -m "not online"
streamlit run app.py
```

主站应显示：

- `VolGuard Pro Live`
- ETF / Options 数据状态
- Quant Engine 四个核心卡片
- Greeks Market Exposure
- Four-pane Risk Chart
- OTM Option Radar

点击“强制更新数据总线”后，页面应保持可用；后台会刷新 ETF 与期权缓存。

## 2. Cloudflare Pages 展示站

Cloudflare Pages 不直接运行 Streamlit，也不直接执行 `arch`、`statsmodels`、`akshare` 等 Python 计算。它只托管 `public/` 静态文件，并读取 JSON 快照。

### 本地生成快照

```bash
python scripts/export_snapshot.py --output public/data/latest.json
```

如果希望先尝试在线拉取：

```bash
python scripts/export_snapshot.py --refresh --output public/data/latest.json
```

### 本地预览

```bash
python -m http.server 8787 --bind 127.0.0.1 --directory public
```

打开 `http://127.0.0.1:8787/`。页面按钮“刷新快照”会重新读取 `data/latest.json`。

## 3. 定时快照 workflow

`.github/workflows/pages-snapshot.yml` 每 30 分钟运行一次，也支持手动触发：

1. 安装 Python 依赖
2. 运行 `python scripts/export_snapshot.py --refresh --output public/data/latest.json`
3. 上传 `public/` artifact
4. 如果配置了 Cloudflare secrets，则部署到 Cloudflare Pages

需要配置：

- Repository secret: `CLOUDFLARE_API_TOKEN`
- Repository variable: `CLOUDFLARE_ACCOUNT_ID`
- Repository variable: `CLOUDFLARE_PAGES_PROJECT`，默认可用 `sh50-volguard`

如果 secrets 未配置，workflow 仍会生成并上传 artifact，但不会部署到 Cloudflare。

## 4. 数据与刷新边界

- Streamlit 主站：按钮触发后台刷新，可实时重新拉数据并重新计算。
- Pages 展示站：按钮只重新拉取最新 JSON；JSON 新旧取决于 GitHub Actions 最近一次生成时间。
- 若 Sina / yfinance 网络失败，主站和快照脚本会回退到本地缓存。

## 5. 排查

### Streamlit 首屏慢

BSADF 已做响应优化：当前信号完整扫描，历史曲线采样扫描。如果仍慢，优先检查数据量、`arch` / `statsmodels` 安装和服务器 CPU。

### Pages 显示 Snapshot load failed

检查：

- `public/data/latest.json` 是否存在
- JSON 是否可通过浏览器访问
- Cloudflare Pages 的部署目录是否为 `public/`

### 期权 GEX/DEX 为 0

确认期权数据至少包含：

- `行权价`
- `隐含波动率`
- `持仓量`
- `名称` 或 `类型` / `option_type`

真实 Sina 缓存中 `名称` 可能只是 `510050`，此时方向会从 `类型` / `option_type` 判断。
