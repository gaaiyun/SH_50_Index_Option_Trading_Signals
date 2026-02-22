# 中证50期权策略看板 - 完整配置文档

## 一、项目概述

这是一个基于GARCH波动率预测的中证50股指期权量化交易策略看板。

**功能**:
- 实时市场数据获取
- BSADF泡沫检验建仓信号
- GARCH波动率预测和VaR计算
- 自动刷新（每5分钟）
- PushPlus消息推送

---

## 二、配置信息汇总

### 2.1 PushPlus推送

| 项目 | 值 |
|------|-----|
| Token | `3660eb1e0b364a78b3beed2f349b29f8` |
| Secret | `ddff31dda80446cc878c163b2410bc5b` |
| 发送方式 | POST请求 |
| 模板 | markdown |

### 2.2 GitHub仓库

| 项目 | 值 |
|------|-----|
| 仓库地址 | `https://github.com/gaaiyun/CSI_50_Index_Option_Trading_Signals` |
| 分支 | main |
| 本地路径 | `C:\Users\gaaiy\Desktop\CSI_50_Index_Option_Trading_Signals` |

### 2.3 Streamlit Cloud

| 项目 | 值 |
|------|-----|
| 部署地址 | https://share.streamlit.io |
| App URL | (部署后获得) |

---

## 三、本地开发指南

### 3.1 环境准备

```bash
# 安装Python依赖
pip install streamlit pandas numpy akshare arch statsmodels scipy requests

# 进入项目目录
cd C:\Users\gaaiy\Desktop\CSI_50_Index_Option_Trading_Signals

# 启动看板
streamlit run app.py
```

### 3.2 本地运行

```bash
# 方式1: 直接运行
streamlit run app.py

# 方式2: 使用代理(中国访问akshare需要)
set HTTP_PROXY=http://127.0.0.1:7890
set HTTPS_PROXY=http://127.0.0.1:7890
streamlit run app.py
```

### 3.3 浏览器访问

本地访问: http://localhost:8501

---

## 四、修改代码流程

### 4.1 修改代码

1. 用文本编辑器打开 `C:\Users\gaaiy\Desktop\CSI_50_Index_Option_Trading_Signals\app.py`
2. 修改代码
3. 保存

### 4.2 推送到GitHub

```bash
# 打开终端，进入项目目录
cd C:\Users\gaaiy\Desktop\CSI_50_Index_Option_Trading_Signals

# 添加所有修改
git add .

# 提交修改
git commit -m "更新说明"

# 推送到GitHub
git push
```

### 4.3 自动部署

推送到GitHub后，Streamlit Cloud会自动重新部署（等待1-2分钟）。

---

## 五、文件说明

| 文件 | 说明 |
|------|------|
| app.py | 主程序看板 |
| push_client.py | 推送模块(备用) |
| strategy/indicators.py | 指标计算 |
| README.md | 基础说明 |
| DEVELOPER.md | 开发文档 |

---

## 六、策略参数

### 6.1 核心参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| GARCH窗口 | 250天 | 滚动计算历史 |
| VaR置信度 | 99% | 风险阈值 |
| 建仓虚值程度 | 11% | 卖出期权的虚值比例 |
| 止损虚值程度 | 6.4% | 触发止损的虚值比例 |

### 6.2 交易规则

**建仓条件**:
- BSADF检验显著（泡沫期）
- 卖出11%虚值程度期权

**止损条件**:
- 虚值程度 < 6.4%
- 高频RV异常

---

## 七、常见问题

### Q1: 数据获取失败
- 本地运行需要开启VPN/代理
- Streamlit Cloud无法访问中国数据源

### Q2: 推送不成功
- 检查Token是否正确
- 检查网络连接

### Q3: GitHub推送失败
- 检查Token权限
- 确保已配置git用户信息

---

## 八、联系方式

- GitHub: gaaiyun
- 项目: CSI_50_Index_Option_Trading_Signals

---

*更新时间: 2026-02-22*
*版本: v2.2*
