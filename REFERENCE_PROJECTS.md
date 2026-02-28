# 可参考的开源期权数据项目

## 1. A-cyan/iVIX ⭐ 推荐

**GitHub**: https://github.com/A-cyan/iVIX  
**Stars**: 16 | **Forks**: 4

### 项目简介
上证50ETF波动率指数计算项目，包含完整的期权数据爬取和 VIX 指数计算流程。

### 核心文件
- `GetiVix.py` - 期权数据爬虫脚本
- `data_process.py` - 数据清洗与预处理
- `VIX_Compute.py` - VIX 指数计算
- `Option_Time_Series_Data_Table.csv` - 期权时间序列数据
- `Option_Basic_Information.csv` - 期权基本信息

### 克隆命令
```bash
git clone https://github.com/A-cyan/iVIX.git
cd iVIX
```

### 适用场景
- 学习 50ETF 期权数据获取方法
- 参考期权数据清洗流程
- 了解 VIX 指数计算逻辑

---

## 2. hanbinggary/options_monitor

**GitHub**: https://github.com/hanbinggary/options_monitor  
**Stars**: 0 | **Forks**: 3

### 项目简介
国内期权监测系统，支持从多个交易所获取期货和期权数据，计算历史波动率和综合隐含波动率。

### 支持的交易所
- 中金所 (CFFEX)
- 上期所 (SHFE)
- 大商所 (DCE)
- 郑商所 (CZCE)

### 核心功能
- 从各交易所获取 K 线数据
- 商品指数计算
- 历史波动率 (HV) 计算
- 综合隐含波动率 (SIV) 计算
- 反爬虫措施（代理池）
- 钉钉消息推送

### 克隆命令
```bash
git clone https://github.com/hanbinggary/options_monitor.git
cd options_monitor
pip install -r requirements.txt
```

### 数据获取方式文档
项目 README 中详细列出了各交易所的数据接口：

**中金所**:
```
http://www.cffex.com.cn/sj/hqsj/rtj/202101/05/index.xml?id=0
```

**上期所**:
```
http://www.shfe.com.cn/data/dailydata/kx/kx20210105.dat
http://www.shfe.com.cn/data/dailydata/option/kx/kx20210105.dat
```

**大商所**:
```
http://www.dce.com.cn/publicweb/quotesdata/dayQuotesCh.html (POST)
```

**郑商所**:
```
http://www.czce.com.cn/cn/DFSStaticFiles/Future/2021/20210105/FutureDataDaily.htm
http://www.czce.com.cn/cn/DFSStaticFiles/Option/2021/20210105/OptionDataDaily.htm
```

### 适用场景
- 学习多交易所数据获取方法
- 参考反爬虫策略实现
- 了解波动率计算方法
- 构建期权监测系统

---

## 3. 其他相关项目

### nkuguanrui/ivx
**GitHub**: https://github.com/nkuguanrui/ivx  
复现中国波指 000188，基于上证50 ETF期权合约。

### leonliu2001/LowRiskArbiTool
**GitHub**: https://github.com/leonliu2001/LowRiskArbiTool  
低风险套利工具，支持可转债、ETF、期货、期权等多种金融产品。

---

## 快速测试建议

### 测试 iVIX 项目
```bash
# 1. 克隆项目
git clone https://github.com/A-cyan/iVIX.git
cd iVIX

# 2. 查看 GetiVix.py 了解数据获取逻辑
cat GetiVix.py

# 3. 运行数据处理
python data_process.py

# 4. 计算 VIX
python VIX_Compute.py
```

### 测试 options_monitor 项目
```bash
# 1. 克隆项目
git clone https://github.com/hanbinggary/options_monitor.git
cd options_monitor

# 2. 安装依赖
pip install -r requirements.txt

# 3. 查看数据获取模块
ls options_monitor/

# 4. 运行测试
python test/test_*.py
```

---

## 与本项目的对比

| 特性 | 本项目 (VolGuard Pro) | iVIX | options_monitor |
|------|----------------------|------|-----------------|
| 数据源 | 新浪财经 + yfinance | 需查看源码 | 多交易所官方接口 |
| 目标市场 | 上证50ETF期权 | 上证50ETF期权 | 国内所有期权品种 |
| 核心功能 | 风控系统 + 交易信号 | VIX 指数计算 | 波动率监测 |
| 部署方式 | Streamlit Cloud | 本地运行 | 本地运行 + 定时任务 |
| 反爬虫 | 请求头 | 未知 | 代理池 |

---

## 数据源整合建议

基于这些项目的实现，可以考虑：

1. **从 iVIX 学习**：
   - 期权数据清洗流程
   - 合约筛选逻辑
   - 数据格式标准化

2. **从 options_monitor 学习**：
   - 多数据源 fallback 策略
   - 反爬虫措施（代理池）
   - 交易所官方接口直连

3. **整合到本项目**：
   - 增加东方财富直连作为 fallback
   - 参考 options_monitor 的交易所接口实现
   - 优化数据清洗和缓存策略
