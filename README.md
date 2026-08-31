# A 股全自动量化交易系统

这是一个模拟盘优先的日线量化系统。默认使用周度动量轮动、内置 SQLite 模拟账户和黑色 Streamlit 控制台。AkShare 是主数据源，配置了 `TUSHARE_TOKEN` 后可在日线接口失败时回退到 Tushare。

> 默认模式始终是 `PAPER`。QMT、PTrade 和 easytrader 文件只是适配边界，不包含可执行实盘下单逻辑。接入真实账户前应完成券商 SDK、交易权限、报单回调、撤单、重连和小资金验收。

## 项目结构

```text
.
├── ashare_quant/
│   ├── brokers/                    # 模拟券商与实盘适配边界（paper/qmt/ptrade/easytrader）
│   ├── data/                       # AkShare/Tushare 数据源、缓存、演示数据
│   ├── services/                   # 信号、交易编排、运行时与控制开关
│   ├── strategies/                 # 因子库与各策略（动量/双均线/多因子等）
│   ├── backtest.py                 # Backtrader A 股回测
│   ├── cli.py                      # 命令行入口
│   ├── config.py                   # YAML 与环境变量配置
│   ├── database.py                 # SQLite 表结构与访问层
│   ├── event_similarity.py         # 事件相似度检索与历史表现回溯
│   ├── fundamental.py              # 事件库与宏观指标采集（关键词打分，可选 LLM）
│   ├── hot_strategy.py             # 短线人气·热度共振（四维共振打分，日线近似回测）
│   ├── industry_chain.py           # 产业链关键词图谱（上下游）
│   ├── lab.py                      # 因子实验室与盘中买点扫描引擎
│   ├── limit_pullback_strategy.py  # 涨停回马枪·冲高回调低吸（日线近似回测）
│   ├── main_rally.py               # 主升浪策略（事件×宏观×产业链×盈利×技术）
│   ├── market_rules.py             # 费用、整手、涨跌停规则
│   ├── ml_model.py                 # 主升概率模型（GBDT 基线 + 重训管线）
│   ├── models.py                   # 数据模型（dataclass）
│   ├── notifications.py            # 企业微信与邮件
│   ├── presentation.py             # 看板展示与字段本地化
│   ├── risk.py                     # 交易前和账户风控
│   ├── scheduler.py                # 盘前/盘中/盘后/重训定时调度
│   ├── trading_calendar.py         # 交易日历（节假日与调休休市）
│   └── utils.py                    # 日期与通用工具
├── config/default.yaml          # 非敏感默认配置
├── deploy/                      # systemd 与 cron 示例
├── scripts/                     # Windows 任务计划 + 历史数据拉取 + 短线策略回测脚本
├── tests/                       # 核心流程测试
├── .streamlit/config.toml       # 黑色主题
├── dashboard.py                 # Streamlit 操作看板
├── Dockerfile
├── .dockerignore                # 排除密钥、数据库与本机缓存
├── docker-compose.yml
├── pyproject.toml
├── requirements.txt
└── .env.example
```

## Windows 安装

要求 Python 3.10 或更高版本。建议使用 3.11。

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
```

编辑 `.env`，至少修改 `DASHBOARD_PASSWORD`。Tushare、企业微信和邮件均可留空。不要把 `.env` 提交到版本库；任何曾粘贴到聊天或日志的 Token 都应先轮换。

## 先跑通模拟盘

以下流程完全使用确定性演示行情，不依赖网络：

```powershell
python -m ashare_quant.cli init-db
python -m ashare_quant.cli seed-demo
python -m ashare_quant.cli backtest --strategy momentum_rotation
python -m ashare_quant.cli paper-demo
python -m ashare_quant.cli status
streamlit run dashboard.py
```

浏览器访问 `http://localhost:8501`。配置了管理员密码时，账户、持仓、委托与回测数据在登录前处于隐藏状态；输入 `.env` 中管理员密码后，策略开关、模拟成交、生成信号和手动平仓才可操作（登录状态在会话内保持）。

## 获取真实行情

首次运行会拉取沪深 A 股与 ETF 基础信息，并尽力获取沪深 300 成分。策略候选池为沪深 300 成分股加 ETF（由 `universe_members` 表强制执行；成分数据缺失时退化为全部 A 股）；ST、退市和停牌标记会被过滤，科创板、创业板、北交所和可转债不会进入 V1 候选池。

```powershell
python -m ashare_quant.cli update-data
python -m ashare_quant.cli generate-signals --strategy momentum_rotation
python -m ashare_quant.cli queue-orders
```

AkShare 接口受上游站点和网络状态影响。批量更新按标的隔离失败并重试，并以 `data.update_concurrency`（默认 4）并发拉取日线，全市场增量更新从原先串行的数小时缩短到约二三十分钟；数据源频繁报错/限流时可把该值调到 1~2（1=串行）。Tushare Token 存在时用于日线回退，但低积分账号仍可能被限流。若使用 Tushare 第三方付费代理，可在 `.env` 设置 `TUSHARE_API_URL`（留空则走官方 `api.tushare.pro`；代理到期接口报错时系统自动退回 AkShare 主源）。生产环境应先用少量代码测试：

```powershell
python -m ashare_quant.cli update-data --codes 600000,600036,510300
```

## 日线运行时序

- `09:25`：盘前主升浪扫描——抓取全球事件（LLM 抽取 + 关键词兜底）→ 更新宏观指标 → 形成盘前候选池 → 推送富文本报告（详见「主升浪盘前扫描与富文本报告」）。
- `09:35`：刷新待执行标的实时快照，执行模拟委托，并按 T+1 更新可卖数量。
- `15:30`：增量更新日线；动量轮动在周五生成下周委托，其他策略按日生成次日委托。
- `15:50`：按收盘价更新账户净值并检查日亏损限制。

> 自 2026-07-06 起，A 股新增盘后固定价格交易时段（15:05–15:30，按当日收盘价成交），
> 因此盘后数据更新与信号生成须推迟到 15:30 之后。以上时间在 `config/default.yaml`
> 的 `scheduler` 下配置，调度器会读取该配置；同日起沪深主板 ST/*ST 涨跌幅由 5% 上调至 10%（`board_price_limit` 已同步）。

内置调度器（按交易所交易日历自动跳过法定节假日与调休休市，接口失败时退回工作日判断）：

```powershell
python -m ashare_quant.cli scheduler
```

Windows 开机任务可选择常驻调度器或三个离散任务，两者不要同时启用：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\install_windows_tasks.ps1 -Mode Scheduler
# 或
powershell -ExecutionPolicy Bypass -File scripts\install_windows_tasks.ps1 -Mode Discrete
```

## 技术面因子与策略

系统内置可复用的技术面因子库（`ashare_quant/strategies/factors.py`），全部基于日线 OHLCV 计算，实盘信号与回测共用同一套打分逻辑。因子原始值在横截面 z-score 后由策略按带符号权重合成（正=越高越好，负=越低越好）。

内置策略（参数在 `config/default.yaml` 的 `strategies` 下调整，看板侧边栏 / CLI `--strategy` 可切换）：

| 策略标识 | 说明 | 使用因子 |
|---|---|---|
| `momentum_rotation` | 周度动量轮动 | 动量 + 成交额过滤 |
| `dual_moving_average` | 双均线趋势 | 快慢均线 |
| `simple_multifactor` | 简易多因子 | 动量、波动率、流动性 |
| `reversal` | 短期反转 | 5 日反转 |
| `low_volatility` | 低波动 | 波动率、ATR |
| `trend` | 趋势跟踪 | 52 周新高、均线多头、MACD |
| `liquidity` | 流动性 | 成交额、Amihud 非流动性 |
| `rsi_mean_reversion` | RSI 超卖反转 | RSI |
| `enhanced_multifactor` | 增强多因子 | 动量、反转、波动率、流动性、52 周新高 |

因子窗口等参数可用 `{因子名}_{参数名}` 覆盖，例如在 `strategies` 下设置 `momentum_window: 120` 或 `macd_fast: 10`。

## 短线情绪策略（人气热度共振 × 涨停回马枪）

系统内置两个短线情绪/形态策略，均为**日线近似回测**（盘中的人气榜、封板时间、炸板次数、换手率等无法历史回填，按日线近似）。核心实现分别在 `ashare_quant/hot_strategy.py` 与 `ashare_quant/limit_pullback_strategy.py`，策略设计要点已并入下方两个小节。

### 短线人气·热度共振

对全市场 A 股用「人气 + 题材 + 涨停 + 资金」四维共振打分，选出总分最高的少数强势股，按状态自动标注参与方式（首板打板 / 半路买入 / 低吸埋伏 / 连板接力）。日线近似下：人气分用当日涨幅横截面分位代理，题材热度用「板块涨停聚集 ≥3 家 + 板块涨幅领先前 10」判定，资金量能用量比。叠加情绪周期风控（冰点空仓）与竞价缺口过滤。

> 注意：该策略是盘中实时型，人气榜只有实时前 100、无法历史回填，日线近似回测会系统性偏向「追涨」，结果不代表真实表现；应按文档 §10 用模拟盘信号记录验证，而非依赖日线回测。

### 涨停回马枪·冲高回调低吸

形态链路：涨停启动(T) → 次日冲高确认(H，冲高 ≥5% 且创 20 日新高) → 缩量回调(H+1~H+8) → 双信号止跌(①缩量十字星/小阳 + ②放量阳线收复 5 日线) → 尾盘低吸。叠加大盘闸门（上证 ≥MA20、前日涨停 ≥50 家、跌停 ≤10 家等）、前高减半 + 移动止盈 + 硬止损风控。

**回测结果**（全市场 5216 只、2024-01~2026-08，前复权，含佣金/印花税/T+1/滑点）：

| 指标 | 优化前 | 优化后 |
|---|---|---|
| 年化收益 | -6.09% | **+8.79%** |
| 最大回撤 | -17.33% | **-5.56%** |
| 胜率 | 35.06% | **62.50%** |

分年度稳健：2024 +1.86% / 2025 +10.51% / 2026 +11.79%（三年全正）。优化要点：修复「前高减半」bug、移动止盈仅在浮盈后生效、信号②放量门槛 1.5→2.0、信号①缩量门槛 0.5→0.4、打分阈值 50→70，并修复 `_load_panels` 日期格式 bug（此前 end_date 所在年份数据被字符串比较误滤）。

### 回测命令与脚本

```powershell
python -m ashare_quant.cli hot-backtest        # 人气热度共振回测（落库供看板展示）
python -m ashare_quant.cli pullback-backtest   # 涨停回马枪回测（落库供看板展示）
```

看板「因子实验室」内置两个策略模板（`engine=hot_score` / `engine=limit_pullback_score`），保存后即可在因子实验室里跑回测。

短线策略需全市场历史日线，演示库（30 只）无法回测，可拉取全市场历史数据到独立库：

```powershell
python scripts/fetch_hist_data.py --start 2023-07-01 --end 2026-08-28   # 拉全市场日线（前复权）到 data/ashare_quant_hist.db
python scripts/backtest_compare.py              # 基线 vs 优化对比
python scripts/scan_pullback_params.py          # 止损/止盈/信号参数扫描
```

`fetch_hist_data.py` 走 Tushare 接口（`.env` 的 `TUSHARE_TOKEN` + 可选 `TUSHARE_API_URL` 第三方代理），按交易日批量拉取并做前复权，流式写入独立库 `data/ashare_quant_hist.db`，不污染默认演示库。默认以 `--workers 4` 并发预取多天数据（按日期顺序消费、保证前复权 `pre_close` 链正确），代理限流时可降到 `1`（串行）。

> `hist` 库与默认演示库有三个数据口径差异，直接写回测/查询脚本时需注意：① `trade_date` 为 `YYYYMMDD` 紧凑格式（SQL 字符串比较勿用 `YYYY-MM-DD`，否则会静默漏掉后续年份数据）；② `amount` 单位是「千元」（默认演示库是「元」），成交额过滤阈值需按千元填（2000 万 = `20000`）；③ 次新股（创业板/科创板）前复权可能因 `adj_factor` 缺失出现单日 ±30% 以上假收益，`ashare_quant/portfolio.py` 加载时已自动过滤。

## 因子实验室与盘中买点扫描

看板「因子实验室」标签页提供一套自建因子的研究闭环：**建因子 → 定买卖条件 → 信号回测 → 逐笔记录**。核心引擎在 `ashare_quant/lab.py`。

### 因子实验室

- **自建/导入因子**：用 pandas 公式定义因子，如 `close/close.shift(5)-1`，可用 `open/high/low/close/volume/amount/pre_close` 列和 `shift/rolling/pct_change/ewm/diff/clip` 等方法；内置 24 个示例因子可一键导入（含箱体突破 `breakout_4`/`box_range_4`、量比 `vol_ratio`、均线偏离 `ma20_dev`/`ma60_dev` 等）。公式在 AST 白名单内求值，仅允许上述列与方法，不执行任意 Python 代码（不支持 `np`）。
- **策略定义**：买入/离场条件用「因子 + 运算符 + 阈值」组合（AND/OR），可选固定止损/止盈、移动止损、跌破箱体高点离场；内置 3 个因子策略模板（箱体突破·放量确认 / 超跌反弹·量比确认 / 主升浪·启动突破）可一键导入，另有 2 个短线情绪策略引擎（`engine=hot_score` 人气热度共振 / `engine=limit_pullback_score` 涨停回马枪）直接走 `hot_strategy.py` / `limit_pullback_strategy.py` 回测。
- **信号回测**：选历史时间段跑多标的逐日回测（第 T 日信号、第 T+1 日开盘成交，无未来函数），含手续费/T+1，输出收益、回撤、胜率、资金曲线与逐笔交易，并写入 SQLite。支持「总仓位比例」控制暴露、压降回撤。

### 盘中买点扫描与邮件

调度器在每个交易日 6 个时点（**09:37 / 10:00 / 10:30 / 14:30 / 14:50 / 15:30**）自动执行盘中扫描：拉取实时行情（新浪源，东方财富接口在部分网络被限），用实时价和当日累计成交量覆盖最新日线后，扫描所有已保存策略的买入候选，**仅在发现候选时发送邮件**。收盘后的盘后流程（15:30）也保留一次扫描。

报告会标注每只候选的价格数据来源与日期：盘中扫描用实时价（标注「实时」）；盘后流程（15:30 `after_close`）不刷新实时行情，直接用日线收盘价，此时报告顶部会注明「行情：日线收盘价（数据截至 YYYY-MM-DD）」，每只候选现价后也会标注对应日期。若日线未更新到最新交易日（例如当日 `update-data` 更新失败），报告会额外给出「⚠️ 行情数据未更新至最新交易日，现价可能过期」警示，避免把陈旧收盘价误当实时价。

扫描先按价格区间（`min_price` / `max_price`）和流动性（近 20 日 `min_average_amount` / `min_average_volume`）过滤，再经过质量过滤（`config/default.yaml` 的 `scan` 下可调）：仅保留当日板块上涨的候选（`min_sector_change`）、可选限定东方财富人气榜前 N（`require_hot_stock` / `hot_rank_limit`）、排除有减持/处罚/立案等负面公告的标的（`exclude_negative_notice`），从而减少出手次数、提高胜率。

命令行可手动跑一次：

```powershell
python -m ashare_quant.cli find-buys
```

看板「因子实验室 → ④ 盘中买点扫描」有「立即扫描买点」按钮可手动触发。

邮件通过 SMTP 发送（默认 QQ 邮箱），在 `.env` 中配置：

```
SMTP_HOST=smtp.qq.com
SMTP_PORT=465
SMTP_USERNAME=发件邮箱地址
SMTP_PASSWORD=QQ 邮箱 SMTP 授权码（在邮箱设置里单独生成，不是登录密码）
EMAIL_TO=收件邮箱地址
```

## 主升浪策略设计（事件驱动 × 宏观周期 × 产业链）

> 本节是《主升浪策略-事件驱动周期分析技术方案》的核心浓缩（原方案文档已并入本节并删除），供快速了解策略定位、评分逻辑与演进路线。状态标记：✅ 已实现 / 🚧 部分实现（MVP）/ ⏳ 待做。

### 核心思想与评分链（§1）

策略不把「新闻」直接等价于「上涨」，而是把
**事件 → 宏观环境 → 资产价格 → 产业链上下游 → 盈利预期 → 资金行为 → 股价周期阶段**
串成一条可计算、可回测、可验证的因果链：

```
事件评分 → 环境评分 → 产业链传导评分 → 个股基本面/预期评分 → 股价周期评分 → 主升浪概率 → 选股排名
```

### 评分公式与选股门槛（§11）

V1 评分公式（✅，`ashare_quant/main_rally.py` 的 `WEIGHTS`）：

```
主升浪评分 = 0.20×事件分 + 0.15×宏观分 + 0.20×产业链分 + 0.20×盈利分 + 0.25×技术分   （0–100）
```

选股门槛（§11.2，✅，报告中以绿/灰徽章标注）：事件分 ≥70、宏观环境分 ≥50、产业链分 ≥70、盈利预期分 ≥60、趋势分 ≥60。
（技术方案原文为 10 因子加权，V1 先落地 5 因子，后续按 §6 因子库逐步扩展。）

### 六阶段模型（§9）

每只股票每日判定阶段：0 潜伏 / 1 启动 / 2 主升 / 3 加速 / 4 高潮 / 5 退潮。模型重点不是预测最高价，而是预测「未来 N 个交易日进入主升（阶段 2/3）的概率」——✅ 由 `ashare_quant/ml_model.py` 的 GBDT 模型输出（见 §10 模型 E）。

### 20 类核心分析因素（§6）

| 状态 | 因素 | 落地情况 |
|---|---|---|
| ✅ | F16 价格趋势（MA/MACD/RSI/ATR/突破/相对强度） | `strategies/factors.py` + `technical_score` |
| 🚧 | F17 成交量与资金行为 | VWAP 偏离 / 量比 / 成交额 z（日线代理，分钟级待做） |
| 🚧 | F01~F05 事件规模/深度/意外/级别/持续 | 关键词打分（`fundamental.py`），可选 LLM 结构化抽取 |
| 🚧 | F06 宏观周期 / F07 利率 / F08 美元流动性 | `macro_regime_score`（GDP/M2/CPI 简版） |
| 🚧 | F12 产业链上下游 / F13 中心性 | `industry_chain.py` 关键词图谱（有向图/GNN 待做） |
| 🚧 | F14 盈利预期变化 | AkShare 财务摘要（营收/净利同比） |
| ⏳ | F15 估值位置 / F18 市场宽度情绪 / F19 拥挤度 / F20 历史相似度 | 部分以评分近似，未独立建模 |

### 事件关系（§7）与反向关系（§12）

关系引擎至少覆盖 12 类（因果/上下游/替代/互补/竞争/成本传导/价格传导/政策传导/资本开支/库存周期/金融传导/替代路径）；反向关系模型区分「受益」与「受损」公司（投入/产出价格弹性、毛利率敏感度）。当前 MVP 用关键词图谱近似，正式有向图/GNN 待做。

### 模型架构（§10）

| 模型 | 任务 | 状态 |
|---|---|---|
| A 事件分类 | 抽取事件类型/方向/强度/持续 | 🚧 关键词兜底 + 可选 LLM 结构化抽取 |
| B 事件相似度 | 相似事件 → 历史表现 | 🚧 `event_similarity.py`（TF-IDF） |
| C 宏观状态 | Macro Regime 概率 | 🚧 规则打分，HMM 待做 |
| D 产业链图 | 节点+边 → GNN | ⏳ |
| E 收益预测 | P(未来20日进入主升) | ✅ `ml_model.py`（GBDT 基线） |

### 事件研究（§13）与反事实（§18）

事件日 T0 前后窗口（T-60 … T+120）回溯个股/行业收益、超额收益、回撤；反事实分析（Synthetic Control / DiD / Causal Forest）用于剥离市场本身上涨、估计事件真实增量贡献。🚧 已实现相似事件 5/10/20/60 日行业收益回溯。

### 分阶段路线图（§21）与实现进度

| 阶段 | 内容 | 状态 |
|---|---|---|
| Phase 1 MVP | 日线/宏观/事件库/因子/基础事件研究 | ✅ 已完成 |
| Phase 2 产业链 | 有向图谱/GNN/公司上下游 | ⏳ 待做 |
| Phase 3 历史模板 | 50 年全球事件库 → 相似事件 | 🚧 相似检索已做，50 年库待建 |
| Phase 4 主升浪预测 | HMM/GNN/时序/因果 | 🚧 GBDT 基线已做，深度模型待做 |
| Phase 5 实时系统 | 实时新闻/行情/盘中评分 | ⏳ 待做 |

### 五层数据分离（§22）与免责声明（§23）

事实层 / 市场层 / 关系层 / 推断层 / 验证层五层分离，避免把「分析结论」当「事实」训练。本系统为量化研究与模拟盘工具，**不构成任何投资建议**；实盘前须通过成本、滑点、停牌、涨跌停、流动性、幸存者偏差与严格样本外测试。

## 主升浪盘前扫描与富文本报告

每日交易日 **09:25**（`scheduler.py` 的 `pre_market_scan`），系统按上文「主升浪策略设计」（§17.2 每日流程、§11.2 选股门槛、§20 输出样例）生成并推送盘前富文本报告。核心实现位于 `ashare_quant/main_rally.py`。

**报告内容（HTML 富文本邮件 + 纯文本兜底，企业微信发纯文本）**：

1. **报告头部总览**：候选总数、「主升浪·启动突破」命中来源、门槛达标数量、宏观环境状态（扩张/中性/收缩）、五分项评分权重。
2. **主升浪候选 Top 10（完整卡片）**：按 §20 输出样例逐股展示——主升浪评分、未来 20 交易日进入主升阶段概率（近似）、驱动事件与强度/持续性、宏观环境、产业链传导描述、盈利预期（营收/净利同比）、趋势分、当前六阶段（潜伏/启动/主升/加速）、事件拥挤度（近似）、主要催化剂与主要风险。
3. **门槛校验徽章**：每只候选标注技术方案 §11.2 选股门槛的通过情况——事件分 ≥70、宏观环境分 ≥50、产业链分 ≥70、盈利预期分 ≥60、趋势分 ≥60（绿色=达标，灰色=未达标）。
4. **高置信事件驱动（事件分 ≥ 70，Top 20）**：对应方案「Top 20 高置信事件驱动」。
5. **产业链核心（产业链分 ≥ 70，Top 10）**：对应方案「Top 10 产业链核心」。
6. **潜伏/启动阶段关注（趋势分 < 60，Top 10）**：事件与基本面达标但趋势尚未确认的标的，适合跟踪观察。
7. **风险提示（命中负面事件，Top 10）**：行业命中减持/处罚/立案等负面事件（`causal_direction=负`）的候选，对应方案「退潮风险」提示。

**评分口径**：`主升浪评分 = 0.20×事件分 + 0.15×宏观分 + 0.20×产业链分 + 0.20×盈利分 + 0.25×技术分`（`main_rally.py` 的 `WEIGHTS`，权重可按样本外回测调整）。主升概率优先由 §10 模型 E 的 GBDT 模型输出（标注「模型」），未训练或特征缺失时回退为评分近似映射（标注「近似」）。

**推送通道**：与买点扫描共用 SMTP 邮件（HTML 富文本优先、纯文本兜底）和企业微信（纯文本），`.env` 配置方式见上节；未配置时仅写日志，不影响调度。

**手动触发**：

```powershell
python -m ashare_quant.cli pre-market        # 抓事件+宏观+候选池，控制台输出报告文本
python -m ashare_quant.cli score-main-rally  # 输出候选完整评分明细 JSON
```

### 主升概率预测模型（P(未来20日进入主升)）

`ashare_quant/ml_model.py` 实现技术方案 §10 模型 E 的 GBDT 基线（Phase 4 MVP）：

- **特征**（13 个，全部日线口径）：动量、反转、波动率、流动性、RSI、MACD、均线多头、52 周新高、ATR、Amihud 非流动性、VWAP 偏离、成交额 z-score、量比。
- **标签**：未来 20 个交易日收盘涨幅 ≥ 10% 记为「进入主升」（V1 阈值，可按样本外回测调整）。
- **训练**：优先 `scikit-learn` 的 `HistGradientBoostingClassifier`；环境无 sklearn 时自动回退到内置 numpy L2 逻辑回归，保证任意环境可运行。按时间切分 80/20 验证，输出验证 AUC/准确率。
- **推理**：盘前报告的「进入主升概率」优先由模型输出（标注「模型」），未训练或特征缺失时回退为评分近似映射（标注「近似」）。报告头部展示模型后端、验证 AUC 与训练日期。
- **持久化**：模型保存至 `data/models/main_rally.pkl`；每次训练写入 `model_runs` 表（样本数、正例率、AUC、准确率）。
- **内存控制**：重训采样标的数由 `config/default.yaml` 的 `model.retrain_sample_symbols` 控制（默认 300）。2G 内存的小服务器建议调到 100~150 并配合 swap，避免 OOM。

### 收盘后模型重训（调度器 16:10）

对应技术方案 §17.2「收盘后：重训/更新」。调度器在每个交易日 16:10（`config/default.yaml` 的 `scheduler.retrain` 可调）自动重训模型；重训失败只记日志，不影响交易与风控。也可手动触发：

```powershell
python -m ashare_quant.cli retrain-model
```

> 模型质量依赖事件库与历史数据规模：当前事件库尚薄、特征以日线技术面为主，AUC 基线可能接近随机。随着事件数据积累，可按方案 §10 逐步引入事件相似度特征、盈利预期特征与 LightGBM/深度模型。

### 相似历史事件回顾（Phase 3 MVP）

`ashare_quant/event_similarity.py` 实现技术方案 §13 事件研究的 MVP 版：

- **相似度检索**：对事件库（标题+摘要）构建字符 2/3-gram TF-IDF 向量，余弦相似度检索 Top-K 相似历史事件（自动排除事件自身）。50 年全球事件库建立后，检索范围自然扩大，无需改代码。
- **历史表现回溯**：对每条相似事件，取其受影响行业（`affected_industries` 匹配 `stock_industry`）的等权组合，回测事件日后 5/10/20/60 个交易日的平均收益（窗口不足时按可得周期部分返回）。
- **报告输出**：盘前报告新增「相似历史事件回顾」板块，展示当前驱动事件 → 相似历史事件 → 相似度 → 后续各周期行业收益。

### VWAP 与资金行为字段（F17 日线代理）

分钟级数据接入前的日线代理口径：**VWAP 偏离** = 收盘价 /（近 20 日成交额 ÷ 成交量）− 1；**量比** = 当日成交量 / 近 20 日均量；**成交额 z-score** = 当日成交额在近 20 日的标准化值。三者出现在候选卡片「资金行为」行，并作为主升概率模型的输入特征。

## 回测规则

Backtrader 在每周首个交易日开盘调仓，最多持有 5 只、单日最多新开 2 只、单票目标权重 20%，下单数量取 100 股整数倍。费用为佣金万 2.5 且最低 5 元，卖出印花税千 1，默认滑点万 5。策略会拦截涨停买入、跌停卖出并记录买入日以限制当日卖出。

```powershell
python -m ashare_quant.cli backtest --strategy momentum_rotation --start 2020-01-01 --end 2026-08-15
python -m ashare_quant.cli backtest --strategy dual_moving_average
python -m ashare_quant.cli backtest --strategy simple_multifactor
```

回测结果和资金曲线写入 SQLite，并在看板“回测”页展示年化收益、最大回撤、夏普、胜率和交易次数。

## 多策略组合回测

除单策略回测外，系统提供一套基于全市场历史库（`data/ashare_quant_hist.db`）的**多策略组合回测**（`ashare_quant/portfolio.py`），用于评估「分散持有多个低相关策略」对年化/回撤的影响：

- 对横截面因子策略（动量/低波动/趋势/反转/流动性等）做宽表向量化的周度调仓回测，产出日度收益序列；再把涨停回马枪的资金曲线也纳入，用等权 / 逆波动率（≈风险平价）/ 低回撤加权合并。
- 严格无未来函数（第 T 日收盘打分、第 T+1 日才计收益）；数据加载时自动过滤复权跳变的次新股（单日 |涨跌幅| > 30% 视为异常）。
- 注意：历史库的 `amount` 单位是「千元」（Tushare 口径），与默认演示库（元）不同，脚本里成交额过滤阈值需按千元填写。

```powershell
python scripts/portfolio_backtest.py
```

输出各策略与组合的年化、最大回撤、夏普对比及策略间相关性矩阵。参考结论（2024-01~2026-08 全市场）：涨停回马枪（年化 8.8%、回撤 5.6%、夏普 1.51）与低波动（年化 14.1%、回撤 17.8%）相关性仅 0.03，二者与流动性等正收益策略按逆波动率组合后年化约 10.8%、回撤约 9.9%；而纯动量/趋势/反转等追涨因子在全市场此区间普遍 -40% 以上，说明演示库（30 只蓝筹）上的动量回测与真实全市场口径差异巨大。

脚本还内置了可选的大盘择时（`apply_market_timing`，全市场等权基准的 MA 均线，熊市降仓）。实测提醒：**对已自带「大盘闸门」的涨停回马枪而言，再叠加 MA 择时是负贡献**（年化从 8.7% 降到约 6%、回撤几乎不降），因为该策略本身已在弱势日空仓；组合整体择时也只能把回撤从 9.9% 压到约 9.6%、却牺牲约 2 个点年化。故默认不开启择时，仅在回测纯 Beta 型策略时可尝试。

## 风控

默认配置位于 `config/default.yaml`：

- 单票和单笔不超过总权益 20%；
- 最多持有 5 只，单日最多新开 2 只；
- 当日权益损失达到 3% 后停止新开仓；
- 连续 3 次执行失败后暂停交易；
- 次一交易日自动重置日内停止状态；
- 卖出检查 T+1 可卖数量，委托数量必须是 100 股整数倍。

涨跌停股票不会从候选池永久排除，但在实际成交时若处于不可成交的一字涨跌停状态，委托会被拒绝并记录风险事件。

## 通知

企业微信机器人和 SMTP 邮件都是可选通道。配置在 `.env` 中；未配置或发送失败不会中断交易与风控。系统发送策略信号、模拟成交报告、行情异常和执行异常。发送在后台线程异步执行，不阻塞信号生成与成交主流程。

## Docker 与 Linux

```bash
cp .env.example .env
docker compose up -d --build
docker compose logs -f scheduler
docker compose exec -T scheduler python -m ashare_quant.cli seed-demo
docker compose exec -T scheduler python -m ashare_quant.cli paper-demo
```

Docker 同时启动看板和调度器，共享一个 SQLite WAL 数据卷。镜像默认通过 DaoCloud 代理获取 Python Slim，并固定为 Docker Hub 官方清单摘要；其他网络环境可通过 `docker compose build --build-arg PYTHON_IMAGE=python:3.11-slim` 切回官方源，pip 依赖默认走清华镜像、可用 `--build-arg PIP_INDEX_URL=...` 覆盖。Linux 原生部署示例位于 `deploy/`：将项目放到 `/opt/ashare-quant`、创建 `quant` 用户和虚拟环境后，复制并启用两个 systemd unit。也可以使用 `deploy/crontab.example`，但不要同时运行 cron 和内置调度器。

## 云服务器部署与更新

云端采用「git 克隆 + Docker Compose」方式部署（生产运行于腾讯云轻量服务器 2核2G，后续可升级配置）。

**首次部署**（详见 `deploy/cloud_setup.sh` 与 `部署更新指南.md`）：

```bash
# 服务器上：拉代码（私有仓库先 git config --global credential.helper store 存好凭据）
git clone https://github.com/<账号>/<仓库>.git ashare-quant
cd ashare-quant
cp .env.example .env && vi .env          # 设置 DASHBOARD_PASSWORD
bash deploy/cloud_setup.sh               # 自动：装 Docker + 2G swap + 调重训采样 + 构建启动
```

**日常更新**（改功能 → 上线）：

```powershell
# 本地（开 Clash 代理后）
git add -A && git commit -m "改动说明" && git push
```

```bash
# 云端
cd ~/ashare-quant && git pull && sudo docker compose up -d --build
```

要点：数据（SQLite + 模型）存于 `quant-data` 数据卷，重建容器不丢数据（切勿 `docker compose down -v`）；`.env` 不进 git，云端本地保留；`config/default.yaml` 是版本化文件，云端若需覆盖参数用 `config/local.yaml` + `.env` 的 `QUANT_CONFIG` 指向，避免 `git pull` 冲突；看板端口 8501 需在云控制台防火墙放行。

## QMT、PTrade 与东方财富

- `ashare_quant/brokers/qmt.py`：Windows miniQMT/xtquant 接入边界。
- `ashare_quant/brokers/ptrade.py`：服务器 PTrade 接入边界。
- `ashare_quant/brokers/easytrader_adapter.py`：东方财富/easytrader 接入边界。

这些适配器不会被默认运行。即使设置 `TRADING_MODE=LIVE`，QMT/PTrade 仍要求额外设置 `ENABLE_LIVE_TRADING=true`，且当前方法会明确抛出 `NotImplementedError`。这使 V1 不可能因配置误改而真实下单。

## 测试

```powershell
python -m pytest -q
```

测试使用临时 SQLite 和演示行情，覆盖费用/整手/涨停规则、信号到模拟成交，以及 Backtrader 回测落库。

## 生产部署检查

已内置：交易日历（`ashare_quant/trading_calendar.py`，法定节假日与调休休市自动跳过，接口失败退回工作日判断）与复权一致性（`pre_close` 统一取同序列上一交易日收盘）。

服务器上线前仍需完成：AkShare 接口巡检、行情时间戳与陈旧报价拦截、SQLite 定时备份、通知送达测试、断网恢复、进程看护、QMT/PTrade 沙箱验收和小资金限额测试。内置交易日历为交易所口径，不能替代券商实盘侧的交易日状态与成交回报。
