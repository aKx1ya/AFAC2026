# v6 — capital_type 多日投票（在 v5=0.5906 基础上）

**待评测**（评测日 20260701 已跑通打包）。**只改 Task2 的 capital_type**，Task1 与意图判断都与 v5 逐行相同（已 diff 验证）。

## 动机
诊断发现 capital_type 是**相对稳定的个股属性**（跨日众数占比均值 0.76），但 v4/v5 每天独立判定，92 只软打分股在「量化↔散户↔游资」间抖动——相邻交易日预测 churn 高达 **29.5%**。赛题按 T+8 实盘回溯真值评分，真值短期稳定，单日噪声纯属损失。

## 方法：证据加权 + 时间衰减投票
- **当日硬证据（strong/mid）股**：直接采用当日判定，**不投票**（当日铁证最权威，且受保护不被翻转）
- **当日软打分（soft）股**：对全部历史窗口按 `证据权重 × decay^Δ天` 加权投票
  - 证据权重：strong 3.0 / mid 2.0 / soft 1.0
  - `decay=0.8`（越近的交易日权重越高）

## 伪真值验证（以 51 只有硬证据股的历史众数为近似真值）
| 指标 | v5（无投票） | v6（投票） |
|---|---|---|
| 软打分股准确率 | 31.8% | **39.2%（+7.4pp）** |
| 全体相邻日 churn | 29.5% | **15.8%** |

## 本地结果（20260701）
- 投票翻转 20 只软打分股；硬证据 9 只全部保持当日判定
- 资金类型：散户 47 / 量化 33 / 游资 20（v5 为 47/32/21）
- 意图、Task1 与 v5 完全一致

## 已验证
- ✅ **无未来函数**：投票只用 ≤target 的历史，0630 全量 vs 截断复算一致
- ✅ **硬证据保护**：9 只硬证据股投票后标签不变
- ✅ Task1（`pattern_reco.csv`）与 v5 逐行相同
- ✅ capital_intention 与 v5 完全一致（只动 capital_type）
- ✅ 格式校验通过，100 行，`transaction_date=20260701`

## 运行
```bash
.venv/bin/python shared/fetch_daily.py --start 20260608 --end <评测日> --out data/daily_hist.csv
.venv/bin/python v6/main_daily.py --target-date <评测日>
.venv/bin/python shared/make_submit.py --dir v6/out --zip v6/submit.zip
```

> 提交后请在 [CHANGELOG.md](../CHANGELOG.md) 回填评测分。若不及 v5，则 v5 仍为基线。
