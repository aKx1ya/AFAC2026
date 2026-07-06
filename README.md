# AFAC2026 赛题一：市场参与者交易行为识别与资金流向分析

> 出题方：蚂蚁集团 ｜ 数据：Tushare 日频微观结构 ｜ **版本记录见 [CHANGELOG.md](CHANGELOG.md)**

---

## 目录结构

```
Zero/
├── CHANGELOG.md          ← 【必读】每次升级与 A榜分数记录
├── README.md             ← 本文件（项目总览）
├── QA.md                 ← 官方答疑
├── requirements.txt
├── init_env.sh
│
├── shared/               ← 各版本共用的数据拉取与提交打包
│   ├── fetch_daily.py    ← 从 Tushare 拉日频数据 → data/
│   ├── make_submit.py    ← 校验格式 + 打包 submit.zip
│   └── paths.py
│
├── data/                 ← 共享数据（生成物，勿提交 git）
│   ├── daily_data.csv    ← 单日数据（v1 用）
│   └── daily_hist.csv    ← 多日历史（v2/v3 用）
│
├── 官方数据/              ← 官方材料（股票样本 + 提交样例）
│   ├── 股票样本.xlsx
│   └── submit/
│
├── baseline_original/    ← 官方原始 baseline（日内 L2 方案，仅存档）
│
├── v1/                   ← 【冻结】首次提交版，得分 0.41
│   ├── README.md
│   ├── main_daily.py
│   └── out/
│
├── v2/                   ← 【冻结】得分 0.5433
│   ├── README.md
│   ├── main_daily.py
│   └── out/
│
└── v3/                   ← 【已评测 0.5311】弱监督分类器（未超 v2）
    ├── README.md
    ├── main_daily.py
    └── out/
```

---

## 快速开始（当前最佳 v5）

```bash
pip install -r requirements.txt   # 或用项目内 .venv

# 设置 Tushare token（勿提交到 Git，见 .env.example）
# 已支持项目根 .env 自动读取；或 PowerShell: $env:TUSHARE_TOKEN="你的token"

python shared/fetch_daily.py --start 20260608 --end 20260701 --out data/daily_hist.csv
python v5/main_daily.py --target-date 20260701
python shared/make_submit.py --dir v5/out --zip v5/submit.zip
```

上传 `v5/submit.zip` 到天池平台。

---

## 赛题一句话

给定官方 **100 只股票**，每个交易日产出：
- **Task1** `pattern_reco.csv`：交易模式 + 解释（权重 40%）
- **Task2** `predict_result.csv`：游资/量化/散户 + 买入/卖出/T0（权重 60%）

---

## 为什么用日频数据？

官方 baseline 面向日内逐笔/L2，但 L2 无法免费获取；Tushare 分钟线被限频 1次/小时。QA 允许使用非 L2 数据。我们用 `moneyflow`（大/中/小/特大单）+ 龙虎榜 + 游资明细等日频微观结构数据。

---

## 版本与分数

| 版本 | 状态 | A榜分 | 说明 |
|:---:|:---:|:---:|:---|
| v1 | 冻结 | **0.41** | 单日规则+聚类 baseline |
| v2 | 冻结 | **0.5433** | 强信号+截面相对+多日画像 |
| v3 | 冻结 | **0.5311** | 弱监督分类器（未超 v2） |
| v4 | 冻结 | **0.5690** | 多源微观结构·证据分层+跨源确认 |
| v5 | **当前最佳** | **0.5906** | 仅改 Task1 聚类空间提纯（@0701；同代码@0702=0.5201） |
| v6 | 冻结 | **0.5708** | capital_type 多日投票（退步，离线指标误导） |
| v7 | 已评测 | 0.5201* | Task1 解释逐股定制化（*@0702，三方对照证实解释文本无权重） |
| v8 | 已评测 | 0.5201* | 意图阈值重标定（*@0702，三方对照证实微改动不可见） |

> ⚠️ **v5/v7/v8 在 0702 同为 0.5201（三方受控对照）**：证实①可解释性文本无可见权重；②个位数意图翻转线上贡献为0。同代码 v5 在 0701/0702 差 0.0705，行情效应 ≫ 任何单次微改动。

每次上传评测后，请在 [CHANGELOG.md](CHANGELOG.md) 中更新分数。

---

## 提交格式（已验证）

| 项 | 要求 |
|---|---|
| `capital_type` | 游资 / 量化 / 散户 |
| `capital_intention` | 买入 / 卖出 / T0交易 |
| `transaction_date` | YYYYMMDD（与平台评测日一致） |
| `stock_code` | 6 位纯数字 |
| zip 结构 | `submit/pattern_reco.csv` + `submit/predict_result.csv` |

---

## 每日提交提醒

- 平台 T+1 滞后：提交**平台要求的评测日**，不是今天
- A榜每天 ≤3 次；每晚约 18 点更新前一日答案
- ⚠️ **评测日每天滚动，跨评测日的分数不可直接比较**（不同交易日真值分布不同）。要验证一次改动的效果，须"同评测日 + 只改一处"两个条件同时满足——否则分差会被市场行情差异污染（见 v7 教训）。
- 详细规则见 `QA.md`
