# AFAC2026 赛题一：市场参与者交易行为识别与资金流向分析

> 出题方：蚂蚁集团 ｜ 数据：Tushare 日频微观结构 + B榜公开行情兜底 ｜ **版本记录见 [CHANGELOG.md](CHANGELOG.md)**

---

## 目录结构

```
Zero/
├── CHANGELOG.md          ← 【必读】升级与 A/B 榜分数
├── README.md             ← 本文件
├── QA.md                 ← 官方答疑
├── requirements.txt
├── init_env.sh
├── .env.example          ← 复制为 .env 填入 TUSHARE_TOKEN（勿提交 .env）
│
├── shared/               ← 拉数 + 提交打包
│   ├── fetch_daily.py
│   ├── make_submit.py
│   └── paths.py
│
├── data/                 ← 生成物（gitignore）
├── 官方数据/
│   ├── 股票样本.xlsx      ← 当前指向的样本（可覆盖）
│   ├── B榜/              ← 按日股票池
│   └── submit/           ← 官方提交样例
│
├── baseline_original/    ← 官方 L2 baseline（存档）
├── v1/ … v8/             ← A榜迭代（v5 A榜最高 0.5906）
├── v9/                   ← B榜旧基线 0.5519（公开行情）
├── v10/                  ← B榜 Task2 硬证据底座（未单独上线）
└── v11/                  ← **B榜当前基线 0.5631**（Task1 提纯）
```

---

## 当前推荐

| 场景 | 推荐 | 分数 |
|---|---|---:|
| **B榜日常提交** | **v11** | **0.5631** @20260714 |
| B榜上一基线 | v9 | 0.5519 @20260713 |
| A榜历史最佳 | v5 | **0.5906** @20260701（不可直接交 B榜） |

> 整迁 v5 到 B榜 = **0.3699**，禁止再整包迁移。

---

## B榜快速开始（v11，当前基线）

```bash
pip install -r requirements.txt
# 复制 .env.example → .env，填入 TUSHARE_TOKEN

# 1) 用【官方】当日股票池拉数（勿用 data/ 下自行另存的 xlsx）
python shared/fetch_daily.py --start 20260608 --end 20260714 \
  --stock-file 官方数据/B榜/股票样本_20260715.xlsx \
  --out data/daily_hist_b_official_20260714.csv

# 2) 推理（transaction_date 以平台为准，常为评测滞后日）
python v11/main_daily.py \
  --input data/daily_hist_b_official_20260714.csv \
  --target-date 20260714 --code-style keep

# 3) 打包提交
python shared/make_submit.py --dir v11/out --zip v11/submit.zip
```

## B榜 + Tushare 底座（v10，Task2 硬证据；未单独上线）

```bash
python v10/main_daily.py --input data/daily_hist_b_official_20260714.csv \
  --target-date 20260714 --code-style keep
python v10/local_eval.py --input data/daily_hist_b_official_20260714.csv
```

旧公开行情应急版见 `v9/`（0.5519 @20260713）。

---

## 赛题一句话

给定官方 **100 只股票**，每个交易日产出：
- **Task1** `pattern_reco.csv`：交易模式 + 解释（权重 40%）
- **Task2** `predict_result.csv`：游资/量化/散户 + 买入/卖出/T0（权重 60%）

---

## 版本与分数

### A榜

| 版本 | 状态 | 分数 | 说明 |
|:---:|:---:|:---:|:---|
| v1 | 冻结 | 0.41 | 单日规则+聚类 |
| v2 | 冻结 | **0.5433** | 强信号+多日画像 |
| v3 | 冻结 | 0.5311 | 弱监督（退步） |
| v4 | 冻结 | **0.5690** | 多源微观结构 |
| v5 | A榜最佳 | **0.5906** | Task1 提纯 @0701 |
| v6 | 冻结 | 0.5708 | 多日投票（退步） |
| v7/v8 | 冻结 | 0.5201* | *@0702，微改动线上不可见 |

### B榜

| 版本 | 状态 | 分数 | 说明 |
|:---:|:---:|:---:|:---|
| v11 | **当前基线** | **0.5631** | Task1 提纯 @20260714；官方池 |
| v9 | 冻结 | 0.5519 | 公开行情日更 @20260713 |
| v5-B | 失败 | 0.3699 | 整迁 v5 |
| v10 | 未单独上线 | — | 硬证据底座，已并入 v11 的 Task2 |

---

## 提交格式

| 项 | A榜 | B榜 |
|---|---|---|
| `stock_code` | 6 位纯数字 | **带后缀**如 `600353.SH` |
| `transaction_date` | 平台评测日 | 平台评测日（常滞后） |
| `capital_type` | 游资/量化/散户 | 同左 |
| `capital_intention` | 买入/卖出/T0交易 | 同左 |
| zip | `submit/*.csv` | 同左 |

详细规则见 `QA.md`；分数变更请写入 `CHANGELOG.md`。
