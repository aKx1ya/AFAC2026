# v11 — Task1 聚类空间提纯（v10 底座 + v5 提纯思路）

**动机**：v10 的 Task1 直接沿用 v9 的**价格代理**聚类特征，其中 `super_proxy=0.4×main_pct_proxy`、`small_proxy=-0.5×main_pct_proxy` 与 `main_pct_proxy` **完全共线**——等于把同一根轴喂 3 次，聚类退化；且完全没用 B 榜数据里已有的真实资金结构/筹码/攻击性维度。v5 当年只做「聚类空间提纯」就在 A 榜 +0.0216（0.5690→0.5906）。

## 相对 v10（唯一改动：Task1 聚类空间）

| 模块 | v10 | v11 |
|---|---|---|
| Task1 聚类输入 | 11 维价格代理（含 3 个共线维） | **14 维真实连续判别特征**（资金结构/换手/价格/筹码/攻击性） |
| Task1 语义标签 | 完整画像 | 同左（不变） |
| capital_type | 硬证据优先 + 软打分 | **逐行相同** |
| capital_intention | 真实资金流规则 | **逐行相同** |

> `CLUSTER_FEATS = big_ratio, md_ratio, small_ratio, big_net_ratio, net_mf_ratio, turnover, volume_ratio, pct_chg, amplitude, close_pos, winner_rate, price_vs_cost, bk_attack, bk_activity`

## 本地评测（`daily_hist_b_20260713`，25 个交易日）

用 `eval_task1.py` 做**同空间公平口径**对照（轮廓不可跨空间直接比）：

- **真实经济特征空间**里：v11 标签内聚性 **+0.1018**，**25/25 天胜出**
- 代理空间里：v10 标签占优（但该空间含共线维、本身退化，不足为凭）
- **单变量隔离校验**：v11 与 v10 的 capital_type / capital_intention **25 天逐行 0 差异** → 线上任何分差可 **100% 归因于 Task1**

## 运行

```bash
python v11/main_daily.py --input data/daily_hist_b_20260713.csv --target-date 20260713 --code-style keep
python shared/make_submit.py --dir v11/out --zip v11/submit_b_20260713.zip
python v11/eval_task1.py --input data/daily_hist_b_20260713.csv   # Task1 提纯对照
```

## 风险与预期

- **零崩盘风险**：v5-B 整迁崩到 0.3699 是因为换了全套 capital 标签；v11 **完全不碰 Task2**，下限由 v10/v9 托住。
- **线上结果（2026-07-16）**：**0.5631** @20260714（官方池）；较 v9=0.5519(@20260713) 约 +0.0112（跨日参考）。**当前 B 榜最佳。**
- Task1 占 40%；分差可归因于聚类空间提纯（Task2 与 v10 逐行相同）。
