# v10 — v9 软打分底座 + Tushare 硬证据/真实资金流

**动机**：B 榜 v9=0.5519，整迁 v5=0.3699。v10 只在硬证据上覆盖 `capital_type`；意图改用真实 moneyflow；Task1 保持 v9 价格特征空间。

## 相对 v9
| 模块 | 做法 |
|---|---|
| 数据 | Tushare `daily_hist` |
| capital_type | 硬证据优先（龙虎榜/机构/连板），其余仍用价格代理软打分 |
| capital_intention | 真实净额 + 封板/炸板/三源一致 |
| Task1 | 与 v9 同构（价格代理特征），避免聚类空间大漂移 |

## 本地评测摘要（`daily_hist_b_20260713`，FWD=5）
- 意图 weighted-F1：v9-soft 0.330 → **v10 0.353**（+0.023，与 v5 意图规则同分）
- 硬证据覆盖约 3.3%；硬证据股上 v9 软标签冲突约 47%
- capital 相对同数据 v9-soft 日均仅改约 1.6 只

## 运行
```bash
python v10/main_daily.py --input data/daily_hist_b_20260713.csv --target-date 20260713 --code-style keep
python shared/make_submit.py --dir v10/out --zip v10/submit_b_20260713.zip
python v10/local_eval.py --input data/daily_hist_b_20260713.csv
```
