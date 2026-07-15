# v9 — B榜日更应急版（公开行情）

**B榜线上基线：0.5519**（评测日 `20260713`，2026-07-15 提交）

## 定位
A榜 v5 流水线依赖 Tushare + 固定股票池；B榜每日换池且当时无 token，故用新浪日线做可提交应急方案。

## 规则
- `stock_code` 须带交易所后缀（如 `600353.SH`）
- `transaction_date` 以平台报错提示为准（当日自然日 ≠ 评测日）
- 股票池用 `官方数据/B榜/股票样本_YYYYMMDD.xlsx`

## 实验记录（同日 `20260713`）
| 版本 | 改动 | 分数 |
|---|---|---:|
| v9 基线 | 公开行情软打分 | **0.5519** |
| mix | 仅调意图分布 | 0.5519（无变化） |
| v5-B | 整迁 v5 多源方案 | **0.3699**（崩） |

## 运行
```bash
python v9/main_daily.py --stock-file 官方数据/B榜/股票样本_20260714.xlsx --target-date 20260713 --feature-date 20260713 --output v9/out
python shared/make_submit.py --dir v9/out --zip v9/submit.zip
```

后续优化见 **v10**（在 v9 风格上叠加 Tushare 硬证据）。
