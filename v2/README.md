# v2 — 强信号 + 截面相对 + 多日画像（已冻结）

**评测分：0.5433**（A榜，20260701）详见 [CHANGELOG.md](../CHANGELOG.md)

## 相对 v1 的升级
1. Task2：强信号决定性 + 当日截面相对排名 + 多日个股画像
2. Task1：8 类互不重复语义标签（greedy 唯一分配）
3. 需要多日历史数据 `data/daily_hist.csv`

## 运行
```bash
# 在项目根目录执行
python shared/fetch_daily.py --start 20260608 --end 20260701 --out data/daily_hist.csv
python v2/main_daily.py --target-date 20260701
python shared/make_submit.py --dir v2/out --zip v2/submit.zip
```

> `transaction_date` 填平台当前评测日（不是今天，是平台要求的那个交易日）。

## 文件
| 文件 | 说明 |
|---|---|
| `main_daily.py` | v2 推理入口 |
| `out/` | 输出 CSV |
| `submit.zip` | 提交包（生成后出现） |
