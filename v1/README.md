# v1 — 日频 baseline（已冻结）

**评测分：0.41**（A榜，20260701）详见 [CHANGELOG.md](../CHANGELOG.md)

## 特点
- 单日特征，不依赖历史窗口
- Task2 易过度判散户（~85%）

## 运行
```bash
# 在项目根目录执行
python shared/fetch_daily.py --dates 20260701 --out data/daily_data.csv
python v1/main_daily.py --input data/daily_data.csv --output v1/out --target-date 20260701
python shared/make_submit.py --dir v1/out --zip v1/submit.zip
```

## 文件
| 文件 | 说明 |
|---|---|
| `main_daily.py` | v1 推理入口（勿改） |
| `out/` | 输出 CSV |
| `submit.zip` | 提交包（生成后出现） |
