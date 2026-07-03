# v3 — 弱监督分类器版（已评测）

**评测分：0.5311**（A榜，20260701）— 较 v2（0.5433）**-0.0122**，未超越 v2。

相对 v2：用历史龙虎榜/游资明细构造弱标签，训练梯度提升分类器替代纯规则 Task2。

## 运行
```bash
# 项目根目录
python shared/fetch_daily.py --start 20260608 --end 20260701 --out data/daily_hist.csv
python v3/main_daily.py --target-date 20260701
python shared/make_submit.py --dir v3/out --zip v3/submit.zip
```

## 方法要点
- **弱标签**：游资明细含「量化」→ 量化；机构席位上榜 → 量化；游资明细/龙虎榜 → 游资；低活跃画像 → 散户；其余用 v2 规则补全（权重较低）
- **模型**：`HistGradientBoostingClassifier`（资金类型 + 意图各一个）
- **训练集**：目标日之前所有交易日（约 16×100 条），无未来信息
- **推理**：模型预测 + 当日极强信号规则覆盖
- **Task1**：沿用 v2 的 8 类 greedy 语义聚类

详见 [CHANGELOG.md](../CHANGELOG.md)
