# AFAC2026 赛题一 — 版本迭代与分数记录

> 总分 = Task1（交易模式识别）× 0.4 + Task2（参与者+意图）× 0.6  
> 记录每次**实际上传评测**的结果，便于对比迭代效果。  
> 历史版本冻结在对应文件夹；**当前最佳：v2（0.5433）**。

---

## 分数总览

| 版本 | 评测日 | 上传日期 | A榜总分 | 备注 |
|:---:|:---:|:---:|:---:|:---|
| v1 | 20260701 | 2026-07-03 | **0.41** | 首次真实数据提交，格式验证通过 |
| v2 | 20260701 | 2026-07-03 | **0.5433** | 强信号+截面相对+多日画像，较 v1 +0.1333 |
| v3 | 20260701 | 2026-07-03 | **0.5311** | 弱监督分类器，较 v2 **-0.0122** |

---

## 版本详情

### v1 — 日频 baseline（冻结）

- **目录**：`v1/`
- **评测分**：**0.41**（A榜，transaction_date=20260701）
- **数据**：Tushare 日频（moneyflow / daily / 龙虎榜 / 游资明细），单日特征
- **Task1**：KMeans 聚类 + 规则匹配语义标签 → 标签易集中到少数模式（如「量化活跃换手」占多数）
- **Task2**：11 维多因子全局 MinMax 打分 → **过度判散户**（约 85% 散户）
- **已知问题**：小单天然占比高导致散户误判；未利用多日行为持续性

**复现**：
```bash
python shared/fetch_daily.py --dates 20260701 --out data/daily_data.csv
python v1/main_daily.py --input data/daily_data.csv --output v1/out --target-date 20260701
python shared/make_submit.py --dir v1/out --zip v1/submit.zip
```

---

### v2 — 强信号 + 截面相对 + 多日画像（冻结）

- **目录**：`v2/`
- **评测分**：**0.5433**（A榜，20260701，较 v1 +0.1333）
- **相对 v1 的改动**：
  1. **Task2 三层决策**：当日强信号（游资明细/龙虎榜/机构席位）→ 截面相对排名 → 17 日个股画像
  2. **Task1**：greedy 唯一分配 8 类互不重复语义标签
  3. **输入**：需多日历史 `data/daily_hist.csv`（A榜窗口 20260608–20260701）
- **本地预期分布**（20260701）：散户 43 / 量化 31 / 游资 26（对比 v1 的 85/11/4）

**运行**：
```bash
python shared/fetch_daily.py --start 20260608 --end 20260701 --out data/daily_hist.csv
python v2/main_daily.py --target-date 20260701
python shared/make_submit.py --dir v2/out --zip v2/submit.zip
```

---

### v3 — 弱监督分类器（已评测，未超越 v2）

- **目录**：`v3/`
- **评测分**：**0.5311**（A榜，20260701，较 v2 **-0.0122**）
- **结论**：弱标签噪声 + 高置信样本仅 5.7%，模型泛化未优于 v2 规则；**暂以 v2 为提交基线**
- **相对 v2 的改动**：
  1. **Task2 弱监督**：历史龙虎榜/游资明细/机构席位 → 加权弱标签（高置信 5.7%）
  2. **HistGradientBoosting** 分别训练 `capital_type` 与 `capital_intention`
  3. 训练集 = 目标日之前全部交易日（1595 条 / 16 天），画像无未来信息
  4. 推理：模型预测 + 当日极强信号规则覆盖
  5. Task1 沿用 v2
- **本地分布**（20260701）：散户 45 / 量化 33 / 游资 22；意图 T0 45 / 买入 37 / 卖出 18

**运行**：
```bash
python shared/fetch_daily.py --start 20260608 --end 20260701 --out data/daily_hist.csv
python v3/main_daily.py --target-date 20260701
python shared/make_submit.py --dir v3/out --zip v3/submit.zip
```

---

## 后续版本规划（模板）

### v4 — （规划中）

---

## 提交注意事项

1. **transaction_date** 必须与平台当前评测日一致（T+1 滞后，见 `QA.md`）
2. **stock_code** 为 6 位纯数字（无 `.SH` 后缀）
3. zip 结构：`submit/pattern_reco.csv` + `submit/predict_result.csv`
4. A榜每天 ≤3 次提交

---

## 变更日志（简表）

| 日期 | 版本 | 事件 |
|:---:|:---:|:---|
| 2026-07-03 | v1 | 首次提交，得分 0.41 |
| 2026-07-03 | v2 | 提交评测，得分 **0.5433**（+0.1333 vs v1） |
| 2026-07-03 | v3 | 提交评测，得分 **0.5311**（-0.0122 vs v2） |
