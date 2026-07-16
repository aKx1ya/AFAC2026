# 本目录存放拉取的真实日频数据（生成物，勿纳入 git）

## 数据文件（gitignore）

- `daily_data.csv` — 单日
- `daily_hist.csv` — A 榜多日历史
- `daily_hist_b_*.csv` — B 榜按股票池拉取的历史
- `daily_hist_b_official_*.csv` — 使用**官方股票池**拉取的 B 榜历史（推荐）

## 股票池（勿放本目录）

B 榜股票池**只认**仓库根下：

```
官方数据/B榜/股票样本_YYYYMMDD.xlsx
官方数据/股票样本.xlsx   ← 当前指针，通常与最新按日池一致
```

**禁止**把自行下载/另存的 xlsx 丢进 `data/`（曾误用 `b20260715.xlsx`，与官方差 60 只，两次拒收）。
