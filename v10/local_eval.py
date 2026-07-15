"""
v10 本地评测：在同一 Tushare B 榜历史上对比 v9-soft / v10 / v5。

代理真值（仅用于离线，不进入推理）：
  - 意图：未来 FWD_N 日累计涨跌方向（与 _intent_harness 同口径）
  - 资金类型：无官方真值；报告硬证据覆盖率、与 v5 硬证据一致率、相对 v9 改动量
  - Task1：轮廓系数 / CH 指数

用法（项目根 Zero/）：
  python v10/local_eval.py --input data/daily_hist_b_20260713.csv
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

warnings.filterwarnings("ignore")

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FWD_N = 5
THR = 3.0


def _load(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def intention_truth(fwd_ret: float):
    if pd.isna(fwd_ret):
        return None
    if fwd_ret > THR:
        return "买入"
    if fwd_ret < -THR:
        return "卖出"
    return "T0交易"


def regime_of(day_df: pd.DataFrame) -> str:
    mu = pd.to_numeric(day_df["pct_chg"], errors="coerce").mean()
    if mu > 0.8:
        return "普涨"
    if mu < -0.8:
        return "普跌"
    return "震荡"


def eval_intent(y_true, y_pred) -> dict:
    labels = ["买入", "卖出", "T0交易"]
    return {
        "f1": float(f1_score(y_true, y_pred, average="weighted", labels=labels, zero_division=0)),
        "acc": float((np.array(y_true) == np.array(y_pred)).mean()),
    }


def soft_only_capital(row):
    """复现 v9 软打分（无硬证据），用于对照。"""
    main = row.get("main_pct_proxy", row["main_pct"])
    super_rk = row.get("rk_super_proxy", row.get("rk_super_pct", 0))
    main_rk = row.get("rk_main_proxy", row.get("rk_main_pct", 0))
    small_rk = row.get("rk_small_proxy", row.get("rk_small_pct", 0))
    scores = {
        "游资": (
            0.30 * row["rk_turnover"]
            + 0.25 * row["rk_amplitude"]
            + 0.20 * max(row.get("rk_ret1", 0), 0)
            + 0.25 * max(super_rk, 0)
        ),
        "量化": (
            0.30 * row["rk_volume_ratio"]
            + 0.25 * row["rk_turnover"]
            + 0.25 * float(np.clip(1 - abs(main) / 30, 0, 1))
            + 0.20 * float(np.clip(1 - abs(row["ret1"]) / 10, 0, 1))
        ),
        "散户": (
            0.35 * (1 - row["rk_turnover"])
            + 0.25 * (1 - row["rk_volume_ratio"])
            + 0.20 * (1 - max(main_rk, 0))
            + 0.20 * max(small_rk, 0)
        ),
    }
    return max(scores.items(), key=lambda x: x[1])[0]


def soft_intention_v9(row):
    main = row.get("main_pct_proxy", row["main_pct"])
    main3 = row.get("main_pct3_proxy", row.get("main_pct3", main))
    direction = 0.55 * main + 0.30 * main3 + 0.15 * row["ret3"]
    if direction >= 2.2 and row["close_pos"] >= 0.35:
        return "买入"
    if direction <= -2.2 and row["close_pos"] <= 0.65:
        return "卖出"
    if main >= 4 and row["ret1"] > -1:
        return "买入"
    if main <= -4 and row["ret1"] < 1:
        return "卖出"
    return "T0交易"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=os.path.join(ROOT, "data", "daily_hist_b_20260713.csv"))
    ap.add_argument("--fwd-n", type=int, default=FWD_N)
    args = ap.parse_args()

    v10 = _load("v10", os.path.join(ROOT, "v10", "main_daily.py"))
    v5 = _load("v5", os.path.join(ROOT, "v5", "main_daily.py"))

    raw = pd.read_csv(args.input)
    raw["dd"] = raw["trade_date"].astype(str).str.replace(r"\.0$", "", regex=True)
    raw["code"] = raw["ts_code"].astype(str)
    raw = raw.sort_values(["code", "dd"])
    raw["pct_chg"] = pd.to_numeric(raw["pct_chg"], errors="coerce")
    raw["fwd_ret"] = raw.groupby("code")["pct_chg"].transform(
        lambda s: sum(s.shift(-k) for k in range(1, args.fwd_n + 1))
    )
    raw["itruth"] = raw["fwd_ret"].map(intention_truth)

    feat_hist = v10.build_hist_features(raw)
    days = sorted(feat_hist["trade_date"].unique())
    valid_days = days[: -args.fwd_n] if len(days) > args.fwd_n else days

    # v5 特征全历史一次
    v5_feat = v5.build_base_features(raw)

    rows = []
    day_metrics = []
    for d in valid_days:
        tf = v10.build_day_frame(feat_hist[feat_hist["trade_date"] <= d], d)
        if tf.empty:
            continue
        truth = raw[raw["dd"] == d][["code", "itruth"]].rename(columns={"code": "ts_code"})
        tf = tf.merge(truth, on="ts_code", how="left")
        tf = tf.dropna(subset=["itruth"])
        if tf.empty:
            continue
        rg = regime_of(raw[raw["dd"] == d])

        # v9-soft on same frame
        pred_v9_cap = tf.apply(soft_only_capital, axis=1)
        pred_v9_int = tf.apply(soft_intention_v9, axis=1)

        # v10
        pairs = tf.apply(v10.classify_capital, axis=1)
        pred_v10_cap = pd.Series([p[0] for p in pairs], index=tf.index)
        pred_v10_ev = pd.Series([p[1] for p in pairs], index=tf.index)
        pred_v10_int = tf.apply(v10.classify_intention, axis=1)
        _, pat_metrics = v10.assign_patterns(tf)

        # v5
        v5_tf = v5.build_day_frame(v5_feat[v5_feat["transaction_date"] <= d], d)
        if not v5_tf.empty:
            v5_tf = v5_tf.copy()
            v5_pairs = v5_tf.apply(v5.classify_capital, axis=1)
            v5_tf["capital_type"] = [p[0] for p in v5_pairs]
            v5_tf["_ev"] = [p[1] for p in v5_pairs]
            v5_tf["capital_intention"] = v5_tf.apply(v5.classify_intention, axis=1)
            v5_map = v5_tf.set_index("stock_code")
            # align codes: v5 stock_code may be bare or ts_code style
            codes = tf["ts_code"].astype(str)
            pred_v5_cap, pred_v5_int, pred_v5_ev = [], [], []
            for c in codes:
                key = c if c in v5_map.index else c.split(".")[0]
                if key in v5_map.index:
                    pred_v5_cap.append(v5_map.loc[key, "capital_type"])
                    pred_v5_int.append(v5_map.loc[key, "capital_intention"])
                    pred_v5_ev.append(v5_map.loc[key, "_ev"])
                else:
                    pred_v5_cap.append(None)
                    pred_v5_int.append(None)
                    pred_v5_ev.append(None)
            pred_v5_cap = pd.Series(pred_v5_cap, index=tf.index)
            pred_v5_int = pd.Series(pred_v5_int, index=tf.index)
            pred_v5_ev = pd.Series(pred_v5_ev, index=tf.index)
        else:
            pred_v5_cap = pred_v5_int = pred_v5_ev = pd.Series([None] * len(tf), index=tf.index)

        for i, row in tf.iterrows():
            rows.append(
                {
                    "day": d,
                    "reg": rg,
                    "ts_code": row["ts_code"],
                    "itruth": row["itruth"],
                    "v9_cap": pred_v9_cap.loc[i],
                    "v9_int": pred_v9_int.loc[i],
                    "v10_cap": pred_v10_cap.loc[i],
                    "v10_int": pred_v10_int.loc[i],
                    "v10_ev": pred_v10_ev.loc[i],
                    "v5_cap": pred_v5_cap.loc[i],
                    "v5_int": pred_v5_int.loc[i],
                    "v5_ev": pred_v5_ev.loc[i],
                }
            )
        day_metrics.append(
            {
                "day": d,
                "reg": rg,
                "n": len(tf),
                "hard_v10": int((pred_v10_ev != "soft").sum()),
                "sil": pat_metrics.get("silhouette"),
                "ch": pat_metrics.get("ch"),
                "cap_diff_v9_v10": int((pred_v9_cap != pred_v10_cap).sum()),
                "int_diff_v9_v10": int((pred_v9_int != pred_v10_int).sum()),
            }
        )

    A = pd.DataFrame(rows)
    D = pd.DataFrame(day_metrics)
    print("=" * 70)
    print(f"v10 本地评测 | 样本 {len(A)} | 交易日 {A['day'].nunique()} | FWD_N={args.fwd_n} THR={THR}")
    print(f"行情分布: {A['reg'].value_counts().to_dict()}")
    print(f"意图真值: {A['itruth'].value_counts().to_dict()}")
    print("=" * 70)

    print("\n【1】意图 weighted-F1 / Acc（相对未来收益代理真值）")
    print(f"{'模型':<8}{'全部F1':>10}{'全部Acc':>10}{'普涨F1':>10}{'震荡F1':>10}{'普跌F1':>10}")
    for name, col in [("v9-soft", "v9_int"), ("v10", "v10_int"), ("v5", "v5_int")]:
        sub = A.dropna(subset=[col])
        all_m = eval_intent(sub["itruth"], sub[col])
        parts = []
        for rg in ["普涨", "震荡", "普跌"]:
            s = sub[sub["reg"] == rg]
            parts.append(eval_intent(s["itruth"], s[col])["f1"] if len(s) else float("nan"))
        print(
            f"{name:<8}{all_m['f1']:>10.4f}{all_m['acc']:>9.1%}"
            f"{parts[0]:>10.4f}{parts[1]:>10.4f}{parts[2]:>10.4f}"
        )
    t0_f1 = f1_score(
        A["itruth"], ["T0交易"] * len(A), average="weighted", labels=["买入", "卖出", "T0交易"], zero_division=0
    )
    print(f"(对照) 全判T0 F1={t0_f1:.4f}")

    print("\n【2】资金类型：硬证据覆盖与改动幅度")
    hard = A[A["v10_ev"] != "soft"]
    print(f"v10 硬/中证据覆盖: {len(hard)}/{len(A)} = {len(hard)/len(A):.1%}")
    if len(hard):
        agree_v5 = (hard["v10_cap"] == hard["v5_cap"]).mean()
        print(f"硬证据日上 v10 与 v5 capital 一致率: {agree_v5:.1%} (期望高，因同源硬规则)")
    print(f"相对 v9-soft，capital 日均改动: {D['cap_diff_v9_v10'].mean():.1f}/100")
    print(f"相对 v9-soft，intent  日均改动: {D['int_diff_v9_v10'].mean():.1f}/100")
    print(f"v10 日均硬证据只数: {D['hard_v10'].mean():.1f}")

    # 硬证据子集上，v9 软打分 vs 硬标签
    if len(hard):
        wrong_v9 = (hard["v9_cap"] != hard["v10_cap"]).mean()
        print(f"硬证据股票上 v9-soft 与硬标签冲突率: {wrong_v9:.1%}  ← 越高说明硬覆盖越有必要")

    print("\n【3】Task1 聚类质量（v10，逐日均值）")
    print(f"轮廓系数均值: {D['sil'].dropna().mean():.4f}")
    print(f"CH 指数均值:   {D['ch'].dropna().mean():.2f}")

    print("\n【4】末交易日快照（最近可评日）")
    last = D.iloc[-1]
    print(
        f"{last['day']} | {last['reg']} | hard={last['hard_v10']} "
        f"| Δcap={last['cap_diff_v9_v10']} Δint={last['int_diff_v9_v10']} "
        f"| sil={last['sil']}"
    )
    # 与线上基线日对照：若数据含 20260713
    if "20260713" in feat_hist["trade_date"].values:
        snap = v10.run(args.input, os.path.join(ROOT, "v10", "out_eval_snap"), "20260713", "keep")
        print("\n20260713 正式跑批分布（提交口径）:")
        print(snap["result"]["capital_type"].value_counts().to_dict())
        print(snap["result"]["capital_intention"].value_counts().to_dict())
        print(f"证据: {pd.Series(snap['evidence']).value_counts().to_dict()}")

    print("\n【结论判据】")
    v9m = eval_intent(A["itruth"], A["v9_int"])
    v10m = eval_intent(A["itruth"], A["v10_int"])
    v5m = eval_intent(A.dropna(subset=["v5_int"])["itruth"], A.dropna(subset=["v5_int"])["v5_int"])
    print(f"意图: v10 相对 v9-soft ΔF1={v10m['f1']-v9m['f1']:+.4f}；相对 v5 ΔF1={v10m['f1']-v5m['f1']:+.4f}")
    if len(hard):
        print(f"硬覆盖纠正 v9 冲突 {wrong_v9:.1%} 的样本 — 这是 v10 相对 v9 的核心增益")
    print("注意：代理真值≠官方 T+8 标签；B榜线上仍以提交验证为准。")


if __name__ == "__main__":
    main()
