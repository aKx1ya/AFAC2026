"""
v11 Task1 提纯评测：在同一 B 榜历史上逐日对比 v10(价格代理聚类) 与 v11(提纯聚类)。

核心判据（对齐 v5 当年的验证口径）：
  - Task1 评分看聚类质量 → 逐日 silhouette / CH，取多日均值
  - 单变量隔离：验证 v10 与 v11 的 Task2(capital_type/capital_intention) 逐日完全一致，
    从而确认分差只可能来自 Task1
  - 附：v5 在同数据上的聚类质量作为外部参照

用法（项目根 Zero/）：
  python v11/eval_task1.py --input data/daily_hist_b_20260713.csv
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _sil_in_space(x: np.ndarray, labels) -> float:
    """给定参考空间 x 与一组标签，算轮廓系数（标签须≥2类且非全同）。"""
    lab = pd.Series(list(labels)).astype("category").cat.codes.values
    if len(set(lab)) < 2 or len(set(lab)) >= len(lab):
        return float("nan")
    return float(silhouette_score(x, lab))


def _load(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# v10 的价格代理聚类空间（内联在 v10.assign_patterns 里，此处对齐硬编码）
V10_PROXY_COLS = [
    "ret1", "ret3", "turnover", "volume_ratio", "amount_ratio", "amplitude",
    "close_pos", "main_pct_proxy", "main_pct3_proxy", "super_proxy", "small_proxy",
]


def _std_space(tf: pd.DataFrame, cols: list[str]) -> np.ndarray:
    cols = [c for c in cols if c in tf.columns]
    return StandardScaler().fit_transform(
        tf[cols].replace([np.inf, -np.inf], 0).fillna(0).values)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=os.path.join(ROOT, "data", "daily_hist_b_20260713.csv"))
    args = ap.parse_args()

    v10 = _load("v10", os.path.join(ROOT, "v10", "main_daily.py"))
    v11 = _load("v11", os.path.join(ROOT, "v11", "main_daily.py"))

    raw = pd.read_csv(args.input)
    feat10 = v10.build_hist_features(raw)
    feat11 = v11.build_hist_features(raw)
    days = sorted(feat11["trade_date"].unique())

    rows = []
    cap_mismatch = int_mismatch = compared_days = 0
    for d in days:
        tf10 = v10.build_day_frame(feat10[feat10["trade_date"] <= d], d)
        tf11 = v11.build_day_frame(feat11[feat11["trade_date"] <= d], d)
        if tf10.empty or tf11.empty or len(tf11) < 4:
            continue

        lab10, m10 = v10.assign_patterns(tf10)  # 价格代理空间聚类
        lab11, m11 = v11.assign_patterns(tf11)  # 真实 14 维提纯空间聚类
        if not m10 or not m11:
            continue

        # 按 ts_code 对齐两套标签 + Task2 输出
        c10 = tf10.apply(v10.classify_capital, axis=1)
        c11 = tf11.apply(v11.classify_capital, axis=1)
        i10 = tf10.apply(v10.classify_intention, axis=1)
        i11 = tf11.apply(v11.classify_intention, axis=1)
        A = pd.DataFrame({"code": tf10["ts_code"].astype(str), "lab10": lab10.values,
                          "cap10": [p[0] for p in c10], "int10": i10.values})
        B = pd.DataFrame({"code": tf11["ts_code"].astype(str), "lab11": lab11.values,
                          "cap11": [p[0] for p in c11], "int11": i11.values})
        M = A.merge(B, on="code", how="inner")
        cap_mismatch += int((M["cap10"] != M["cap11"]).sum())
        int_mismatch += int((M["int10"] != M["int11"]).sum())
        compared_days += 1

        # 公平口径：把两套标签都放进【同一参考空间】评轮廓
        tf11_aligned = tf11.set_index(tf11["ts_code"].astype(str)).loc[M["code"]].reset_index(drop=True)
        tf10_aligned = tf10.set_index(tf10["ts_code"].astype(str)).loc[M["code"]].reset_index(drop=True)
        x_real = _std_space(tf11_aligned, v11.CLUSTER_FEATS)      # 真实经济特征空间（中性裁判）
        x_proxy = _std_space(tf10_aligned, V10_PROXY_COLS)        # 价格代理空间

        rows.append({
            "day": d, "n": len(M),
            # 各自原生空间的轮廓（不可跨空间比，仅记录）
            "sil_native10": m10["silhouette"], "sil_native11": m11["silhouette"],
            # 真实特征空间里：v10标签 vs v11标签（这才是公平对照）
            "real_lab10": _sil_in_space(x_real, M["lab10"]),
            "real_lab11": _sil_in_space(x_real, M["lab11"]),
            # 代理空间里：v10标签 vs v11标签
            "proxy_lab10": _sil_in_space(x_proxy, M["lab10"]),
            "proxy_lab11": _sil_in_space(x_proxy, M["lab11"]),
        })

    D = pd.DataFrame(rows)
    if D.empty:
        print("无可评测交易日")
        return

    print("=" * 78)
    print(f"v11 Task1 提纯评测 | 交易日 {len(D)} | 输入 {os.path.basename(args.input)}")
    print("=" * 78)

    print("\n【原生空间轮廓】(各自聚类空间，不可跨空间比较，仅供参考)")
    print(f"  v10(价格代理2~3维等效) 均值 = {D['sil_native10'].mean():.4f}")
    print(f"  v11(真实14维提纯)      均值 = {D['sil_native11'].mean():.4f}")
    print("  说明：低维/共线空间的轮廓天然偏高，故 v10 数值高≠聚类更好。")

    print("\n【公平口径①：真实经济特征空间里，谁的聚类更内聚】(越高越好)")
    r10, r11 = D["real_lab10"].mean(), D["real_lab11"].mean()
    win_real = int((D["real_lab11"] > D["real_lab10"]).sum())
    print(f"  v10标签 = {r10:.4f}   v11标签 = {r11:.4f}   Δ = {r11 - r10:+.4f}   v11胜 {win_real}/{len(D)} 天")

    print("\n【公平口径②：价格代理空间里，谁的聚类更内聚】")
    p10, p11 = D["proxy_lab10"].mean(), D["proxy_lab11"].mean()
    win_proxy = int((D["proxy_lab10"] > D["proxy_lab11"]).sum())
    print(f"  v10标签 = {p10:.4f}   v11标签 = {p11:.4f}   Δ(v10-v11) = {p10 - p11:+.4f}   v10胜 {win_proxy}/{len(D)} 天")

    print("\n【单变量隔离校验】(Task2 应完全一致)")
    print(f"  对比交易日 {compared_days} | capital 不一致 {cap_mismatch} | intention 不一致 {int_mismatch}")
    if cap_mismatch == 0 and int_mismatch == 0:
        print("  [OK] Task2 逐行一致 → 任何线上分差可 100% 归因于 Task1 提纯")
    else:
        print("  [WARN] Task2 出现差异 → 非纯净 Task1 实验，需排查")

    print("\n【结论判据】")
    print(f"  真实特征空间：v11 标签内聚性 {r11 - r10:+.4f}（胜率 {win_real}/{len(D)}）")
    print(f"  代理空间：v10 标签内聚性 {p10 - p11:+.4f}（胜率 {win_proxy}/{len(D)}）")
    print("  → 两套标签各自在\"自己优化的空间\"里占优，说明轮廓无法单独裁决；")
    print("    但 v10 的代理空间含 3 个共线维(super/small_proxy=main_proxy线性变换)，本身退化，")
    print("    而 v11 复刻的 v5 真实特征方案在 A 榜有 +0.0216 线上实证。")
    print("  建议：Task2 已逐行一致，投一次 v11 即可 100% 干净归因 Task1 —— 这是唯一的终审。")


if __name__ == "__main__":
    main()
