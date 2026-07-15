"""
AFAC2026 赛题一 —— v10：v9 软打分底座 + Tushare 硬证据/真实资金流覆盖

设计原则（吸取 B 榜教训）：
  - 线上已验证基线是 v9=0.5519；整套迁移 v5 崩到 0.3699
  - 意图微调线上不可见；资金类型/Task1 大改风险高
  - v10 只在「硬证据」上覆盖 capital_type，其余保留 v9 风格软打分
  - 意图改用 Tushare 真实 moneyflow（封板/炸板/净额），不再用涨跌×活跃度伪主力
  - Task1 在 v9 聚类特征上加入真实大单/净额，但不照搬 v5 全套

用法：
  python shared/fetch_daily.py --start 20260608 --end <日> --stock-file <样本> --out data/xxx.csv
  python v10/main_daily.py --input data/xxx.csv --target-date 20260713
  python shared/make_submit.py --dir v10/out --zip v10/submit.zip
"""
from __future__ import annotations

import argparse
import os
import re
import warnings

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score, silhouette_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

VERSION = "v10"
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
EPS = 1e-8
RANDOM_SEED = 42
N_CLUSTERS = 8

LB_HOT, LB_QUANT, LB_RETAIL = "游资", "量化", "散户"
VALID_CAPITAL = {LB_HOT, LB_QUANT, LB_RETAIL}
VALID_INTENT = {"买入", "卖出", "T0交易"}

PATTERNS = {
    "游资抢筹拉升": "大额资金净流入，量价同步走强，短线资金抢筹特征明显",
    "游资高位出货": "大额资金净流出且价格波动放大，存在高位派发特征",
    "量化高频换手": "成交活跃、换手偏高而方向性较弱，呈程序化双向交易特征",
    "主力大单吸筹": "主力资金持续净流入，价格稳步走强，呈分批吸筹特征",
    "放量剧烈震荡": "成交显著放大且振幅较高，多空资金博弈激烈",
    "散户情绪博弈": "主力方向不强、短期波动由中小资金和情绪交易主导",
    "获利盘活跃换手": "近期上涨后维持较高换手，浮盈筹码交易活跃",
    "缩量平静整理": "成交与波动均偏低，资金观望并维持区间整理",
}


def _num(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce").fillna(default)
    return pd.Series(default, index=df.index, dtype=float)


def build_hist_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Tushare 日频表 → 逐股逐日特征（含真实资金流 + 硬证据）。"""
    df = raw.copy()
    df["ts_code"] = df["ts_code"].astype(str)
    df["trade_date"] = df["trade_date"].astype(str).str.replace(r"\.0$", "", regex=True)

    tiers = ["sm", "md", "lg", "elg"]
    for t in tiers:
        df[f"b_{t}"] = _num(df, f"buy_{t}_amount")
        df[f"s_{t}"] = _num(df, f"sell_{t}_amount")
    total = sum(df[f"b_{t}"] + df[f"s_{t}"] for t in tiers) + EPS

    f = pd.DataFrame({"ts_code": df["ts_code"], "trade_date": df["trade_date"]})
    for t in tiers:
        f[f"{t}_ratio"] = (df[f"b_{t}"] + df[f"s_{t}"]) / total
        f[f"{t}_net_ratio"] = (df[f"b_{t}"] - df[f"s_{t}"]) / total
    f["big_ratio"] = f["lg_ratio"] + f["elg_ratio"]
    f["small_ratio"] = f["sm_ratio"]
    f["big_net_ratio"] = ((df["b_lg"] + df["b_elg"]) - (df["s_lg"] + df["s_elg"])) / total
    f["net_mf_ratio"] = _num(df, "net_mf_amount") / total

    pre_close = _num(df, "pre_close", np.nan)
    high, low, close, openp = (_num(df, c, np.nan) for c in ["high", "low", "close", "open"])
    f["pct_chg"] = _num(df, "pct_chg")
    f["abs_ret"] = f["pct_chg"].abs()
    f["amplitude"] = (high - low) / (pre_close + EPS) * 100
    f["close_pos"] = (close - low) / ((high - low) + EPS)
    f["turnover_rate"] = _num(df, "turnover_rate")
    f["volume_ratio"] = _num(df, "volume_ratio")
    vol = _num(df, "vol")
    amount = _num(df, "amount")
    f["volume"] = vol
    f["amount"] = amount

    circ = _num(df, "circ_mv", np.nan)
    circ_safe = circ.where(circ > 0, np.nan)

    # 硬证据
    f["on_top_list"] = _num(df, "on_top_list")
    f["inst_seats"] = _num(df, "inst_seats")
    f["hm_count"] = _num(df, "hm_count")
    f["hm_has_quant"] = _num(df, "hm_has_quant")
    f["ll_is_up"] = _num(df, "ll_is_up")
    f["ll_is_dn"] = _num(df, "ll_is_dn")
    f["ll_is_zhaban"] = _num(df, "ll_is_zhaban")
    f["ll_limit_times"] = _num(df, "ll_limit_times")
    f["ll_open_times"] = _num(df, "ll_open_times")
    f["ll_fd_amount_norm"] = (_num(df, "ll_fd_amount") / circ_safe).fillna(0)

    up_lim, dn_lim = _num(df, "up_limit", np.nan), _num(df, "down_limit", np.nan)
    f["at_up_limit"] = ((close >= up_lim - 0.01) & up_lim.notna()).astype(float)
    f["at_dn_limit"] = ((close <= dn_lim + 0.01) & dn_lim.notna()).astype(float)

    # 三源资金流
    f["dc_net_rate"] = _num(df, "dc_net_rate")
    sign_mf = np.sign(f["net_mf_ratio"])
    sign_dc = np.sign(_num(df, "dc_net_amount"))
    sign_ths = np.sign(_num(df, "ths_net_amount"))
    f["flow_agree"] = ((sign_mf == sign_dc).astype(float) + (sign_mf == sign_ths).astype(float))
    f["flow_agree_dir"] = sign_mf * (f["flow_agree"] >= 2).astype(float)
    f["ths_d5_dir"] = np.sign((_num(df, "ths_net_d5_amount") / circ_safe).fillna(0))

    # v9 风格代理主力（无 moneyflow 时兜底；有真实净额时优先用真实）
    f["main_pct_proxy"] = (f["pct_chg"] * f["volume_ratio"].clip(0.2, 5)).clip(-20, 20)
    f["main_pct"] = np.where(
        f["net_mf_ratio"].abs() > 1e-6,
        (f["net_mf_ratio"] * 100).clip(-20, 20),
        f["main_pct_proxy"],
    )
    f["big_pct"] = (f["big_net_ratio"] * 100).clip(-20, 20)
    f["super_pct"] = (f["elg_net_ratio"] * 100).clip(-20, 20)
    f["small_pct"] = (f["sm_net_ratio"] * 100).clip(-20, 20)

    return f.replace([np.inf, -np.inf], 0).fillna(0)


def build_day_frame(feat_hist: pd.DataFrame, day: str) -> pd.DataFrame:
    hist = feat_hist[feat_hist["trade_date"] <= day].copy()
    tf = hist[hist["trade_date"] == day].copy()
    if tf.empty:
        return tf

    rows = []
    for code, g in hist.groupby("ts_code"):
        g = g.sort_values("trade_date")
        if g.iloc[-1]["trade_date"] != day:
            continue
        last = g.iloc[-1]
        vol_base = g["volume"].iloc[-6:-1].mean() if len(g) > 1 else last["volume"]
        amt_base = g["amount"].iloc[-6:-1].mean() if len(g) > 1 else last["amount"]
        rows.append(
            {
                "ts_code": code,
                "ret1": last["pct_chg"],
                "ret3": g["pct_chg"].tail(3).sum(),
                "ret5": g["pct_chg"].tail(5).sum(),
                "turnover": last["turnover_rate"],
                "volume_ratio": last["volume"] / (vol_base + EPS),
                "amount_ratio": last["amount"] / (amt_base + EPS),
                "amplitude": last["amplitude"],
                "close_pos": last["close_pos"],
                "main_pct": last["main_pct"],
                "main_pct_proxy": last["main_pct_proxy"],
                "main_pct3": g["main_pct"].tail(3).mean(),
                "main_pct3_proxy": g["main_pct_proxy"].tail(3).mean(),
                "big_pct": last["big_pct"],
                "super_pct": last["super_pct"],
                "small_pct": last["small_pct"],
                "super_proxy": last["main_pct_proxy"] * 0.4,
                "small_proxy": -last["main_pct_proxy"] * 0.5,
                "big_ratio": last["big_ratio"],
                "small_ratio": last["small_ratio"],
                "big_net_ratio": last["big_net_ratio"],
                "net_mf_ratio": last["net_mf_ratio"],
                "elg_ratio": last["elg_ratio"],
                "md_ratio": last["md_ratio"],
                "on_top_list": last["on_top_list"],
                "inst_seats": last["inst_seats"],
                "hm_count": last["hm_count"],
                "hm_has_quant": last["hm_has_quant"],
                "ll_limit_times": last["ll_limit_times"],
                "ll_open_times": last["ll_open_times"],
                "ll_is_zhaban": last["ll_is_zhaban"],
                "ll_fd_amount_norm": last["ll_fd_amount_norm"],
                "at_up_limit": last["at_up_limit"],
                "at_dn_limit": last["at_dn_limit"],
                "flow_agree_dir": last["flow_agree_dir"],
                "ths_d5_dir": last["ths_d5_dir"],
                "pct_chg": last["pct_chg"],
                "p_top_freq": g["on_top_list"].mean(),
                "p_limit_freq": g["ll_is_up"].mean() if "ll_is_up" in g.columns else 0.0,
            }
        )

    tf = pd.DataFrame(rows)
    if tf.empty:
        return tf

    for col in [
        "turnover", "volume_ratio", "amount_ratio", "amplitude", "ret1", "abs_ret",
        "big_ratio", "small_ratio", "main_pct", "super_pct", "big_pct", "small_pct",
        "net_mf_ratio", "elg_ratio", "md_ratio",
        "main_pct_proxy", "super_proxy", "small_proxy",
    ]:
        if col == "abs_ret":
            tf["abs_ret"] = tf["ret1"].abs()
        if col in tf.columns or col == "abs_ret":
            c = "abs_ret" if col == "abs_ret" else col
            tf[f"rk_{c}"] = tf[c].rank(pct=True)
    # 软打分专用分位（价格代理）
    tf["rk_main_proxy"] = tf["rk_main_pct_proxy"]
    tf["rk_super_proxy"] = tf["rk_super_proxy"]
    tf["rk_small_proxy"] = tf["rk_small_proxy"]
    return tf.reset_index(drop=True)


def soft_capital_scores(row: pd.Series) -> dict[str, float]:
    """v9 风格软打分：软路径刻意用价格代理，避免真实资金流大面积改写 soft 标签。"""
    main = row.get("main_pct_proxy", row["main_pct"])
    super_rk = row.get("rk_super_proxy", row.get("rk_super_pct", 0))
    main_rk = row.get("rk_main_proxy", row.get("rk_main_pct", 0))
    small_rk = row.get("rk_small_proxy", row.get("rk_small_pct", 0))
    hot = (
        0.30 * row["rk_turnover"]
        + 0.25 * row["rk_amplitude"]
        + 0.20 * max(row.get("rk_ret1", 0), 0)
        + 0.25 * max(super_rk, 0)
    )
    quant = (
        0.30 * row["rk_volume_ratio"]
        + 0.25 * row["rk_turnover"]
        + 0.25 * float(np.clip(1 - abs(main) / 30, 0, 1))
        + 0.20 * float(np.clip(1 - abs(row["ret1"]) / 10, 0, 1))
    )
    retail = (
        0.35 * (1 - row["rk_turnover"])
        + 0.25 * (1 - row["rk_volume_ratio"])
        + 0.20 * (1 - max(main_rk, 0))
        + 0.20 * max(small_rk, 0)
    )
    return {LB_HOT: hot, LB_QUANT: quant, LB_RETAIL: retail}


def hard_capital_override(row: pd.Series) -> tuple[str | None, str]:
    """仅返回硬/中证据标签；无证据则 (None, soft)。"""
    if row["hm_has_quant"] > 0:
        return LB_QUANT, "strong"
    if row["inst_seats"] > 0 and row["on_top_list"] > 0:
        return LB_QUANT, "strong"
    if row["ll_limit_times"] >= 1 or row["hm_count"] > 0:
        return LB_HOT, "strong"
    if row["on_top_list"] > 0:
        return LB_HOT, "strong"
    if (
        row["at_up_limit"] > 0
        and row["ll_fd_amount_norm"] > 0.005
        and row.get("p_limit_freq", 0) > 0.02
    ):
        return LB_HOT, "mid"
    return None, "soft"


def classify_capital(row: pd.Series) -> tuple[str, str]:
    hard, level = hard_capital_override(row)
    if hard is not None:
        return hard, level
    scores = soft_capital_scores(row)
    return max(scores.items(), key=lambda x: x[1])[0], "soft"


def classify_intention(row: pd.Series) -> str:
    """真实资金流意图：封板 → 炸板 → 三源一致 → 单源净额 → T0。"""
    net, big = row["net_mf_ratio"], row["big_net_ratio"]
    pct = row["pct_chg"] if "pct_chg" in row.index else row.get("ret1", 0)
    if row["at_up_limit"] > 0:
        return "买入"
    if row["at_dn_limit"] > 0:
        return "卖出"
    if row["ll_is_zhaban"] > 0 and row["ll_open_times"] >= 2 and pct < 3:
        return "卖出"
    if row["flow_agree_dir"] > 0 and net > 0.02 and pct > -1:
        return "买入"
    if row["flow_agree_dir"] < 0 and net < -0.02 and pct < 1:
        return "卖出"
    if (net > 0.03 or big > 0.04) and pct > -1:
        return "买入"
    if (net < -0.03 or big < -0.04) and pct < 1:
        return "卖出"
    if row["ths_d5_dir"] > 0 and net > 0.03 and pct > 2:
        return "买入"
    if row["ths_d5_dir"] < 0 and net < -0.03 and pct < -2:
        return "卖出"
    return "T0交易"


def assign_patterns(tf: pd.DataFrame) -> tuple[pd.Series, dict]:
    cols = [
        "ret1", "ret3", "turnover", "volume_ratio", "amount_ratio", "amplitude",
        "close_pos", "main_pct_proxy", "main_pct3_proxy", "super_proxy", "small_proxy",
    ]
    cols = [c for c in cols if c in tf.columns]
    work = tf.copy()
    x = StandardScaler().fit_transform(work[cols].replace([np.inf, -np.inf], 0).fillna(0))
    n = len(work)
    nc = 1 if n < 3 else min(N_CLUSTERS, n - 1)
    cluster = KMeans(n_clusters=nc, random_state=RANDOM_SEED, n_init=30).fit_predict(x)
    work["cluster"] = cluster
    centers = work.groupby("cluster").mean(numeric_only=True)

    def scores(c: pd.Series) -> dict[str, float]:
        return {
            "游资抢筹拉升": c.get("rk_ret1", 0) + c.get("rk_turnover", 0) + c.get("rk_super_proxy", 0) + c.get("rk_main_pct_proxy", 0),
            "游资高位出货": c.get("rk_turnover", 0) + c.get("rk_amplitude", 0) + (1 - c.get("rk_main_pct_proxy", 0)) + (1 - c.get("rk_ret1", 0)),
            "量化高频换手": c.get("rk_volume_ratio", 0) + c.get("rk_turnover", 0) + (1 - abs(c.get("main_pct_proxy", 0)) / 30),
            "主力大单吸筹": c.get("main_pct3_proxy", 0) / 10 + c.get("rk_super_proxy", 0) + c.get("close_pos", 0),
            "放量剧烈震荡": c.get("rk_amount_ratio", 0) + c.get("rk_amplitude", 0) + c.get("rk_volume_ratio", 0),
            "散户情绪博弈": c.get("rk_small_proxy", 0) + (1 - c.get("rk_main_pct_proxy", 0)) + c.get("rk_amplitude", 0),
            "获利盘活跃换手": c.get("ret5", 0) / 10 + c.get("rk_turnover", 0) + c.get("close_pos", 0),
            "缩量平静整理": (1 - c.get("rk_volume_ratio", 0)) + (1 - c.get("rk_amplitude", 0)) + (1 - c.get("rk_turnover", 0)),
        }

    candidates = [
        (value, cid, name)
        for cid, center in centers.iterrows()
        for name, value in scores(center).items()
    ]
    mapping, used_c, used_n = {}, set(), set()
    for _, cid, name in sorted(candidates, reverse=True):
        if cid not in used_c and name not in used_n:
            mapping[cid] = name
            used_c.add(cid)
            used_n.add(name)

    metrics = {}
    if 2 <= pd.Series(cluster).nunique() <= n - 1:
        metrics["silhouette"] = float(silhouette_score(x, cluster))
        metrics["ch"] = float(calinski_harabasz_score(x, cluster))
    return pd.Series(cluster, index=tf.index).map(mapping), metrics


def _norm_code(v: str, style: str = "keep") -> str:
    t = re.sub(r"\.0$", "", str(v).strip())
    if style == "bare6":
        m = re.match(r"^(\d{1,6})(\.(SH|SZ|BJ))?$", t, re.IGNORECASE)
        return m.group(1).zfill(6) if m else t
    return t


def run(input_csv: str, out_dir: str, target_date: str | None = None, code_style: str = "keep"):
    print(f"\n{'=' * 60}\nAFAC2026 赛题一 · {VERSION}\n{'=' * 60}\n")
    raw = pd.read_csv(input_csv)
    feat_hist = build_hist_features(raw)
    if target_date is None:
        target_date = sorted(feat_hist["trade_date"].unique())[-1]
    feat_hist = feat_hist[feat_hist["trade_date"] <= target_date]
    print(
        f"目标日 {target_date} | 历史 {feat_hist['trade_date'].nunique()} 天 | "
        f"样本 {(feat_hist['trade_date'] == target_date).sum()} 只"
    )
    tf = build_day_frame(feat_hist, target_date)
    if tf.empty:
        raise ValueError(f"目标日 {target_date} 无数据")

    patterns, metrics = assign_patterns(tf)
    if metrics:
        print(f"Task1 聚类 | 轮廓:{metrics['silhouette']:.4f} CH:{metrics['ch']:.4f}")

    pairs = tf.apply(classify_capital, axis=1)
    capital = [p[0] for p in pairs]
    evidence = [p[1] for p in pairs]
    intention = tf.apply(classify_intention, axis=1)

    ev = pd.Series(evidence).value_counts().to_dict()
    print(f"证据层级: 硬={ev.get('strong', 0)} 中={ev.get('mid', 0)} 软={ev.get('soft', 0)}")

    pattern = pd.DataFrame(
        {
            "stock_code": [_norm_code(c, code_style) for c in tf["ts_code"]],
            "transaction_date": target_date,
            "pattern_type": patterns.values,
            "pattern_explanation": patterns.map(PATTERNS).values,
        }
    )
    result = pd.DataFrame(
        {
            "stock_code": [_norm_code(c, code_style) for c in tf["ts_code"]],
            "transaction_date": target_date,
            "capital_type": capital,
            "capital_intention": intention,
        }
    )
    result["capital_type"] = result["capital_type"].where(
        result["capital_type"].isin(VALID_CAPITAL), LB_RETAIL
    )
    result["capital_intention"] = result["capital_intention"].where(
        result["capital_intention"].isin(VALID_INTENT), "T0交易"
    )

    os.makedirs(out_dir, exist_ok=True)
    pattern.to_csv(os.path.join(out_dir, "pattern_reco.csv"), index=False, encoding="utf-8-sig")
    result.to_csv(os.path.join(out_dir, "predict_result.csv"), index=False, encoding="utf-8-sig")
    print(f"资金类型:\n{result['capital_type'].value_counts().to_string()}")
    print(f"交易意图:\n{result['capital_intention'].value_counts().to_string()}")
    print(f"模式分布:\n{pattern['pattern_type'].value_counts().to_string()}")
    print(f"已保存 {out_dir}/ ({len(pattern)} 行)")
    return {"pattern": pattern, "result": result, "tf": tf, "evidence": evidence, "metrics": metrics}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=f"AFAC2026 {VERSION}")
    ap.add_argument("--input", "-i", default=os.path.join(ROOT, "data", "daily_hist_b_20260713.csv"))
    ap.add_argument("--output", "-o", default=os.path.join(os.path.dirname(__file__), "out"))
    ap.add_argument("--target-date", default=None)
    ap.add_argument("--code-style", default="keep", choices=["keep", "bare6"])
    args = ap.parse_args()
    run(args.input, args.output, args.target_date, args.code_style)
