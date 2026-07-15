"""AFAC2026 B榜日更版：用前一交易日公开行情预测目标日标签。"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import requests
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.optimize import linear_sum_assignment


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
KLINE_URL = "https://push2his.eastmoney.com/api/qt/stock/kline/get"
FLOW_URL = "https://push2his.eastmoney.com/api/qt/stock/fflow/daykline/get"
SINA_KLINE_URL = "https://quotes.sina.cn/cn/api/jsonp_v2.php/var%20_data=/CN_MarketDataService.getKLineData"
HEADERS = {"User-Agent": "Mozilla/5.0"}

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


def _secid(ts_code: str) -> str:
    code, suffix = ts_code.split(".")
    return f"{1 if suffix.upper() == 'SH' else 0}.{code}"


def _get_json(url: str, params: dict, retries: int = 3) -> dict:
    for attempt in range(retries):
        try:
            response = requests.get(url, params=params, headers=HEADERS, timeout=15)
            response.raise_for_status()
            payload = response.json()
            if payload.get("data"):
                return payload["data"]
        except (requests.RequestException, ValueError):
            if attempt + 1 == retries:
                raise
            time.sleep(1.5 * (attempt + 1))
    return {}


def _fetch_stock(ts_code: str, begin: str, end: str) -> pd.DataFrame:
    code, suffix = ts_code.split(".")
    symbol = ("sh" if suffix.upper() == "SH" else "sz") + code
    response = requests.get(
        SINA_KLINE_URL,
        params={"symbol": symbol, "scale": "240", "ma": "no", "datalen": "60"},
        headers={**HEADERS, "Referer": "https://finance.sina.com.cn/"},
        timeout=20,
    )
    response.raise_for_status()
    match = re.search(r"\((\[.*\])\)", response.text)
    if not match:
        raise ValueError(f"{ts_code} 行情响应格式异常")
    price = pd.DataFrame(json.loads(match.group(1))).rename(columns={"day": "date", "vol": "volume"})
    if price.empty:
        raise ValueError(f"{ts_code} 无日线行情")
    for column in ["open", "close", "high", "low", "volume"]:
        price[column] = pd.to_numeric(price[column], errors="coerce")
    previous = price["close"].shift()
    price["pct_chg"] = (price["close"] / previous - 1) * 100
    price["amplitude"] = (price["high"] - price["low"]) / previous * 100
    price["turnover"] = price["volume"] / price["volume"].rolling(20, min_periods=5).mean()
    price["amount"] = price["volume"] * (price["open"] + price["close"]) / 2
    signed_activity = (price["pct_chg"] * price["turnover"]).clip(-20, 20)
    price["main_pct"] = signed_activity
    price["big_pct"] = signed_activity * 0.6
    price["super_pct"] = signed_activity * 0.4
    price["small_pct"] = -signed_activity * 0.5
    price["ts_code"] = ts_code
    return price


def load_pool(path: str) -> list[str]:
    df = pd.read_excel(path, dtype=str)
    codes = df.iloc[:, 0].astype(str).str.strip()
    codes = codes[codes.str.fullmatch(r"\d{6}\.(SH|SZ|BJ)", case=False, na=False)]
    if codes.nunique() != 100:
        raise ValueError(f"股票样本应有100只，实际有效代码 {codes.nunique()} 只")
    return codes.drop_duplicates().tolist()


def fetch_history(codes: list[str], begin: str, feature_date: str) -> pd.DataFrame:
    end = f"{feature_date[:4]}-{feature_date[4:6]}-{feature_date[6:]}"
    frames, failures = [], []
    with ThreadPoolExecutor(max_workers=10) as executor:
        jobs = {executor.submit(_fetch_stock, code, begin, feature_date): code for code in codes}
        for job in as_completed(jobs):
            code = jobs[job]
            try:
                frames.append(job.result())
            except Exception as exc:
                failures.append((code, str(exc)))
    if failures:
        detail = ", ".join(f"{c}: {e}" for c, e in failures[:5])
        raise RuntimeError(f"行情拉取失败 {len(failures)} 只：{detail}")
    hist = pd.concat(frames, ignore_index=True)
    hist["date"] = pd.to_datetime(hist["date"], errors="coerce")
    cutoff = pd.Timestamp(end)
    hist = hist[hist["date"] <= cutoff].copy()
    return hist.sort_values(["ts_code", "date"]).reset_index(drop=True)


def build_features(hist: pd.DataFrame, codes: list[str], feature_date: str) -> pd.DataFrame:
    numeric = [c for c in hist.columns if c not in {"date", "ts_code"}]
    hist[numeric] = hist[numeric].apply(pd.to_numeric, errors="coerce")
    rows = []
    for code, group in hist.groupby("ts_code"):
        g = group.sort_values("date").copy()
        if g.empty or g.iloc[-1]["date"] != pd.to_datetime(feature_date):
            raise ValueError(f"{code} 缺少特征日 {feature_date} 行情")
        last = g.iloc[-1]
        volume_base = g["volume"].iloc[-6:-1].mean()
        amount_base = g["amount"].iloc[-6:-1].mean()
        rows.append(
            {
                "ts_code": code,
                "ret1": last["pct_chg"],
                "ret3": g["pct_chg"].tail(3).sum(),
                "ret5": g["pct_chg"].tail(5).sum(),
                "turnover": last["turnover"],
                "volume_ratio": last["volume"] / (volume_base + 1e-8),
                "amount_ratio": last["amount"] / (amount_base + 1e-8),
                "amplitude": last["amplitude"],
                "close_pos": (last["close"] - last["low"]) / (last["high"] - last["low"] + 1e-8),
                "main_pct": last.get("main_pct", 0),
                "main_pct3": g["main_pct"].tail(3).mean(),
                "big_pct": last.get("big_pct", 0),
                "super_pct": last.get("super_pct", 0),
                "small_pct": last.get("small_pct", 0),
            }
        )
    feat = pd.DataFrame(rows).set_index("ts_code").reindex(codes).reset_index()
    value_cols = [c for c in feat.columns if c != "ts_code"]
    feat[value_cols] = feat[value_cols].replace([np.inf, -np.inf], np.nan)
    feat[value_cols] = feat[value_cols].fillna(feat[value_cols].median()).fillna(0)
    for col in value_cols:
        feat[f"rk_{col}"] = feat[col].rank(pct=True)
    return feat


def capital_scores(row: pd.Series) -> dict[str, float]:
    hot = (
        0.30 * row["rk_turnover"]
        + 0.25 * row["rk_amplitude"]
        + 0.20 * max(row["rk_ret1"], 0)
        + 0.25 * max(row["rk_super_pct"], 0)
    )
    quant = (
        0.30 * row["rk_volume_ratio"]
        + 0.25 * row["rk_turnover"]
        + 0.25 * np.clip(1 - abs(row["main_pct"]) / 30, 0, 1)
        + 0.20 * np.clip(1 - abs(row["ret1"]) / 10, 0, 1)
    )
    retail = (
        0.35 * (1 - row["rk_turnover"])
        + 0.25 * (1 - row["rk_volume_ratio"])
        + 0.20 * (1 - max(row["rk_main_pct"], 0))
        + 0.20 * max(row["rk_small_pct"], 0)
    )
    return {"游资": hot, "量化": quant, "散户": retail}


def classify_capital(row: pd.Series) -> str:
    return max(capital_scores(row).items(), key=lambda item: item[1])[0]


def classify_capital_quota(feat: pd.DataFrame) -> pd.Series:
    """按历史最佳 v5 的类别先验做全局最优分配：游资22、量化31、散户47。"""
    labels = ["游资"] * 22 + ["量化"] * 31 + ["散户"] * 47
    score_matrix = np.array(
        [[capital_scores(row)[label] for label in labels] for _, row in feat.iterrows()]
    )
    row_ind, col_ind = linear_sum_assignment(-score_matrix)
    assigned = pd.Series(index=range(len(feat)), dtype=object)
    assigned.iloc[row_ind] = [labels[i] for i in col_ind]
    return assigned


def classify_intention(
    row: pd.Series,
    buy_direction: float = 2.2,
    sell_direction: float = 2.2,
    buy_single: float = 4.0,
    sell_single: float = 4.0,
) -> str:
    direction = 0.55 * row["main_pct"] + 0.30 * row["main_pct3"] + 0.15 * row["ret3"]
    if direction >= buy_direction and row["close_pos"] >= 0.35:
        return "买入"
    if direction <= -sell_direction and row["close_pos"] <= 0.65:
        return "卖出"
    if row["main_pct"] >= buy_single and row["ret1"] > -1:
        return "买入"
    if row["main_pct"] <= -sell_single and row["ret1"] < 1:
        return "卖出"
    return "T0交易"


def assign_patterns(feat: pd.DataFrame) -> pd.Series:
    cols = [
        "ret1", "ret3", "turnover", "volume_ratio", "amount_ratio", "amplitude",
        "close_pos", "main_pct", "main_pct3", "big_pct", "super_pct", "small_pct",
    ]
    x = StandardScaler().fit_transform(feat[cols])
    cluster = KMeans(n_clusters=8, random_state=42, n_init=30).fit_predict(x)
    work = feat.copy()
    work["cluster"] = cluster
    centers = work.groupby("cluster").mean(numeric_only=True)

    def scores(c: pd.Series) -> dict[str, float]:
        return {
            "游资抢筹拉升": c["rk_ret1"] + c["rk_turnover"] + c["rk_super_pct"] + c["rk_main_pct"],
            "游资高位出货": c["rk_turnover"] + c["rk_amplitude"] + (1 - c["rk_main_pct"]) + (1 - c["rk_ret1"]),
            "量化高频换手": c["rk_volume_ratio"] + c["rk_turnover"] + (1 - abs(c["main_pct"]) / 30),
            "主力大单吸筹": c["rk_main_pct3"] + c["rk_big_pct"] + c["rk_close_pos"],
            "放量剧烈震荡": c["rk_amount_ratio"] + c["rk_amplitude"] + c["rk_volume_ratio"],
            "散户情绪博弈": c["rk_small_pct"] + (1 - c["rk_main_pct"]) + c["rk_amplitude"],
            "获利盘活跃换手": c["rk_ret5"] + c["rk_turnover"] + c["rk_close_pos"],
            "缩量平静整理": (1 - c["rk_volume_ratio"]) + (1 - c["rk_amplitude"]) + (1 - c["rk_turnover"]),
        }

    candidates = [
        (value, cid, name)
        for cid, center in centers.iterrows()
        for name, value in scores(center).items()
    ]
    mapping, used_clusters, used_names = {}, set(), set()
    for _, cid, name in sorted(candidates, reverse=True):
        if cid not in used_clusters and name not in used_names:
            mapping[cid] = name
            used_clusters.add(cid)
            used_names.add(name)
    return pd.Series(cluster).map(mapping)


def run(
    stock_file: str,
    target_date: str,
    feature_date: str,
    out_dir: str,
    buy_direction: float = 2.2,
    sell_direction: float = 2.2,
    buy_single: float = 4.0,
    sell_single: float = 4.0,
    capital_mode: str = "free",
) -> None:
    codes = load_pool(stock_file)
    hist = fetch_history(codes, "20260601", feature_date)
    feat = build_features(hist, codes, feature_date)
    patterns = assign_patterns(feat)

    transaction_date = str(target_date)
    pattern = pd.DataFrame(
        {
            "stock_code": feat["ts_code"],
            "transaction_date": transaction_date,
            "pattern_type": patterns,
            "pattern_explanation": patterns.map(PATTERNS),
        }
    )
    capital_type = (
        classify_capital_quota(feat)
        if capital_mode == "quota"
        else feat.apply(classify_capital, axis=1)
    )
    result = pd.DataFrame(
        {
            "stock_code": feat["ts_code"],
            "transaction_date": transaction_date,
            "capital_type": capital_type,
            "capital_intention": feat.apply(
                classify_intention,
                axis=1,
                args=(buy_direction, sell_direction, buy_single, sell_single),
            ),
        }
    )
    os.makedirs(out_dir, exist_ok=True)
    pattern.to_csv(os.path.join(out_dir, "pattern_reco.csv"), index=False, encoding="utf-8-sig")
    result.to_csv(os.path.join(out_dir, "predict_result.csv"), index=False, encoding="utf-8-sig")
    print(f"已生成 {len(pattern)} 行，目标日 {target_date}，特征截止日 {feature_date}")
    print(result["capital_type"].value_counts().to_dict())
    print(result["capital_intention"].value_counts().to_dict())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stock-file",
        default=os.path.join(ROOT, "官方数据", "股票样本.xlsx"),
    )
    parser.add_argument("--target-date", default="20260715")
    parser.add_argument("--feature-date", default="20260714")
    parser.add_argument("--output", default=os.path.join(os.path.dirname(__file__), "out"))
    parser.add_argument("--buy-direction", type=float, default=2.2)
    parser.add_argument("--sell-direction", type=float, default=2.2)
    parser.add_argument("--buy-single", type=float, default=4.0)
    parser.add_argument("--sell-single", type=float, default=4.0)
    parser.add_argument("--capital-mode", choices=["free", "quota"], default="free")
    args = parser.parse_args()
    run(
        args.stock_file,
        args.target_date,
        args.feature_date,
        args.output,
        args.buy_direction,
        args.sell_direction,
        args.buy_single,
        args.sell_single,
        args.capital_mode,
    )
