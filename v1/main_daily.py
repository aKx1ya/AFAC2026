"""
AFAC2026 赛题一 —— 日频真实数据版 v1（已冻结，评测分 0.41）

特点：
  - 单日特征工程（28 维），不依赖多日历史画像
  - Task1：KMeans + 规则匹配语义标签（易集中到少数模式）
  - Task2：11 维多因子全局 MinMax 打分（易过度判为散户）

用法（在项目根目录）：
  python v1/main_daily.py --input data/daily_data.csv --output v1/out
  python shared/make_submit.py --dir v1/out --zip v1/submit.zip
"""
import os
import re
import argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import warnings
warnings.filterwarnings('ignore')

VERSION = 'v1'
RANDOM_SEED = 42
N_CLUSTERS = 8
EPS = 1e-8
LB_HOT, LB_QUANT, LB_RETAIL = '游资', '量化', '散户'
PATTERN_COLUMNS = ['stock_code', 'transaction_date', 'pattern_type', 'pattern_explanation']
RESULT_COLUMNS = ['stock_code', 'transaction_date', 'capital_type', 'capital_intention']


def _num(df, col, default=0.0):
    if col in df.columns:
        return pd.to_numeric(df[col], errors='coerce').fillna(default)
    return pd.Series(default, index=df.index)


def build_features(df_raw):
    df = df_raw.copy()
    df['stock_code'] = df['ts_code'].astype(str)
    df['transaction_date'] = df['trade_date'].astype(str).str.replace(r'\.0$', '', regex=True)
    tiers = ['sm', 'md', 'lg', 'elg']
    for t in tiers:
        df[f'b_{t}'] = _num(df, f'buy_{t}_amount')
        df[f's_{t}'] = _num(df, f'sell_{t}_amount')
    total = sum(df[f'b_{t}'] + df[f's_{t}'] for t in tiers) + EPS
    f = pd.DataFrame({'stock_code': df['stock_code'], 'transaction_date': df['transaction_date']})
    for t in tiers:
        f[f'{t}_ratio'] = (df[f'b_{t}'] + df[f's_{t}']) / total
        f[f'{t}_net_ratio'] = (df[f'b_{t}'] - df[f's_{t}']) / total
    f['big_ratio'] = f['lg_ratio'] + f['elg_ratio']
    f['small_ratio'] = f['sm_ratio']
    f['big_net_ratio'] = ((df['b_lg'] + df['b_elg']) - (df['s_lg'] + df['s_elg'])) / total
    f['net_mf_ratio'] = _num(df, 'net_mf_amount') / total
    pre_close = _num(df, 'pre_close', np.nan)
    high, low, close, openp = (_num(df, c, np.nan) for c in ['high', 'low', 'close', 'open'])
    f['pct_chg'] = _num(df, 'pct_chg')
    f['amplitude'] = (high - low) / (pre_close + EPS) * 100
    f['gap'] = (openp - pre_close) / (pre_close + EPS) * 100
    f['close_pos'] = (close - low) / ((high - low) + EPS)
    f['turnover_rate'] = _num(df, 'turnover_rate')
    f['volume_ratio'] = _num(df, 'volume_ratio')
    circ = _num(df, 'circ_mv', np.nan)
    circ_safe = circ.where(circ > 0, np.nan)
    f['on_top_list'] = _num(df, 'on_top_list')
    f['top_net_rate'] = _num(df, 'top_net_rate')
    f['inst_seats'] = _num(df, 'inst_seats')
    f['inst_net_norm'] = (_num(df, 'inst_net') / circ_safe).fillna(0)
    f['hm_count'] = _num(df, 'hm_count')
    f['hm_has_quant'] = _num(df, 'hm_has_quant')
    f['hm_net_norm'] = (_num(df, 'hm_net') / circ_safe).fillna(0)
    f['hm_signal'] = (f['hm_count'] > 0).astype(float) + f['hm_net_norm'].abs().clip(0, 1)
    f['inst_signal'] = (f['inst_seats'] > 0).astype(float) + f['inst_net_norm'].abs().clip(0, 1)
    return f.replace([np.inf, -np.inf], 0).fillna(0)


PATTERN_RULES = [
    ('游资抢筹拉升', '特大/大单净流入显著、放量高换手、股价上行',
     [('big_net_ratio', 'gt', 0.06), ('pct_chg', 'gt', 2.0), ('turnover_rate', 'gt', 2.0)]),
    ('主力资金吸筹', '大单温和净流入、换手适中、方向偏多',
     [('big_net_ratio', 'gt', 0.03), ('big_ratio', 'gt', 0.35), ('turnover_rate', 'lt', 3.0)]),
    ('主力大单出货', '大/特大单净流出明显、股价走弱',
     [('big_net_ratio', 'lt', -0.05), ('pct_chg', 'lt', 0.0)]),
    ('量化活跃换手', '中单为主、换手与量比高、净额接近零',
     [('md_ratio', 'gt', 0.28), ('turnover_rate', 'gt', 3.0), ('net_mf_ratio', 'abs_lt', 0.03)]),
    ('散户主导跟风', '小单占比高、大单参与弱',
     [('small_ratio', 'gt', 0.30), ('big_ratio', 'lt', 0.30)]),
    ('机构席位调仓', '龙虎榜现机构专用席位',
     [('inst_signal', 'gt', 0.5), ('on_top_list', 'gt', 0.5)]),
    ('放量剧烈震荡', '振幅大、换手高但净额不明显',
     [('amplitude', 'gt', 5.0), ('volume_ratio', 'gt', 1.2), ('big_net_ratio', 'abs_lt', 0.04)]),
    ('缩量平静整理', '换手低、量比低、振幅小',
     [('turnover_rate', 'lt', 1.0), ('volume_ratio', 'lt', 1.0), ('amplitude', 'lt', 3.0)]),
]
PATTERN_NAMES = [p[0] for p in PATTERN_RULES]
PATTERN_DESC = {p[0]: p[1] for p in PATTERN_RULES}
PATTERN_COND = {p[0]: p[2] for p in PATTERN_RULES}
DEFAULT_PATTERN = '缩量平静整理'


def _check(val, op, thr):
    try:
        v = float(val)
    except (ValueError, TypeError):
        return False
    if np.isnan(v):
        return False
    if op == 'gt': return v > thr
    if op == 'lt': return v < thr
    if op == 'abs_lt': return abs(v) < thr
    return False


def _match_pattern(row):
    scores = {n: sum(1 for c, op, t in PATTERN_COND[n] if c in row.index and _check(row[c], op, t))
              for n in PATTERN_NAMES}
    mx = max(scores.values())
    return max(scores, key=scores.get) if mx >= 2 else DEFAULT_PATTERN


def task1_clustering(feat):
    print("【2/4】Task1 交易模式聚类")
    cols = [c for c in feat.columns if c not in ('stock_code', 'transaction_date')]
    X = StandardScaler().fit_transform(feat[cols].fillna(0).values)
    n = X.shape[0]
    nc = 1 if n < 3 else min(N_CLUSTERS, n - 1)
    km = KMeans(n_clusters=nc, random_state=RANDOM_SEED, n_init=10)
    feat = feat.copy()
    feat['cluster_id'] = km.fit_predict(X)
    akeys = [c for c in ['big_ratio', 'elg_ratio', 'small_ratio', 'big_net_ratio', 'net_mf_ratio',
                         'turnover_rate', 'volume_ratio', 'pct_chg', 'amplitude', 'hm_signal',
                         'inst_signal', 'on_top_list'] if c in feat.columns]
    profile = feat.groupby('cluster_id')[akeys].mean().round(3)
    print("===== 聚类中心画像 =====\n" + profile.to_string())
    pmap = {cid: _match_pattern(profile.loc[cid]) for cid in profile.index}
    dp = feat[['stock_code', 'transaction_date']].copy()
    dp['pattern_type'] = feat['cluster_id'].map(pmap)
    dp['pattern_explanation'] = dp['pattern_type'].map(PATTERN_DESC)
    if 2 <= feat['cluster_id'].nunique() <= n - 1:
        print(f"聚类 | 轮廓:{silhouette_score(X, feat['cluster_id']):.4f}")
    print(f"模式分布:\n{dp['pattern_type'].value_counts().to_string()}")
    return dp[PATTERN_COLUMNS]


def task2_recognition(feat):
    print("【3/4】Task2 参与者识别")
    df = feat.copy()
    dims = [
        ['big_ratio', 'elg_ratio'], ['turnover_rate', 'volume_ratio'],
        ['md_ratio'], ['big_net_ratio', 'net_mf_ratio'],
        ['hm_signal'], ['inst_signal'], ['on_top_list'],
        ['small_ratio'], ['amplitude'], ['pct_chg'],
    ]
    yz_like = {0, 3, 4, 5}
    wyz = [0.18, 0.15, 0.12, 0.20, 0.15, 0.10, 0.10]
    wqt = [0.10, 0.18, 0.15, 0.12, 0.12, 0.18, 0.15]
    vdims, vi, wyz_u, wqt_u = [], [], [], []
    for i, d in enumerate(dims):
        if all(c in df.columns for c in d):
            vdims.append(d)
            vi.append(i)
    if not vdims:
        df['capital_type'] = LB_RETAIL
    else:
        wyz_u = [wyz[min(i, len(wyz)-1)] for i in range(len(vdims))]
        wqt_u = [wqt[min(i, len(wqt)-1)] for i in range(len(vdims))]
        sw, sq = sum(wyz_u), sum(wqt_u)
        wyz_u = [x/sw for x in wyz_u]
        wqt_u = [x/sq for x in wqt_u]
        all_cols = list({c for d in vdims for c in d})
        dfn = df.copy()
        for c in all_cols:
            v = np.nan_to_num(dfn[c].values.astype(float), nan=0)
            dfn[c] = (v - v.min()) / (v.max() - v.min()) if v.max() > v.min() else 0.5

        def score(row):
            sy, sq = 0.0, 0.0
            for j, dcols in enumerate(vdims):
                ds = np.mean([row[c] for c in dcols])
                if j in yz_like:
                    sy += ds * wyz_u[j]
                    sq += (1 - ds) * wqt_u[j]
                else:
                    sy += (1 - ds) * wyz_u[j]
                    sq += ds * wqt_u[j]
            return LB_HOT if sy >= sq else LB_QUANT

        df['capital_type'] = dfn.apply(score, axis=1)
        df.loc[df['hm_has_quant'] > 0, 'capital_type'] = LB_QUANT
        df.loc[(df['hm_count'] > 0) & (df['hm_has_quant'] == 0), 'capital_type'] = LB_HOT
        df.loc[(df['small_ratio'] > 0.52) & (df['big_ratio'] < 0.22) & (df['turnover_rate'] < 2.5),
               'capital_type'] = LB_RETAIL

    def intention(r):
        net, big = r['net_mf_ratio'], r['big_net_ratio']
        if (net > 0.03 or big > 0.05) and r['pct_chg'] > -1:
            return '买入'
        if (net < -0.03 or big < -0.05) and r['pct_chg'] < 1:
            return '卖出'
        return 'T0交易'

    df['capital_intention'] = df.apply(intention, axis=1)
    res = df[RESULT_COLUMNS]
    print(f"资金类型:\n{res['capital_type'].value_counts().to_string()}")
    print(f"交易意图:\n{res['capital_intention'].value_counts().to_string()}")
    return res


def _norm_code(v):
    t = re.sub(r'\.0$', '', str(v).strip())
    m = re.match(r'^(\d{1,6})(\.(SH|SZ|BJ))?$', t, re.IGNORECASE)
    return m.group(1).zfill(6) if m else t


def save_results(df_pat, df_res, out_dir):
    print("【4/4】结果保存")
    os.makedirs(out_dir, exist_ok=True)
    for d in (df_pat, df_res):
        d['stock_code'] = d['stock_code'].map(_norm_code)
        d['transaction_date'] = d['transaction_date'].astype(str)
    df_pat.to_csv(os.path.join(out_dir, 'pattern_reco.csv'), index=False, encoding='utf-8-sig')
    df_res.to_csv(os.path.join(out_dir, 'predict_result.csv'), index=False, encoding='utf-8-sig')
    print(f"已保存 {out_dir}/ ({len(df_pat)} 行)")


def run(input_csv, out_dir, target_date=None):
    print(f"\n{'='*60}\nAFAC2026 赛题一 · {VERSION}\n{'='*60}\n")
    raw = pd.read_csv(input_csv)
    if target_date:
        raw = raw[raw['trade_date'].astype(str).str.replace(r'\.0$', '', regex=True) == target_date]
    print("【1/4】特征工程（单日）")
    feat = build_features(raw)
    print(f"样本 {len(feat)} | {feat['stock_code'].nunique()} 股 | {feat['transaction_date'].nunique()} 天")
    save_results(task1_clustering(feat), task2_recognition(feat), out_dir)


if __name__ == '__main__':
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    ap = argparse.ArgumentParser(description=f'AFAC2026 赛题一 {VERSION}')
    ap.add_argument('--input', '-i', default=os.path.join(root, 'data', 'daily_data.csv'))
    ap.add_argument('--output', '-o', default=os.path.join(os.path.dirname(__file__), 'out'))
    ap.add_argument('--target-date', default=None, help='只处理该交易日 YYYYMMDD')
    args = ap.parse_args()
    run(args.input, args.output, args.target_date)
