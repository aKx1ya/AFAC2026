"""
AFAC2026 赛题一 —— 日频真实数据版 v2（当前开发版）

相比 v1 的升级见项目根目录 CHANGELOG.md。

用法（在项目根目录）：
  python v2/main_daily.py --target-date 20260701
  python shared/make_submit.py --dir v2/out --zip v2/submit.zip
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

VERSION = 'v2'
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
RANDOM_SEED = 42
N_CLUSTERS = 8
EPS = 1e-8

LB_HOT, LB_QUANT, LB_RETAIL = '游资', '量化', '散户'
VALID_CAPITAL_TYPES = {LB_HOT, LB_QUANT, LB_RETAIL}
VALID_INTENTIONS = {'买入', '卖出', 'T0交易'}
PATTERN_COLUMNS = ['stock_code', 'transaction_date', 'pattern_type', 'pattern_explanation']
RESULT_COLUMNS = ['stock_code', 'transaction_date', 'capital_type', 'capital_intention']


def _num(df, col, default=0.0):
    if col in df.columns:
        return pd.to_numeric(df[col], errors='coerce').fillna(default)
    return pd.Series(default, index=df.index)


def build_base_features(df_raw):
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
    f['abs_ret'] = f['pct_chg'].abs()
    f['amplitude'] = (high - low) / (pre_close + EPS) * 100
    f['gap'] = (openp - pre_close) / (pre_close + EPS) * 100
    f['close_pos'] = (close - low) / ((high - low) + EPS)
    f['turnover_rate'] = _num(df, 'turnover_rate')
    f['volume_ratio'] = _num(df, 'volume_ratio')
    circ = _num(df, 'circ_mv', np.nan)
    circ_safe = circ.where(circ > 0, np.nan)
    f['on_top_list'] = _num(df, 'on_top_list')
    f['inst_seats'] = _num(df, 'inst_seats')
    f['inst_net_norm'] = (_num(df, 'inst_net') / circ_safe).fillna(0)
    f['hm_count'] = _num(df, 'hm_count')
    f['hm_has_quant'] = _num(df, 'hm_has_quant')
    f['hm_net_norm'] = (_num(df, 'hm_net') / circ_safe).fillna(0)
    f['hm_signal'] = (f['hm_count'] > 0).astype(float) + f['hm_net_norm'].abs().clip(0, 1)
    f['inst_signal'] = (f['inst_seats'] > 0).astype(float) + f['inst_net_norm'].abs().clip(0, 1)
    return f.replace([np.inf, -np.inf], 0).fillna(0)


def build_stock_profile(feat_hist):
    g = feat_hist.groupby('stock_code')
    prof = pd.DataFrame({
        'p_top_freq': g['on_top_list'].mean(),
        'p_hm_freq': g['hm_count'].apply(lambda s: (s > 0).mean()),
        'p_hmq_freq': g['hm_has_quant'].apply(lambda s: (s > 0).mean()),
        'p_inst_freq': g['inst_seats'].apply(lambda s: (s > 0).mean()),
        'p_turn_mean': g['turnover_rate'].mean(),
        'p_vr_mean': g['volume_ratio'].mean(),
        'p_big_mean': g['big_ratio'].mean(),
        'p_sm_mean': g['small_ratio'].mean(),
    }).reset_index()
    for c in ['p_turn_mean', 'p_vr_mean', 'p_big_mean', 'p_sm_mean']:
        prof[c + '_rk'] = prof[c].rank(pct=True)
    return prof


def _pct_rank(s):
    return s.rank(pct=True)


def build_target_frame(feat_hist, target_date):
    prof = build_stock_profile(feat_hist)
    tf = feat_hist[feat_hist['transaction_date'] == target_date].copy()
    if tf.empty:
        raise ValueError(f"目标日 {target_date} 无数据")
    tf = tf.merge(prof, on='stock_code', how='left')
    for c in ['turnover_rate', 'volume_ratio', 'big_ratio', 'elg_ratio', 'md_ratio',
              'small_ratio', 'amplitude', 'abs_ret']:
        tf['rk_' + c] = _pct_rank(tf[c])
    tf['rk_absnet'] = _pct_rank(tf['net_mf_ratio'].abs())
    tf['rk_bignet'] = _pct_rank(tf['big_net_ratio'].abs())
    return tf.reset_index(drop=True)


PATTERN_LIB = [
    ('游资抢筹拉升', '特大/大单净流入、放量高换手、股价强势上行，游资短线抢筹拉升',
     lambda p: 2*p['big_net_ratio'] + p['rk_turnover_rate'] + 0.02*p['pct_chg'] + p['rk_elg_ratio']),
    ('游资高位出货', '大/特大单净流出、股价冲高回落，游资资金派发出货',
     lambda p: -2*p['big_net_ratio'] + p['rk_big_ratio'] - 0.02*p['pct_chg'] + p['rk_absnet']),
    ('量化高频换手', '中单为主、换手与量比双高、净额接近零，程序化双向高频',
     lambda p: p['rk_turnover_rate'] + p['rk_volume_ratio'] + p['rk_md_ratio'] - p['rk_absnet']),
    ('机构资金调仓', '龙虎榜现机构专用席位、净额显著，机构资金进出调仓',
     lambda p: 2*p['inst_signal'] + p['on_top_list']),
    ('主力大单吸筹', '大单温和净流入、换手适中、收盘偏强，主力分批建仓',
     lambda p: p['big_net_ratio'] + p['rk_big_ratio'] + p['close_pos'] - 0.3*p['rk_turnover_rate']),
    ('放量剧烈震荡', '振幅大、量比高但净额不明显，多空剧烈博弈',
     lambda p: p['rk_amplitude'] + p['rk_volume_ratio'] - p['rk_bignet']),
    ('散户情绪博弈', '小单占比相对高、大单参与弱，散户情绪主导的零散交易',
     lambda p: p['rk_small_ratio'] - p['rk_big_ratio'] - 0.3*p['rk_turnover_rate']),
    ('缩量平静整理', '换手低、量比低、振幅小，资金关注度低的平静整理',
     lambda p: -p['rk_turnover_rate'] - p['rk_volume_ratio'] - p['rk_amplitude']),
]
PATTERN_DESC = {n: d for n, d, _ in PATTERN_LIB}


def task1_clustering(tf):
    print("【2/4】Task1 交易模式聚类（8类唯一语义标签）")
    feat_cols = ['big_ratio', 'elg_ratio', 'lg_ratio', 'md_ratio', 'small_ratio',
                 'big_net_ratio', 'net_mf_ratio', 'turnover_rate', 'volume_ratio',
                 'pct_chg', 'amplitude', 'close_pos', 'hm_signal', 'inst_signal']
    X = StandardScaler().fit_transform(tf[feat_cols].replace([np.inf, -np.inf], 0).fillna(0).values)
    n = X.shape[0]
    nc = 1 if n < 3 else min(N_CLUSTERS, n - 1)
    km = KMeans(n_clusters=nc, random_state=RANDOM_SEED, n_init=20)
    tf['cluster_id'] = km.fit_predict(X)
    prof_cols = feat_cols + ['rk_turnover_rate', 'rk_volume_ratio', 'rk_big_ratio',
                             'rk_elg_ratio', 'rk_md_ratio', 'rk_small_ratio',
                             'rk_amplitude', 'rk_absnet', 'rk_bignet', 'on_top_list']
    centers = tf.groupby('cluster_id')[prof_cols].mean()
    score = {(cid, name): fn(centers.loc[cid]) for cid in centers.index for name, _, fn in PATTERN_LIB}
    pairs = sorted(score.items(), key=lambda kv: kv[1], reverse=True)
    assigned, used_names, used_cids = {}, set(), set()
    for (cid, name), _ in pairs:
        if cid in used_cids or name in used_names:
            continue
        assigned[cid] = name
        used_cids.add(cid)
        used_names.add(name)
    for cid in centers.index:
        assigned.setdefault(cid, PATTERN_LIB[-1][0])
    dp = tf[['stock_code', 'transaction_date']].copy()
    dp['pattern_type'] = tf['cluster_id'].map(assigned)
    dp['pattern_explanation'] = dp['pattern_type'].map(PATTERN_DESC)
    if 2 <= pd.Series(tf['cluster_id']).nunique() <= n - 1:
        print(f"聚类 | 轮廓:{silhouette_score(X, tf['cluster_id']):.4f} "
              f"CH:{calinski_harabasz_score(X, tf['cluster_id']):.4f}")
    print(f"模式分布:\n{dp['pattern_type'].value_counts().to_string()}")
    return dp[PATTERN_COLUMNS]


def task2_recognition(tf):
    print("【3/4】Task2 参与者识别（强信号 + 截面相对 + 多日画像）")
    df = tf.copy()

    def classify(r):
        if r['hm_has_quant'] > 0:
            return LB_QUANT
        if r['inst_seats'] > 0 and r['on_top_list'] > 0:
            return LB_QUANT
        if r['hm_count'] > 0 or r['on_top_list'] > 0:
            return LB_HOT
        p_top = r.get('p_top_freq', 0) or 0
        p_hmq = r.get('p_hmq_freq', 0) or 0
        p_turn = r.get('p_turn_mean_rk', 0.5)
        s_hot = (0.28*r['rk_big_ratio'] + 0.22*r['rk_elg_ratio'] + 0.20*r['rk_bignet'] +
                 0.15*r['rk_abs_ret'] + 0.15*min(p_top*3, 1))
        s_quant = (0.30*r['rk_turnover_rate'] + 0.22*r['rk_volume_ratio'] + 0.18*r['rk_md_ratio'] +
                   0.15*(1-r['rk_absnet']) + 0.15*max(p_hmq*2, p_turn))
        s_retail = (0.42*r['rk_small_ratio'] + 0.30*(1-r['rk_turnover_rate']) + 0.28*(1-r['rk_big_ratio']))
        return max(((s_hot, LB_HOT), (s_quant, LB_QUANT), (s_retail, LB_RETAIL)), key=lambda x: x[0])[1]

    df['capital_type'] = df.apply(classify, axis=1)

    def intention(r):
        net, big = r['net_mf_ratio'], r['big_net_ratio']
        if (net > 0.03 or big > 0.04) and r['pct_chg'] > -1:
            return '买入'
        if (net < -0.03 or big < -0.04) and r['pct_chg'] < 1:
            return '卖出'
        return 'T0交易'

    df['capital_intention'] = df.apply(intention, axis=1)
    res = df[RESULT_COLUMNS]
    print(f"资金类型:\n{res['capital_type'].value_counts().to_string()}")
    print(f"交易意图:\n{res['capital_intention'].value_counts().to_string()}")
    return res


def _norm_code(v, style='bare6'):
    t = re.sub(r'\.0$', '', str(v).strip())
    if style == 'bare6':
        m = re.match(r'^(\d{1,6})(\.(SH|SZ|BJ))?$', t, re.IGNORECASE)
        return m.group(1).zfill(6) if m else t
    return t.zfill(6) if re.fullmatch(r'\d{1,6}', t) else t


def save_results(df_pat, df_res, out_dir, code_style='bare6'):
    print("【4/4】结果保存")
    os.makedirs(out_dir, exist_ok=True)
    for d in (df_pat, df_res):
        d['stock_code'] = d['stock_code'].map(lambda x: _norm_code(x, code_style))
        d['transaction_date'] = d['transaction_date'].astype(str)
    df_pat = df_pat.drop_duplicates(['stock_code', 'transaction_date'], keep='last')
    df_res = df_res.drop_duplicates(['stock_code', 'transaction_date'], keep='last')
    df_pat['pattern_explanation'] = df_pat['pattern_explanation'].fillna(df_pat['pattern_type'].map(PATTERN_DESC))
    df_res['capital_type'] = df_res['capital_type'].where(df_res['capital_type'].isin(VALID_CAPITAL_TYPES), LB_RETAIL)
    df_res['capital_intention'] = df_res['capital_intention'].where(df_res['capital_intention'].isin(VALID_INTENTIONS), 'T0交易')
    df_pat.to_csv(os.path.join(out_dir, 'pattern_reco.csv'), index=False, encoding='utf-8-sig')
    df_res.to_csv(os.path.join(out_dir, 'predict_result.csv'), index=False, encoding='utf-8-sig')
    print(f"已保存 {out_dir}/ ({len(df_pat)} 行)")


def run(input_csv, out_dir, target_date=None, code_style='bare6'):
    print(f"\n{'='*60}\nAFAC2026 赛题一 · {VERSION}\n{'='*60}\n")
    print("【1/4】特征工程（多日画像 + 当日截面）")
    raw = pd.read_csv(input_csv)
    feat_hist = build_base_features(raw)
    if target_date is None:
        target_date = sorted(feat_hist['transaction_date'].unique())[-1]
    feat_hist = feat_hist[feat_hist['transaction_date'] <= target_date]
    print(f"目标日 {target_date} | 历史 {feat_hist['transaction_date'].nunique()} 天 | "
          f"样本 {(feat_hist['transaction_date']==target_date).sum()} 只")
    tf = build_target_frame(feat_hist, target_date)
    save_results(task1_clustering(tf), task2_recognition(tf), out_dir, code_style)
    print("\n下一步: python shared/make_submit.py --dir v2/out --zip v2/submit.zip")


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=f'AFAC2026 赛题一 {VERSION}')
    ap.add_argument('--input', '-i', default=os.path.join(ROOT, 'data', 'daily_hist.csv'))
    ap.add_argument('--output', '-o', default=os.path.join(os.path.dirname(__file__), 'out'))
    ap.add_argument('--target-date', default=None)
    ap.add_argument('--code-style', default='bare6', choices=['bare6', 'keep'])
    args = ap.parse_args()
    run(args.input, args.output, args.target_date, args.code_style)
