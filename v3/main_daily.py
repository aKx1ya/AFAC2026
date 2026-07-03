"""
AFAC2026 赛题一 —— v3：弱监督分类器版

相对 v2 的核心升级（Task2 占 60% 权重）：
  1. 用历史龙虎榜/游资明细/机构席位构造弱标签（高置信样本加权更高）
  2. 在 target_date 之前全部交易日上训练 HistGradientBoosting 分类器
  3. 个股画像仅用 <= 当日 的历史（无未来函数）；推理日用模型替代纯规则打分
  4. 当日极强信号仍保留规则覆盖（与弱标签定义一致，防模型欠拟合稀有类）
  5. Task1 沿用 v2 的 8 类 greedy 语义聚类

用法（项目根目录）：
  python v3/main_daily.py --target-date 20260701
  python shared/make_submit.py --dir v3/out --zip v3/submit.zip
"""
import os
import re
import argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import warnings
warnings.filterwarnings('ignore')

VERSION = 'v3'
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
RANDOM_SEED = 42
N_CLUSTERS = 8
EPS = 1e-8

LB_HOT, LB_QUANT, LB_RETAIL = '游资', '量化', '散户'
LABELS = [LB_HOT, LB_QUANT, LB_RETAIL]
INTENTS = ['买入', '卖出', 'T0交易']
VALID_CAPITAL_TYPES = set(LABELS)
VALID_INTENTIONS = set(INTENTS)
PATTERN_COLUMNS = ['stock_code', 'transaction_date', 'pattern_type', 'pattern_explanation']
RESULT_COLUMNS = ['stock_code', 'transaction_date', 'capital_type', 'capital_intention']

# Task2 模型特征（数值列，不含键与标签）
ML_FEATURE_COLS = [
    'sm_ratio', 'md_ratio', 'lg_ratio', 'elg_ratio', 'big_ratio', 'small_ratio',
    'big_net_ratio', 'net_mf_ratio', 'sm_net_ratio', 'md_net_ratio',
    'pct_chg', 'abs_ret', 'amplitude', 'gap', 'close_pos',
    'turnover_rate', 'volume_ratio',
    'on_top_list', 'hm_count', 'hm_has_quant', 'hm_signal', 'inst_signal', 'inst_seats',
    'hm_net_norm', 'inst_net_norm',
    'p_top_freq', 'p_hm_freq', 'p_hmq_freq', 'p_inst_freq',
    'p_turn_mean', 'p_vr_mean', 'p_big_mean', 'p_sm_mean',
    'p_turn_mean_rk', 'p_vr_mean_rk', 'p_big_mean_rk', 'p_sm_mean_rk',
    'rk_turnover_rate', 'rk_volume_ratio', 'rk_big_ratio', 'rk_elg_ratio', 'rk_md_ratio',
    'rk_small_ratio', 'rk_amplitude', 'rk_abs_ret', 'rk_absnet', 'rk_bignet',
]


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


def build_day_frame(feat_hist, day):
    """单日截面特征 + 截至当日的个股画像（无未来信息）。"""
    hist = feat_hist[feat_hist['transaction_date'] <= day]
    tf = hist[hist['transaction_date'] == day].copy()
    if tf.empty:
        return tf
    tf = tf.merge(build_stock_profile(hist), on='stock_code', how='left')
    for c in ['turnover_rate', 'volume_ratio', 'big_ratio', 'elg_ratio', 'md_ratio',
              'small_ratio', 'amplitude', 'abs_ret']:
        tf['rk_' + c] = _pct_rank(tf[c])
    tf['rk_absnet'] = _pct_rank(tf['net_mf_ratio'].abs())
    tf['rk_bignet'] = _pct_rank(tf['big_net_ratio'].abs())
    return tf.reset_index(drop=True)


def build_training_pool(feat_hist, target_date):
    """target_date 之前每个交易日一帧，用于弱监督训练。"""
    days = sorted(d for d in feat_hist['transaction_date'].unique() if d < target_date)
    frames = [build_day_frame(feat_hist, d) for d in days]
    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


# ── 弱标签 ──────────────────────────────────────────────
def _rule_capital_fallback(r):
    """v2 风格规则，用于无强信号日的弱标签补全。"""
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


def weak_capital_label_weight(r):
    """返回 (标签, 样本权重)。强信号来自龙虎榜/游资明细，权重更高。"""
    if r['hm_has_quant'] > 0:
        return LB_QUANT, 4.0
    if r['inst_seats'] > 0 and r['on_top_list'] > 0:
        return LB_QUANT, 3.5
    if r['hm_count'] > 0:
        return LB_HOT, 4.0
    if r['on_top_list'] > 0:
        return LB_HOT, 3.0
    p_top = r.get('p_top_freq', 0) or 0
    if p_top < 0.05 and r['rk_turnover_rate'] < 0.30 and r['rk_big_ratio'] < 0.30:
        return LB_RETAIL, 2.0
    return _rule_capital_fallback(r), 1.0


def weak_intention_label(r):
    net, big, ret = r['net_mf_ratio'], r['big_net_ratio'], r['pct_chg']
    if (net > 0.04 or big > 0.05) and ret > 0.5:
        return '买入'
    if (net < -0.04 or big < -0.05) and ret < -0.5:
        return '卖出'
    if ret > 2.0 and net > 0.02:
        return '买入'
    if ret < -2.0 and net < -0.02:
        return '卖出'
    return 'T0交易'


def _ml_matrix(df, cols):
    X = df.reindex(columns=cols).replace([np.inf, -np.inf], 0).fillna(0).values.astype(float)
    return X


def train_task2_models(train_df):
    """训练资金类型 + 意图两个弱监督分类器。"""
    cap_labels, cap_weights = [], []
    for _, r in train_df.iterrows():
        lb, w = weak_capital_label_weight(r)
        cap_labels.append(lb)
        cap_weights.append(w)
    train_df = train_df.copy()
    train_df['_cap_y'] = cap_labels
    train_df['_cap_w'] = cap_weights
    train_df['_int_y'] = train_df.apply(weak_intention_label, axis=1)

    X = _ml_matrix(train_df, ML_FEATURE_COLS)
    le_cap = LabelEncoder().fit(LABELS)
    le_int = LabelEncoder().fit(INTENTS)

    print(f"  弱标签训练集: {len(train_df)} 条 | "
          f"交易日 {train_df['transaction_date'].nunique()} 天")
    print(f"  capital 分布:\n{pd.Series(cap_labels).value_counts().to_string()}")
    print(f"  intention 分布:\n{train_df['_int_y'].value_counts().to_string()}")
    strong = (train_df['_cap_w'] >= 3.0).sum()
    print(f"  高置信弱标签(权重>=3): {strong} 条 ({strong/len(train_df)*100:.1f}%)")

    m_cap = HistGradientBoostingClassifier(
        max_depth=5, learning_rate=0.08, max_iter=200,
        min_samples_leaf=8, random_state=RANDOM_SEED,
    )
    m_cap.fit(X, le_cap.transform(train_df['_cap_y']), sample_weight=train_df['_cap_w'].values)

    int_w = np.where(train_df['_int_y'] != 'T0交易', 1.8, 1.0)
    m_int = HistGradientBoostingClassifier(
        max_depth=4, learning_rate=0.08, max_iter=180,
        min_samples_leaf=10, random_state=RANDOM_SEED + 1,
    )
    m_int.fit(X, le_int.transform(train_df['_int_y']), sample_weight=int_w)

    return m_cap, le_cap, m_int, le_int


def predict_task2(tf, m_cap, le_cap, m_int, le_int):
    """模型预测 + 当日极强信号规则覆盖。"""
    X = _ml_matrix(tf, ML_FEATURE_COLS)
    cap_pred = le_cap.inverse_transform(m_cap.predict(X))
    int_pred = le_int.inverse_transform(m_int.predict(X))
    df = tf.copy()
    df['capital_type'] = cap_pred
    df['capital_intention'] = int_pred

    # 极强信号覆盖（与弱标签同源，提升 rare class 精度）
    for i, r in df.iterrows():
        if r['hm_has_quant'] > 0:
            df.at[i, 'capital_type'] = LB_QUANT
        elif r['inst_seats'] > 0 and r['on_top_list'] > 0:
            df.at[i, 'capital_type'] = LB_QUANT
        elif r['hm_count'] > 0 or r['on_top_list'] > 0:
            df.at[i, 'capital_type'] = LB_HOT

    # 意图：强净流入/流出时微调
    for i, r in df.iterrows():
        if r['net_mf_ratio'] > 0.08 and r['pct_chg'] > 1:
            df.at[i, 'capital_intention'] = '买入'
        elif r['net_mf_ratio'] < -0.08 and r['pct_chg'] < -1:
            df.at[i, 'capital_intention'] = '卖出'

    res = df[RESULT_COLUMNS]
    print(f"资金类型(模型+覆盖):\n{res['capital_type'].value_counts().to_string()}")
    print(f"交易意图:\n{res['capital_intention'].value_counts().to_string()}")
    return res


# ── Task1（同 v2）──────────────────────────────────────
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
    print("  KMeans + 8类唯一语义标签")
    feat_cols = ['big_ratio', 'elg_ratio', 'lg_ratio', 'md_ratio', 'small_ratio',
                 'big_net_ratio', 'net_mf_ratio', 'turnover_rate', 'volume_ratio',
                 'pct_chg', 'amplitude', 'close_pos', 'hm_signal', 'inst_signal']
    X = StandardScaler().fit_transform(tf[feat_cols].fillna(0).values)
    n = X.shape[0]
    nc = 1 if n < 3 else min(N_CLUSTERS, n - 1)
    km = KMeans(n_clusters=nc, random_state=RANDOM_SEED, n_init=20)
    tf = tf.copy()
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
    if 2 <= tf['cluster_id'].nunique() <= n - 1:
        print(f"聚类 | 轮廓:{silhouette_score(X, tf['cluster_id']):.4f} "
              f"CH:{calinski_harabasz_score(X, tf['cluster_id']):.4f}")
    print(f"模式分布:\n{dp['pattern_type'].value_counts().to_string()}")
    return dp[PATTERN_COLUMNS]


def _norm_code(v, style='bare6'):
    t = re.sub(r'\.0$', '', str(v).strip())
    if style == 'bare6':
        m = re.match(r'^(\d{1,6})(\.(SH|SZ|BJ))?$', t, re.IGNORECASE)
        return m.group(1).zfill(6) if m else t
    return t.zfill(6) if re.fullmatch(r'\d{1,6}', t) else t


def save_results(df_pat, df_res, out_dir, code_style='bare6'):
    print("【5/5】结果保存")
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
    raw = pd.read_csv(input_csv)
    feat_hist = build_base_features(raw)
    if target_date is None:
        target_date = sorted(feat_hist['transaction_date'].unique())[-1]
    feat_hist = feat_hist[feat_hist['transaction_date'] <= target_date]

    print("【1/5】特征工程")
    train_df = build_training_pool(feat_hist, target_date)
    tf = build_day_frame(feat_hist, target_date)
    if tf.empty:
        raise ValueError(f"目标日 {target_date} 无数据")
    print(f"目标日 {target_date} | 训练池 {len(train_df)} 条 / {train_df['transaction_date'].nunique() if len(train_df) else 0} 天")

    print("【2/5】Task1 交易模式聚类")
    df_pat = task1_clustering(tf)

    print("【3/5】Task2 弱监督训练（HistGradientBoosting）")
    if len(train_df) < 50:
        raise ValueError("训练样本不足，请拉取更多历史日频数据")
    m_cap, le_cap, m_int, le_int = train_task2_models(train_df)

    print("【4/5】Task2 推理")
    df_res = predict_task2(tf, m_cap, le_cap, m_int, le_int)

    save_results(df_pat, df_res, out_dir, code_style)
    print("\n下一步: python shared/make_submit.py --dir v3/out --zip v3/submit.zip")


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=f'AFAC2026 赛题一 {VERSION}')
    ap.add_argument('--input', '-i', default=os.path.join(ROOT, 'data', 'daily_hist.csv'))
    ap.add_argument('--output', '-o', default=os.path.join(os.path.dirname(__file__), 'out'))
    ap.add_argument('--target-date', default=None)
    ap.add_argument('--code-style', default='bare6', choices=['bare6', 'keep'])
    args = ap.parse_args()
    run(args.input, args.output, args.target_date, args.code_style)
