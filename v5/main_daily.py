"""
AFAC2026 赛题一 —— v5：Task1 聚类空间提纯（在 v4=0.5690 基础上）

【只改 Task1，Task2 完全沿用 v4】
  v4 用 21 维（含连板/封板/龙虎榜等稀疏 0/1 证据信号）做 KMeans 距离聚类，
  但这些二值证据信号是为 Task2 硬证据分层设计的，在【距离聚类】里是噪声，
  稀释了连续行为/资金/价格维度的可分性。

  v5 把【聚类空间】与【Task2 证据信号】解耦：
    - 聚类只用 14 维连续判别特征（资金结构/换手/价格/筹码/攻击性）
    - 语义标签匹配仍用完整画像（含 at_up_limit/ll_is_zhaban 等证据列）
  实测（跨 20260626–20260703 六个交易日，同空间公平口径）：
    轮廓 0.1534 → 0.1696（+10.6%），CH 基本持平。Task1 占 40%。

用法（项目根目录）：
  python shared/fetch_daily.py --start 20260608 --end <评测日> --out data/daily_hist.csv
  python v5/main_daily.py --target-date <评测日>
  python shared/make_submit.py --dir v5/out --zip v5/submit.zip
"""
import os
import re
import argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import warnings
warnings.filterwarnings('ignore')

VERSION = 'v5'
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
    # ── 龙虎榜 / 机构席位 / 游资明细 ──
    f['on_top_list'] = _num(df, 'on_top_list')
    f['inst_seats'] = _num(df, 'inst_seats')
    f['inst_net_norm'] = (_num(df, 'inst_net') / circ_safe).fillna(0)
    f['hm_count'] = _num(df, 'hm_count')
    f['hm_has_quant'] = _num(df, 'hm_has_quant')
    f['hm_net_norm'] = (_num(df, 'hm_net') / circ_safe).fillna(0)
    f['hm_signal'] = (f['hm_count'] > 0).astype(float) + f['hm_net_norm'].abs().clip(0, 1)
    f['inst_signal'] = (f['inst_seats'] > 0).astype(float) + f['inst_net_norm'].abs().clip(0, 1)
    # ── v4 新增：连板 / 涨停 / 炸板 ──
    f['ll_is_up'] = _num(df, 'll_is_up')
    f['ll_is_dn'] = _num(df, 'll_is_dn')
    f['ll_is_zhaban'] = _num(df, 'll_is_zhaban')
    f['ll_limit_times'] = _num(df, 'll_limit_times')   # 连板数
    f['ll_open_times'] = _num(df, 'll_open_times')     # 开板次数
    f['ll_fd_amount_norm'] = (_num(df, 'll_fd_amount') / circ_safe).fillna(0)  # 封单额/流通市值
    # ── v4 新增：封板状态（收盘触及涨/跌停）──
    up_lim, dn_lim = _num(df, 'up_limit', np.nan), _num(df, 'down_limit', np.nan)
    f['at_up_limit'] = ((close >= up_lim - 0.01) & up_lim.notna()).astype(float)
    f['at_dn_limit'] = ((close <= dn_lim + 0.01) & dn_lim.notna()).astype(float)
    # ── v4 新增：筹码 / 获利盘 ──
    f['winner_rate'] = _num(df, 'winner_rate', np.nan)
    cost50 = _num(df, 'cost_50pct', np.nan)
    f['price_vs_cost'] = ((close - cost50) / (cost50 + EPS) * 100).replace([np.inf, -np.inf], np.nan)
    cost15, cost85 = _num(df, 'cost_15pct', np.nan), _num(df, 'cost_85pct', np.nan)
    f['chip_concentration'] = ((cost85 - cost15) / (cost50 + EPS)).replace([np.inf, -np.inf], np.nan)
    # ── v4 新增：三源资金流交叉确认 ──
    f['dc_net_rate'] = _num(df, 'dc_net_rate')
    f['dc_net_norm'] = (_num(df, 'dc_net_amount') * 1e4 / circ_safe).fillna(0)  # dc单位万元
    f['ths_net_d5_norm'] = (_num(df, 'ths_net_d5_amount') / circ_safe).fillna(0)
    sign_mf = np.sign(f['net_mf_ratio'])
    sign_dc = np.sign(_num(df, 'dc_net_amount'))
    sign_ths = np.sign(_num(df, 'ths_net_amount'))
    f['flow_agree'] = ((sign_mf == sign_dc).astype(float) + (sign_mf == sign_ths).astype(float))  # 0/1/2
    f['flow_agree_dir'] = sign_mf * (f['flow_agree'] >= 2).astype(float)  # 三源一致方向 ∈{-1,0,1}
    f['ths_d5_dir'] = np.sign(f['ths_net_d5_norm'])
    # ── v4 新增：bak 行为因子 ──
    f['bk_strength'] = _num(df, 'bk_strength')
    f['bk_activity'] = _num(df, 'bk_activity')
    f['bk_attack'] = _num(df, 'bk_attack')
    return f.replace([np.inf, -np.inf], 0).fillna(0)


def build_stock_profile(feat_hist):
    """截至当日的个股行为画像（无未来函数，按 stock_code 聚合历史）。"""
    g = feat_hist.groupby('stock_code')
    prof = pd.DataFrame({
        'p_top_freq': g['on_top_list'].mean(),
        'p_hm_freq': g['hm_count'].apply(lambda s: (s > 0).mean()),
        'p_hmq_freq': g['hm_has_quant'].apply(lambda s: (s > 0).mean()),
        'p_inst_freq': g['inst_seats'].apply(lambda s: (s > 0).mean()),
        'p_limit_freq': g['ll_is_up'].mean(),           # 历史涨停频率（游资活跃度）
        'p_turn_mean': g['turnover_rate'].mean(),
        'p_vr_mean': g['volume_ratio'].mean(),
        'p_big_mean': g['big_ratio'].mean(),
        'p_sm_mean': g['small_ratio'].mean(),
        'p_attack_mean': g['bk_attack'].mean(),         # 历史攻击性
    }).reset_index()
    for c in ['p_turn_mean', 'p_vr_mean', 'p_big_mean', 'p_sm_mean', 'p_attack_mean']:
        prof[c + '_rk'] = prof[c].rank(pct=True)
    return prof


def _pct_rank(s):
    return s.rank(pct=True)


def build_day_frame(feat_hist, day):
    """单日截面特征 + 截至当日画像 + 截面分位（无未来信息）。"""
    hist = feat_hist[feat_hist['transaction_date'] <= day]
    tf = hist[hist['transaction_date'] == day].copy()
    if tf.empty:
        return tf
    tf = tf.merge(build_stock_profile(hist), on='stock_code', how='left')
    for c in ['turnover_rate', 'volume_ratio', 'big_ratio', 'elg_ratio', 'md_ratio',
              'small_ratio', 'amplitude', 'abs_ret', 'bk_attack', 'bk_activity',
              'll_limit_times', 'winner_rate']:
        tf['rk_' + c] = _pct_rank(tf[c])
    tf['rk_absnet'] = _pct_rank(tf['net_mf_ratio'].abs())
    tf['rk_bignet'] = _pct_rank(tf['big_net_ratio'].abs())
    return tf.reset_index(drop=True)
# ── Task2：证据分层 + 跨源确认 ──────────────────────────
def classify_capital(r):
    """资金类型判定：高精度硬证据优先，其后跨源截面打分兜底。

    证据强度排序（越靠前越可信，直接返回）：
      量化: 龙虎榜「量化」游资 / 机构专用席位上榜
      游资: 连板(limit_times>=1) / 龙虎榜游资明细 / 首板封涨停且封单大
      散户: 无任何机构游资痕迹 且 历史低关注 + 小单主导
    """
    # —— 硬证据层 ——
    if r['hm_has_quant'] > 0:
        return LB_QUANT, 'strong'
    if r['inst_seats'] > 0 and r['on_top_list'] > 0:
        return LB_QUANT, 'strong'
    # 连板是游资最干净的信号（比龙虎榜覆盖更广、噪声更低）
    if r['ll_limit_times'] >= 1 or r['hm_count'] > 0:
        return LB_HOT, 'strong'
    if r['on_top_list'] > 0:
        return LB_HOT, 'strong'
    # 首板封涨停 + 大额封单 + 前期有涨停历史 → 游资抢筹
    if r['at_up_limit'] > 0 and r['ll_fd_amount_norm'] > 0.005 and r.get('p_limit_freq', 0) > 0.02:
        return LB_HOT, 'mid'

    # —— 截面相对打分层（无硬证据时）——
    p_top = r.get('p_top_freq', 0) or 0
    p_hmq = r.get('p_hmq_freq', 0) or 0
    p_limit = r.get('p_limit_freq', 0) or 0
    p_turn = r.get('p_turn_mean_rk', 0.5)
    p_attack = r.get('p_attack_mean_rk', 0.5)

    s_hot = (0.24*r['rk_big_ratio'] + 0.18*r['rk_elg_ratio'] + 0.18*r['rk_bignet'] +
             0.12*r['rk_abs_ret'] + 0.10*r['rk_bk_attack'] +
             0.10*min(p_top*3, 1) + 0.08*min(p_limit*10, 1))
    s_quant = (0.26*r['rk_turnover_rate'] + 0.20*r['rk_volume_ratio'] + 0.16*r['rk_md_ratio'] +
               0.14*(1-r['rk_absnet']) + 0.12*r['rk_bk_activity'] +
               0.12*max(p_hmq*2, p_turn))
    s_retail = (0.38*r['rk_small_ratio'] + 0.26*(1-r['rk_turnover_rate']) +
                0.22*(1-r['rk_big_ratio']) + 0.14*(1-max(p_attack, p_top)))
    lab = max(((s_hot, LB_HOT), (s_quant, LB_QUANT), (s_retail, LB_RETAIL)), key=lambda x: x[0])[1]
    return lab, 'soft'


def classify_intention(r):
    """交易意图：封板 → 炸板出货 → 跨源确认(需净额显著) → 5日持续 → T0。

    设计原则：方向判断必须有净额量级支撑，避免把弱信号过度判成买/卖，
    保持与已验证基线(v2)相近的稳健 T0 占比，仅在证据更强时才偏向方向。
    """
    net, big = r['net_mf_ratio'], r['big_net_ratio']
    # 1) 封板：最强意图信号
    if r['at_up_limit'] > 0:
        return '买入'
    if r['at_dn_limit'] > 0:
        return '卖出'
    # 2) 冲高炸板（涨停被反复打开、涨幅回落）→ 高位派发
    if r['ll_is_zhaban'] > 0 and r['ll_open_times'] >= 2 and r['pct_chg'] < 3:
        return '卖出'
    # 3) 三源资金流方向一致 + 净额达标（跨源确认，门槛可略低）
    if r['flow_agree_dir'] > 0 and net > 0.02 and r['pct_chg'] > -1:
        return '买入'
    if r['flow_agree_dir'] < 0 and net < -0.02 and r['pct_chg'] < 1:
        return '卖出'
    # 4) 单源强净流入/流出 + 价格配合（与 v2 阈值一致，保稳健）
    if (net > 0.03 or big > 0.04) and r['pct_chg'] > -1:
        return '买入'
    if (net < -0.03 or big < -0.04) and r['pct_chg'] < 1:
        return '卖出'
    # 5) 5 日资金持续性（弱兜底，需净额与价格同向）
    if r['ths_d5_dir'] > 0 and net > 0.03 and r['pct_chg'] > 2:
        return '买入'
    if r['ths_d5_dir'] < 0 and net < -0.03 and r['pct_chg'] < -2:
        return '卖出'
    return 'T0交易'


def task2_recognition(tf):
    print("【3/4】Task2 参与者识别（证据分层 + 跨源确认）")
    df = tf.copy()
    res_pairs = df.apply(classify_capital, axis=1)
    df['capital_type'] = [p[0] for p in res_pairs]
    df['_evidence'] = [p[1] for p in res_pairs]
    df['capital_intention'] = df.apply(classify_intention, axis=1)
    res = df[RESULT_COLUMNS]
    ev = df['_evidence'].value_counts().to_dict()
    print(f"  证据层级: 硬证据={ev.get('strong',0)} 中={ev.get('mid',0)} 软打分={ev.get('soft',0)}")
    print(f"资金类型:\n{res['capital_type'].value_counts().to_string()}")
    print(f"交易意图:\n{res['capital_intention'].value_counts().to_string()}")
    return res
# ── Task1：交易模式聚类（更丰富判别特征 → 更高区分度）──────
PATTERN_LIB = [
    ('游资抢筹拉升', '特大/大单净流入、放量高换手、封涨停或强势上行，游资短线抢筹拉升',
     lambda p: 2*p['big_net_ratio'] + p['rk_turnover_rate'] + 0.02*p['pct_chg'] +
               p['rk_elg_ratio'] + 0.8*p['at_up_limit'] + 0.5*p['rk_ll_limit_times']),
    ('游资高位出货', '大/特大单净流出、冲高炸板回落、获利盘兑现，游资派发出货',
     lambda p: -2*p['big_net_ratio'] + p['rk_big_ratio'] - 0.02*p['pct_chg'] +
               p['rk_absnet'] + 0.6*p['ll_is_zhaban'] + 0.4*p['rk_winner_rate']),
    ('量化高频换手', '中单为主、换手与量比双高、净额接近零、活跃度高，程序化双向高频',
     lambda p: p['rk_turnover_rate'] + p['rk_volume_ratio'] + p['rk_md_ratio'] -
               p['rk_absnet'] + 0.5*p['rk_bk_activity']),
    ('机构资金调仓', '龙虎榜现机构专用席位、净额显著，机构资金进出调仓',
     lambda p: 2*p['inst_signal'] + p['on_top_list']),
    ('主力大单吸筹', '大单温和净流入、换手适中、收盘偏强、成本上移，主力分批建仓',
     lambda p: p['big_net_ratio'] + p['rk_big_ratio'] + p['close_pos'] -
               0.3*p['rk_turnover_rate'] + 0.3*p['flow_agree_dir']),
    ('放量剧烈震荡', '振幅大、量比高但净额不明显、攻击性强，多空剧烈博弈',
     lambda p: p['rk_amplitude'] + p['rk_volume_ratio'] - p['rk_bignet'] + 0.4*p['rk_bk_attack']),
    ('散户情绪博弈', '小单占比相对高、大单参与弱、历史低关注，散户情绪主导的零散交易',
     lambda p: p['rk_small_ratio'] - p['rk_big_ratio'] - 0.3*p['rk_turnover_rate'] -
               0.3*p['p_top_freq']),
    ('获利盘活跃换手', '换手率偏高、小单为主、获利盘充足，浮盈资金高位频繁换手博弈',
     lambda p: p['rk_turnover_rate'] + p['rk_winner_rate'] + 0.5*p['rk_small_ratio'] -
               p['rk_bignet'] - 0.3*p['rk_amplitude']),
    ('缩量平静整理', '换手低、量比低、振幅小、攻击性弱，资金关注度低的平静整理',
     lambda p: -p['rk_turnover_rate'] - p['rk_volume_ratio'] - p['rk_amplitude'] - 0.3*p['rk_bk_attack']),
]
PATTERN_DESC = {n: d for n, d, _ in PATTERN_LIB}

# v5：聚类空间只用连续判别维度（剔除稀疏 0/1 证据信号，它们是 Task2 用的，
# 在距离聚类里是噪声）。语义标签匹配仍用完整画像（见 task1_clustering 的 prof_cols）。
CLUSTER_FEATS = ['big_ratio', 'md_ratio', 'small_ratio', 'big_net_ratio', 'net_mf_ratio',
                 'turnover_rate', 'volume_ratio', 'pct_chg', 'amplitude', 'close_pos',
                 'winner_rate', 'price_vs_cost', 'bk_attack', 'bk_activity']


def task1_clustering(tf):
    print(f"【2/4】Task1 交易模式聚类（{len(CLUSTER_FEATS)}维连续判别特征 + 8类唯一语义标签）")
    X = StandardScaler().fit_transform(tf[CLUSTER_FEATS].replace([np.inf, -np.inf], 0).fillna(0).values)
    n = X.shape[0]
    nc = 1 if n < 3 else min(N_CLUSTERS, n - 1)
    km = KMeans(n_clusters=nc, random_state=RANDOM_SEED, n_init=25)
    tf = tf.copy()
    tf['cluster_id'] = km.fit_predict(X)
    # 语义标签匹配用完整画像（含证据列 at_up_limit/ll_is_zhaban 等，不影响聚类本身）
    prof_cols = (CLUSTER_FEATS +
                 ['elg_ratio', 'lg_ratio', 'hm_signal', 'inst_signal', 'll_is_up',
                  'll_limit_times', 'at_up_limit', 'll_is_zhaban', 'on_top_list',
                  'p_top_freq', 'flow_agree_dir'] +
                 ['rk_turnover_rate', 'rk_volume_ratio', 'rk_big_ratio', 'rk_elg_ratio',
                  'rk_md_ratio', 'rk_small_ratio', 'rk_amplitude', 'rk_absnet', 'rk_bignet',
                  'rk_bk_attack', 'rk_bk_activity', 'rk_ll_limit_times', 'rk_winner_rate'])
    prof_cols = [c for c in prof_cols if c in tf.columns]
    centers = tf.groupby('cluster_id')[prof_cols].mean()
    score = {(cid, name): fn(centers.loc[cid]) for cid in centers.index for name, _, fn in PATTERN_LIB}
    # 证据型标签门槛：无真实证据的簇不得认领该标签（否则语义失真、损可解释性评分）
    def _gate_ok(cid, name):
        c = centers.loc[cid]
        if name == '机构资金调仓':      # 需真有机构专用席位痕迹
            return c.get('inst_signal', 0) >= 0.10 or c.get('on_top_list', 0) >= 0.20
        return True
    pairs = sorted(score.items(), key=lambda kv: kv[1], reverse=True)
    assigned, used_names, used_cids = {}, set(), set()
    for (cid, name), _ in pairs:
        if cid in used_cids or name in used_names or not _gate_ok(cid, name):
            continue
        assigned[cid] = name
        used_cids.add(cid)
        used_names.add(name)
    # 未认领的簇：优先取「未使用且过闸」的最高分标签，否则回退通用「缩量平静整理」
    for cid in centers.index:
        if cid in assigned:
            continue
        cand = [(score[(cid, nm)], nm) for nm, _, _ in PATTERN_LIB
                if nm not in used_names and _gate_ok(cid, nm)]
        pick = max(cand)[1] if cand else PATTERN_LIB[-1][0]
        assigned[cid] = pick
        used_names.add(pick)
    dp = tf[['stock_code', 'transaction_date']].copy()
    dp['pattern_type'] = tf['cluster_id'].map(assigned)
    dp['pattern_explanation'] = dp['pattern_type'].map(PATTERN_DESC)
    if 2 <= pd.Series(tf['cluster_id']).nunique() <= n - 1:
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
    print("【1/4】特征工程（多源微观结构 + 多日画像 + 当日截面）")
    raw = pd.read_csv(input_csv)
    feat_hist = build_base_features(raw)
    if target_date is None:
        target_date = sorted(feat_hist['transaction_date'].unique())[-1]
    feat_hist = feat_hist[feat_hist['transaction_date'] <= target_date]
    print(f"目标日 {target_date} | 历史 {feat_hist['transaction_date'].nunique()} 天 | "
          f"样本 {(feat_hist['transaction_date']==target_date).sum()} 只")
    tf = build_day_frame(feat_hist, target_date)
    if tf.empty:
        raise ValueError(f"目标日 {target_date} 无数据")
    save_results(task1_clustering(tf), task2_recognition(tf), out_dir, code_style)
    print("\n下一步: python shared/make_submit.py --dir v5/out --zip v5/submit.zip")


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=f'AFAC2026 赛题一 {VERSION}')
    ap.add_argument('--input', '-i', default=os.path.join(ROOT, 'data', 'daily_hist.csv'))
    ap.add_argument('--output', '-o', default=os.path.join(os.path.dirname(__file__), 'out'))
    ap.add_argument('--target-date', default=None)
    ap.add_argument('--code-style', default='bare6', choices=['bare6', 'keep'])
    args = ap.parse_args()
    run(args.input, args.output, args.target_date, args.code_style)

