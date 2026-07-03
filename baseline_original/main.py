"""
================================================================================
AFAC2026 赛题一：市场参与者交易行为识别与资金流向分析 — Baseline
================================================================================
赛题任务：
  Task 1：交易模式识别（无监督聚类，输出聚类结果及语义解释）
  Task 2：参与者识别（游资 / 量化机构） + 意图识别（买入 / 卖出 / T0交易）

输出文件（4列固定格式，字段名称和顺序不可更改）：
  pattern_reco.csv : stock_code, transaction_date, pattern_type, pattern_explanation
  predict_result.csv: stock_code, transaction_date, capital_type, capital_intention
  - capital_type     ∈ {游资, 量化机构}
  - capital_intention ∈ {买入, 卖出, T0交易}

数据认知（关键！）：
  1. volume/amount/transactions/bigordervolume 为当日累计值，必须 diff 得逐笔量
  2. date 为 UTC 毫秒时间戳，hh 为北京时间小时；时段判断必须用 hh 列
  3. bids/asks 为 JSON 字符串，Excel 中双引号转义为 ""，需还原后解析
  4. 赛题提供参考特征集（OSS/RS/CB/AP/OBP/PD/PI），原始数据需自行获取

架构：特征工程(8类52维) → KMeans聚类(Task1) + 11维多因子打分(Task2)
数据：样例集 1只(2026/05/07) | A榜 100只(2026/06/08-07/10) | B榜 100只(2026/07/13-07/24)
评分：A/B榜总得分 = 交易模式识别分×0.4 + 参与者识别分×0.6
 - Task1(40%)：综合轮廓系数、CH指数、Wasserstein距离、DTW距离评估聚类区分度
 - Task2(60%)：基于T+8日实盘回溯真实标签计算加权F1-Score
提交：A榜每天≤3次(23:59前) | B榜每个交易日必须提交(≥8次)
运行：python main.py | python main.py --input testA.xlsx -o out/ | python main.py --input "data/*.xlsx" -o out/

合规（依据赛题5.6节）：禁止硬编码；决策基于Level-2行情特征；使用相对路径；
  main.py为入口文件；B榜TOP15需含init_env.sh；方案与代码脱节将取消评奖资格
================================================================================
"""

import sys, os, glob, argparse, json
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import warnings
warnings.filterwarnings('ignore')

N_CLUSTERS = 8          # 预设聚类数（8 种常见交易模式）
RANDOM_SEED = 42
# 聚类中心画像关键特征（用于模式匹配和离线评估）
KEY_COLS = [
    'oss_mega_amount_pct', 'oss_large_amount_pct', 'oss_medium_amount_pct',
    'oss_small_amount_pct', 'oss_mega_count_pct',
    'rs_interval_cv', 'rs_split_similarity', 'rs_burst_ratio',
    'cb_fast_cancel_ratio',
    'ap_active_buy_pct', 'ap_active_sell_pct', 'ap_active_net_pct',
    'ap_unilateral_intensity', 'ap_active_buy_run_max',
    'spread', 'book_imbalance',
    'pi_time_concentration', 'pi_price_std_pct', 'pd_impact',
]
# ===================== 工具函数 =====================
def parse_order_book(json_str):
    """解析 bids/asks 十档盘口 JSON（{price, volume}[]），返回 (prices, volumes)"""
    try:
        data = json.loads(str(json_str).replace('""', '"'))
        return [it['price'] for it in data], [it['volume'] for it in data]
    except Exception:
        return [], []


def get_book_feat(bids_str, asks_str):
    """从首条 bids/asks 快照提取盘口特征：spread、book_imbalance、大单占比"""
    bp, bv = parse_order_book(bids_str)
    ap, av = parse_order_book(asks_str)
    f = {}
    f['bid1'] = bp[0] if bp else np.nan
    f['ask1'] = ap[0] if ap else np.nan
    f['spread'] = f['ask1'] - f['bid1'] if (not np.isnan(f['bid1']) and not np.isnan(f['ask1'])) else np.nan
    tb, ta = sum(bv), sum(av)
    tt = tb + ta + 1e-8
    f['book_imbalance'] = (tb - ta) / tt       # 正值=买盘强，负值=卖盘强
    f['big_bid_ratio'] = sum(v for v in bv if v >= 50000) / (tb + 1e-8)
    f['big_ask_ratio'] = sum(v for v in av if v >= 50000) / (ta + 1e-8)
    return f

# ===================== 数据预处理 =====================
def load_and_preprocess(input_path):
    """加载并清洗数据：时区处理、异常值过滤、时序排序"""
    print(f"【1/5】数据预处理 | {input_path}")
    df = pd.read_excel(input_path, engine='openpyxl')
    required = ['symbol', 'date', 'price', 'volume', 'amount', 'bids', 'asks']
    df = df.dropna(subset=[c for c in required if c in df.columns]).copy()

    # 日期时间标准化 — date=UTC毫秒时间戳，hh=北京时间小时（时段判断必须用 hh 列！）
    df['datetime'] = pd.to_datetime(df['date'], unit='ms')
    df['transaction_date'] = df['dt'].astype(str) if 'dt' in df.columns else \
        (df['datetime'] + pd.Timedelta(hours=8)).dt.strftime('%Y%m%d')
    df['hour'] = df['hh'] if 'hh' in df.columns else \
        (df['datetime'] + pd.Timedelta(hours=8)).dt.hour
    df['minute'] = df['datetime'].dt.minute
    df = df.rename(columns={'symbol': 'stock_code'})
    df = df[(df['price'] > 0) & (df['volume'] >= 0) & (df['amount'] >= 0)]
    df = df.sort_values(['stock_code', 'transaction_date', 'datetime']).reset_index(drop=True)
    print(f"预处理完成 | {df.shape[0]}行 | {df['stock_code'].nunique()}股 | "
          f"{df['transaction_date'].nunique()}天")
    return df

def extract_all_feature(df_raw):
    """按个股+交易日聚合，提取 8 大类（OSS/TRD/RS/CB/AP/PI/OBP/PD）特征"""
    print("【2/5】特征提取")
    grouped = df_raw.groupby(['stock_code', 'transaction_date'])
    n_groups = grouped.ngroups
    feature_list = []

    for idx, ((sc, td), g) in enumerate(grouped):
        if (idx + 1) % 100 == 0 or (idx + 1) == n_groups:
            print(f"  进度: {idx + 1}/{n_groups}")
        f = {'stock_code': sc, 'transaction_date': td}
        g = g.copy()

        # ── 累计值转逐笔量（volume/amount/transactions/bigordervolume 均为累计值）──
        g['tick_volume'] = g['volume'].diff().fillna(0).clip(lower=0)
        g['tick_amount'] = g['amount'].diff().fillna(0).clip(lower=0)
        g['tick_transactions'] = g['transactions'].diff().fillna(0).clip(lower=0) \
            if 'transactions' in g.columns else 0
        if 'bigordervolume' in g.columns:
            g['tick_big_order_volume'] = g['bigordervolume'].diff().fillna(0).clip(lower=0)

        n = g.shape[0]
        ta = g['tick_amount'].sum() + 1e-8
        tv = g['tick_volume'].sum() + 1e-8

        # === 1. OSS 大单分级（8维）=== 超大≥50000 大≥10000 中≥1000 小<1000 ===
        mega = g['tick_volume'] >= 50000
        large = (g['tick_volume'] >= 10000) & (g['tick_volume'] < 50000)
        mid = (g['tick_volume'] >= 1000) & (g['tick_volume'] < 10000)
        small = g['tick_volume'] < 1000
        for mask, key in [(mega, 'oss_mega'), (large, 'oss_large'),
                           (mid, 'oss_medium'), (small, 'oss_small')]:
            f[f'{key}_amount_pct'] = g.loc[mask, 'tick_amount'].sum() / ta
        f['oss_mega_count_pct'] = mega.sum() / n if n > 0 else 0
        f['oss_large_count_pct'] = large.sum() / n if n > 0 else 0
        f['oss_small_count_pct'] = small.sum() / n if n > 0 else 0
        # 游资活跃成交：大单+价格明显变动
        hot = (g['tick_volume'] >= 10000) & (g['price'].diff().abs() > 0.01)
        f['oss_hot_money_count_pct'] = hot.sum() / n if n > 0 else 0

        # === 2. TRD 交易结构（6维）===
        nt = g['tick_transactions'].sum() + 1
        f['trd_avg_trade_size'] = tv / nt
        f['trd_avg_trade_amount'] = ta / nt
        vt = g[g['tick_transactions'] > 0]
        f['trd_trade_size_std'] = (vt['tick_volume'] / (vt['tick_transactions'] + 1)).std() \
            if len(vt) > 0 else 0
        if 'tick_big_order_volume' in g.columns:
            f['trd_big_order_ratio'] = g['tick_big_order_volume'].sum() / (tv + 1e-8)
        f['trd_change_percent'] = g['changepercent'].iloc[-1] \
            if 'changepercent' in g.columns and len(g) > 0 else 0
        f['trd_range_percent'] = g['rangepercent'].max() \
            if 'rangepercent' in g.columns and len(g) > 0 else 0

        # === 3. RS 订单时序（6维）=== 间隔越均匀→量化；爆发密集→游资 ===
        g['interval_ms'] = g['datetime'].diff().dt.total_seconds() * 1000
        im, istd = g['interval_ms'].mean(), g['interval_ms'].std()
        f['rs_interval_cv'] = istd / im if im and im > 0 else 0
        f['rs_split_similarity'] = max(0, 1 - f['rs_interval_cv'])
        f['rs_burst_ratio'] = (g['interval_ms'] < 100).sum() / n if n > 0 else 0

        # 解析买卖方向（优先使用 side 字段，否则用价格涨跌推断）
        if 'side' in g.columns:
            side = g['side'].astype(str).str.upper()
            bm, sm = side.isin(['B', 'BUY', '1']), side.isin(['S', 'SELL', '-1'])
            for name, mask in [('buy', bm), ('sell', sm)]:
                sg = g[mask]
                if len(sg) > 1:
                    iv = sg['datetime'].diff().dt.total_seconds() * 1000
                    f[f'rs_{name}_interval_cv'] = iv.std() / iv.mean() \
                        if iv.mean() and iv.mean() > 0 else 0
                else:
                    f[f'rs_{name}_interval_cv'] = 0
            f['rs_split_run_ratio'] = 0.0
        else:
            for k in ['rs_buy_interval_cv', 'rs_sell_interval_cv', 'rs_split_run_ratio']:
                f[k] = 0.0

        # === 4. CB 撤单行为（5维）=== 快照数据无撤单明细，暂置0；有逐笔撤单数据时可补充 ===
        for col in ['cb_fast_cancel_ratio', 'cb_cancel_amount_ratio', 'cb_buy_cancel_ratio',
                     'cb_sell_cancel_ratio', 'cb_cancel_interval_cv']:
            f[col] = 0.0

        # === 5. AP 主动成交（7维）=== 优先 side 字段，否则用价格涨跌方向推断 ===
        g['price_change'] = g['price'].diff()
        if 'side' in g.columns:
            side = g['side'].astype(str).str.upper()
            bm = side.isin(['B', 'BUY', '1'])
            sm = side.isin(['S', 'SELL', '-1'])
            ba, sa = g.loc[bm, 'tick_amount'].sum(), g.loc[sm, 'tick_amount'].sum()
        else:
            ba = g.loc[g['price_change'] > 0, 'tick_amount'].sum()
            sa = g.loc[g['price_change'] < 0, 'tick_amount'].sum()
        at = ba + sa + 1e-8
        f['ap_active_buy_pct'] = ba / at
        f['ap_active_sell_pct'] = sa / at
        f['ap_active_net_pct'] = (ba - sa) / ta
        up = (g['price_change'] > 0).astype(int)
        dn = (g['price_change'] < 0).astype(int)
        f['ap_active_buy_run_max'] = up.groupby((up == 0).cumsum()).cumsum().max()
        f['ap_active_sell_run_max'] = dn.groupby((dn == 0).cumsum()).cumsum().max()
        f['ap_unilateral_intensity'] = abs(f['ap_active_net_pct'])

        # === 6. PI 日内时段（4维）=== 游资开盘/尾盘集中，量化全天均匀 ===
        open30 = g[((g['hour'] == 9) & (g['minute'] >= 30)) | ((g['hour'] == 10) & (g['minute'] == 0))]
        close10 = g[(g['hour'] == 14) & (g['minute'] >= 50)]
        f['pi_open_30min_amount_pct'] = open30['tick_amount'].sum() / ta
        f['pi_close_10min_amount_pct'] = close10['tick_amount'].sum() / ta
        f['pi_time_concentration'] = f['pi_open_30min_amount_pct'] + f['pi_close_10min_amount_pct']
        f['pi_price_std_pct'] = g['price'].std() / (g['price'].mean() + 1e-6)

        # === 7. OBP 盘口衍生（14维）=== 方案A:首条快照(4维) + 方案B:全天统计(10维) ===
        if 'bids' in g.columns and 'asks' in g.columns:
            f.update(get_book_feat(g['bids'].iloc[0], g['asks'].iloc[0]))
        if 'totalbidvolume' in g.columns and 'totalaskvolume' in g.columns:
            tb_v, ta_v = g['totalbidvolume'].values, g['totalaskvolume'].values
            imb = (tb_v - ta_v) / (tb_v + ta_v + 1e-8)
            f['obp_imbalance_mean'] = np.nanmean(imb)
            f['obp_imbalance_std'] = np.nanstd(imb)
            f['obp_imbalance_max'] = np.nanmax(imb)
            f['obp_imbalance_min'] = np.nanmin(imb)
            f['obp_total_bid_mean'] = np.nanmean(tb_v)
            f['obp_total_ask_mean'] = np.nanmean(ta_v)
            f['obp_bid_ask_ratio'] = np.nanmean(tb_v) / (np.nanmean(ta_v) + 1e-8)
        if 'weightedbidprice' in g.columns and 'weightedaskprice' in g.columns:
            ws = g['weightedaskprice'].values - g['weightedbidprice'].values
            f['obp_weighted_spread_mean'] = np.nanmean(ws)
            f['obp_weighted_spread_std'] = np.nanstd(ws)
        for col, key in [('bidaskrate', 'obp_bid_ask_rate'),
                         ('bidaskdifference', 'obp_bid_ask_diff')]:
            if col in g.columns:
                f[f'{key}_mean'] = np.nanmean(g[col].values)
                f[f'{key}_std'] = np.nanstd(g[col].values)

        # === 8. PD 价格发现（2维）===
        f['pd_impact'] = abs(f['ap_active_net_pct']) / (f['pi_price_std_pct'] + 1e-6)
        bi = f.get('book_imbalance', 0)
        f['pd_Q1_ratio'] = abs(bi) if not (isinstance(bi, float) and np.isnan(bi)) else 0

        feature_list.append(f)

    df_feat = pd.DataFrame(feature_list)
    for c in df_feat.columns:
        if c not in ('stock_code', 'transaction_date') and df_feat[c].isnull().any():
            df_feat[c] = df_feat[c].fillna(df_feat[c].median())
    df_feat = df_feat.fillna(0).replace([np.inf, -np.inf], 0)
    print(f"特征提取完成 | {df_feat.shape[1] - 2}维 | {df_feat.shape[0]}样本")
    return df_feat

# ===================== Task 1：交易模式聚类 =====================
# 模式: (名称, 解释, [条件列表]), 条件操作符: gt/lt/abs_lt/diff_lt
PATTERN_RULES = [
    ('游资强势连板拉升', '超大单占比高、盘口买盘失衡、主动买入偏多，游资短线集中拉升',
     [('oss_mega_amount_pct', 'gt', 0.12), ('book_imbalance', 'gt', 0.2),
      ('ap_active_buy_pct', 'gt', 0.55), ('pi_time_concentration', 'gt', 0.3)]),
    ('量化高频T0套利', '小单为主、拆单均匀、窄价差、撤单频繁，程序化全天T0套利',
     [('oss_small_amount_pct', 'gt', 0.7), ('rs_split_similarity', 'gt', 0.7),
      ('spread', 'lt', 0.02), ('cb_fast_cancel_ratio', 'gt', 0.1)]),
    ('尾盘资金突袭', '开盘+尾盘成交集中、大单参与、方向性强，游资尾盘突击',
     [('pi_time_concentration', 'gt', 0.35), ('oss_mega_amount_pct', 'gt', 0.1),
      ('ap_unilateral_intensity', 'gt', 0.2)]),
    ('主力分批吸筹', '大单稳步进场、买方占优、时段不集中，主力分批建仓',
     [('oss_mega_amount_pct', 'gt', 0.08), ('book_imbalance', 'gt', 0.15),
      ('ap_active_buy_pct', 'gt', 0.5), ('pi_time_concentration', 'lt', 0.3)]),
    ('日内均衡T0套利', '盘口多空均衡、买卖对称、中单为主，短线日内换手套利',
     [('book_imbalance', 'abs_lt', 0.06), ('ap_active_buy_pct', 'diff_lt', 0.08),
      ('oss_medium_amount_pct', 'gt', 0.4)]),
    ('对倒洗盘', '净买入趋零、大单频繁、挂撤单交替，主力对倒洗盘震仓',
     [('ap_active_net_pct', 'abs_lt', 0.05), ('oss_large_amount_pct', 'gt', 0.3),
      ('cb_fast_cancel_ratio', 'gt', 0.2)]),
    ('散户零散交易', '小单为主、价差宽、间隔不规则，无主力参与',
     [('oss_small_amount_pct', 'gt', 0.85), ('spread', 'gt', 0.05),
      ('rs_interval_cv', 'gt', 0.8)]),
    ('机构长线配置', '波动小、方向弱、无大单集中，公募等长线缓慢布局',
     [('pi_price_std_pct', 'lt', 0.02), ('ap_unilateral_intensity', 'lt', 0.1),
      ('oss_mega_amount_pct', 'lt', 0.05)]),
]

PATTERN_NAMES = [p[0] for p in PATTERN_RULES]
PATTERN_DESC = {p[0]: p[1] for p in PATTERN_RULES}
PATTERN_CONDITIONS = {p[0]: p[2] for p in PATTERN_RULES}


def _check_condition(val, op, threshold):
    """检查条件: gt=大于, lt=小于, abs_lt=绝对值小于, diff_lt=与0.5差值小于"""
    try:
        v = float(val) if not np.isnan(float(val)) else 0.0
    except (ValueError, TypeError):
        return False
    if op == 'gt': return v > threshold
    if op == 'lt': return v < threshold
    if op == 'abs_lt': return abs(v) < threshold
    if op == 'diff_lt': return abs(v - 0.5) < threshold
    return False


def _match_pattern(profile_row):
    """对聚类中心行进行多条件联合匹配（≥3条件命中才生效），返回语义模式名"""
    scores = {}
    for name in PATTERN_NAMES:
        scores[name] = sum(1 for col, op, thresh in PATTERN_CONDITIONS[name]
                          if col in profile_row.index
                          and _check_condition(profile_row[col], op, thresh))
    mx = max(scores.values())
    return max(scores, key=scores.get) if mx >= 3 else '机构长线配置'


def task1_trade_pattern_clustering(df_feat):
    """KMeans 无监督聚类 + 多条件联合匹配赋予语义解释"""
    print("【3/5】Task1 交易模式聚类")
    feat_cols = [c for c in df_feat.columns if c not in ('stock_code', 'transaction_date')]
    X = StandardScaler().fit_transform(df_feat[feat_cols].values)

    nc = min(N_CLUSTERS, X.shape[0])
    if nc < N_CLUSTERS:
        print(f">>> 样本数({X.shape[0]}) < 预设聚类数({N_CLUSTERS})，调整为 {nc}")

    kmeans = KMeans(n_clusters=nc, random_state=RANDOM_SEED, n_init=10)
    df_feat['cluster_id'] = kmeans.fit_predict(X)

    akeys = [c for c in KEY_COLS if c in df_feat.columns]
    profile = df_feat.groupby('cluster_id')[akeys].mean().round(3)
    print("===== 聚类中心画像 =====\n" + profile.to_string())

    # 对每个聚类中心匹配语义模式
    pmap = {cid: _match_pattern(profile.loc[cid]) for cid in range(nc)}
    df_pat = df_feat[['stock_code', 'transaction_date']].copy()
    df_pat['pattern_type'] = df_feat['cluster_id'].map(pmap)
    df_pat['pattern_explanation'] = df_pat['pattern_type'].map(PATTERN_DESC)
    df_pat = df_pat[['stock_code', 'transaction_date', 'pattern_type', 'pattern_explanation']]

    if nc > 1:
        sil = silhouette_score(X, df_feat['cluster_id'])
        ch = calinski_harabasz_score(X, df_feat['cluster_id'])
        db = davies_bouldin_score(X, df_feat['cluster_id'])
        print(f"聚类完成 | 轮廓系数:{sil:.4f} CH:{ch:.4f} DB:{db:.4f}")
    else:
        print("聚类完成 | 样本数不足，仅生成 1 个聚类")
    print(f"模式分布:\n{df_pat['pattern_type'].value_counts().to_string()}")
    return df_pat
# ===================== Task 2：参与者识别 & 意图识别 =====================
def task2_capital_recognition(df_feat):
    """11维多因子打分（游资/量化机构）+ 双源信号联合意图判定（买入/卖出/T0交易）

    核心思路：不同参与者在盘口微观上呈现系统性差异
    - 游资：大单集中、拆单少、方向激进、尾盘突击、撤单率低
    - 量化机构：小单为主、拆单均匀、买卖对称、全天运作、撤单频繁
    """
    print("【4/5】Task2 资金与意图识别")
    df = df_feat.copy()

    # ── 11 维度定义 ── 每个维度含1-2个相关特征，维度内取均值
    dims = [
        ['oss_mega_amount_pct', 'oss_large_amount_pct'],  # 1. 大额成交
        ['rs_split_similarity', 'rs_burst_ratio'],         # 2. 拆单时序
        ['cb_fast_cancel_ratio', 'cb_buy_cancel_ratio'],   # 3. 撤单行为
        ['ap_active_buy_pct', 'ap_active_net_pct'],        # 4. 主动单边
        ['spread', 'book_imbalance'],                      # 5. 盘口结构
        ['pd_impact', 'pd_Q1_ratio'],                      # 6. 价格冲击
        ['pi_time_concentration', 'pi_price_std_pct'],     # 7. 时段波动
        ['ap_active_buy_run_max'],                         # 8. 连续买入
        ['big_bid_ratio', 'big_ask_ratio'],                # 9. 盘口大单
        ['cb_sell_cancel_ratio'],                          # 10. 卖出撤单
        ['ap_unilateral_intensity'],                       # 11. 单边强度
    ]
    # 游资倾向维度索引：值越大越像游资（大额/单边/冲击/时段集中）
    yz_like = {0, 3, 5, 6}

    # 权重配置（游资权重 vs 量化权重，互补设计）
    wyz_full = [0.15, 0.10, 0.08, 0.18, 0.15, 0.10, 0.12, 0.06, 0.06, 0.05, 0.05]
    wqt_full = [0.08, 0.18, 0.12, 0.09, 0.18, 0.11, 0.07, 0.05, 0.09, 0.08, 0.05]

    # 过滤不存在的维度（兼容不同数据版本）
    vdims, vi = [], []
    for i, d in enumerate(dims):
        if all(c in df.columns for c in d):
            vdims.append(d)
            vi.append(i)
    wyz = [wyz_full[i] / sum(wyz_full[j] for j in vi) for i in vi]
    wqt = [wqt_full[i] / sum(wqt_full[j] for j in vi) for i in vi]

    # 跨样本全局 MinMax 归一化
    all_cols = list({c for d in vdims for c in d})
    dfn = df.copy()
    for c in all_cols:
        v = np.nan_to_num(dfn[[c]].values.astype(float), nan=0, posinf=0, neginf=0)
        dfn[c] = (v - v.min()) / (v.max() - v.min()) if v.max() > v.min() else 0.5

    def calc_score(row):
        """加权打分：游资倾向维度值越大游资分越高，量化倾向维度反之"""
        sy, sq = 0.0, 0.0
        for j, dcols in enumerate(vdims):
            di = vi[j]
            ds = np.mean([row[c] for c in dcols])
            if di in yz_like:
                sy += ds * wyz[j]; sq += (1 - ds) * wqt[j]
            else:
                sy += (1 - ds) * wyz[j]; sq += ds * wqt[j]
        return '游资' if sy >= sq else '量化机构'

    df['capital_type'] = dfn.apply(calc_score, axis=1)

    # ── 意图判定：双源信号联合 ── 首条快照失衡(0.4) + 全天均值失衡(0.6)
    # 买入：主动买入>60% 且 综合买盘失衡>0.08
    # 卖出：主动卖出>60% 且 综合卖盘失衡<-0.08
    # T0交易：其余情况（多空均衡、日内套利）
    def get_intention(row):
        bp = row.get('ap_active_buy_pct', 0.5)
        sp = row.get('ap_active_sell_pct', 0.5)
        imb = 0.4 * row.get('book_imbalance', 0) + 0.6 * row.get('obp_imbalance_mean', 0)
        if bp > 0.6 and imb > 0.08: return '买入'
        if sp > 0.6 and imb < -0.08: return '卖出'
        return 'T0交易'

    df['capital_intention'] = df.apply(get_intention, axis=1)
    df_r = df[['stock_code', 'transaction_date', 'capital_type', 'capital_intention']]

    print(f"识别完成\n资金类型:\n{df_r['capital_type'].value_counts().to_string()}")
    print(f"交易意图:\n{df_r['capital_intention'].value_counts().to_string()}")
    return df_r
# ===================== 结果保存与评估 =====================
def save_results(df_pat, df_res, df_feat, out_dir):
    """保存 CSV（格式校验 + 合法值检查）+ 离线评估"""
    print("【5/5】结果保存与评估")
    os.makedirs(out_dir, exist_ok=True)
    pp = os.path.join(out_dir, 'pattern_reco.csv')
    rp = os.path.join(out_dir, 'predict_result.csv')

    # 格式校验
    assert list(df_pat.columns) == ['stock_code', 'transaction_date',
                                     'pattern_type', 'pattern_explanation'], \
        f"Task1 字段错误: {list(df_pat.columns)}"
    assert list(df_res.columns) == ['stock_code', 'transaction_date',
                                     'capital_type', 'capital_intention'], \
        f"Task2 字段错误: {list(df_res.columns)}"
    assert df_res['capital_type'].isin(['游资', '量化机构']).all(), \
        f"非法 capital_type: {df_res['capital_type'].unique()}"
    assert df_res['capital_intention'].isin(['买入', '卖出', 'T0交易']).all(), \
        f"非法 capital_intention: {df_res['capital_intention'].unique()}"

    df_pat.fillna('机构长线配置').to_csv(pp, index=False, encoding='utf-8-sig')
    df_res.fillna('T0交易').to_csv(rp, index=False, encoding='utf-8-sig')
    print(f"已保存: {os.path.basename(pp)}, {os.path.basename(rp)}")

    # 离线评估（仅在含 cluster_id 时执行）
    if 'cluster_id' not in df_feat.columns:
        print("跳过评估（无 cluster_id，可能为打分模式）")
        return
    print("===== 离线评估 =====")
    fc = [c for c in df_feat.columns if c not in ('stock_code', 'transaction_date', 'cluster_id')]
    Xs = StandardScaler().fit_transform(df_feat[fc].values)
    nu = df_feat['cluster_id'].nunique()
    if nu > 1:
        print(f"Task1: 轮廓系数={silhouette_score(Xs, df_feat['cluster_id']):.4f} "
              f"CH={calinski_harabasz_score(Xs, df_feat['cluster_id']):.4f} "
              f"DB={davies_bouldin_score(Xs, df_feat['cluster_id']):.4f}")
    else:
        print(f"Task1: 聚类数={nu}，跳过评估")
    n = len(df_res)
    yz_pct = (df_res['capital_type'] == '游资').sum() / n * 100
    qt_pct = (df_res['capital_type'] == '量化机构').sum() / n * 100
    print(f"Task2: 游资占比={yz_pct:.1f}% 量化机构占比={qt_pct:.1f}%")

# ===================== 主流程 =====================
def run_pipeline(input_path, output_dir):
    """执行 Baseline：预处理→特征提取→Task1聚类→Task2识别→结果保存"""
    print(f"\n{'=' * 60}\nAFAC2026 赛题一 Baseline\n"
          f"输入: {input_path}\n输出: {output_dir}\n{'=' * 60}\n")
    df_raw = load_and_preprocess(input_path)
    df_feat = extract_all_feature(df_raw)
    df_pat = task1_trade_pattern_clustering(df_feat)
    df_res = task2_capital_recognition(df_feat)
    save_results(df_pat, df_res, df_feat, output_dir)
    print(f"\n流程完成！打包 pattern_reco.csv + predict_result.csv 为 submit.zip 提交")
    return df_pat, df_res

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='AFAC2026 赛题一 Baseline')
    parser.add_argument('--input', '-i', default='AFAC2026赛题一训练数据.xlsx',
                        help='输入文件路径，支持 glob 通配符（默认: AFAC2026赛题一训练数据.xlsx）')
    parser.add_argument('--output', '-o', default='./', help='输出目录（默认: ./）')
    args = parser.parse_args()

    # 解析输入文件（支持 glob 通配符批量处理，适配测试集A/B）
    input_files = sorted(glob.glob(args.input))
    if not input_files:
        if os.path.exists(args.input):
            input_files = [args.input]
        else:
            print(f"错误: 未找到输入文件 '{args.input}'")
            sys.exit(1)
    print(f"找到 {len(input_files)} 个输入文件")

    all_pat, all_res = [], []
    for i, f in enumerate(input_files):
        if len(input_files) > 1:
            print(f"\n处理 [{i + 1}/{len(input_files)}]: {f}")
        df_pat, df_res = run_pipeline(f, args.output)
        all_pat.append(df_pat)
        all_res.append(df_res)

    # 多文件合并
    if len(input_files) > 1:
        mp = pd.concat(all_pat, ignore_index=True)
        mr = pd.concat(all_res, ignore_index=True)
        os.makedirs(args.output, exist_ok=True)
        mp.to_csv(os.path.join(args.output, 'pattern_reco.csv'), index=False, encoding='utf-8-sig')
        mr.to_csv(os.path.join(args.output, 'predict_result.csv'), index=False, encoding='utf-8-sig')
        print(f"合并完成: {len(mp)}行 | {mr['stock_code'].nunique()}股 | "
              f"{mr['transaction_date'].nunique()}天")