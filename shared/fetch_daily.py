"""
从 Tushare 拉取【真实】日频微观结构数据，合并为"逐股-逐日"特征就绪表。

用法（在项目根目录执行）：
  python shared/fetch_daily.py --dates 20260701 --out data/daily_data.csv
  python shared/fetch_daily.py --start 20260608 --end 20260701 --out data/daily_hist.csv
"""
import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
import tushare as ts

sys.path.insert(0, os.path.dirname(__file__))
from paths import ROOT, STOCK_FILE


def _load_dotenv(path):
    """轻量读取 .env（无需 python-dotenv 依赖）；已存在的环境变量优先。"""
    if not os.path.exists(path):
        return
    with open(path, encoding='utf-8') as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith('#') or '=' not in line:
                continue
            k, v = line.split('=', 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


# 优先环境变量；否则回退项目根目录 .env（勿提交，见 .env.example）
_load_dotenv(os.path.join(ROOT, '.env'))
TOKEN = os.environ.get('TUSHARE_TOKEN', '')

DEFAULT_POOL = [
    '600000.SH', '600519.SH', '600036.SH', '601318.SH', '600030.SH',
    '600887.SH', '601012.SH', '603259.SH', '600276.SH', '601899.SH',
    '000001.SZ', '000002.SZ', '000651.SZ', '000858.SZ', '002415.SZ',
    '002594.SZ', '300750.SZ', '300059.SZ', '300760.SZ', '002230.SZ',
]


def get_pro():
    if not TOKEN:
        raise SystemExit(
            '未设置 TUSHARE_TOKEN。请复制 .env.example 为 .env 并填入 token，\n'
            '或在 PowerShell 中: $env:TUSHARE_TOKEN="你的token"'
        )
    ts.set_token(TOKEN)
    return ts.pro_api()


def _safe(pro, api_name, **kwargs):
    fn = getattr(pro, api_name)
    for attempt in range(4):
        try:
            df = fn(**kwargs)
            return df if df is not None else pd.DataFrame()
        except Exception as e:
            wait = 15 if ('频率' in str(e) or 'limit' in str(e).lower()) else 3
            print(f"  [retry {attempt+1}] {api_name}: {e} → {wait}s")
            time.sleep(wait)
    print(f"  [fail] {api_name} 最终失败，返回空表")
    return pd.DataFrame()


def resolve_dates(pro, args):
    if args.dates:
        want = [d.strip() for d in args.dates.split(',') if d.strip()]
        cal = _safe(pro, 'trade_cal', exchange='SSE',
                    start_date=min(want), end_date=max(want))
        opens = set(cal[cal['is_open'] == 1]['cal_date']) if not cal.empty else set(want)
        return [d for d in want if d in opens] or want
    if args.start and args.end:
        cal = _safe(pro, 'trade_cal', exchange='SSE',
                    start_date=args.start, end_date=args.end)
        return sorted(cal[cal['is_open'] == 1]['cal_date'].tolist())
    raise ValueError("请用 --dates 或 --start/--end 指定交易日")


def _read_codes_from_xlsx(path):
    df = pd.read_excel(path, engine='openpyxl', dtype=str)
    col = next((c for c in df.columns if '代码' in str(c)), df.columns[0])
    return [str(x).strip() for x in df[col] if str(x).strip() and str(x).strip().lower() != 'nan']


def load_pool(args):
    sf = args.stock_file
    if sf and os.path.exists(sf):
        if sf.lower().endswith(('.xlsx', '.xls')):
            pool = _read_codes_from_xlsx(sf)
            print(f"股票池: 来自官方样本 {sf}，{len(pool)} 只")
        else:
            with open(sf, encoding='utf-8') as fh:
                pool = [ln.strip() for ln in fh if ln.strip() and not ln.startswith('#')]
            print(f"股票池: 来自 {sf}，{len(pool)} 只")
        return pool
    print(f"股票池: 未找到 {sf}，使用默认演示池 {len(DEFAULT_POOL)} 只")
    return DEFAULT_POOL


def agg_top_inst(df):
    if df.empty:
        return pd.DataFrame(columns=['ts_code', 'inst_buy', 'inst_sell', 'inst_net', 'inst_seats'])
    d = df.copy()
    for c in ['buy', 'sell', 'net_buy']:
        d[c] = pd.to_numeric(d.get(c), errors='coerce').fillna(0)
    d = d[d['exalter'].astype(str).str.contains('机构专用')]
    if d.empty:
        return pd.DataFrame(columns=['ts_code', 'inst_buy', 'inst_sell', 'inst_net', 'inst_seats'])
    return d.groupby('ts_code').agg(
        inst_buy=('buy', 'sum'), inst_sell=('sell', 'sum'),
        inst_net=('net_buy', 'sum'), inst_seats=('exalter', 'count')).reset_index()


def agg_hm(df):
    if df.empty:
        return pd.DataFrame(columns=['ts_code', 'hm_buy', 'hm_sell', 'hm_net', 'hm_count', 'hm_has_quant'])
    d = df.copy()
    for c in ['buy_amount', 'sell_amount', 'net_amount']:
        d[c] = pd.to_numeric(d.get(c), errors='coerce').fillna(0)
    d['is_quant'] = d['hm_name'].astype(str).str.contains('量化')
    return d.groupby('ts_code').agg(
        hm_buy=('buy_amount', 'sum'), hm_sell=('sell_amount', 'sum'),
        hm_net=('net_amount', 'sum'), hm_count=('hm_name', 'nunique'),
        hm_has_quant=('is_quant', 'max')).reset_index()


def agg_top_list(df):
    if df.empty:
        return pd.DataFrame(columns=['ts_code', 'top_net_amount', 'top_net_rate', 'on_top_list'])
    d = df.copy()
    for c in ['net_amount', 'net_rate']:
        d[c] = pd.to_numeric(d.get(c), errors='coerce').fillna(0)
    g = d.groupby('ts_code').agg(top_net_amount=('net_amount', 'sum'), top_net_rate=('net_rate', 'mean'))
    g['on_top_list'] = 1
    return g.reset_index()


def agg_limit(df):
    """涨停/连板明细（limit_list_d）→ 游资核心信号。"""
    cols = ['ts_code', 'll_is_up', 'll_is_dn', 'll_is_zhaban', 'll_limit_times',
            'll_open_times', 'll_fd_amount']
    if df.empty:
        return pd.DataFrame(columns=cols)
    d = df.copy()
    lim = d['limit'].astype(str).str.upper()
    d['ll_is_up'] = (lim == 'U').astype(int)
    d['ll_is_dn'] = (lim == 'D').astype(int)
    d['ll_is_zhaban'] = (lim == 'Z').astype(int)
    d['ll_limit_times'] = pd.to_numeric(d.get('limit_times'), errors='coerce').fillna(0)
    d['ll_open_times'] = pd.to_numeric(d.get('open_times'), errors='coerce').fillna(0)
    d['ll_fd_amount'] = pd.to_numeric(d.get('fd_amount'), errors='coerce').fillna(0)
    return d.groupby('ts_code').agg(
        ll_is_up=('ll_is_up', 'max'), ll_is_dn=('ll_is_dn', 'max'),
        ll_is_zhaban=('ll_is_zhaban', 'max'), ll_limit_times=('ll_limit_times', 'max'),
        ll_open_times=('ll_open_times', 'max'), ll_fd_amount=('ll_fd_amount', 'sum')).reset_index()


def sel_cyq(df):
    """筹码分布（cyq_perf）→ 意图/主力成本信号。"""
    cols = ['ts_code', 'winner_rate', 'cost_50pct', 'cost_15pct', 'cost_85pct', 'weight_avg']
    if df.empty:
        return pd.DataFrame(columns=cols)
    d = df.copy()
    for c in cols[1:]:
        d[c] = pd.to_numeric(d.get(c), errors='coerce')
    return d[[c for c in cols if c in d.columns]].drop_duplicates('ts_code')


def sel_mf_dc(df):
    """东财资金流（moneyflow_dc）→ 第三方交叉验证。"""
    if df.empty:
        return pd.DataFrame(columns=['ts_code', 'dc_net_amount', 'dc_net_rate'])
    d = df.rename(columns={'net_amount': 'dc_net_amount', 'net_amount_rate': 'dc_net_rate'})
    for c in ['dc_net_amount', 'dc_net_rate']:
        d[c] = pd.to_numeric(d.get(c), errors='coerce')
    return d[['ts_code', 'dc_net_amount', 'dc_net_rate']].drop_duplicates('ts_code')


def sel_mf_ths(df):
    """同花顺资金流（moneyflow_ths）→ 5日净额=持续性信号。"""
    if df.empty:
        return pd.DataFrame(columns=['ts_code', 'ths_net_amount', 'ths_net_d5_amount'])
    d = df.rename(columns={'net_amount': 'ths_net_amount', 'net_d5_amount': 'ths_net_d5_amount'})
    for c in ['ths_net_amount', 'ths_net_d5_amount']:
        d[c] = pd.to_numeric(d.get(c), errors='coerce')
    return d[['ts_code', 'ths_net_amount', 'ths_net_d5_amount']].drop_duplicates('ts_code')


def sel_bak(df):
    """备用行情（bak_daily）→ 攻击性/强度/活跃度行为因子。"""
    ren = {'strength': 'bk_strength', 'activity': 'bk_activity',
           'attack': 'bk_attack', 'swing': 'bk_swing'}
    if df.empty:
        return pd.DataFrame(columns=['ts_code'] + list(ren.values()))
    d = df.rename(columns=ren)
    for c in ren.values():
        d[c] = pd.to_numeric(d.get(c), errors='coerce')
    return d[['ts_code'] + [c for c in ren.values() if c in d.columns]].drop_duplicates('ts_code')


def sel_stk_limit(df):
    """涨跌停价（stk_limit）→ 与收盘价比对判断封板意图。"""
    if df.empty:
        return pd.DataFrame(columns=['ts_code', 'up_limit', 'down_limit'])
    d = df.copy()
    for c in ['up_limit', 'down_limit']:
        d[c] = pd.to_numeric(d.get(c), errors='coerce')
    return d[['ts_code', 'up_limit', 'down_limit']].drop_duplicates('ts_code')


def fetch_one_day(pro, date_str, pool_set):
    print(f"  拉取 daily/basic/moneyflow/top_list/top_inst/hm_detail + limit/cyq/dc/ths/bak/stk_limit ...")
    daily = _safe(pro, 'daily', trade_date=date_str)
    basic = _safe(pro, 'daily_basic', trade_date=date_str,
                  fields='ts_code,trade_date,turnover_rate,volume_ratio,total_mv,circ_mv,pe')
    mf = _safe(pro, 'moneyflow', trade_date=date_str)
    tl = agg_top_list(_safe(pro, 'top_list', trade_date=date_str))
    ti = agg_top_inst(_safe(pro, 'top_inst', trade_date=date_str))
    hm = agg_hm(_safe(pro, 'hm_detail', trade_date=date_str))
    # v4 新增微观结构数据源
    ll = agg_limit(_safe(pro, 'limit_list_d', trade_date=date_str))
    cyq = sel_cyq(_safe(pro, 'cyq_perf', trade_date=date_str))
    dc = sel_mf_dc(_safe(pro, 'moneyflow_dc', trade_date=date_str))
    ths = sel_mf_ths(_safe(pro, 'moneyflow_ths', trade_date=date_str))
    bak = sel_bak(_safe(pro, 'bak_daily', trade_date=date_str))
    slim = sel_stk_limit(_safe(pro, 'stk_limit', trade_date=date_str))
    if daily.empty:
        print(f"  [warn] {date_str} 无日线数据，跳过")
        return pd.DataFrame()
    df = daily.copy()
    if not basic.empty:
        df = df.merge(basic.drop(columns=['trade_date'], errors='ignore'), on='ts_code', how='left')
    if not mf.empty:
        df = df.merge(mf.drop(columns=['trade_date'], errors='ignore'), on='ts_code', how='left')
    for extra in (tl, ti, hm, ll, cyq, dc, ths, bak, slim):
        if not extra.empty:
            df = df.merge(extra, on='ts_code', how='left')
    df = df[df['ts_code'].isin(pool_set)].copy()
    zero_fill = ['on_top_list', 'top_net_amount', 'top_net_rate',
                 'inst_buy', 'inst_sell', 'inst_net', 'inst_seats',
                 'hm_buy', 'hm_sell', 'hm_net', 'hm_count', 'hm_has_quant',
                 'll_is_up', 'll_is_dn', 'll_is_zhaban', 'll_limit_times',
                 'll_open_times', 'll_fd_amount']
    for c in zero_fill:
        df[c] = pd.to_numeric(df.get(c), errors='coerce').fillna(0) if c in df.columns else 0
    print(f"  {date_str}: 命中股票池 {df['ts_code'].nunique()} / {len(pool_set)} 只")
    return df


def main():
    ap = argparse.ArgumentParser(description='Tushare 日频微观结构 → AFAC2026 特征表')
    ap.add_argument('--dates', default='', help='逗号分隔交易日 YYYYMMDD')
    ap.add_argument('--start', default='', help='起始日 YYYYMMDD')
    ap.add_argument('--end', default='', help='结束日 YYYYMMDD')
    ap.add_argument('--stock-file', default=STOCK_FILE, help='股票池文件')
    ap.add_argument('--out', default=os.path.join(ROOT, 'data', 'daily_hist.csv'), help='输出 CSV')
    ap.add_argument('--sleep', type=float, default=0.6, help='每日间隔秒数')
    args = ap.parse_args()

    pro = get_pro()
    dates = resolve_dates(pro, args)
    pool = load_pool(args)
    pool_set = set(pool)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    print(f"交易日={dates} | 股票池={len(pool)} | 输出={args.out}")

    frames = []
    for i, d in enumerate(dates):
        print(f"[{i+1}/{len(dates)}] {d}")
        one = fetch_one_day(pro, d, pool_set)
        if not one.empty:
            frames.append(one)
        if i < len(dates) - 1:
            time.sleep(args.sleep)

    if not frames:
        raise SystemExit("未获取到任何数据")
    out = pd.concat(frames, ignore_index=True).sort_values(['ts_code', 'trade_date']).reset_index(drop=True)
    out.to_csv(args.out, index=False, encoding='utf-8-sig')
    print(f"\n已保存: {args.out} | {len(out)}行 | {out['ts_code'].nunique()}股 | {out['trade_date'].nunique()}天")


if __name__ == '__main__':
    main()
