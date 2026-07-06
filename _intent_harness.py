"""
意图(capital_intention)离线验证框架 —— v8 探路,防 v6 式陷阱。

代理真值：每只股票当日之后 N 个交易日的累计涨跌方向
  未来N日累计收益 > +THR  → 真值「买入」
  未来N日累计收益 < -THR  → 真值「卖出」
  其余                    → 真值「T0交易」

严格防未来函数：特征只用 <= 当日；代理真值只用于【离线评估阈值好坏】，
不进入任何推理路径。时间切分 train/test 检验阈值泛化。
分行情(普涨/震荡/普跌)报告，确保跨行情稳健(而非单日过拟合)。
"""
import importlib.util, warnings, numpy as np, pandas as pd
from itertools import product
from sklearn.metrics import f1_score
warnings.filterwarnings('ignore')

spec = importlib.util.spec_from_file_location('v5', 'v5/main_daily.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

DATA = 'data/daily_ext.csv'
FWD_N = 8          # 未来交易日数(模拟 T+8)
THR = 3.0          # 方向阈值(%): |累计涨跌|>3% 才算有方向

raw = pd.read_csv(DATA)
raw['dd'] = raw['trade_date'].astype(str).str.replace(r'\.0$', '', regex=True)
raw['code'] = raw['ts_code'].astype(str)
raw = raw.sort_values(['code', 'dd'])
# 未来 N 日累计收益(shift 负 = 取未来)
raw['fwd_ret'] = raw.groupby('code')['pct_chg'].transform(
    lambda s: sum(s.shift(-k) for k in range(1, FWD_N + 1)))

def truth_dir(fr):
    if pd.isna(fr): return None
    return '买入' if fr > THR else ('卖出' if fr < -THR else 'T0交易')
raw['itruth'] = raw['fwd_ret'].map(truth_dir)
retmap = raw.set_index(['code', 'dd'])[['fwd_ret', 'itruth']]

feat = m.build_base_features(raw)
days = sorted(feat['transaction_date'].unique())
# 只保留能算出未来N日真值的交易日(末尾 N 天无真值)
valid_days = days[:-FWD_N] if len(days) > FWD_N else days

def regime(day):
    s = raw[raw['dd'] == day]; mu = pd.to_numeric(s['pct_chg'], errors='coerce').mean()
    return '普涨' if mu > 0.8 else ('普跌' if mu < -0.8 else '震荡')

# 逐日构造当日截面(无未来)，附代理真值
rows = []
for d in valid_days:
    tf = m.build_day_frame(feat[feat['transaction_date'] <= d], d)
    if tf.empty: continue
    tf = tf.merge(retmap, left_on=['stock_code', 'transaction_date'], right_index=True, how='left')
    tf['reg'] = regime(d)
    rows.append(tf)
A = pd.concat(rows, ignore_index=True).dropna(subset=['itruth'])
print(f"验证池: {len(A)} 样本 | {A['transaction_date'].nunique()} 交易日 | FWD_N={FWD_N} THR={THR}")
print(f"行情分布(样本数): {A['reg'].value_counts().to_dict()}")
print(f"代理真值分布: {A['itruth'].value_counts().to_dict()}")

def eval_rule(df, rule_fn):
    pred = df.apply(rule_fn, axis=1)
    f1 = f1_score(df['itruth'], pred, average='weighted', labels=['买入', '卖出', 'T0交易'])
    acc = (pred == df['itruth']).mean()
    return f1, acc, pred

print("\n" + "=" * 60)
print("A. 当前 v5/v4 意图规则 baseline (分行情)")
print("=" * 60)
cur_f1, cur_acc, cur_pred = eval_rule(A, m.classify_intention)
print(f"{'行情':<8}{'样本':>7}{'当前F1':>10}{'当前Acc':>10}")
for rg in ['普涨', '震荡', '普跌', '全部']:
    sub = A if rg == '全部' else A[A['reg'] == rg]
    f1, acc, _ = eval_rule(sub, m.classify_intention)
    print(f"{rg:<8}{len(sub):>7}{f1:>10.4f}{acc:>9.1%}")

# 全判 T0 基线
base_f1 = f1_score(A['itruth'], ['T0交易'] * len(A), average='weighted', labels=['买入', '卖出', 'T0交易'])
print(f"\n(对照) 全判T0基线 weighted-F1: {base_f1:.4f}")
print(f"当前规则相对基线: {'✓+' if cur_f1 > base_f1 else '✗'}{cur_f1 - base_f1:+.4f}")


# ── 候选新规则：参数化阈值 ──────────────────────────────
def make_rule(net_hi, big_hi, net_lo_agree, pct_gate):
    """结构同 v5，仅阈值参数化。net_hi=单源净额门槛, big_hi=大单净额门槛,
    net_lo_agree=三源一致时的净额门槛, pct_gate=价格确认带宽。"""
    def rule(r):
        net, big = r['net_mf_ratio'], r['big_net_ratio']
        if r['at_up_limit'] > 0:
            return '买入'
        if r['at_dn_limit'] > 0:
            return '卖出'
        if r['ll_is_zhaban'] > 0 and r['ll_open_times'] >= 2 and r['pct_chg'] < 3:
            return '卖出'
        if r['flow_agree_dir'] > 0 and net > net_lo_agree and r['pct_chg'] > -pct_gate:
            return '买入'
        if r['flow_agree_dir'] < 0 and net < -net_lo_agree and r['pct_chg'] < pct_gate:
            return '卖出'
        if (net > net_hi or big > big_hi) and r['pct_chg'] > -pct_gate:
            return '买入'
        if (net < -net_hi or big < -big_hi) and r['pct_chg'] < pct_gate:
            return '卖出'
        if r['ths_d5_dir'] > 0 and net > net_hi and r['pct_chg'] > 2:
            return '买入'
        if r['ths_d5_dir'] < 0 and net < -net_hi and r['pct_chg'] < -2:
            return '卖出'
        return 'T0交易'
    return rule

# 当前规则等价参数: net_hi=0.03 big_hi=0.04 net_lo_agree=0.02 pct_gate=1.0
print("\n" + "=" * 60)
print("B. 阈值网格搜索 + 时间切分 train/test + 跨行情稳健性")
print("=" * 60)
split = valid_days[len(valid_days) * 2 // 3]  # 前2/3训练, 后1/3测试
A_tr, A_te = A[A['transaction_date'] < split], A[A['transaction_date'] >= split]
print(f"时间切分: 训练<{split} ({A_tr['transaction_date'].nunique()}天/{len(A_tr)}样本) | "
      f"测试>={split} ({A_te['transaction_date'].nunique()}天/{len(A_te)}样本)")

grid = list(product([0.02, 0.03, 0.04, 0.05], [0.03, 0.04, 0.05, 0.06],
                    [0.015, 0.02, 0.03], [0.5, 1.0, 1.5]))
results = []
for nh, bh, nla, pg in grid:
    rule = make_rule(nh, bh, nla, pg)
    f1_tr, _, _ = eval_rule(A_tr, rule)
    results.append((f1_tr, nh, bh, nla, pg))
results.sort(reverse=True)

cur_tr_f1, _, _ = eval_rule(A_tr, m.classify_intention)
cur_te_f1, _, _ = eval_rule(A_te, m.classify_intention)
print(f"\n当前规则: train-F1={cur_tr_f1:.4f} | test-F1={cur_te_f1:.4f}")
print(f"\nTop5 训练集最优阈值 → 测试集泛化 + 分行情:")
print(f"{'net_hi':>7}{'big_hi':>7}{'agr':>6}{'gate':>6}{'trF1':>8}{'teF1':>8}{'普涨te':>8}{'震荡te':>8}{'普跌te':>8}")
for f1_tr, nh, bh, nla, pg in results[:5]:
    rule = make_rule(nh, bh, nla, pg)
    f1_te, _, _ = eval_rule(A_te, rule)
    reg_f1 = {}
    for rg in ['普涨', '震荡', '普跌']:
        sub = A_te[A_te['reg'] == rg]
        reg_f1[rg] = eval_rule(sub, rule)[0] if len(sub) else float('nan')
    print(f"{nh:>7}{bh:>7}{nla:>6}{pg:>6}{f1_tr:>8.4f}{f1_te:>8.4f}"
          f"{reg_f1['普涨']:>8.4f}{reg_f1['震荡']:>8.4f}{reg_f1['普跌']:>8.4f}")

# ── C. 结构性诊断：为什么普跌日烂？看混淆矩阵 ──
print("\n" + "=" * 60)
print("C. 结构诊断：当前规则在普跌日的错误结构")
print("=" * 60)
from sklearn.metrics import confusion_matrix
for rg in ['普涨', '普跌']:
    sub = A[A['reg'] == rg]
    pred = sub.apply(m.classify_intention, axis=1)
    labs = ['买入', '卖出', 'T0交易']
    cm = confusion_matrix(sub['itruth'], pred, labels=labs)
    print(f"\n[{rg}] 行=真值 列=预测  {labs}")
    for i, lb in enumerate(labs):
        row = cm[i]
        tot = row.sum()
        print(f"  真{lb:<4}(n={tot:>4}): " + " ".join(f"{v:>4}" for v in row) +
              f"  召回={row[i]/tot:.2f}" if tot else "")
    # 真值分布
    print(f"  真值分布: {sub['itruth'].value_counts().to_dict()}")
    print(f"  预测分布: {pred.value_counts().to_dict()}")

# ── D. 最佳候选 vs 当前：全窗口 + 是否稳健胜出 ──
print("\n" + "=" * 60)
print("D. 决策：最佳候选(0.02/0.04/0.015/1.0) vs 当前规则")
print("=" * 60)
best = make_rule(0.02, 0.04, 0.015, 1.0)
print(f"{'口径':<12}{'当前F1':>10}{'候选F1':>10}{'增益':>9}")
for rg in ['普涨', '震荡', '普跌', '全部']:
    sub = A if rg == '全部' else A[A['reg'] == rg]
    cf = eval_rule(sub, m.classify_intention)[0]
    bf = eval_rule(sub, best)[0]
    flag = ' ✓' if bf > cf + 0.003 else (' ✗' if bf < cf - 0.003 else ' ~')
    print(f"{rg:<12}{cf:>10.4f}{bf:>10.4f}{bf-cf:>+9.4f}{flag}")
# 测试集(未见)上的净增益 —— 这才是能否上线的判据
bf_te = eval_rule(A_te, best)[0]
cf_te = eval_rule(A_te, m.classify_intention)[0]
print(f"\n关键判据 — 测试集(未见数据)净增益: {bf_te-cf_te:+.4f}")
print("决策规则: 测试集增益>+0.01 且 无行情为负 → 上线v8; 否则判负、不上线(防v6式陷阱)")


# ── E. 结构性修复：逆势资金流入买入信号（攻普跌日买入召回0.12）──
# 病根：绝对价格门槛 pct_chg>-1 在普跌日杀掉买入。改用【当日截面相对强度】：
#   全场普跌时，某股大单/三源净流入 且 相对抗跌(收盘位置高 或 涨幅居截面前列)
#   → 判买入，即便绝对涨幅为负。无未来函数：相对强度只用当日截面。
print("\n" + "=" * 60)
print("E. 结构修复：逆势资金流入买入信号")
print("=" * 60)

# 当日截面 pct_chg 分位（无未来：只用当日100只的相对位置）
A = A.copy()
A['rk_pct'] = A.groupby('transaction_date')['pct_chg'].rank(pct=True)
A_tr = A[A['transaction_date'] < split]; A_te = A[A['transaction_date'] >= split]

def make_rule_v2(net_hi=0.02, big_hi=0.04, nla=0.015, pg=1.0,
                 rev_net=0.03, rev_pos=0.6, rev_rk=0.6, rev_floor=-5.0):
    """在最佳阈值基础上，新增'逆势资金流入'买入分支。
    rev_*：逆势买入门槛——净流入达标 + (收盘位置高 或 截面相对强) + 跌幅未破底线。"""
    def rule(r):
        net, big = r['net_mf_ratio'], r['big_net_ratio']
        if r['at_up_limit'] > 0:
            return '买入'
        if r['at_dn_limit'] > 0:
            return '卖出'
        if r['ll_is_zhaban'] > 0 and r['ll_open_times'] >= 2 and r['pct_chg'] < 3:
            return '卖出'
        # 逆势资金流入买入（新）：负收益但资金逆势进 + 相对抗跌
        if (r['pct_chg'] <= -pg and r['pct_chg'] > rev_floor and
                (net > rev_net or big > rev_net or r['flow_agree_dir'] > 0) and
                (r['close_pos'] >= rev_pos or r.get('rk_pct', 0.5) >= rev_rk)):
            return '买入'
        if r['flow_agree_dir'] > 0 and net > nla and r['pct_chg'] > -pg:
            return '买入'
        if r['flow_agree_dir'] < 0 and net < -nla and r['pct_chg'] < pg:
            return '卖出'
        if (net > net_hi or big > big_hi) and r['pct_chg'] > -pg:
            return '买入'
        if (net < -net_hi or big < -big_hi) and r['pct_chg'] < pg:
            return '卖出'
        if r['ths_d5_dir'] > 0 and net > net_hi and r['pct_chg'] > 2:
            return '买入'
        if r['ths_d5_dir'] < 0 and net < -net_hi and r['pct_chg'] < -2:
            return '卖出'
        return 'T0交易'
    return rule

# 小网格调逆势分支参数（在训练集选，测试集验证）
rev_grid = list(product([0.02, 0.03, 0.04], [0.55, 0.6, 0.7], [0.6, 0.7, 0.8], [-4.0, -5.0, -6.0]))
rev_res = []
for rn, rp, rk, rf in rev_grid:
    rule = make_rule_v2(rev_net=rn, rev_pos=rp, rev_rk=rk, rev_floor=rf)
    rev_res.append((eval_rule(A_tr, rule)[0], rn, rp, rk, rf))
rev_res.sort(reverse=True)

def buy_recall(df, rule):
    pred = df.apply(rule, axis=1)
    sub = df[df['itruth'] == '买入']
    return (pred[sub.index] == '买入').mean() if len(sub) else float('nan')

print(f"{'rev_net':>8}{'pos':>6}{'rk':>6}{'floor':>7}{'trF1':>8}{'teF1':>8}{'普跌teF1':>9}{'普跌买召回':>10}")
cur_pd_recall = buy_recall(A_te[A_te['reg']=='普跌'], m.classify_intention)
for f1_tr, rn, rp, rk, rf in rev_res[:6]:
    rule = make_rule_v2(rev_net=rn, rev_pos=rp, rev_rk=rk, rev_floor=rf)
    f1_te = eval_rule(A_te, rule)[0]
    pd_te = A_te[A_te['reg'] == '普跌']
    pd_f1 = eval_rule(pd_te, rule)[0]
    pd_rec = buy_recall(pd_te, rule)
    print(f"{rn:>8}{rp:>6}{rk:>6}{rf:>7}{f1_tr:>8.4f}{f1_te:>8.4f}{pd_f1:>9.4f}{pd_rec:>10.2f}")
print(f"\n当前规则普跌日: 测试集买入召回={cur_pd_recall:.2f} (对比上面候选)")



