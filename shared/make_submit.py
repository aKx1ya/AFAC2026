"""
提交文件格式校验 + 打包工具（AFAC2026 赛题一）。

用法（在项目根目录执行）：
  python shared/make_submit.py --dir v2/out --zip v2/submit.zip
"""
import os
import re
import sys
import argparse
import zipfile
import pandas as pd

PATTERN_FILE = 'pattern_reco.csv'
RESULT_FILE = 'predict_result.csv'
PATTERN_COLUMNS = ['stock_code', 'transaction_date', 'pattern_type', 'pattern_explanation']
RESULT_COLUMNS = ['stock_code', 'transaction_date', 'capital_type', 'capital_intention']
DATE_RE = re.compile(r'^\d{8}$')
CODE_RE = re.compile(r'^(\d{6}|\d{6}\.(SH|SZ|BJ))$')


class FormatError(Exception):
    pass


def _read_csv(path):
    if not os.path.exists(path):
        raise FormatError(f"缺少文件: {path}")
    return pd.read_csv(path, dtype=str, encoding='utf-8-sig', keep_default_na=False)


def validate(sub_dir, capital_types, intentions):
    print(f"===== 校验提交文件 @ {sub_dir} =====")
    pat = _read_csv(os.path.join(sub_dir, PATTERN_FILE))
    res = _read_csv(os.path.join(sub_dir, RESULT_FILE))

    if list(pat.columns) != PATTERN_COLUMNS:
        raise FormatError(f"{PATTERN_FILE} 列错误: {list(pat.columns)}")
    if list(res.columns) != RESULT_COLUMNS:
        raise FormatError(f"{RESULT_FILE} 列错误: {list(res.columns)}")

    for df, fname in ((pat, PATTERN_FILE), (res, RESULT_FILE)):
        for c in df.columns:
            if df[c].astype(str).str.strip().eq('').any():
                raise FormatError(f"{fname} 列 '{c}' 存在空值")
        if not df['transaction_date'].map(lambda x: bool(DATE_RE.match(str(x)))).all():
            raise FormatError(f"{fname} transaction_date 非 YYYYMMDD")
        if df.duplicated(['stock_code', 'transaction_date']).any():
            raise FormatError(f"{fname} 存在重复键")

    bad_ct = set(res['capital_type']) - set(capital_types)
    if bad_ct:
        raise FormatError(f"非法 capital_type: {bad_ct}")
    bad_it = set(res['capital_intention']) - set(intentions)
    if bad_it:
        raise FormatError(f"非法 capital_intention: {bad_it}")

    kp = set(map(tuple, pat[['stock_code', 'transaction_date']].values))
    kr = set(map(tuple, res[['stock_code', 'transaction_date']].values))
    if kp != kr:
        raise FormatError("两文件键集合不一致")

    print(f"  [OK] {len(pat)} 行 | capital_type={sorted(set(res['capital_type']))}")
    print(f"  [OK] transaction_date={sorted(set(res['transaction_date']))}")
    return pat, res


def repackage(sub_dir, pat, res, encoding, zip_root, zip_path):
    pat.to_csv(os.path.join(sub_dir, PATTERN_FILE), index=False, encoding=encoding)
    res.to_csv(os.path.join(sub_dir, RESULT_FILE), index=False, encoding=encoding)
    os.makedirs(os.path.dirname(os.path.abspath(zip_path)) or '.', exist_ok=True)
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for fname in (PATTERN_FILE, RESULT_FILE):
            arc = os.path.join(zip_root, fname) if zip_root else fname
            zf.write(os.path.join(sub_dir, fname), arcname=arc)
    print(f"已打包: {zip_path} → {zipfile.ZipFile(zip_path).namelist()}")


def main():
    ap = argparse.ArgumentParser(description='AFAC2026 提交格式校验+打包')
    ap.add_argument('--dir', required=True, help='结果目录（含两份 csv）')
    ap.add_argument('--zip', required=True, help='输出 zip 路径')
    ap.add_argument('--encoding', default='utf-8-sig')
    ap.add_argument('--zip-root', default='submit')
    ap.add_argument('--capital-types', default='游资,量化,散户')
    ap.add_argument('--intentions', default='买入,卖出,T0交易')
    args = ap.parse_args()

    cts = [s.strip() for s in args.capital_types.split(',') if s.strip()]
    its = [s.strip() for s in args.intentions.split(',') if s.strip()]
    try:
        pat, res = validate(args.dir, cts, its)
    except FormatError as e:
        print(f"\n[格式校验失败] {e}")
        sys.exit(1)
    repackage(args.dir, pat, res, args.encoding, args.zip_root, args.zip)
    print("校验通过。")


if __name__ == '__main__':
    main()
