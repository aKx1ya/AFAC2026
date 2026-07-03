"""项目根目录与常用路径（各版本脚本共用）。"""
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR = os.path.join(ROOT, 'data')
STOCK_FILE = os.path.join(ROOT, '官方数据', '股票样本.xlsx')
OFFICIAL_SUBMIT_SAMPLE = os.path.join(ROOT, '官方数据', 'submit')
