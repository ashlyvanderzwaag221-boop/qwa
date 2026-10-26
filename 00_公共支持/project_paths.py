"""Canonical paths shared by the three responsibility folders.

Paths are derived from this file, so the project can be copied as a whole.
The numerical methods and evaluation protocol are unchanged by the folder layout.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COMMON = ROOT / '00_公共支持'
DATA_STAGE = ROOT / '01_数据处理与特征'
TRAIN_STAGE = ROOT / '02_模型训练'
EVAL_STAGE = ROOT / '03_评估与汇报'
DATA_DIR = DATA_STAGE / 'data'
FEATURES_DIR = DATA_STAGE / 'features'
DATA_RESULTS = DATA_STAGE / 'results'
TRAIN_RESULTS = TRAIN_STAGE / 'results'
MODELS_DIR = TRAIN_STAGE / 'models'
EVAL_RESULTS = EVAL_STAGE / 'results'
REPORTS_DIR = EVAL_STAGE / 'reports'
EVIDENCE_DIR = EVAL_STAGE / 'evidence'
TESTS_DIR = EVAL_STAGE / 'tests'
SUBMISSION_DIR = EVAL_STAGE / 'submission'
LOGS_DIR = COMMON / 'logs'
