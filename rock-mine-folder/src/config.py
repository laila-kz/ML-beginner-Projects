"""Configuration settings for the rock vs mine project."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
DATA_PATH = DATA_DIR / "sonar_data.csv"
TARGET_COLUMN = 60
TRAIN_TEST_SPLIT = 0.1
RANDOM_STATE = 1
