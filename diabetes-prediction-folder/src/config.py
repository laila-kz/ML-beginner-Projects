"""Configuration settings for the diabetes prediction project."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
DATA_PATH = DATA_DIR / "diabetes.csv"
TARGET_COLUMN = "Outcome"
TRAIN_TEST_SPLIT = 0.2
RANDOM_STATE = 2
