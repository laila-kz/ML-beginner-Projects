# ML Beginner Projects

A polished portfolio of beginner-friendly machine learning exercises converted into a cleaner, more maintainable, and presentation-ready repository.

## Overview

This repository contains three small but instructive machine learning projects covering classification and regression workflows:

- Diabetes prediction using Support Vector Machines
- House price prediction using XGBoost regression
- Rock-vs-mine detection using Logistic Regression

The repository is designed to showcase how educational notebooks/scripts can be upgraded into a production-style project structure with modular code, explicit configuration, relative paths, and documentation.

## Architecture

```text
ML-beginner-Projects/
├── README.md                     # Executive overview and repo navigation
├── requirements.txt              # Reproducible Python dependencies
├── .gitignore                    # Ignore generated files and local artifacts
├── diabetes-prediction-folder/   # SVM classification workflow
│   ├── README.md
│   ├── src/
│   ├── data/
│   └── tests/
├── house-price-folder/           # XGBoost regression workflow
│   ├── README.md
│   ├── src/
│   └── tests/
├── rock-mine-folder/             # Logistic regression workflow
│   ├── README.md
│   ├── src/
│   ├── data/
│   └── tests/
└── .venv/                        # Local environment (ignored by Git)
```

## Table of Contents

| Project | Problem | Algorithm | Key metrics (from original educational scripts) | Link |
| --- | --- | --- | --- | --- |
| Diabetes Prediction | Binary classification | SVM with linear kernel | Training accuracy: ~0.7866; Test accuracy: ~0.7727 | [diabetes-prediction-folder](diabetes-prediction-folder/README.md) |
| House Price Prediction | Regression | XGBoost Regressor | R² and MAE benchmark tracked in model evaluation | [house-price-folder](house-price-folder/README.md) |
| Rock vs Mine | Binary classification | Logistic Regression | Training accuracy: ~0.8342; Test accuracy: ~0.7619 | [rock-mine-folder](rock-mine-folder/README.md) |

## Prerequisites

- Python 3.10+
- pip or conda
- Optional: virtual environment manager such as `venv` or `conda`

## Quickstart

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Navigate to any project folder and run the module entry point:

```bash
python -m src.diabetes_prediction
python -m src.house_price
python -m src.rock_mine
```

## Repository Goals

This project intentionally balances educational clarity with professional polish by focusing on the following improvements:

- Modular pipeline design and reusable functions
- Relative paths and centralized configuration
- Structured logging and explicit error handling
- Clear documentation and reproducible environment setup
- Basic `pytest` coverage for preprocessing and utility logic

## Project Highlights

- Beginner-friendly modeling workflows with clean, maintainable Python structure
- Portable path management using `pathlib.Path`
- Project-specific documentation for dataset context and evaluation approach
- Simple reproducibility profile for local experimentation and portfolio presentation
