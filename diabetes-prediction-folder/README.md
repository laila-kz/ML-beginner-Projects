# Diabetes Prediction

## Problem Statement

This project predicts whether a patient is diabetic based on clinical features such as glucose level, blood pressure, BMI, insulin, and age-related health indicators.

## Dataset

The project uses the Pima Indians Diabetes dataset stored in the local data folder. It contains diagnostic measurements and a binary target column labeled `Outcome`.

- `0`: non-diabetic
- `1`: diabetic

## Workflow

1. Load the dataset from the project data directory.
2. Inspect the target distribution and feature statistics.
3. Separate features (`X`) from labels (`y`).
4. Standardize numerical features with `StandardScaler`.
5. Split into training and testing sets with stratification.
6. Train a linear SVM classifier.
7. Evaluate model performance using accuracy.
8. Run a sample prediction using a known patient profile.

## Model

Algorithm: Support Vector Machine (SVM)

- Kernel: linear
- Standardization: applied before model fitting
- Train/test split: 80/20 with stratification by target

## Evaluation

The original educational workflow reported:

- Training accuracy: approximately 0.7866
- Test accuracy: approximately 0.7727

These values serve as an instructional baseline for the pipeline and should be updated with a fresh run in a local environment.

## Sample Prediction

A sample patient vector can be passed to the prediction function to obtain a binary result:

- `0`: no diabetes predicted
- `1`: diabetes predicted

## Project Structure

```text
diabetes-prediction-folder/
├── data/
│   └── diabetes.csv
├── src/
│   ├── config.py
│   └── diabetes_prediction.py
├── tests/
│   └── test_diabetes_prediction.py
└── README.md
```
