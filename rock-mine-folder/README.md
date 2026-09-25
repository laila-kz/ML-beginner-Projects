# Rock vs Mine Detection

## Problem Statement

This project classifies sonar returns as either a rock or a mine based on signal energy patterns across multiple frequency bins.

## Dataset

The project uses a sonar dataset with 60 numeric features and a final categorical label column indicating `R` (rock) or `M` (mine).

## Workflow

1. Load the sonar dataset from the project data directory.
2. Inspect the target frequency and class distribution.
3. Separate features from labels.
4. Split data into training and test sets with class stratification.
5. Train a logistic regression classifier.
6. Evaluate accuracy on training and test data.
7. Make a single-sample prediction based on a sonar feature vector.

## Model

Algorithm: Logistic Regression

- Binary classification task
- Stratified train/test split
- Evaluation metric: accuracy

## Evaluation

The original notebook reported:

- Training accuracy: approximately 0.8342
- Test accuracy: approximately 0.7619

These values represent the baseline from the original beginner tutorial and can be refined through local model runs.

## Sample Prediction

The model predicts a label from the set:

- `R` for rock
- `M` for mine

## Project Structure

```text
rock-mine-folder/
├── data/
│   └── sonar_data.csv
├── src/
│   ├── config.py
│   └── rock_mine.py
├── tests/
│   └── test_rock_mine.py
├── README.md
└── notebooks/
```
